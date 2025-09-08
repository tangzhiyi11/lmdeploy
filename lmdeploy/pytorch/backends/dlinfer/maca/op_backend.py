# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import math

import torch

from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig
from lmdeploy.utils import get_logger

from ..op_backend import DlinferOpsBackend

logger = get_logger('lmdeploy')

def flashinfer_available():
    """Check if flashinfer is available."""
    # use flashinfer by default if it is installed
    # return False
    use_flashinfer = False
    try:
        import flashinfer  # noqa
        use_flashinfer = True
    except ImportError:
        logger.warning('For higher performance, please install flashinfer https://github.com/flashinfer-ai/flashinfer')
    return use_flashinfer


class FlashInferMeta:
    is_mla = None
    zero_tensor = None
    q_max_arange_tensor = None
    q_max_arange_size = 64
    mask_max_arange_tensor = None
    mask_max_arange_size = 4100
    flashinfer_decode_wrapper = None
    _lock = None

    @classmethod
    def _get_lock(cls):
        """Get thread lock for thread-safe operations."""
        if cls._lock is None:
            import threading
            cls._lock = threading.Lock()
        return cls._lock

    @classmethod
    def _get_is_mla(cls, k_head_size, v_head_size):
        """Thread-safe setter for is_mla flag."""
        if cls.is_mla is None:
            with cls._get_lock():
                # Double-check pattern to avoid race conditions
                if cls.is_mla is None:
                    cls.is_mla = (k_head_size != v_head_size)
        return cls.is_mla

    @classmethod
    def _get_zero_tensor(cls, device):
        if cls.zero_tensor is None or cls.zero_tensor.device != device:
            with cls._get_lock():
                # Double-check pattern for thread safety
                if cls.zero_tensor is None or cls.zero_tensor.device != device:
                    cls.zero_tensor = torch.tensor([0], device=device)
        return cls.zero_tensor

    @classmethod
    def _get_q_indptr_arrange(cls, device, size):
        if cls.q_max_arange_tensor is None or size > cls.q_max_arange_size:
            with cls._get_lock():
                # Double-check pattern for thread safety
                if cls.q_max_arange_tensor is None or size > cls.q_max_arange_size:
                    if size > cls.q_max_arange_size:
                        cls.q_max_arange_size = size * 2
                    cls.q_max_arange_tensor = torch.arange(0, cls.q_max_arange_size, dtype=torch.int32, device=device)
        return cls.q_max_arange_tensor[:size]

    @classmethod
    def _get_mask_arrange(cls, device, size):
        if cls.mask_max_arange_tensor is None or size > cls.mask_max_arange_size:
            with cls._get_lock():
                # Double-check pattern for thread safety
                if cls.mask_max_arange_tensor is None or size > cls.mask_max_arange_size:
                    if size > cls.mask_max_arange_size:
                        cls.mask_max_arange_size = size * 2
                    cls.mask_max_arange_tensor = torch.arange(0, cls.mask_max_arange_size, dtype=torch.int32, device=device)
        return cls.mask_max_arange_tensor[:size]

    @classmethod
    def _get_flashinfer_decode_wrapper(cls,):
        """Get flashinfer decode wrapper. TODO: Implement this method."""
        if cls.flashinfer_decode_wrapper is None:
            with cls._get_lock():
                import flashinfer
                if cls.is_mla:
                    cls.flashinfer_decode_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
                                torch.empty(128 * 1024 * 1024, dtype=torch.int8).to(0),
                                backend="auto",
                            )
                else:
                    cls.flashinfer_decode_wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
                                torch.empty(128 * 1024 * 1024, dtype=torch.int8).to(0),
                                # backend="auto",
                            )
        return cls.flashinfer_decode_wrapper

    @classmethod
    def _get_sm_scale(cls):
        """Get sm scale."""
        return 0.114721386792


class MacaOpsBackend(DlinferOpsBackend):
    """Maca layer backend."""
    total_slots = None
    enable_graph = False
    use_flashinfer = flashinfer_available()

    @staticmethod
    def get_name() -> str:
        """Backend name."""
        return 'maca'

    @classmethod
    def get_k_block_shape(
        cls,
        block_size: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> Tuple[int, ...]:
        if cls.use_flashinfer:
            return (num_heads, block_size, head_size)
        else:
            x = 16
            return (num_heads, head_size // x, block_size, x)

    @staticmethod
    def get_v_block_shape(
        block_size: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> Tuple[int, ...]:
        return (num_heads, block_size, head_size)

    @classmethod
    def update_step_context(cls, step_context):
        """Update step context."""

        def get_total_slots():
            if cls.total_slots is None:
                cls.total_slots = torch.arange(block_num * block_size,
                                               dtype=torch.long,
                                               device=step_context.block_offsets.device)
                cls.total_slots = cls.total_slots.view(block_num, block_size)
            return cls.total_slots

        kv_start_indices, attention_mask = [], []
        block_num, _, block_size, v_head_size = step_context.kv_caches[0][1].shape
        device = step_context.block_offsets.device

        is_unpaged_prefill = False
        if not step_context.is_decoding:
            is_unpaged_prefill = \
               all((step_context.q_seqlens ==
                    step_context.kv_seqlens).tolist())
        q_start_loc = torch.cat((torch.tensor([0], device=device), step_context.q_seqlens.cumsum(0))).int()
        q_seqlens = step_context.q_seqlens.int()
        kv_seqlens = step_context.kv_seqlens.int()
        max_q_seq_len = torch.max(q_seqlens).item()
        max_kv_seq_len = torch.max(kv_seqlens).item()

        if step_context.is_decoding:
            # collect kv_start_indices without using a for-loop,
            # (fill kv-cache for just ONE token during the decoding phase)
            idx = (step_context.kv_seqlens - 1) % block_size
            b_num = (step_context.kv_seqlens - 1) // block_size
            last_block = step_context.block_offsets.gather(1, b_num.view(-1, 1)).view(-1)
            kv_start_indices = (last_block * block_size + idx).reshape((-1, 1))

            if cls.use_flashinfer:
                _, _, _, k_head_size = step_context.kv_caches[0][0].shape
                is_mla = FlashInferMeta._get_is_mla(k_head_size, v_head_size)

                if not cls.enable_graph:
                    eager_decode_wrapper = FlashInferMeta._get_flashinfer_decode_wrapper()
                    page_size = block_size

                    zero_tensor = FlashInferMeta._get_zero_tensor(device)
                    tmp_cumsum = kv_seqlens.cumsum(0)
                    kv_indptr = ((torch.cat((zero_tensor, tmp_cumsum)) + page_size - 1)  // page_size).int()

                    max_blocks = step_context.block_offsets.shape[1]
                    blocks_needed = (kv_seqlens + block_size - 1) // block_size
                    mask = FlashInferMeta._get_mask_arrange(device, max_blocks)[None, :] < blocks_needed[:, None]
                    kv_indices = step_context.block_offsets[mask].int()

                    from lmdeploy.pytorch.distributed import get_tp_world_rank
                    tp_size, _ = get_tp_world_rank()
                    kv_lens = kv_seqlens
                    num_local_heads = step_context.model_config.num_attention_heads // tp_size

                    if is_mla:
                        q_indptr = FlashInferMeta._get_q_indptr_arrange(device, kv_seqlens.shape[0] + 1)
                        sm_scale = FlashInferMeta._get_sm_scale()
                        head_dim_ckv = step_context.model_config.hf_config.kv_lora_rank
                        head_dim_kpe = step_context.model_config.hf_config.qk_rope_head_dim
                        eager_decode_wrapper.plan(
                            q_indptr,
                            kv_indptr,
                            kv_indices,
                            kv_lens,
                            num_local_heads,
                            head_dim_ckv,
                            head_dim_kpe,
                            page_size,
                            False,  # causal
                            sm_scale,
                            torch.bfloat16,
                            torch.bfloat16,
                        )
                    else:
                        num_local_kv_heads = step_context.model_config.num_key_value_heads // tp_size
                        sm_scale = float(1 / math.sqrt(step_context.model_config.head_dim))
                        eager_decode_wrapper.plan(
                            indptr=kv_indptr,
                            indices=kv_indices,
                            last_page_len=idx,
                            num_qo_heads=num_local_heads,
                            num_kv_heads=num_local_kv_heads,
                            head_dim=step_context.model_config.k_head_dim,
                            page_size=page_size,
                            pos_encoding_mode='NONE',
                            window_left=-1,
                            q_data_type=torch.bfloat16,
                            kv_data_type=torch.bfloat16,
                            sm_scale=sm_scale,
                        )

        else:
            for i in range(step_context.q_start_loc.size(0)):
                q_seq_len = int(step_context.q_seqlens[i])
                kv_seq_len = int(step_context.kv_seqlens[i])
                # collect kv start indices during the prefill phase.
                history_length = kv_seq_len - q_seq_len
                total_slots = get_total_slots()
                slot_tables = total_slots[step_context.block_offsets[i]].view(-1)
                slots = slot_tables[history_length:kv_seq_len]
                kv_start_indices.append(slots)
            kv_start_indices = torch.cat(kv_start_indices)

        attn_meta_cls = cls.get_attention_metadata_cls()
        attn_metadata = attn_meta_cls(
            step_context.is_decoding,
            step_context.block_offsets.int(),
            q_start_loc=q_start_loc,
            q_seqlens=q_seqlens,
            kv_seqlens=kv_seqlens,
            kv_start_indices=kv_start_indices,
            block_size=block_size,
            attention_mask=attention_mask,
            is_unpaged_prefill=is_unpaged_prefill,
            max_q_seq_len=max_q_seq_len,
            max_kv_seq_len=max_kv_seq_len,
        )

        if step_context.is_decoding and cls.use_flashinfer and not cls.enable_graph:
            attn_metadata.flashinfer_wrapper = eager_decode_wrapper

        step_context.attn_metadata = attn_metadata
        return step_context

    @staticmethod
    def build_graph_runner(model: torch.nn.Module, model_config: ModelConfig, cache_config: CacheConfig,
                           backend_config: BackendConfig, device: torch.device):
        """Build graph runner."""
        from lmdeploy.pytorch.backends.dlinfer.maca.graph_runner import MacaGraphRunner
        maca_graph_runner = MacaGraphRunner(model, model_config, cache_config, backend_config, device)
        MacaOpsBackend.enable_graph = maca_graph_runner.enable_graph_mode
        return maca_graph_runner

    @staticmethod
    def support_ray():
        """Support ray."""
        return True
