# Copyright (c) OpenMMLab. All rights reserved.

import inspect
import os
import time

import torch
from torch.profiler import record_function

from lmdeploy.utils import get_logger

from ..backends import get_backend
from ..config import BackendConfig, CacheConfig, MiscConfig, ModelConfig, SpecDecodeConfig
from ..distributed import get_dist_manager
from ..engine.cache_engine import CacheEngine
from ..engine.logits_process import FusedLogitsProcessor, SamplingInputs, _torch_topk
from ..engine.model_agent.agent import BatchedLogProbs
from ..model_inputs import ModelInputs
from ..strategies.ar_spec.model_agent import ARSpecExtraInputs
from ..strategies.base.model_agent import ExtraInputs
from .base import BaseSpecModelAgent
from .proposers.base import build_specdecode_proposer
from .reject_sampler import RejectionSampler

logger = get_logger('lmdeploy')


def _expand_sampling_inputs(sampling_inputs: SamplingInputs, num_tokens: int) -> SamplingInputs:
    """Expand per-batch SamplingInputs to per-token by repeating each batch
    element num_tokens times via repeat_interleave.

    Args:
        sampling_inputs: SamplingInputs with batch_size elements.
        num_tokens: Number of tokens per batch element.

    Returns:
        New SamplingInputs with batch_size * num_tokens elements.
    """
    if num_tokens == 1:
        return sampling_inputs

    from dataclasses import fields
    out_dict = {}
    for f in fields(sampling_inputs):
        k = f.name
        v = getattr(sampling_inputs, k)
        if isinstance(v, torch.Tensor):
            v = v.repeat_interleave(num_tokens, dim=0)
            if k == 'random_offsets':
                # Each token position needs a different offset for
                # reproducible but distinct random sampling
                arange = torch.arange(num_tokens, device=v.device)
                v = v + arange.repeat(sampling_inputs.batch_size)
        out_dict[k] = v

    out_dict['batch_size'] = sampling_inputs.batch_size * num_tokens
    return SamplingInputs(**out_dict)


def _slice_sampling_inputs(sampling_inputs: SamplingInputs, num_tokens: int, is_last: bool = True) -> SamplingInputs:
    """Slice expanded SamplingInputs.

    After _expand_sampling_inputs repeats each batch element num_tokens
    times, this function extracts a subset per batch element.

    Args:
        sampling_inputs: Expanded SamplingInputs with
            batch_size * num_tokens elements.
        num_tokens: Number of tokens per batch element.
        is_last: If True (default), take the last token per batch element
            (for bonus token sampling), returning batch_size elements.
            If False, take the first num_tokens-1 tokens per batch element
            (all except the last), returning
            batch_size * (num_tokens - 1) elements.

    Returns:
        Sliced SamplingInputs.
    """
    if num_tokens == 1:
        return sampling_inputs

    from dataclasses import fields

    batch_size = sampling_inputs.batch_size // num_tokens
    out_dict = {}
    for f in fields(sampling_inputs):
        k = f.name
        v = getattr(sampling_inputs, k)
        if isinstance(v, torch.Tensor):
            if is_last:
                v = v[num_tokens - 1::num_tokens]
            else:
                shape = v.shape
                v = v.view(batch_size, num_tokens, *shape[1:])
                v = v[:, :-1].reshape(batch_size * (num_tokens - 1), *shape[1:])
        out_dict[k] = v

    if is_last:
        out_dict['batch_size'] = batch_size
    else:
        out_dict['batch_size'] = batch_size * (num_tokens - 1)
    return SamplingInputs(**out_dict)


class SpecModelAgent(BaseSpecModelAgent):
    """Speculative model agent."""

    def __init__(
        self,
        specdecode_config: SpecDecodeConfig,
        backend_config: BackendConfig,
        inputs_strategy,
        agent_strategy,
        misc_config: MiscConfig,
        device: str = 'cuda',
    ):
        super().__init__(specdecode_config, enable=True)

        self.backend_config = backend_config
        self.device = device
        self.cache_engine = None
        self.inputs_strategy = inputs_strategy
        self.agent_strategy = agent_strategy
        self.misc_config = misc_config
        # Non-TP draft model: each rank holds full weights and runs
        # independently — no collective ops, no broadcast needed.
        self.is_draft_tp = False
        self.rejection_sampler = RejectionSampler()
        self.proposer = build_specdecode_proposer(specdecode_config, device=device)
        self.method = specdecode_config.method
        self.model_config = specdecode_config.model_config
        self.cache_config = specdecode_config.cache_config
        self._skip_warmup = False
        if self.model_config is not None and self.cache_config is not None:
            self.model_config.block_size = self.cache_config.block_size

        # make dummy meta
        self.make_dummy_meta = self.inputs_strategy.create_make_dummy_meta(self.model_config)
        # for long context carry-over in chunked decoding
        self._prev_chunk_last = {}

    def set_cache_config(self, cache_config: CacheConfig):
        """Set all cache config."""
        self.cache_config = cache_config
        if self.model_config is not None and self.cache_config is not None:
            self.model_config.block_size = self.cache_config.block_size
            self.specdecode_config.model_config = self.model_config

    def set_model_config(self, model_config: ModelConfig):
        """Set model config."""
        self.model_config = model_config
        if model_config is not None:
            if self.cache_config is not None:
                self.model_config.block_size = self.cache_config.block_size
            self.specdecode_config.model_config = self.model_config
            # make dummy meta
            self.make_dummy_meta = self.inputs_strategy.create_make_dummy_meta(self.model_config)

    def build_model(self, empty_init: bool, target_model=None, build_model_ctx=None):
        """Build draft model."""
        self.proposer.build_model(empty_init, target_model=target_model, build_model_ctx=build_model_ctx)

    def build_graph_runner(self):
        """Build graph runner."""
        self._skip_warmup = False
        backend = get_backend()
        self.proposer.model = backend.build_graph_runner(self.proposer.model,
                                                         model_config=self.model_config,
                                                         cache_config=self.cache_config,
                                                         backend_config=self.backend_config,
                                                         device=self.device)

    def build_cache_engine(self, cache_stream: torch.cuda.Stream):
        """Build cache engine."""
        if self.cache_config is not None:
            # Draft model runs without TP (is_tp=False); each rank holds a full
            # copy so the cache engine uses world_size=1 regardless of global TP.
            self.cache_engine = CacheEngine(self.cache_config,
                                            self.model_config,
                                            rank=0,
                                            tp_rank=0,
                                            world_size=1,
                                            cache_stream=cache_stream)

    def _prepare_inputs_from_main(self, model_inputs: ModelInputs, extra_inputs: ExtraInputs):
        """Update inputs from main model inputs."""
        spec_debug = os.environ.get('LMDEPLOY_SPEC_DEBUG', '0') == '1'
        spec_debug_steps = int(os.environ.get('LMDEPLOY_SPEC_DEBUG_STEPS', '4'))
        if not hasattr(self, '_spec_debug_prepare_step'):
            self._spec_debug_prepare_step = 0

        next_token_ids = extra_inputs.next_token_ids
        last_token_indices = extra_inputs.last_token_indices
        # create new inputs for draft model (offset by 1 from main model)
        target_hidden_states = extra_inputs.target_hidden_states
        target_position_ids = extra_inputs.target_position_ids
        target_inputs_embeds = extra_inputs.target_inputs_embeds
        mrope_pos_ids = model_inputs.mrope_pos_ids
        seq_length = model_inputs.seq_length
        max_q_seqlen = model_inputs.max_q_seqlen
        max_kv_seqlen = model_inputs.max_kv_seqlen
        sum_kv_seqlen = model_inputs.sum_kv_seqlen
        history_lengths = model_inputs.history_lengths.clone()

        if not model_inputs.is_chunk:
            # Case A: non-chunked — shift left by 1, place next_token at end
            if spec_debug and self._spec_debug_prepare_step < spec_debug_steps:
                dist_ctx = get_dist_manager().current_context()
                rank = 0 if dist_ctx is None else dist_ctx.rank
                if rank == 0:
                    print(
                        f'[SPEC_DEBUG_PREP][step={self._spec_debug_prepare_step}] '
                        f'is_decoding={model_inputs.is_decoding} '
                        f'input_ids_shape={tuple(model_inputs.input_ids.shape)} '
                        f'seq_length={model_inputs.seq_length.tolist()} '
                        f'last_token_indices={last_token_indices.tolist() if last_token_indices is not None else None} '
                        f'next_token_ids={next_token_ids.tolist() if next_token_ids is not None else None} '
                        f'mrope_pos_ids_shape={None if mrope_pos_ids is None else tuple(mrope_pos_ids.shape)}',
                        flush=True,
                    )
            input_ids = model_inputs.input_ids.clone()
            # In spec decoding, the draft model input window has a fixed width
            # (num_spec_tokens + 1). We always place the sampled next token at
            # the last position, regardless of how many speculative tokens were
            # rejected this step.
            num_seqs = seq_length.size(0)
            total_tokens = input_ids.size(-1)
            window_size = total_tokens // num_seqs
            if num_seqs == 1:
                # Single-sequence: simple global shift works
                input_ids[:, :-1] = model_inputs.input_ids[:, 1:]
                input_ids[:, -1] = next_token_ids
            elif not model_inputs.is_decoding:
                # Prefill multi-batch: per-sequence shift-left preserving total
                # token count so input_ids shape matches target_hidden_states.
                # For each sequence [t0, t1, ..., tL-1] → [t1, ..., tL-1, next].
                seq_lengths = model_inputs.seq_length
                starts = torch.zeros_like(seq_lengths)
                starts[1:] = seq_lengths[:-1].cumsum(0)
                ends = seq_lengths.cumsum(0)
                shifted_parts = []
                for i in range(num_seqs):
                    s, e = starts[i].item(), ends[i].item()
                    shifted_parts.append(torch.cat([
                        model_inputs.input_ids[0, s + 1:e],
                        next_token_ids[i:i + 1],
                    ]))
                input_ids = torch.cat(shifted_parts).unsqueeze(0)
            else:
                # Decode multi-batch: window-based rearrangement.
                # Each seq has window_size = num_spec_tokens + 1 tokens.
                # Verify layout: [next_seq0, d0_seq0, next_seq1, d0_seq1, ...]
                # Draft layout:  [d0_seq0, new_seq0, d0_seq1, new_seq1, ...]
                d0_old = model_inputs.input_ids[0,
                            torch.arange(num_seqs, device=input_ids.device) * window_size + 1]
                draft_tokens = torch.stack([d0_old, next_token_ids], dim=1).flatten()
                input_ids = draft_tokens.unsqueeze(0)

            if target_inputs_embeds is not None:
                next_token_embeds = self.proposer.embed_input_ids(next_token_ids)
                if num_seqs == 1:
                    input_embeds = target_inputs_embeds.clone()
                    input_embeds[:, :-1, :] = target_inputs_embeds[:, 1:, :]
                    input_embeds[:, -1, :] = next_token_embeds
                    target_inputs_embeds = input_embeds
                elif not model_inputs.is_decoding:
                    # Prefill multi-batch: per-sequence shift for embeds
                    seq_lengths = model_inputs.seq_length
                    starts = torch.zeros_like(seq_lengths)
                    starts[1:] = seq_lengths[:-1].cumsum(0)
                    ends = seq_lengths.cumsum(0)
                    shifted_embed_parts = []
                    for i in range(num_seqs):
                        s, e = starts[i].item(), ends[i].item()
                        shifted_embed_parts.append(torch.cat([
                            target_inputs_embeds[0, s + 1:e],
                            next_token_embeds[i:i + 1].unsqueeze(0),
                        ], dim=0))
                    target_inputs_embeds = torch.cat(shifted_embed_parts).unsqueeze(0)
                else:
                    # Decode multi-batch: extract d0 embeds, interleave with next_token_embeds
                    d0_embeds = target_inputs_embeds[0,
                                torch.arange(num_seqs, device=target_inputs_embeds.device) * window_size + 1]
                    draft_embeds = torch.stack([d0_embeds, next_token_embeds], dim=1).flatten(0, 1)
                    target_inputs_embeds = draft_embeds.unsqueeze(0)

        else:
            if model_inputs.is_first_chunk:
                # Case B: first chunk — skip first token, save last for next chunk
                input_ids = model_inputs.input_ids[:, 1:]
                seq_length = model_inputs.seq_length - 1
                max_q_seqlen = model_inputs.max_q_seqlen - 1
                max_kv_seqlen = model_inputs.max_kv_seqlen - 1
                sum_kv_seqlen = model_inputs.sum_kv_seqlen - 1

                target_hidden_states = self._prepare_long_context_chunk_save_last('hidden_states', target_hidden_states)
                if target_position_ids is not None:
                    target_position_ids = self._prepare_long_context_chunk_save_last(
                        'position_ids', target_position_ids)
                if target_inputs_embeds is not None:
                    target_inputs_embeds = target_inputs_embeds[:, 1:]
                if mrope_pos_ids is not None:
                    mrope_pos_ids = self._prepare_long_context_chunk_save_last('mrope_pos_ids', mrope_pos_ids)

            elif model_inputs.is_last_chunk:
                # Case C: last chunk — prepend saved last, append next_token
                seq_length = model_inputs.seq_length + 1
                max_q_seqlen = model_inputs.max_q_seqlen + 1
                last_token_indices = last_token_indices + 1
                max_kv_seqlen = model_inputs.max_kv_seqlen - 1
                sum_kv_seqlen = model_inputs.sum_kv_seqlen - 1
                history_lengths = model_inputs.history_lengths - 1
                input_ids = torch.cat([model_inputs.input_ids, next_token_ids.unsqueeze(0)], dim=-1)

                target_hidden_states = self._prepare_long_context_chunk_prepend_saved('hidden_states',
                                                                                      target_hidden_states,
                                                                                      save_last=False)
                if target_position_ids is not None:
                    target_position_ids = self._prepare_long_context_chunk_prepend_saved('position_ids',
                                                                                         target_position_ids,
                                                                                         save_last=False)
                if target_inputs_embeds is not None:
                    next_token_embeds = self.proposer.embed_input_ids(next_token_ids)[None]
                    target_inputs_embeds = torch.cat(
                        [target_inputs_embeds, next_token_embeds], dim=1)
                if mrope_pos_ids is not None:
                    mrope_pos_ids = self._prepare_long_context_chunk_prepend_saved('mrope_pos_ids',
                                                                                   mrope_pos_ids,
                                                                                   save_last=False)

                # clear cross-chunk state
                self._prev_chunk_last.clear()
            else:
                # Case D: middle chunk — prepend saved last, save current last
                input_ids = model_inputs.input_ids
                max_kv_seqlen = model_inputs.max_kv_seqlen - 1
                sum_kv_seqlen = model_inputs.sum_kv_seqlen - 1
                history_lengths = model_inputs.history_lengths - 1

                target_hidden_states = self._prepare_long_context_chunk_prepend_saved(
                    'hidden_states', target_hidden_states)
                if target_position_ids is not None:
                    target_position_ids = self._prepare_long_context_chunk_prepend_saved(
                        'position_ids', target_position_ids)
                if mrope_pos_ids is not None:
                    mrope_pos_ids = self._prepare_long_context_chunk_prepend_saved('mrope_pos_ids', mrope_pos_ids)

        new_model_inputs = ModelInputs(
            input_ids=input_ids,
            seq_length=seq_length,
            max_kv_seqlen=max_kv_seqlen,
            max_q_seqlen=max_q_seqlen,
            sum_kv_seqlen=sum_kv_seqlen,
            history_lengths=history_lengths,
            block_offsets=model_inputs.block_offsets,
            num_ignored_history=model_inputs.num_ignored_history,
            is_decoding=model_inputs.is_decoding,
            target_hidden_states=target_hidden_states,
            target_position_ids=target_position_ids,
            target_inputs_embeds=target_inputs_embeds,
            mrope_pos_ids=mrope_pos_ids,
            is_chunk=model_inputs.is_chunk,
            is_first_chunk=model_inputs.is_first_chunk,
            is_last_chunk=model_inputs.is_last_chunk,
        )

        new_extra_inputs = extra_inputs.clone(
            target_hidden_states=None,
            target_inputs_embeds=None,
            target_position_ids=None,
            last_token_indices=last_token_indices,
        )
        if spec_debug and self._spec_debug_prepare_step < spec_debug_steps:
            dist_ctx = get_dist_manager().current_context()
            rank = 0 if dist_ctx is None else dist_ctx.rank
            if rank == 0:
                ids_preview = new_model_inputs.input_ids[0, :min(new_model_inputs.input_ids.shape[1], 16)].tolist()
                ths_shape = tuple(target_hidden_states.shape) if target_hidden_states is not None else None
                tie_shape = tuple(target_inputs_embeds.shape) if target_inputs_embeds is not None else None
                print(
                    f'[SPEC_DEBUG_PREP][step={self._spec_debug_prepare_step}] '
                    f'new_input_ids_preview={ids_preview} '
                    f'input_ids_shape={tuple(new_model_inputs.input_ids.shape)} '
                    f'target_hidden_states_shape={ths_shape} '
                    f'target_inputs_embeds_shape={tie_shape} '
                    f'is_decoding={model_inputs.is_decoding} '
                    f'seq_length={seq_length.tolist()}',
                    flush=True,
                )
        self._spec_debug_prepare_step += 1
        return new_model_inputs, new_extra_inputs

    def _prepare_long_context_chunk_save_last(self, key, tensor):
        """Save the last entry of a tensor for cross-chunk carry-over."""
        self._prev_chunk_last[key] = tensor[:, -1:]
        return tensor[:, :-1]

    def _prepare_long_context_chunk_prepend_saved(self, key, tensor, save_last=True):
        """Prepend saved last entry from previous chunk."""
        saved = self._prev_chunk_last[key]
        if save_last:
            self._prev_chunk_last[key] = tensor[:, -1:]
            tensor = tensor[:, :-1]
        else:
            self._prev_chunk_last.pop(key, None)
        return torch.cat([saved, tensor], dim=1)

    async def async_sampling_logits(self, target_logits: torch.Tensor, sampling_inputs: SamplingInputs):
        """Process target logits and sample bonus token using
        FusedLogitsProcessor.

        Args:
            target_logits: [batch_size, num_tokens, vocab_size]
                num_tokens = 1 + num_spec_tokens (decoding) or 1 (prefill)
            sampling_inputs: SamplingInputs — already expanded by
                make_sampling_inputs to batch_size * (num_spec_tokens + 1)

        Returns:
            processed_logits: [batch_size, num_tokens, vocab_size]
            next_token_ids: [batch_size] — sampled from the bonus (last) position
            logprobs: BatchedLogProbs or None
        """
        with record_function('spec_sampling_logits'):
            batch_size, num_tokens, vocab_size = target_logits.shape
            inv_debug = os.environ.get("LMDEPLOY_SPEC_INVARIANT_DEBUG", "0") == "1"
            inv_steps = int(os.environ.get("LMDEPLOY_SPEC_INVARIANT_STEPS", "2"))
            if inv_debug and not hasattr(self, "_inv_sampling_step"):
                self._inv_sampling_step = 0

            # Reshape to 2D: [batch * num_tokens, vocab]
            flat_logits = target_logits.reshape(-1, vocab_size)

            # TODO: guided decoding not supported yet for spec decoding
            # sampling_inputs is already expanded to batch_size * num_tokens
            logits_processor = FusedLogitsProcessor(
                sampling_inputs,
                logprobs_mode=self.misc_config.logprobs_mode,
            )
            processed_logits, raw_logprobs = await logits_processor(flat_logits)

            # Slice bonus (last) position logits for each batch element
            bonus_logits = processed_logits[num_tokens - 1::num_tokens]  # [batch_size, vocab]
            if inv_debug and self._inv_sampling_step < inv_steps:
                dist_ctx = get_dist_manager().current_context()
                rank = 0 if dist_ctx is None else dist_ctx.rank
                if rank == 0:
                    try:
                        # Show which flat indices correspond to batch0 columns.
                        idxs = list(range(0, num_tokens))
                        bonus_flat_idx = num_tokens - 1
                        print(
                            f"[INV][sampling_logits][step={self._inv_sampling_step}] "
                            f"target_logits_shape={tuple(target_logits.shape)} "
                            f"flat_logits_shape={tuple(flat_logits.shape)} "
                            f"batch0_flat_cols={idxs} bonus_flat_col={bonus_flat_idx}",
                            flush=True,
                        )
                    except Exception:
                        pass

            # Create a per-batch processor for bonus token sampling
            # by slicing the expanded sampling_inputs back to batch_size
            bonus_sampling_inputs = _slice_sampling_inputs(sampling_inputs, num_tokens)
            bonus_processor = FusedLogitsProcessor(
                bonus_sampling_inputs,
                logprobs_mode=self.misc_config.logprobs_mode,
            )
            # Sample next token from bonus position
            next_token_ids = bonus_processor.sampling(bonus_logits)  # [batch_size]

            # Reshape back to 3D
            processed_logits = processed_logits.view(batch_size, num_tokens, vocab_size)
            if inv_debug and self._inv_sampling_step < inv_steps:
                dist_ctx = get_dist_manager().current_context()
                rank = 0 if dist_ctx is None else dist_ctx.rank
                if rank == 0:
                    try:
                        # Argmax of bonus column for batch0, purely for alignment sanity.
                        bonus_argmax = int(processed_logits[0, -1].argmax(dim=-1).item())
                        print(
                            f"[INV][sampling_logits][step={self._inv_sampling_step}] "
                            f"batch0_bonus_argmax={bonus_argmax}",
                            flush=True,
                        )
                    except Exception:
                        pass
                self._inv_sampling_step += 1

        return processed_logits, next_token_ids, raw_logprobs

    async def _rejection_sampling(self, model_inputs: 'ModelInputs', extra_inputs: ARSpecExtraInputs,
                                  sampling_inputs: SamplingInputs):
        """Do rejection sampling."""
        spec_debug = os.environ.get('LMDEPLOY_SPEC_DEBUG', '0') == '1'
        spec_debug_steps = int(os.environ.get('LMDEPLOY_SPEC_DEBUG_STEPS', '4'))
        if not hasattr(self, '_spec_debug_step'):
            self._spec_debug_step = 0

        @torch.inference_mode()
        def __compute_logprobs(raw_logprobs: torch.Tensor, token_ids: torch.LongTensor,
                               sampling_inputs: SamplingInputs):
            """Compute logprobs."""
            if raw_logprobs is None or sampling_inputs.max_num_logprobs <= 0:
                return None

            indices = token_ids.flatten().unsqueeze(-1)
            clamped_indices = indices.clamp_min(0)
            logprobs = raw_logprobs.gather(-1, clamped_indices)
            num_logprobs = sampling_inputs.max_num_logprobs
            topk_logprobs, topk_indices = _torch_topk(raw_logprobs, num_logprobs, dim=-1)
            logprobs = torch.cat([logprobs, topk_logprobs], dim=-1)
            indices = torch.cat([indices, topk_indices], dim=-1).to(torch.int32)
            output_logprobs = BatchedLogProbs(
                vals=logprobs,
                indices=indices,
            )
            return output_logprobs

        # Process target_logits via FusedLogitsProcessor for BOTH prefill and decoding
        target_logits = extra_inputs.target_logits
        num_tokens = target_logits.shape[1]
        vocab_size = target_logits.shape[-1]
        expanded_sampling_inputs = _expand_sampling_inputs(sampling_inputs, num_tokens)
        processed_logits, next_token_ids, raw_logprobs = await self.async_sampling_logits(
            target_logits, expanded_sampling_inputs)

        num_rejected_tokens = torch.zeros_like(model_inputs.seq_length)
        output_token_ids = next_token_ids.unsqueeze(-1)
        last_token_indices = model_inputs.seq_length.cumsum(0) - 1

        if model_inputs.is_decoding:
            # Rejection sampling on processed logits (exclude bonus position)
            target_logits = processed_logits[:, :-1].contiguous()  # [batch, num_spec, vocab]
            num_tokens = self.num_spec_tokens + 1
            batch_sampling_inputs = _slice_sampling_inputs(expanded_sampling_inputs, num_tokens, is_last=False)
            # Process draft logits with the same sampling parameters to get q(x).
            draft_probs = None
            if extra_inputs.output_draft_logits is not None:
                flat_draft = extra_inputs.output_draft_logits.reshape(-1, vocab_size)
                draft_processor = FusedLogitsProcessor(
                    batch_sampling_inputs,
                    logprobs_mode=self.misc_config.logprobs_mode,
                )
                processed_draft_flat, _ = await draft_processor(flat_draft)
                batch_size = target_logits.shape[0]
                processed_draft = processed_draft_flat.view(batch_size, -1, vocab_size)
                draft_probs = processed_draft.softmax(dim=-1, dtype=torch.float32).contiguous()
            output_token_ids, num_rejected_tokens, next_token_ids = self.rejection_sampler(
                target_logits,
                extra_inputs.output_draft_token_ids,
                next_token_ids,
                sampling_inputs=batch_sampling_inputs,
                draft_probs=draft_probs,
            )
            # update last token indices
            last_token_indices = last_token_indices - num_rejected_tokens

        if spec_debug and self._spec_debug_step < spec_debug_steps:
            dist_ctx = get_dist_manager().current_context()
            rank = 0 if dist_ctx is None else dist_ctx.rank
            if rank == 0:
                print(
                    f'[SPEC_DEBUG][step={self._spec_debug_step}] '
                    f'is_decoding={model_inputs.is_decoding} '
                    f'seq_length={model_inputs.seq_length.tolist()} '
                    f'last_token_indices={last_token_indices.tolist()} '
                    f'num_rejected_tokens={num_rejected_tokens.tolist()} '
                    f'next_token_ids={next_token_ids.tolist()} '
                    f'output_token_ids={output_token_ids.tolist()}',
                    flush=True,
                )

        logprobs = __compute_logprobs(raw_logprobs, output_token_ids, sampling_inputs)

        new_extra_inputs = extra_inputs.clone(
            next_token_ids=next_token_ids,
            last_token_indices=last_token_indices,
            num_rejected_tokens=num_rejected_tokens,
            output_token_ids=output_token_ids,
            target_logits=None,  # clear for next step
            logprobs=logprobs,
        )
        self._spec_debug_step += 1
        return new_extra_inputs

    def _forward_impl(self, inputs: ModelInputs):
        """Forward impl."""
        return self.proposer._forward(inputs, cache_engine=self.cache_engine)

    async def _async_model_forward(self, inputs: ModelInputs, extra_inputs: ARSpecExtraInputs,
                                   sampling_inputs: SamplingInputs):
        """Model forward.

        Args:
            inputs (dict): The input data comes from _make_inputs.
        """
        t_fwd0 = time.perf_counter()
        outputs = self._forward_impl(inputs)
        t_fwd1 = time.perf_counter()
        print(f'[DRAFT_FWD] 1st forward: {1000*(t_fwd1-t_fwd0):.1f}ms', flush=True)
        if inputs.is_chunk and not inputs.is_last_chunk:
            # create dummy draft tokens
            batch_size = inputs.seq_length.size(0)
            output_draft_ids = inputs.input_ids.new_zeros(batch_size, self.num_spec_tokens)
        else:
            loop_count = self.num_spec_tokens - 1
            draft_token_ids, draft_logits, model_metas, target_hidden_states = self.proposer.get_outputs(
                outputs, inputs, extra_inputs)
            if os.environ.get('LMDEPLOY_SPEC_DEBUG', '0') == '1':
                dist_ctx = get_dist_manager().current_context()
                rank = 0 if dist_ctx is None else dist_ctx.rank
                if rank == 0:
                    try:
                        print(f'[SPEC_DEBUG_DRAFT] draft_token_ids[0]={draft_token_ids[0].tolist()}', flush=True)
                    except Exception:
                        print(f'[SPEC_DEBUG_DRAFT] draft_token_ids_shape={tuple(draft_token_ids.shape)}', flush=True)
            draft_tokens_li = [draft_token_ids]
            draft_logits_li = [draft_logits.unsqueeze(1)]
            if loop_count > 0:
                inputs = self.proposer.update_inputs_decoding(inputs, extra_inputs, draft_token_ids.transpose(0, 1),
                                                              target_hidden_states, model_metas)
                # set last_token_indices to None for decoding
                extra_inputs.last_token_indices = None

                for loop_idx in range(loop_count):
                    outputs = self._forward_impl(inputs)
                    draft_token_ids, draft_logits, model_metas, target_hidden_states = self.proposer.get_outputs(
                        outputs, inputs)
                    draft_tokens_li.append(draft_token_ids)
                    draft_logits_li.append(draft_logits.unsqueeze(1))
                    if loop_idx < loop_count - 1:
                        step_seqlens = inputs.seq_length.new_ones(inputs.seq_length.size(0))
                        inputs = inputs.step(draft_token_ids.transpose(0, 1), step_seqlens)
                        inputs.model_metas = model_metas
                        inputs.target_hidden_states = target_hidden_states
                        if inputs.target_position_ids is not None:
                            inputs.target_position_ids += 1
                        if inputs.mrope_pos_ids is not None:
                            inputs.mrope_pos_ids += 1

            output_draft_ids = torch.cat(draft_tokens_li, dim=-1)
            output_draft_logits = torch.cat(draft_logits_li, dim=1)

        # create new extra inputs
        extra_inputs = ARSpecExtraInputs(
            output_draft_token_ids=output_draft_ids,
            output_draft_logits=output_draft_logits if not (inputs.is_chunk and not inputs.is_last_chunk) else None,
            next_token_ids=extra_inputs.next_token_ids,
            num_rejected_tokens=extra_inputs.num_rejected_tokens,
            output_token_ids=extra_inputs.output_token_ids,
            logprobs=extra_inputs.logprobs,
        )
        return extra_inputs

    def _sync_draft_runtime_inputs(self, model_inputs: ModelInputs,
                                   extra_inputs: ARSpecExtraInputs) -> ARSpecExtraInputs:
        """Share leader rejection outputs before all-rank draft forward."""
        dist_ctx = get_dist_manager().current_context()
        if dist_ctx is None or dist_ctx.dist_config.attn_tp <= 1:
            return extra_inputs.ensure_draft_runtime_inputs(model_inputs)

        # Non-TP draft model: each rank independently computes the same
        # rejection-sampling results from identical target_logits, so no
        # broadcast is needed.  Skipping the broadcast eliminates a
        # synchronisation barrier that would otherwise stall all ranks.
        if not self.is_draft_tp:
            return extra_inputs.ensure_draft_runtime_inputs(model_inputs)

        sync_fn = getattr(self.agent_strategy, 'sync_draft_runtime_inputs', None)
        if sync_fn is None:
            return extra_inputs.ensure_draft_runtime_inputs(model_inputs)
        return sync_fn(extra_inputs, model_inputs, dist_ctx)

    async def run_rejection_sampling(self,
                                     model_inputs: ModelInputs,
                                     extra_inputs: ARSpecExtraInputs,
                                     sampling_inputs: SamplingInputs,
                                     enabled: bool) -> ARSpecExtraInputs:
        """Run leader-only rejection sampling and share its outputs."""
        if enabled:
            extra_inputs = await self._rejection_sampling(model_inputs, extra_inputs, sampling_inputs)
        else:
            extra_inputs = extra_inputs.clone(next_token_ids=None,
                                              last_token_indices=None,
                                              num_rejected_tokens=None,
                                              output_token_ids=None,
                                              logprobs=None)
        return self._sync_draft_runtime_inputs(model_inputs, extra_inputs)

    async def run_draft_forward(self,
                                model_inputs: ModelInputs,
                                extra_inputs: ARSpecExtraInputs,
                                sampling_inputs: SamplingInputs):
        """Run draft forward on every TP rank after sync."""
        draft_model_inputs, draft_extra_inputs = self._prepare_inputs_from_main(model_inputs, extra_inputs)
        return await self._async_model_forward(draft_model_inputs, draft_extra_inputs, sampling_inputs)

    async def async_model_forward(
        self,
        model_inputs: ModelInputs,
        extra_inputs: ExtraInputs,
        sampling_inputs: SamplingInputs,
        do_rejection_sampling: bool = True,
    ):
        """Draft model forward."""
        t0 = time.perf_counter()
        draft_extra_inputs = await self.run_rejection_sampling(model_inputs,
                                                               extra_inputs,
                                                               sampling_inputs,
                                                               enabled=do_rejection_sampling)
        t1 = time.perf_counter()
        result = await self.run_draft_forward(model_inputs, draft_extra_inputs, sampling_inputs)
        t2 = time.perf_counter()
        print(f'[SPEC_PERF] rejection={1000*(t1-t0):.1f}ms  draft_fwd={1000*(t2-t1):.1f}ms  '
              f'total={1000*(t2-t0):.1f}ms  is_decoding={model_inputs.is_decoding}', flush=True)
        return result

    def _make_warmup_inputs(
        self,
        batch_size: int,
        is_decoding: bool,
        target_hidden_size: int,
        max_q_seqlen: int = 1,
    ) -> ModelInputs:
        """Create warmup inputs for draft model across old/new strategy APIs."""
        make_dummy = self.inputs_strategy.make_dummy
        sig = inspect.signature(make_dummy)
        kwargs = dict(
            batch_size=batch_size,
            is_decoding=is_decoding,
            device='cuda',
            vocab_size=self.model_config.vocab_size,
            max_q_seqlen=max_q_seqlen,
            target_hidden_size=target_hidden_size,
            target_dtype=self.model_config.dtype,
            meta=self.make_dummy_meta,
        )
        if 'target_hidden_size' in sig.parameters:
            return make_dummy(**kwargs)

        logger.warning(
            'Inputs strategy %s does not support draft warmup kwargs; '
            'falling back to ARSpecModelInputsStrategy.',
            type(self.inputs_strategy).__name__,
        )
        from ..strategies.ar_spec.model_inputs import ARSpecModelInputsStrategy

        fallback = ARSpecModelInputsStrategy(self.num_spec_tokens)
        return fallback.make_dummy(**kwargs)

    def warmup(self, max_batches: int, target_model_config: ModelConfig):
        """warmup."""
        if self._skip_warmup:
            logger.info('Skip draft warmup for eager qwen3_5_mtp runner on Ascend.')
            return

        target_hidden_size = self.proposer.get_target_hidden_size(target_model_config)

        # warmup prefill
        inputs = self._make_warmup_inputs(
            max_batches,
            is_decoding=False,
            target_hidden_size=target_hidden_size,
        )

        self._forward_impl(inputs)

        capture_batch_sizes = self.proposer.model.get_capture_batch_sizes()
        capture_batch_sizes = sorted(capture_batch_sizes, reverse=True)

        for batch_size in capture_batch_sizes:
            # decode with num_spec_tokens + 1 per seq
            # Multi-token shapes bypass graph capture in AscendGraphRunner
            # and run in eager mode.
            inputs = self._make_warmup_inputs(
                batch_size,
                is_decoding=True,
                max_q_seqlen=self.num_spec_tokens + 1,
                target_hidden_size=target_hidden_size,
            )
            self._forward_impl(inputs)
            # decode 1 tokens per sequence
            inputs = self._make_warmup_inputs(
                batch_size,
                is_decoding=True,
                max_q_seqlen=1,
                target_hidden_size=self.model_config.hidden_size,
            )
            self._forward_impl(inputs)

    def reset_graph_runner(self):
        """Reset graph runner."""
        if self.proposer.model is not None and hasattr(self.proposer.model, 'reset'):
            self.proposer.model.reset()

    def get_model(self):
        """Get model."""
        return self.proposer.model.get_model()
