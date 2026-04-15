# Copyright (c) OpenMMLab. All rights reserved.

import os
import torch

from lmdeploy.utils import get_logger

from ...model_inputs import ModelInputs
from ...strategies.ar_spec.model_agent import ARSpecExtraInputs
from .base import SPEC_PROPOSERS, BaseSpecProposer

logger = get_logger('lmdeploy')


@SPEC_PROPOSERS.register_module(name='deepseek_mtp')
class DeepseekMTP(BaseSpecProposer):

    def get_outputs(self,
                    model_outputs: dict[str, torch.Tensor],
                    model_inputs: ModelInputs,
                    extra_inputs: ARSpecExtraInputs = None):
        """Get outputs."""
        hidden_states = model_outputs['hidden_states']
        model_metas = model_outputs['model_metas']
        spec_debug = os.environ.get("LMDEPLOY_SPEC_DEBUG", "0") == "1"
        spec_debug_steps = int(os.environ.get("LMDEPLOY_SPEC_DEBUG_STEPS", "2"))
        if spec_debug and not hasattr(self, "_spec_debug_draft_step"):
            self._spec_debug_draft_step = 0
        if extra_inputs is not None and extra_inputs.last_token_indices is not None:
            # for long input
            if (not model_inputs.is_decoding) and model_inputs.seq_length.size(0) == 1:
                hidden_states = hidden_states[:, -1:]
            else:
                last_token_loc = extra_inputs.last_token_indices
                if spec_debug and model_inputs.is_decoding and hidden_states.size(1) > 1:
                    # Compare conditioning choices in decoding:
                    # - index-based slice (current behavior)
                    # - always take the last token in the window (often the "bonus" position)
                    try:
                        logits_idx = self.get_logits(hidden_states[:, last_token_loc])[0]
                        logits_last = self.get_logits(hidden_states[:, -1])[0]
                        d_idx = logits_idx.argmax(dim=-1).tolist()
                        d_last = logits_last.argmax(dim=-1).tolist()
                        print(
                            f"[DRAFT_COND_DEBUG] is_decoding={model_inputs.is_decoding} "
                            f"hidden_states_shape={tuple(hidden_states.shape)} "
                            f"last_token_indices={last_token_loc.tolist()} "
                            f"draft_argmax_by_idx={d_idx} draft_argmax_by_last={d_last}",
                            flush=True,
                        )
                    except Exception:
                        pass
                hidden_states = hidden_states[:, last_token_loc]

        logits = self.get_logits(hidden_states)[0]
        if spec_debug and self._spec_debug_draft_step < spec_debug_steps:
            # Print draft top-k for batch0 to estimate proposer quality.
            # This is a cheap way to tell whether target argmax tends to land in
            # draft top-k (weak proposer) or not even top-k (likely misalignment).
            try:
                # Best-effort rank0-only gating to reduce log spam.
                dist_ctx = None
                try:
                    from lmdeploy.pytorch.distributed import get_dist_manager  # local import
                    dist_ctx = get_dist_manager().current_context()
                except Exception:
                    dist_ctx = None
                rank = 0 if dist_ctx is None else dist_ctx.rank
                if rank == 0:
                    topk = int(os.environ.get("LMDEPLOY_DRAFT_TOPK", "5"))
                    v, i = torch.topk(logits[0], k=topk, dim=-1)
                    print(
                        f"[DRAFT_TOPK][step={self._spec_debug_draft_step}] "
                        f"is_decoding={model_inputs.is_decoding} "
                        f"topk_ids={i.tolist()} topk_vals={v.float().tolist()}",
                        flush=True,
                    )
            except Exception:
                pass
            self._spec_debug_draft_step += 1
        draft_token_ids = logits.argmax(dim=-1, keepdim=True)
        return draft_token_ids, logits, model_metas, hidden_states
