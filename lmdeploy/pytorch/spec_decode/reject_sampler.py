# Copyright (c) OpenMMLab. All rights reserved.
import torch
import triton
import triton.language as tl
from torch import LongTensor, Tensor, nn
from torch.profiler import record_function

from lmdeploy.pytorch.engine.logits_process import SamplingInputs

PLACEHOLDER_TOKEN_ID = -1


class RejectionSampler(nn.Module):
    """Rejection sampler for speculative decoding.

    Implements the rejection sampling algorithm from the speculative decoding paper (
    https://arxiv.org/abs/2211.17192).
    Supports both greedy (argmax)
    and random (probabilistic) rejection, with per-sequence greedy detection
    via sampling_inputs.top_k.
    """

    def forward(
        self,
        target_logits: Tensor,
        draft_token_ids: LongTensor,
        bonus_token_ids: LongTensor,
        sampling_inputs: SamplingInputs,
        draft_probs: Tensor | None = None,
    ):
        """forward.

        Args:
            target_logits (Tensor): Processed target logits in shape of
                [batch_size, num_spec_tokens, vocab_size].
            draft_token_ids (LongTensor): The input draft tokens in shape of
                [batch_size, num_spec_tokens].
            bonus_token_ids (LongTensor): The bonus token ids in shape of
                [batch_size].
            sampling_inputs (SamplingInputs): Sampling parameters.
            draft_probs (Tensor): The probability of draft model in shape of
                [batch_size, num_spec_tokens, vocab_size]. Default to ``None``.
        """
        output_token_ids, num_rejected_tokens, last_token_ids = rejection_sample(
            target_logits,
            draft_token_ids,
            bonus_token_ids,
            sampling_inputs=sampling_inputs,
            draft_probs=draft_probs,
        )
        return output_token_ids, num_rejected_tokens, last_token_ids


@record_function('rejection_sample')
def torch_greedy_rejection_sample(
    target_probs: Tensor,
    draft_token_ids: LongTensor,
    bonus_token_ids: LongTensor,
    sampling_inputs: SamplingInputs = None,
    draft_probs: Tensor | None = None,
):
    """Greedy reject sampler
    1. keep targets tokens that are equal to draft tokens
    2. keep first not equal target tokens
    3. add bonus tokens if all equal
    Args:
        target_probs: (batch_size, num_spec_tokens, vocab_size)
        draft_token_ids: (batch_size, num_spec_tokens)
        bonus_token_ids: (batch_size, 1)
    Returns:
        output_token_ids: (batch_size, num_spec_tokens + 1)
    """
    assert draft_probs is None or draft_probs.is_contiguous()
    if bonus_token_ids.ndim == 1:
        bonus_token_ids = bonus_token_ids.unsqueeze(-1)
    target_token_ids = target_probs.argmax(dim=-1)

    masks = draft_token_ids == target_token_ids
    batch_size, num_spec_tokens = draft_token_ids.shape
    # check rest draft tokens
    range_data = torch.arange(num_spec_tokens, device=draft_token_ids.device)[None, :]
    equals = (masks.cumsum(dim=1) - 1) == range_data
    num_rejected_tokens = num_spec_tokens - equals.sum(dim=1)
    first_diff_indices = torch.argmin(equals.int(), dim=1, keepdim=True)
    keeps = range_data.repeat(batch_size, 1) <= first_diff_indices
    keeps = keeps | equals
    keep_token_ids = torch.where(keeps, target_token_ids, -1)
    # add bonus tokens
    keep_bonus_ids = torch.where(equals[:, -1:], bonus_token_ids, -1)
    output_token_ids = torch.cat([keep_token_ids, keep_bonus_ids], dim=1)
    # get last token ids
    last_indices = (torch.cat([keeps, equals[:, -1:]], dim=1).cumsum(dim=1) - 1)[:, -1].flatten()
    last_token_ids = output_token_ids[torch.arange(batch_size, device=draft_token_ids.device), last_indices]
    return output_token_ids, num_rejected_tokens, last_token_ids


def _extract_outputs(output_token_ids: Tensor, num_spec_tokens: int):
    """Extract num_rejected_tokens and last_token_ids from output_token_ids.

    Args:
        output_token_ids: [batch_size, num_spec_tokens + 1]
        num_spec_tokens: number of speculative tokens

    Returns:
        output_token_ids, num_rejected_tokens, last_token_ids
    """
    batch_size = output_token_ids.size(0)
    valid_mask = output_token_ids != PLACEHOLDER_TOKEN_ID
    num_accepted = valid_mask.sum(dim=1)
    num_rejected_tokens = num_spec_tokens + 1 - num_accepted
    last_token_ids = output_token_ids[torch.arange(batch_size, device=output_token_ids.device), num_accepted - 1]
    return output_token_ids, num_rejected_tokens, last_token_ids


def _use_ascend_rejection_sampler(device: torch.device) -> bool:
    return device.type in ('npu', 'privateuseone')


def _resolve_is_greedy_mask(sampling_inputs: SamplingInputs, batch_size: int, device: torch.device) -> torch.Tensor:
    if sampling_inputs.max_top_k == 1:
        return torch.ones(batch_size, dtype=torch.bool, device=device)
    if sampling_inputs.top_k is not None:
        return (sampling_inputs.top_k == 1).to(device=device)
    return torch.zeros(batch_size, dtype=torch.bool, device=device)


def _flatten_spec_inputs(
    target_logits: Tensor,
    draft_token_ids: LongTensor,
    draft_probs: Tensor | None = None,
):
    batch_size, num_spec_tokens = draft_token_ids.shape
    vocab_size = target_logits.shape[-1]
    device = draft_token_ids.device
    cu_num_draft_tokens = torch.arange(1, batch_size + 1, device=device, dtype=torch.long)
    cu_num_draft_tokens.mul_(num_spec_tokens)
    draft_token_ids_flat = draft_token_ids.reshape(-1).contiguous()
    target_logits_flat = target_logits.reshape(-1, vocab_size).contiguous()
    draft_probs_flat = None
    if draft_probs is not None:
        draft_probs_flat = draft_probs.reshape(-1, vocab_size).contiguous()
    return cu_num_draft_tokens, draft_token_ids_flat, target_logits_flat, draft_probs_flat


def _rejection_greedy_sample_pytorch_flat(
    output_token_ids: Tensor,
    cu_num_draft_tokens: Tensor,
    draft_token_ids: LongTensor,
    target_argmax: LongTensor,
    bonus_token_ids: LongTensor,
    max_spec_len: int,
    is_greedy: torch.Tensor,
):
    """Adapted from vllm-ascend's greedy rejection sampler."""
    batch_size = output_token_ids.size(0)
    device = output_token_ids.device
    zero = cu_num_draft_tokens.new_zeros(1)
    cu_start = torch.cat([zero, cu_num_draft_tokens[:-1]])
    draft_tokens_per_req = cu_num_draft_tokens - cu_start
    token_req_ids = torch.repeat_interleave(torch.arange(batch_size, device=device), draft_tokens_per_req)
    token_positions = torch.arange(draft_token_ids.size(0), device=device) - cu_start[token_req_ids]

    mismatch_global = draft_token_ids != target_argmax
    if max_spec_len == 0:
        first_mismatch_pos_per_req = torch.zeros(batch_size, dtype=torch.long, device=device)
    else:
        pos_matrix = torch.full((batch_size, max_spec_len), -1, dtype=torch.long, device=device)
        pos_matrix[token_req_ids, token_positions] = token_positions
        mismatch_matrix = torch.zeros((batch_size, max_spec_len), dtype=torch.bool, device=device)
        mismatch_matrix[token_req_ids, token_positions] = mismatch_global
        mismatch_positions = torch.where(mismatch_matrix, pos_matrix, max_spec_len * 2)
        first_mismatch_pos_per_req, _ = torch.min(mismatch_positions, dim=1)
        no_mismatch_mask = first_mismatch_pos_per_req == max_spec_len * 2
        first_mismatch_pos_per_req[no_mismatch_mask] = draft_tokens_per_req[no_mismatch_mask]

    copy_len = torch.minimum(first_mismatch_pos_per_req + 1, draft_tokens_per_req)
    copy_indices = torch.arange(max_spec_len + 1, device=device).expand(batch_size, -1)
    copy_mask = copy_indices < copy_len.unsqueeze(1)
    final_copy_mask = copy_mask & is_greedy.unsqueeze(1)
    global_idx = cu_start.unsqueeze(1) + copy_indices
    output_token_ids[final_copy_mask] = target_argmax[global_idx[final_copy_mask]].to(output_token_ids.dtype)

    needs_bonus = is_greedy & (first_mismatch_pos_per_req >= draft_tokens_per_req)
    if torch.any(needs_bonus):
        bonus_rows = torch.where(needs_bonus)[0]
        bonus_cols = draft_tokens_per_req[bonus_rows]
        bonus_values = bonus_token_ids.squeeze(-1) if bonus_token_ids.ndim > 1 else bonus_token_ids
        output_token_ids[bonus_rows, bonus_cols] = bonus_values[bonus_rows]


def _sample_recovered_tokens_pytorch_flat(
    output_token_ids: Tensor,
    cu_num_draft_tokens: Tensor,
    draft_token_ids: LongTensor,
    draft_probs: Tensor | None,
    target_probs: Tensor,
    q: Tensor,
    is_ngram: bool = False,
):
    """Adapted from vllm-ascend's flattened recovered-token sampler."""
    device = output_token_ids.device
    num_tokens = output_token_ids.shape[0]
    if num_tokens == 0:
        return

    zero = cu_num_draft_tokens.new_zeros(1)
    cu_start = torch.cat([zero, cu_num_draft_tokens[:-1]])
    token_indices = torch.arange(num_tokens, device=device)
    in_range_mask = ((token_indices[:, None] >= cu_start[None, :]) &
                     (token_indices[:, None] < cu_num_draft_tokens[None, :]))
    token_to_batch = torch.argmax(in_range_mask.to(torch.int32), dim=1)
    token_to_batch = torch.where(in_range_mask.any(dim=1), token_to_batch, token_to_batch.new_zeros(num_tokens))

    if is_ngram:
        prob = target_probs.clone()
        prob[token_indices, draft_token_ids] = 0
    else:
        prob = torch.maximum(target_probs - draft_probs, target_probs.new_zeros(1))

    q_values = q[token_to_batch]
    epsilon = torch.tensor(1e-10, dtype=q_values.dtype, device=device)
    q_values_safe = torch.where(q_values == 0, epsilon, q_values)
    q_values_safe = torch.where(torch.isinf(q_values), epsilon, q_values_safe)
    prob_over_q = prob / q_values_safe
    invalid_q = (q_values == 0) | torch.isinf(q_values)
    prob_over_q = torch.where(invalid_q, prob_over_q.new_full((1, ), -1e10), prob_over_q)
    output_token_ids.copy_(torch.argmax(prob_over_q, dim=1))


def _rejection_random_sample_pytorch_flat(
    output_token_ids: Tensor,
    cu_num_draft_tokens: Tensor,
    draft_token_ids: LongTensor,
    draft_probs: Tensor | None,
    target_probs: Tensor,
    bonus_token_ids: LongTensor,
    recovered_token_ids: LongTensor,
    uniform_probs: Tensor,
    is_greedy: torch.Tensor,
    max_spec_len: int,
    is_ngram: bool = False,
):
    """Adapted from vllm-ascend's vectorized random rejection sampler."""
    batch_size = output_token_ids.shape[0]
    device = output_token_ids.device
    zero = cu_num_draft_tokens.new_zeros(1)
    cu_start = torch.cat([zero, cu_num_draft_tokens[:-1]])
    num_draft_per_batch = cu_num_draft_tokens - cu_start
    pos_indices = torch.arange(max_spec_len, device=device)[None, :]
    valid_mask = pos_indices < num_draft_per_batch[:, None]
    global_token_indices = (cu_start[:, None] + pos_indices).clamp(0, draft_token_ids.shape[0] - 1)
    draft_tokens = draft_token_ids[global_token_indices]

    if is_ngram:
        draft_token_probs = torch.ones_like(draft_tokens, dtype=torch.float32)
    else:
        flat_indices = global_token_indices.flatten()
        flat_draft_tokens = draft_tokens.flatten()
        draft_token_probs = draft_probs[flat_indices, flat_draft_tokens].view(batch_size, max_spec_len)

    flat_indices = global_token_indices.flatten()
    flat_draft_tokens = draft_tokens.flatten()
    target_token_probs = target_probs[flat_indices, flat_draft_tokens].view(batch_size, max_spec_len)
    uniform_token_probs = uniform_probs[global_token_indices]
    recovered_tokens = recovered_token_ids[global_token_indices]

    safe_draft_token_probs = torch.where(draft_token_probs > 0, draft_token_probs, torch.ones_like(draft_token_probs))
    acceptance_condition = (draft_token_probs > 0) & (target_token_probs / safe_draft_token_probs >= uniform_token_probs)
    first_rejection = (~acceptance_condition) & valid_mask
    default_pos = torch.full((batch_size, 1), max_spec_len, dtype=torch.long, device=device)
    first_reject_pos = torch.where(first_rejection.any(dim=1, keepdim=True),
                                   first_rejection.to(torch.float32).argmax(dim=1, keepdim=True),
                                   default_pos)
    should_skip = (pos_indices >= first_reject_pos) & valid_mask
    final_acceptance = acceptance_condition & (~should_skip)
    non_greedy_mask = ~is_greedy
    update_mask = non_greedy_mask[:, None] & valid_mask & (~should_skip)
    first_reject_mask = (pos_indices == first_reject_pos) & valid_mask & non_greedy_mask[:, None]
    final_update_mask = update_mask | first_reject_mask
    final_tokens = torch.where(first_reject_mask, recovered_tokens,
                               torch.where(final_acceptance, draft_tokens, output_token_ids[:, :max_spec_len]))
    output_token_ids[:, :max_spec_len] = torch.where(final_update_mask,
                                                     final_tokens,
                                                     output_token_ids[:, :max_spec_len])

    no_rejection = first_reject_pos.squeeze(1) >= num_draft_per_batch
    should_add_bonus = non_greedy_mask & no_rejection
    bonus_positions = num_draft_per_batch
    all_positions = torch.arange(output_token_ids.shape[1], device=device)[None, :]
    bonus_pos_mask = (all_positions == bonus_positions[:, None]) & should_add_bonus[:, None]
    output_token_ids[:] = torch.where(bonus_pos_mask,
                                      bonus_token_ids.view(-1, 1).expand_as(output_token_ids),
                                      output_token_ids)


def _rejection_random_sample_block_verify_pytorch_flat(
    output_token_ids: Tensor,
    cu_num_draft_tokens: Tensor,
    draft_token_ids: LongTensor,
    draft_probs: Tensor | None,
    target_probs: Tensor,
    bonus_token_ids: LongTensor,
    recovered_token_ids: LongTensor,
    uniform_probs: Tensor,
    is_greedy: torch.Tensor,
    max_spec_len: int,
    is_ngram: bool = False,
):
    """Adapted from vllm-ascend's block-verify rejection sampler."""
    batch_size = output_token_ids.shape[0]
    device = output_token_ids.device
    zero = cu_num_draft_tokens.new_zeros(1)
    cu_start = torch.cat([zero, cu_num_draft_tokens[:-1]])
    num_draft_per_batch = (cu_num_draft_tokens - cu_start)[:, None]
    pos_indices = torch.arange(max_spec_len, device=device)[None, :]
    valid_mask = pos_indices < num_draft_per_batch
    global_token_indices = (cu_start[:, None] + pos_indices).clamp(0, draft_token_ids.shape[0] - 1)
    draft_tokens = draft_token_ids[global_token_indices]

    if is_ngram:
        draft_token_probs = torch.ones_like(draft_tokens, dtype=torch.float32)
    else:
        flat_indices = global_token_indices.flatten()
        flat_draft_tokens = draft_tokens.flatten()
        draft_token_probs = draft_probs[flat_indices, flat_draft_tokens].view(batch_size, max_spec_len)

    flat_indices = global_token_indices.flatten()
    flat_draft_tokens = draft_tokens.flatten()
    target_token_probs = target_probs[flat_indices, flat_draft_tokens].view(batch_size, max_spec_len)
    uniform_token_probs = uniform_probs[global_token_indices]
    recovered_tokens = recovered_token_ids[global_token_indices]

    safe_draft_token_probs = torch.where(draft_token_probs > 0, draft_token_probs, torch.ones_like(draft_token_probs))
    pi = (target_token_probs / safe_draft_token_probs).clamp(max=1.0)
    pi = torch.cumprod(pi, dim=-1)
    uniform_token_probs = torch.cumprod(uniform_token_probs, dim=-1)
    legal_mask = (draft_token_probs > 0) & (pi >= uniform_token_probs) & valid_mask
    last_accept_pos = torch.where(
        legal_mask.any(dim=-1, keepdim=True),
        max_spec_len - legal_mask.flip(dims=[-1]).to(torch.float32).argmax(dim=-1, keepdim=True) - 1,
        torch.full((batch_size, 1), -1, dtype=torch.long, device=device),
    )
    non_greedy_mask = (~is_greedy)[:, None]
    accept_mask = (pos_indices <= last_accept_pos) & valid_mask & non_greedy_mask
    output_token_ids[:, :max_spec_len] = torch.where(accept_mask,
                                                     draft_tokens,
                                                     output_token_ids[:, :max_spec_len])
    reject_mask = (pos_indices == last_accept_pos + 1) & valid_mask & non_greedy_mask
    output_token_ids[:, :max_spec_len] = torch.where(reject_mask,
                                                     recovered_tokens,
                                                     output_token_ids[:, :max_spec_len])
    bonus_mask = (last_accept_pos + 1 >= num_draft_per_batch) & non_greedy_mask
    all_positions = torch.arange(max_spec_len + 1, device=device)[None, :]
    bonus_pos_match = all_positions == num_draft_per_batch
    output_token_ids[:] = torch.where(bonus_mask & bonus_pos_match,
                                      bonus_token_ids.view(-1, 1).expand_as(output_token_ids),
                                      output_token_ids)


def _rejection_sample_ascend(
    target_logits: Tensor,
    draft_token_ids: LongTensor,
    bonus_token_ids: LongTensor,
    sampling_inputs: SamplingInputs,
    draft_probs: Tensor | None = None,
):
    """Ascend-specific rejection sampler adapted from vllm-ascend.

    Keep the vllm-ascend flatten + PyTorch fallback path on Ascend for
    stability. The default Triton kernels in this file still work for CUDA.
    """
    if not draft_token_ids.is_contiguous():
        draft_token_ids = draft_token_ids.contiguous()
    if not target_logits.is_contiguous():
        target_logits = target_logits.contiguous()
    if draft_probs is not None and not draft_probs.is_contiguous():
        draft_probs = draft_probs.contiguous()
    if not bonus_token_ids.is_contiguous():
        bonus_token_ids = bonus_token_ids.contiguous()

    batch_size, num_spec_tokens = draft_token_ids.shape
    vocab_size = target_logits.shape[-1]
    device = target_logits.device
    cu_num_draft_tokens, draft_token_ids_flat, target_logits_flat, draft_probs_flat = _flatten_spec_inputs(
        target_logits, draft_token_ids, draft_probs)
    bonus_token_ids = bonus_token_ids.view(-1, 1)

    output_token_ids = torch.full((batch_size, num_spec_tokens + 1),
                                  PLACEHOLDER_TOKEN_ID,
                                  dtype=torch.long,
                                  device=device)
    is_greedy = _resolve_is_greedy_mask(sampling_inputs, batch_size, device)
    is_all_greedy = bool(is_greedy.all().item())
    is_all_random = bool((~is_greedy).all().item())
    using_block_verify = num_spec_tokens >= 3 and draft_probs_flat is not None

    if not is_all_random:
        target_argmax = target_logits_flat.argmax(dim=-1)
        _rejection_greedy_sample_pytorch_flat(output_token_ids,
                                              cu_num_draft_tokens,
                                              draft_token_ids_flat,
                                              target_argmax,
                                              bonus_token_ids,
                                              num_spec_tokens,
                                              is_greedy)
        if is_all_greedy:
            return _extract_outputs(output_token_ids, num_spec_tokens)

    target_probs_flat = target_logits_flat.softmax(dim=-1, dtype=torch.float32)
    uniform_probs = torch.rand((draft_token_ids_flat.shape[0], ), dtype=torch.float64, device=device)
    q = torch.empty((batch_size, vocab_size), dtype=torch.float32, device=device)
    q.exponential_()
    recovered_token_ids_flat = torch.empty_like(draft_token_ids_flat)
    _sample_recovered_tokens_pytorch_flat(recovered_token_ids_flat,
                                          cu_num_draft_tokens,
                                          draft_token_ids_flat,
                                          draft_probs_flat,
                                          target_probs_flat,
                                          q,
                                          is_ngram=draft_probs_flat is None)

    if using_block_verify:
        _rejection_random_sample_block_verify_pytorch_flat(output_token_ids,
                                                           cu_num_draft_tokens,
                                                           draft_token_ids_flat,
                                                           draft_probs_flat,
                                                           target_probs_flat,
                                                           bonus_token_ids,
                                                           recovered_token_ids_flat,
                                                           uniform_probs,
                                                           is_greedy,
                                                           num_spec_tokens,
                                                           is_ngram=draft_probs_flat is None)
    else:
        _rejection_random_sample_pytorch_flat(output_token_ids,
                                              cu_num_draft_tokens,
                                              draft_token_ids_flat,
                                              draft_probs_flat,
                                              target_probs_flat,
                                              bonus_token_ids,
                                              recovered_token_ids_flat,
                                              uniform_probs,
                                              is_greedy,
                                              num_spec_tokens,
                                              is_ngram=draft_probs_flat is None)
    return _extract_outputs(output_token_ids, num_spec_tokens)


@record_function('rejection_sample')
def rejection_sample(
    target_logits: Tensor,
    draft_token_ids: LongTensor,
    bonus_token_ids: LongTensor,
    sampling_inputs: SamplingInputs,
    draft_probs: Tensor | None = None,
):
    """Rejection sampling.

    Args:
        target_logits (Tensor): Processed target logits in shape of
            [batch_size, num_spec_tokens, vocab_size]. Already processed
            by FusedLogitsProcessor (temperature, top-k, top-p applied).
        draft_token_ids (LongTensor): [batch_size, num_spec_tokens]
        bonus_token_ids (LongTensor): [batch_size]
        sampling_inputs (SamplingInputs): Sampling parameters.
        draft_probs (Tensor): [batch_size, num_spec_tokens, vocab_size] or None.
    """
    assert draft_probs is None or draft_probs.is_contiguous()
    if not draft_token_ids.is_contiguous():
        draft_token_ids = draft_token_ids.contiguous()

    if not target_logits.is_contiguous():
        target_logits = target_logits.contiguous()

    batch_size, num_spec_tokens = draft_token_ids.shape
    vocab_size = target_logits.shape[-1]
    device = target_logits.device

    if _use_ascend_rejection_sampler(device):
        return _rejection_sample_ascend(target_logits,
                                        draft_token_ids,
                                        bonus_token_ids,
                                        sampling_inputs=sampling_inputs,
                                        draft_probs=draft_probs)

    # Determine sampling policy
    is_all_greedy = (sampling_inputs.max_top_k == 1)
    is_all_random = False
    is_greedy = None
    if not is_all_greedy:
        if sampling_inputs.top_k is not None:
            is_greedy = (sampling_inputs.top_k == 1)
            is_all_random = not is_greedy.any().item()
        else:
            is_all_random = True

    # Create output buffer
    output_token_ids = torch.full(
        (batch_size, num_spec_tokens + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.long,
        device=device,
    )

    # 1. Greedy path (skip if all_random)
    if not is_all_random:
        target_argmax = target_logits.argmax(dim=-1)  # [batch, num_spec]
        rejection_greedy_sample_kernel[(batch_size, )](
            output_token_ids,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            is_greedy,
            num_spec_tokens,
        )
        if is_all_greedy:
            return _extract_outputs(output_token_ids, num_spec_tokens)

    # 2. Compute target probs from processed logits
    target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)

    # 3. Uniform random [batch, num_spec] (float64 to avoid exact 0.0)
    uniform_probs = torch.rand(
        (batch_size, num_spec_tokens),
        dtype=torch.float64,
        device=device,
    )

    # 4. Recovered tokens via Gumbel-max trick
    q = torch.empty(
        (batch_size, vocab_size),
        dtype=torch.float32,
        device=device,
    )
    q.exponential_()
    inv_q = q.reciprocal()

    recovered_token_ids = torch.empty(
        (batch_size, num_spec_tokens),
        dtype=torch.long,
        device=device,
    )
    # Ascend Triton kernels can overflow UB with the upstream 8K tile size.
    # Match the vllm-ascend / triton-ascend-kernels adaptation and scan the
    # vocab in smaller 4K chunks instead.
    SUB_BLOCK = 4 * 1024
    kernel_kwargs = dict(NO_DRAFT_PROBS=draft_probs is None)
    if device.type in ('npu', 'privateuseone'):
        # Match the vllm-ascend launch config for this kernel.
        kernel_kwargs['multibuffer'] = False
    sample_recovered_tokens_kernel[(batch_size, num_spec_tokens)](
        recovered_token_ids,
        draft_token_ids,
        draft_probs,
        target_probs,
        inv_q,
        num_spec_tokens,
        vocab_size,
        SUB_BLOCK,
        **kernel_kwargs,
    )

    # 5. Random rejection
    rejection_random_sample_kernel[(batch_size, )](
        output_token_ids,
        draft_token_ids,
        draft_probs,
        target_probs,
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        num_spec_tokens,
        vocab_size,
        NO_DRAFT_PROBS=draft_probs is None,
    )

    return _extract_outputs(output_token_ids, num_spec_tokens)


@triton.jit(do_not_specialize=['num_spec_tokens'])
def rejection_greedy_sample_kernel(
    output_token_ids_ptr,  # [batch_size, num_spec_tokens + 1]
    draft_token_ids_ptr,  # [batch_size, num_spec_tokens]
    target_argmax_ptr,  # [batch_size, num_spec_tokens]
    bonus_token_ids_ptr,  # [batch_size]
    is_greedy_ptr,  # [batch_size] or None
    num_spec_tokens,
):
    """Greedy rejection sampling kernel.

    Grid: (batch_size,)
    For each request: if greedy, accept matching tokens, reject at first
    mismatch.
    """
    req_idx = tl.program_id(0)
    if is_greedy_ptr is not None:
        is_greedy = tl.load(is_greedy_ptr + req_idx).to(tl.int1)
        if not is_greedy:
            return

    out_stride = num_spec_tokens + 1
    draft_stride = num_spec_tokens

    rejected = False
    for pos in range(num_spec_tokens):
        if not rejected:
            draft_token_id = tl.load(draft_token_ids_ptr + req_idx * draft_stride + pos)
            target_argmax_id = tl.load(target_argmax_ptr + req_idx * draft_stride + pos)
            tl.store(
                output_token_ids_ptr + req_idx * out_stride + pos,
                target_argmax_id,
            )
            if draft_token_id != target_argmax_id:
                rejected = True

    if not rejected:
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * out_stride + num_spec_tokens,
            bonus_token_id,
        )


@triton.jit(do_not_specialize=['num_spec_tokens'])
def rejection_random_sample_kernel(
    output_token_ids_ptr,  # [batch_size, num_spec_tokens + 1]
    draft_token_ids_ptr,  # [batch_size, num_spec_tokens]
    draft_probs_ptr,  # [batch_size, num_spec_tokens, vocab_size] or None
    target_probs_ptr,  # [batch_size, num_spec_tokens, vocab_size]
    bonus_token_ids_ptr,  # [batch_size]
    recovered_token_ids_ptr,  # [batch_size, num_spec_tokens]
    uniform_probs_ptr,  # [batch_size, num_spec_tokens]
    is_greedy_ptr,  # [batch_size]
    num_spec_tokens,
    vocab_size,
    NO_DRAFT_PROBS: tl.constexpr,
):
    """Random rejection sampling kernel.

    Grid: (batch_size,)
    For each non-greedy request: accept if target_prob/draft_prob >=
    uniform, else use recovered token.
    """
    req_idx = tl.program_id(0)
    if is_greedy_ptr is not None:
        is_greedy = tl.load(is_greedy_ptr + req_idx).to(tl.int1)
        if is_greedy:
            return

    out_stride = num_spec_tokens + 1
    draft_stride = num_spec_tokens

    rejected = False
    for pos in range(num_spec_tokens):
        if not rejected:
            draft_token_id = tl.load(draft_token_ids_ptr + req_idx * draft_stride + pos)
            if NO_DRAFT_PROBS:
                draft_prob = 1
            else:
                draft_prob = tl.load(draft_probs_ptr + (req_idx * draft_stride + pos) * vocab_size + draft_token_id)
            target_prob = tl.load(target_probs_ptr + (req_idx * draft_stride + pos) * vocab_size + draft_token_id)
            uniform_prob = tl.load(uniform_probs_ptr + req_idx * draft_stride + pos)

            if draft_prob > 0 and target_prob / draft_prob >= uniform_prob:
                token_id = draft_token_id
            else:
                rejected = True
                token_id = tl.load(recovered_token_ids_ptr + req_idx * draft_stride + pos)
            tl.store(
                output_token_ids_ptr + req_idx * out_stride + pos,
                token_id,
            )

    if not rejected:
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * out_stride + num_spec_tokens,
            bonus_token_id,
        )


@triton.jit
def sample_recovered_tokens_kernel(
    output_token_ids_ptr,  # [batch_size, num_spec_tokens]
    draft_token_ids_ptr,  # [batch_size, num_spec_tokens]
    draft_probs_ptr,  # [batch_size, num_spec_tokens, vocab_size] or None
    target_probs_ptr,  # [batch_size, num_spec_tokens, vocab_size]
    inv_q_ptr,  # [batch_size, vocab_size]
    num_spec_tokens,
    vocab_size,
    SUB_BLOCK: tl.constexpr,
    NO_DRAFT_PROBS: tl.constexpr,
):
    """Recovered token sampling kernel via Gumbel-max trick.

    Grid: (batch_size, num_spec_tokens)
    For each position: find argmax of max(0, target_prob - draft_prob)
    * inv_q.
    """
    req_idx = tl.program_id(0)
    pos = tl.program_id(1)
    if pos >= num_spec_tokens:
        return

    draft_stride = num_spec_tokens
    token_idx = req_idx * draft_stride + pos

    if NO_DRAFT_PROBS:
        draft_token_id = tl.load(draft_token_ids_ptr + token_idx)

    max_val = float('-inf')
    recovered_id = 0
    loop = (vocab_size + SUB_BLOCK - 1) // SUB_BLOCK
    for loop_i in range(loop):
        v = loop_i * SUB_BLOCK
        vocab_offset = v + tl.arange(0, SUB_BLOCK)
        vocab_mask = vocab_offset < vocab_size

        if NO_DRAFT_PROBS:
            prob = tl.load(
                target_probs_ptr + token_idx * vocab_size + vocab_offset,
                mask=(vocab_mask & (vocab_offset != draft_token_id)),
                other=0.0,
            )
        else:
            draft_prob = tl.load(
                draft_probs_ptr + token_idx * vocab_size + vocab_offset,
                mask=vocab_mask,
                other=0.0,
            )
            target_prob = tl.load(
                target_probs_ptr + token_idx * vocab_size + vocab_offset,
                mask=vocab_mask,
                other=0.0,
            )
            prob = tl.maximum(target_prob - draft_prob, 0.0)

        inv_q = tl.load(
            inv_q_ptr + req_idx * vocab_size + vocab_offset,
            mask=vocab_mask,
            other=0.0,
        )

        score = prob * inv_q
        local_max = tl.max(score, axis=0)
        local_id = tl.argmax(score, axis=0)

        if local_max > max_val:
            max_val = local_max
            recovered_id = v + local_id

    tl.store(output_token_ids_ptr + token_idx, recovered_id)
