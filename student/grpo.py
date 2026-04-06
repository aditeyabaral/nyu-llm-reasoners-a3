from typing import Callable, Literal

import torch

from student.sft import masked_normalize


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    masked = tensor * mask
    if dim is None:
        return masked.sum() / mask.sum()
    return masked.sum(dim=dim) / mask.sum(dim=dim)


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    raw = [
        reward_fn(resp, gt)["reward"]
        for resp, gt in zip(rollout_responses, repeated_ground_truths)
    ]
    raw_rewards = torch.tensor(raw, dtype=torch.float32)

    n_prompts = len(rollout_responses) // group_size
    advantages = torch.zeros_like(raw_rewards)

    for i in range(n_prompts):
        start, end = i * group_size, (i + 1) * group_size
        group = raw_rewards[start:end]
        mean = group.mean()
        if normalize_by_std:
            std = group.std() + advantage_eps
            advantages[start:end] = (group - mean) / std
        else:
            advantages[start:end] = group - mean

    metadata = {
        "mean_reward": raw_rewards.mean().item(),
        "std_reward": raw_rewards.std().item(),
        "max_reward": raw_rewards.max().item(),
        "min_reward": raw_rewards.min().item(),
    }
    return advantages, raw_rewards, metadata


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - cliprange, 1 + cliprange)
    loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
    is_clipped = (ratio < 1 - cliprange) | (ratio > 1 + cliprange)
    return loss, {"is_clipped": is_clipped}


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    return -raw_rewards_or_advantages * policy_log_probs


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_type == "no_baseline":
        assert raw_rewards is not None
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}
    elif loss_type == "grpo_clip":
        assert (
            advantages is not None
            and old_log_probs is not None
            and cliprange is not None
        )
        return compute_grpo_clip_loss(
            advantages, policy_log_probs, old_log_probs, cliprange
        )
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    normalize_constant: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    if normalize_constant is not None:
        per_example_loss = masked_normalize(
            per_token_loss,
            response_mask,
            dim=1,
            normalize_constant=normalize_constant,
        )
    else:
        per_example_loss = masked_mean(per_token_loss, response_mask, dim=1)
    loss = per_example_loss.mean() / gradient_accumulation_steps
    loss.backward()
    return loss, metadata
