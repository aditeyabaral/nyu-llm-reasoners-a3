import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    prompt_token_ids = [
        tokenizer.encode(p, add_special_tokens=False) for p in prompt_strs
    ]
    output_token_ids = [
        tokenizer.encode(o, add_special_tokens=False) for o in output_strs
    ]

    concatenated_ids = [p + o for p, o in zip(prompt_token_ids, output_token_ids)]
    prompt_lengths = [len(p) for p in prompt_token_ids]
    output_lengths = [len(o) for o in output_token_ids]

    # Pad the full concatenation first, then slice into input_ids and labels
    max_len = max(len(c) for c in concatenated_ids)
    pad_id = tokenizer.eos_token_id
    padded = torch.full((len(concatenated_ids), max_len), pad_id, dtype=torch.long)
    for i, ids in enumerate(concatenated_ids):
        padded[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

    input_ids = padded[:, :-1]
    labels = padded[:, 1:]

    seq_len = max_len - 1
    response_mask = torch.zeros(len(concatenated_ids), seq_len, dtype=torch.long)
    for i, (prompt_len, output_len) in enumerate(zip(prompt_lengths, output_lengths)):
        response_mask[i, prompt_len - 1 : prompt_len + output_len - 1] = 1

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    # log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    return -(probs * log_probs).sum(dim=-1)


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits  # (batch_size, seq_len, vocab_size)
    log_probs = F.log_softmax(logits, dim=-1)  # (batch_size, seq_len, vocab_size)
    # Gather log probs at the label token indices
    token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(
        -1
    )  # (batch_size, seq_len)

    result = {"log_probs": token_log_probs}
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)

    return result


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    masked = tensor * mask
    if dim is None:
        return masked.sum() / normalize_constant
    return masked.sum(dim=dim) / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss = (
        -masked_normalize(
            tensor=policy_log_probs,
            mask=response_mask,
            normalize_constant=normalize_constant,
            dim=1,
        ).mean()
        / gradient_accumulation_steps
    )

    loss.backward()

    return loss, {}
