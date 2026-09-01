"""Small utilities for semantic-calibration mechanism diagnostics."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def roll_candidate_context(context: torch.Tensor, shift: int = 1) -> torch.Tensor:
    """Replace candidate ``i`` with context ``i-shift`` (cyclically)."""
    if context.ndim != 3:
        raise ValueError("context must be [B,Q,D]")
    if not 0 < int(shift) < context.shape[1]:
        raise ValueError("roll shift must be in [1, Q-1]")
    return torch.roll(context, shifts=int(shift), dims=1)


def derangement_indices(
    batch_size: int,
    num_candidates: int,
    *,
    seed: int,
    sample_offset: int = 0,
    device=None,
) -> torch.Tensor:
    """Generate deterministic per-sample permutations with no fixed points."""
    if num_candidates < 2:
        raise ValueError("a derangement requires at least two candidates")
    identity = torch.arange(num_candidates)
    rows = []
    for sample_idx in range(batch_size):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + int(sample_offset) + sample_idx)
        for _ in range(1000):
            permutation = torch.randperm(num_candidates, generator=generator)
            if not torch.any(permutation == identity):
                rows.append(permutation)
                break
        else:
            raise RuntimeError("failed to generate a random derangement")
    return torch.stack(rows, dim=0).to(device=device)


def gather_candidate_context(context: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather one source context for each ``[B,Q]`` candidate index."""
    if context.ndim != 3 or indices.shape != context.shape[:2]:
        raise ValueError("context must be [B,Q,D] and indices must be [B,Q]")
    gather_indices = indices.to(context.device).unsqueeze(-1).expand_as(context)
    return torch.gather(context, dim=1, index=gather_indices)


def random_deranged_context(
    context: torch.Tensor,
    *,
    seed: int,
    sample_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = derangement_indices(
        context.shape[0], context.shape[1], seed=seed,
        sample_offset=sample_offset, device=context.device,
    )
    return gather_candidate_context(context, indices), indices


def farthest_candidate_context(context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace each candidate by the least cosine-similar other context."""
    if context.ndim != 3 or context.shape[1] < 2:
        raise ValueError("context must be [B,Q,D] with Q >= 2")
    normalized = F.normalize(context, p=2, dim=-1)
    similarity = torch.bmm(normalized, normalized.transpose(1, 2))
    diagonal = torch.eye(context.shape[1], device=context.device, dtype=torch.bool)
    similarity = similarity.masked_fill(diagonal.unsqueeze(0), float("inf"))
    indices = similarity.argmin(dim=-1)
    return gather_candidate_context(context, indices), indices


def semantic_ranking_margin(semantic_scores, matched_indices):
    """Compute positive-vs-hard-negative semantic margins for a batch."""
    margins = []
    for scores, pair in zip(semantic_scores, matched_indices):
        positive_queries = pair[0].to(scores.device)
        if positive_queries.numel() == 0:
            continue
        hard_negative = scores.masked_fill(
            torch.zeros_like(scores, dtype=torch.bool).scatter_(0, positive_queries, True),
            float("-inf"),
        ).max()
        margins.extend((scores[positive_queries] - hard_negative).detach().flatten().tolist())
    return margins
