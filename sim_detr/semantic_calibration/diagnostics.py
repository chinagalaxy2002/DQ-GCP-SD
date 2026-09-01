"""Small utilities for semantic-calibration mechanism diagnostics."""

from __future__ import annotations

import torch


def roll_candidate_context(context: torch.Tensor) -> torch.Tensor:
    """Return ``[v_Q, v_1, ..., v_{Q-1}]`` as specified by context-roll."""
    return torch.roll(context, shifts=1, dims=1)


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
