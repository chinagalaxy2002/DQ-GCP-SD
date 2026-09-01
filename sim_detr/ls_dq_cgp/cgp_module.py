"""Late-semantic candidate-specific text adaptation and matching."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class LSDQCGPOutput(NamedTuple):
    adapted_semantic: Tensor
    basis_weights: Tensor
    pooled_prompt: Tensor
    semantic_scores: Tensor
    pred_logits: Tensor


class LateSemanticCGP(nn.Module):
    """Map ``(stopgrad(V_q), E_static)`` to query-specific text features."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_basis: int = 16,
        prompt_length: int = 6,
        router_hidden_dim: int = 256,
        frf_hidden_dim: int = 512,
        temperature: float = 1.0,
        initial_scale: float = 10.0,
        initial_bias: float = -2.0,
    ) -> None:
        super().__init__()
        if min(hidden_dim, num_basis, prompt_length) <= 0:
            raise ValueError("hidden_dim, num_basis and prompt_length must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.hidden_dim = int(hidden_dim)
        self.num_basis = int(num_basis)
        self.prompt_length = int(prompt_length)
        self.temperature = float(temperature)

        self.router = nn.Sequential(
            nn.Linear(2 * hidden_dim, router_hidden_dim),
            nn.ReLU(),
            nn.Linear(router_hidden_dim, num_basis),
        )
        self.basis_prompts = nn.Parameter(
            torch.empty(num_basis, prompt_length, hidden_dim)
        )
        self.basis_norm = nn.LayerNorm(hidden_dim)
        self.frf_v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.frf = nn.Sequential(
            nn.Linear(3 * hidden_dim, frf_hidden_dim),
            nn.ReLU(),
            nn.Linear(frf_hidden_dim, hidden_dim),
        )
        self.frf_norm = nn.LayerNorm(hidden_dim)
        self.visual_proj = nn.Linear(hidden_dim, hidden_dim)
        self.semantic_proj = nn.Linear(hidden_dim, hidden_dim)
        self.logit_scale = nn.Parameter(torch.tensor(float(initial_scale)))
        self.logit_bias = nn.Parameter(torch.tensor(float(initial_bias)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.basis_prompts)
        for module in (self.router, self.frf):
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        for layer in (self.frf_v_proj, self.visual_proj, self.semantic_proj):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(
        self,
        visual_context: Tensor,
        static_semantic: Tensor,
        query_states: Tensor,
        static_bypass: bool = False,
    ) -> LSDQCGPOutput:
        if visual_context.ndim != 3 or query_states.shape != visual_context.shape:
            raise ValueError("visual_context and query_states must share shape [B,Q,D]")
        batch_size, num_queries, dim = visual_context.shape
        if static_semantic.shape != (batch_size, dim) or dim != self.hidden_dim:
            raise ValueError("static_semantic must have shape [B,D] matching hidden_dim")

        static = static_semantic.unsqueeze(1).expand(-1, num_queries, -1)
        local = visual_context.detach()
        route_logits = self.router(torch.cat([local, static], dim=-1))
        basis_weights = F.softmax(route_logits / self.temperature, dim=-1)
        prompt_sequence = torch.einsum("bqk,kpd->bqpd", basis_weights, self.basis_prompts)
        pooled_prompt = self.basis_norm(prompt_sequence.mean(dim=2))

        fusion = torch.cat([pooled_prompt, static, self.frf_v_proj(local)], dim=-1)
        adapted = self.frf_norm(static + self.frf(fusion))
        match_semantic = static if static_bypass else adapted
        visual_metric = F.normalize(self.visual_proj(query_states), p=2, dim=-1)
        semantic_metric = F.normalize(self.semantic_proj(match_semantic), p=2, dim=-1)
        semantic_scores = self.logit_scale * (visual_metric * semantic_metric).sum(-1)
        semantic_scores = semantic_scores + self.logit_bias
        pred_logits = torch.stack([semantic_scores, torch.zeros_like(semantic_scores)], dim=-1)
        return LSDQCGPOutput(
            adapted, basis_weights, pooled_prompt, semantic_scores, pred_logits
        )
