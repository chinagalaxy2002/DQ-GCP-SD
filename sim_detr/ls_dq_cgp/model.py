"""Two-layer Sim-DETR with late-semantic candidate re-ranking."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from sim_detr.model import SimDETR, inverse_sigmoid
from .attention_capture import NativeD1AttentionCapture
from .cgp_module import LateSemanticCGP


class SimDETRWithLSDQCGP(SimDETR):
    """Preserve Sim-DETR localization and replace only final classification."""

    def __init__(
        self,
        *args,
        ls_num_basis: int = 16,
        ls_prompt_length: int = 6,
        ls_router_hidden_dim: int = 256,
        ls_frf_hidden_dim: int = 512,
        ls_temperature: float = 1.0,
        ls_initial_scale: float = 10.0,
        ls_initial_bias: float = -2.0,
        use_ls_exist_head: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if int(self.transformer.decoder.num_layers) != 2:
            raise ValueError("LS-DQ-CGP on Sim-DETR is defined for dec_layers=2")
        self.ls_cgp = LateSemanticCGP(
            hidden_dim=self.hidden_dim,
            num_basis=ls_num_basis,
            prompt_length=ls_prompt_length,
            router_hidden_dim=ls_router_hidden_dim,
            frf_hidden_dim=ls_frf_hidden_dim,
            temperature=ls_temperature,
            initial_scale=ls_initial_scale,
            initial_bias=ls_initial_bias,
        )
        self.d1_capture = NativeD1AttentionCapture(self.transformer.decoder)
        self.static_bypass = False
        self.context_roll = False
        self.use_ls_exist_head = bool(use_ls_exist_head)
        self.ls_exist_head = None
        if self.use_ls_exist_head:
            self.ls_exist_head = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1),
            )

    @staticmethod
    def _semantic_mask(src_txt, src_txt_mask, explicit_mask=None):
        if src_txt_mask.shape != src_txt.shape[:2]:
            raise ValueError("src_txt_mask must match src_txt's first two dimensions")
        padding_valid = src_txt_mask.to(src_txt.device).bool()
        if explicit_mask is None:
            # Input text dropout writes complete token rows to zero.
            semantic_mask = padding_valid & (src_txt.detach().abs().sum(-1) > 0)
        else:
            if explicit_mask.shape != src_txt.shape[:2]:
                raise ValueError("src_txt_semantic_mask must have shape [B,L]")
            semantic_mask = explicit_mask.to(src_txt.device).bool() & padding_valid
        empty = semantic_mask.sum(1) == 0
        if bool(empty.any()):
            ids = empty.nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(f"no valid semantic tokens for batch indices {ids}")
        return semantic_mask

    def forward(
        self,
        src_txt,
        src_txt_mask,
        src_vid,
        src_vid_mask,
        src_aud=None,
        src_aud_mask=None,
        src_txt_semantic_mask=None,
    ):
        del src_aud_mask
        if src_aud is not None:
            src_vid = torch.cat([src_vid, src_aud], dim=2)
        batch_size, text_length = src_txt.shape[:2]
        semantic_mask = self._semantic_mask(
            src_txt, src_txt_mask, src_txt_semantic_mask
        )
        self.d1_capture.reset(src_vid_mask)

        video_projected = self.input_vid_proj(src_vid)
        text_projected = self.input_txt_proj(src_txt)
        semantic_weight = semantic_mask.to(text_projected.dtype).unsqueeze(-1)
        static_semantic = (text_projected * semantic_weight).sum(1)
        static_semantic = static_semantic / semantic_weight.sum(1).clamp_min(1)

        reversed_mask = src_txt_mask.flip(dims=[1])
        eos_index = text_length - reversed_mask.argmax(dim=1) - 1
        batch_index = torch.arange(batch_size, device=text_projected.device)
        sentence_feature = text_projected[batch_index, eos_index]

        # Keep the original CTC/VTC feature paths.
        src_txt_ed = text_projected
        src_vid_ed = video_projected
        src_vid_cls_ed = video_projected.mean(1)
        src_txt_cls_ed = sentence_feature

        video_fused = self.fuser(
            video_projected, text_projected, src_vid_mask, src_txt_mask
        )
        source = torch.cat([video_fused, text_projected], dim=1)
        valid_mask = torch.cat([src_vid_mask, src_txt_mask], dim=1).bool()
        video_position = self.position_embed(video_fused, src_vid_mask)
        text_position = (
            self.txt_position_embed(text_projected)
            if self.use_txt_pos
            else torch.zeros_like(text_projected)
        )
        position = torch.cat([video_position, text_position], dim=1)
        video_length = video_fused.shape[1]

        hs, reference, memory, saliency_scores = self.transformer(
            source,
            ~valid_mask,
            self.query_embed.weight,
            position,
            self.saliency_proj1,
            video_length=video_length,
            sent_feat=sentence_feature,
        )
        if hs.shape[0] != 2:
            raise RuntimeError(f"expected exactly D1,D2 states; got {hs.shape[0]} layers")

        native_class = self.class_embed(hs)
        iou_scores = self.iou_head(hs)
        mask_features = hs + self.mask_head(hs)
        query_norm = F.normalize(mask_features, p=2, dim=-1)
        video_norm = F.normalize(memory, p=2, dim=-1).unsqueeze(0).expand(2, -1, -1, -1)
        masks = (self.logit_scale.exp() * torch.matmul(
            query_norm, video_norm.transpose(-2, -1)
        )).sigmoid()

        coordinates = self.span_embed(hs) + inverse_sigmoid(reference)
        if self.span_loss_type == "l1":
            coordinates = coordinates.sigmoid()

        d1_attention = self.d1_capture.normalized()
        visual_context = torch.bmm(d1_attention, memory)
        if self.context_roll:
            visual_context = visual_context.roll(shifts=1, dims=1)
        ls_output = self.ls_cgp(
            visual_context=visual_context,
            static_semantic=static_semantic,
            query_states=hs[-1],
            static_bypass=self.static_bypass,
        )

        out = {
            "pred_logits": ls_output.pred_logits,
            "pred_spans": coordinates[-1],
            "pred_masks": masks[-2],
            # The stock evaluator multiplies foreground probability by
            # sigmoid(iou_scores).  A constant factor keeps ordering strictly
            # semantic-only, as defined by LS-DQ-CGP; the trained native head
            # remains available to the criterion and diagnostics below.
            "iou_scores": torch.full_like(iou_scores[-1], 20.0),
            "native_iou_scores": iou_scores[-1],
            "saliency_scores": saliency_scores,
            "video_mask": src_vid_mask,
            "ls_d1_attention": d1_attention,
            "ls_basis_weights": ls_output.basis_weights,
            "ls_semantic_scores": ls_output.semantic_scores,
            "src_txt_ed": src_txt_ed,
            "src_vid_ed": src_vid_ed,
            "src_vid_cls_ed": src_vid_cls_ed,
            "src_txt_cls_ed": src_txt_cls_ed,
        }
        if self.ls_exist_head is not None:
            pooled_queries = hs[-1].max(dim=1).values
            out["pred_exist_logits"] = self.ls_exist_head(pooled_queries).squeeze(-1)

        if self.contrastive_align_loss:
            projected_queries = F.normalize(
                self.contrastive_align_projection_query(hs), p=2, dim=-1
            )
            # Sim-DETR's decoder memory is video-only after T2V selection.
            projected_video = F.normalize(
                self.contrastive_align_projection_vid(memory), p=2, dim=-1
            )
            projected_text = F.normalize(
                self.contrastive_align_projection_txt(memory[:, 0:0]), p=2, dim=-1
            )
            out.update(
                proj_queries=projected_queries[-1],
                proj_txt_mem=projected_text,
                proj_vid_mem=projected_video,
            )

        if self.aux_loss:
            out["aux_outputs"] = [{
                "pred_logits": native_class[0],
                "pred_spans": coordinates[0],
                "pred_masks": masks[0],
                "iou_scores": iou_scores[0],
            }]
            if self.contrastive_align_loss:
                out["aux_outputs"][0].update(
                    proj_queries=projected_queries[0], proj_txt_mem=projected_text
                )
        return out
