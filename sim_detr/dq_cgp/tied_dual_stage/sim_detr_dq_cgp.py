"""Sim-DETR with one DQ-CGP reused at two decoder depths.

This class **subclasses SimDETR** without touching any original source file.
It injects DQ-CGP between decoder layers 1/2 and layers 3/4 by:

1. Calling ``super().__init__()`` to build the complete Sim-DETR model.
2. Installing a state-dict-preserving forward hook on the existing
   ``self.transformer.decoder`` instance.
3. Overriding ``forward`` to:
   - Compute ``query_semantic`` via masked mean-pooling of projected text.
   - Push the semantic and video_length into the decoder hook before the
     transformer call (side-channel via ``set_adapter_kwargs``).
   - Snapshot separate early/late diagnostics for averaged V3 losses.

Checkpoint compatibility
------------------------
* ``--resume`` (strict=True): restores a checkpoint built with the same
  architecture.  An active DQ-CGP model therefore resumes from a DQ-CGP
  checkpoint, not from a baseline checkpoint.
* ``--init_from`` (strict=False, only allows missing ``query_cgp.*``): used to
  warm-start DQ-CGP from a baseline Sim-DETR checkpoint.

The model is architecture-backwards-compatible: setting ``use_query_cgp=False``
(or not using this class at all) restores vanilla Sim-DETR.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from sim_detr.model import SimDETR
from .query_cgp import DETRQueryCGP
from .decoder_wrapper import (
    install_dq_cgp_decoder_hooks,
    get_dq_cgp_stage_outputs,
    set_dq_cgp_adapter_kwargs,
)
from .utils import masked_mean_pool
from .loss import loss_query_cgp


ARCHITECTURE_ID = "sim_detr_tied_dual_dq_cgp_v1"


class SimDETRWithTiedDualDQCGP(SimDETR):
    """Sim-DETR extended with a parameter-tied two-call DQ-CGP adapter.

    All original Sim-DETR parameters (encoder, T2V, HD2MR, decoder layers,
    reference-point heads, class/span/iou/mask heads) are untouched.  Only
    the native decoder forward is extended by a state-dict-preserving hook.

    Parameters
    ----------
    transformer, position_embed, txt_position_embed, txt_dim, vid_dim,
    num_queries, input_dropout, aux_loss, contrastive_align_loss,
    contrastive_hdim, max_v_l, span_loss_type, use_txt_pos, n_input_proj,
    aud_dim, clip_len :
        Identical to ``SimDETR.__init__`` – forwarded via ``**kwargs``.

    use_query_cgp : bool
        Master switch.  When False this class behaves identically to SimDETR.
    query_cgp_num_basis : int
        Number of shared prompt bases (K).  Default 16.
    query_cgp_prompt_length : int
        Tokens per basis prompt.  Default 6.
    query_cgp_router_hidden_dim : int
        Router MLP hidden dimension.  Default 256.
    query_cgp_frf_hidden_dim : int
        FRF MLP hidden dimension.  Default 512.
    query_cgp_temperature : float
        Routing softmax temperature.  Default 1.0.
    query_cgp_beta : float
        Fixed residual injection strength.  Default 0.05.
    query_cgp_after_layer : int
        Early DQ-CGP insertion layer (0-indexed).  Default 0.
    query_cgp_late_after_layer : int
        Late DQ-CGP insertion layer (0-indexed).  Default 2.
    query_cgp_binding_loss_coef : float
        Coefficient for the temporal binding loss.  Default 0.2.
    query_cgp_route_loss_coef : float
        Coefficient for the routing diversity loss.  Default 0.01.
    """

    def __init__(
        self,
        transformer,
        position_embed,
        txt_position_embed,
        txt_dim: int,
        vid_dim: int,
        num_queries: int,
        input_dropout: float = 0.5,
        aux_loss: bool = False,
        contrastive_align_loss: bool = False,
        contrastive_hdim: int = 64,
        max_v_l: int = 75,
        span_loss_type: str = "l1",
        use_txt_pos: bool = False,
        n_input_proj: int = 2,
        aud_dim: int = 0,
        clip_len: int = 2,
        # ---- DQ-CGP configuration ----
        use_query_cgp: bool = False,
        query_cgp_num_basis: int = 16,
        query_cgp_prompt_length: int = 6,
        query_cgp_router_hidden_dim: int = 256,
        query_cgp_frf_hidden_dim: int = 512,
        query_cgp_temperature: float = 1.0,
        query_cgp_beta: float = 0.05,
        query_cgp_after_layer: int = 0,
        query_cgp_late_after_layer: int = 2,
        query_cgp_binding_loss_coef: float = 0.2,
        query_cgp_route_loss_coef: float = 0.01,
    ) -> None:
        super().__init__(
            transformer=transformer,
            position_embed=position_embed,
            txt_position_embed=txt_position_embed,
            txt_dim=txt_dim,
            vid_dim=vid_dim,
            num_queries=num_queries,
            input_dropout=input_dropout,
            aux_loss=aux_loss,
            contrastive_align_loss=contrastive_align_loss,
            contrastive_hdim=contrastive_hdim,
            max_v_l=max_v_l,
            span_loss_type=span_loss_type,
            use_txt_pos=use_txt_pos,
            n_input_proj=n_input_proj,
            aud_dim=aud_dim,
            clip_len=clip_len,
        )

        self.use_query_cgp = use_query_cgp
        self.query_cgp_after_layer = int(query_cgp_after_layer)
        self.query_cgp_late_after_layer = int(query_cgp_late_after_layer)
        self.query_cgp_binding_loss_coef = float(query_cgp_binding_loss_coef)
        self.query_cgp_route_loss_coef = float(query_cgp_route_loss_coef)
        self._clip_len = int(clip_len)

        if use_query_cgp:
            hidden_dim = transformer.d_model  # 256 for Sim-DETR

            if self.query_cgp_after_layer == self.query_cgp_late_after_layer:
                raise ValueError("early and late DQ-CGP insertion layers must differ")

            # The sole registered DQ-CGP parameter set is reused at both depths.
            self.query_cgp = DETRQueryCGP(
                hidden_dim=hidden_dim,
                num_basis=query_cgp_num_basis,
                prompt_length=query_cgp_prompt_length,
                router_hidden_dim=query_cgp_router_hidden_dim,
                frf_hidden_dim=query_cgp_frf_hidden_dim,
                temperature=query_cgp_temperature,
                beta=query_cgp_beta,
            )

            # ---- 2. Install an inter-layer hook on the native decoder ----
            # At this point SimDETR has already set:
            #   self.transformer.decoder.class_embed
            #   self.transformer.decoder.span_embed
            #   self.transformer.decoder.iou_head
            # The native decoder object remains in place.  Consequently its
            # public API and all transformer.decoder.* checkpoint keys remain
            # identical to the baseline architecture.
            install_dq_cgp_decoder_hooks(
                decoder=self.transformer.decoder,
                adapters_by_layer={
                    self.query_cgp_after_layer: self.query_cgp,
                    self.query_cgp_late_after_layer: self.query_cgp,
                },
            )

        else:
            # Keep the attribute present for uniform checkpoint handling.
            self.query_cgp = None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

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
        """Forward pass: standard Sim-DETR + optional DQ-CGP adaptation.

        The only additions over the base class are:
        1. Compute ``query_semantic`` from the projected text tokens.
        2. Register it with the decoder hook before the transformer call.
        3. Append DQ-CGP diagnostics and loss inputs to the output dict.

        All original Sim-DETR outputs are preserved unchanged.
        """
        # ------------------------------------------------------------------
        # Optional audio concatenation (identical to SimDETR.forward)
        # ------------------------------------------------------------------
        if src_aud is not None:
            src_vid = torch.cat([src_vid, src_aud], dim=2)

        b, l_txt = src_txt.shape[:2]

        # ``txt_drop_ratio`` zeros complete rows in the raw offline text
        # features.  A padding-only mask cannot identify those dropped rows
        # after input projection because LinearLayer may add a non-zero bias.
        # Derive the semantic mask before projection unless the data pipeline
        # provides the more explicit mask itself.
        semantic_mask = None
        if self.use_query_cgp and self.query_cgp is not None:
            if src_txt_mask.shape != src_txt.shape[:2]:
                raise ValueError(
                    "src_txt_mask must have shape "
                    "[batch, text_length] matching src_txt"
                )
            padding_valid = src_txt_mask.to(device=src_txt.device).bool()
            if src_txt_semantic_mask is None:
                raw_nonzero = src_txt.detach().abs().sum(dim=-1) > 0
                semantic_mask = padding_valid & raw_nonzero
            else:
                if src_txt_semantic_mask.shape != src_txt.shape[:2]:
                    raise ValueError(
                        "src_txt_semantic_mask must have shape "
                        "[batch, text_length] matching src_txt"
                    )
                semantic_mask = (
                    src_txt_semantic_mask.to(device=src_txt.device).bool()
                    & padding_valid
                )

            empty_rows = semantic_mask.sum(dim=1) == 0
            if empty_rows.any():
                bad_indices = empty_rows.nonzero(as_tuple=False).flatten().tolist()
                raise ValueError(
                    "DQ-CGP query semantic is empty for batch indices "
                    f"{bad_indices}; each query requires at least one valid, "
                    "non-dropped text token"
                )

        # ------------------------------------------------------------------
        # Feature projection (same as SimDETR)
        # ------------------------------------------------------------------
        src_vid_proj = self.input_vid_proj(src_vid)
        src_txt_proj = self.input_txt_proj(src_txt)

        # ------------------------------------------------------------------
        # Query semantic: masked mean-pool of PROJECTED text tokens.
        # Computed BEFORE fuser so the semantic is independent of the
        # video-text fusion path (parallels the DQ-CGP original design).
        # src_txt_mask: 1=valid, 0=padding  (Sim-DETR convention)
        # ------------------------------------------------------------------
        if self.use_query_cgp and self.query_cgp is not None:
            query_semantic = masked_mean_pool(src_txt_proj, semantic_mask)
            # Register with the native decoder hook without changing
            # Transformer.forward's public call signature.
            set_dq_cgp_adapter_kwargs(
                self.transformer.decoder,
                query_semantic=query_semantic,
                # video_length=None means "use full memory" (Sim-DETR path)
            )
            # Clear stale diagnostics from the previous step.
            self.query_cgp.clear_diagnostics()
        
        # ------------------------------------------------------------------
        # Run the rest of SimDETR.forward via super() – but we have already
        # projected src_vid/src_txt, so we need to call the sub-steps
        # directly to avoid double projection.
        # ------------------------------------------------------------------
        # We replicate SimDETR.forward from the fuser onwards so we can reuse
        # the already-projected features (src_vid_proj, src_txt_proj).
        out = {}
        src_txt_mask_reversed = src_txt_mask.flip(dims=[1])
        eos_idx = l_txt - src_txt_mask_reversed.argmax(dim=1) - 1
        src_sent = src_txt_proj[torch.arange(b), eos_idx]   # [B, D]

        # CTC / VTC inputs
        src_txt_ed = src_txt_proj
        src_vid_ed = src_vid_proj
        src_vid_cls_ed = src_vid_proj.mean(1)
        src_txt_cls_ed = src_sent

        # Fuser
        src_vid_fused = self.fuser(src_vid_proj, src_txt_proj, src_vid_mask, src_txt_mask)

        # Concatenate video+text for the transformer
        src = torch.cat([src_vid_fused, src_txt_proj], dim=1)
        mask = torch.cat([src_vid_mask, src_txt_mask], dim=1).bool()
        pos_vid = self.position_embed(src_vid_fused, src_vid_mask)
        pos_txt = (
            self.txt_position_embed(src_txt_proj)
            if self.use_txt_pos
            else torch.zeros_like(src_txt_proj)
        )
        pos = torch.cat([pos_vid, pos_txt], dim=1)

        video_length = src_vid_fused.shape[1]

        hs, reference, memory, saliency_scores = self.transformer(
            src, ~mask,
            self.query_embed.weight,
            pos,
            self.saliency_proj1,
            video_length=video_length,
            sent_feat=src_sent,
        )

        # ------------------------------------------------------------------
        # Heads – identical to SimDETR.forward
        # ------------------------------------------------------------------
        n_layers = hs.shape[0]
        outputs_class = self.class_embed(hs)
        output_iou_scores = self.iou_head(hs)

        hs_res = self.mask_head(hs)
        hs_mask = hs + hs_res
        query_norm = F.normalize(hs_mask, p=2, dim=-1)
        vid_norm = (
            F.normalize(memory[:, :video_length], p=2, dim=-1)
            .unsqueeze(0)
            .repeat(n_layers, 1, 1, 1)
        )
        mask_logit = torch.matmul(query_norm, vid_norm.transpose(-2, -1))
        mask_norm = (self.logit_scale.exp() * mask_logit).sigmoid()
        out["pred_masks"] = mask_norm[-2]

        from sim_detr.model import inverse_sigmoid as _inv_sig  # local import avoids circular
        reference_before_sigmoid = _inv_sig(reference)
        outputs_coord = self.span_embed(hs) + reference_before_sigmoid
        if self.span_loss_type == "l1":
            outputs_coord = outputs_coord.sigmoid()

        out["pred_logits"] = outputs_class[-1]
        out["pred_spans"] = outputs_coord[-1]
        out["iou_scores"] = output_iou_scores[-1]

        txt_mem = memory[:, video_length:]
        vid_mem = memory[:, :video_length]

        if self.contrastive_align_loss:
            proj_queries = F.normalize(
                self.contrastive_align_projection_query(hs), p=2, dim=-1
            )
            proj_txt_mem = F.normalize(
                self.contrastive_align_projection_txt(txt_mem), p=2, dim=-1
            )
            proj_vid_mem = F.normalize(
                self.contrastive_align_projection_vid(vid_mem), p=2, dim=-1
            )
            out.update(
                proj_queries=proj_queries[-1],
                proj_txt_mem=proj_txt_mem,
                proj_vid_mem=proj_vid_mem,
            )

        out["saliency_scores"] = saliency_scores
        out["video_mask"] = src_vid_mask

        if self.aux_loss:
            out["aux_outputs"] = [
                {
                    "pred_logits": a,
                    "pred_spans": b_,
                    "pred_masks": c,
                    "iou_scores": d,
                }
                for a, b_, c, d in zip(
                    outputs_class[:-1],
                    outputs_coord[:-1],
                    mask_norm[:-1],
                    output_iou_scores[:-1],
                )
            ]
            if self.contrastive_align_loss:
                for idx, d in enumerate(proj_queries[:-1]):
                    out["aux_outputs"][idx].update(
                        proj_queries=d, proj_txt_mem=proj_txt_mem
                    )

        # CTC / VTC features (same as SimDETR)
        out["src_txt_ed"] = src_txt_ed
        out["src_vid_ed"] = src_vid_ed
        out["src_vid_cls_ed"] = src_vid_cls_ed
        out["src_txt_cls_ed"] = src_txt_cls_ed

        # ------------------------------------------------------------------
        # DQ-CGP diagnostics (only when adapter is active and produced output)
        # ------------------------------------------------------------------
        if self.use_query_cgp and self.query_cgp is not None:
            stage_outputs = get_dq_cgp_stage_outputs(self.transformer.decoder)
            early_output = stage_outputs.get(self.query_cgp_after_layer)
            late_output = stage_outputs.get(self.query_cgp_late_after_layer)
            if early_output is not None:
                # Preserve V3's original diagnostic names for the early stage.
                out["query_cgp_temporal_attention"] = (
                    early_output.temporal_attention
                )
                out["query_cgp_basis_weights"] = early_output.basis_weights
            if late_output is not None:
                out["query_cgp_late_temporal_attention"] = (
                    late_output.temporal_attention
                )
                out["query_cgp_late_basis_weights"] = late_output.basis_weights
            if early_output is not None or late_output is not None:
                out["query_cgp_video_mask"] = src_vid_mask.bool()

        return out

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    @classmethod
    def load_from_baseline_checkpoint(
        cls,
        checkpoint_path: str,
        model: "SimDETRWithTiedDualDQCGP",
        device: str = "cpu",
    ) -> None:
        """Initialise DQ-CGP model from a baseline Sim-DETR checkpoint.

        Only ``query_cgp.*`` keys may be missing. All original Sim-DETR
        weights must be present and unmodified.

        Parameters
        ----------
        checkpoint_path : str
            Path to the baseline ``.ckpt`` file.
        model : SimDETRWithTiedDualDQCGP
            The model instance to load into.
        device : str
            Map location for ``torch.load``.
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        source_state = checkpoint.get("model", checkpoint)
        state_dict = {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in source_state.items()
        }
        dq_prefix = "query_cgp."
        source_dq_keys = [key for key in state_dict if key.startswith(dq_prefix)]
        if source_dq_keys:
            raise RuntimeError(
                "--init_from expects a baseline Sim-DETR checkpoint without "
                "query_cgp.* tensors. Use --resume for a DQ-CGP checkpoint."
            )

        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        # Validate: the only allowed missing keys are DQ-CGP parameters.
        disallowed_missing = [k for k in missing if not k.startswith(dq_prefix)]
        if disallowed_missing:
            raise RuntimeError(
                f"Baseline checkpoint is missing non-DQ-CGP keys:\n"
                f"{disallowed_missing}\n"
                "This indicates an architecture mismatch."
            )
        if unexpected:
            raise RuntimeError(
                f"Baseline checkpoint has unexpected keys:\n{unexpected}"
            )
        n_cgp = len([k for k in missing if k.startswith(dq_prefix)])
        print(
            f"[DQ-CGP] Loaded baseline checkpoint. "
            f"{n_cgp} tied DQ-CGP tensors keep their initial values."
        )


__all__ = ["ARCHITECTURE_ID", "SimDETRWithTiedDualDQCGP"]
