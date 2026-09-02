"""Options added only for the isolated QVHighlights experiment."""

from __future__ import annotations

from sim_detr.config import BaseOptions


class ExperimentOptions(BaseOptions):
    def initialize(self):
        super().initialize()
        group = self.parser.add_argument_group(
            "QVHighlights Full D1-attention Binding"
        )
        group.add_argument("--semantic_variant", choices=("full",), default="full")
        group.add_argument(
            "--semantic_evidence_source",
            choices=("d1_attention",),
            default="d1_attention",
        )
        group.add_argument(
            "--semantic_context_variant", choices=("aligned",), default="aligned"
        )
        group.add_argument("--binding_loss_coef", type=float, default=0.2)
        group.add_argument("--semantic_hidden_dim", type=int, default=256)
        group.add_argument("--semantic_dropout", type=float, default=0.1)
        group.add_argument("--semantic_scale_init", type=float, default=1.0)
        group.add_argument(
            "--semantic_no_detach_support",
            dest="semantic_detach_support",
            action="store_false",
            default=True,
        )
        return self.parser
