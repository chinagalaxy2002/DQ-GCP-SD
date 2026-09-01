"""Semantic-calibration-only option extensions."""

from __future__ import annotations

import argparse
import sys

from sim_detr.config import BaseOptions, TestOptions


def add_semantic_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group("Candidate-Conditioned Semantic Calibration")
    group.add_argument("--semantic_variant", choices=("native", "static", "full"), default="full")
    group.add_argument("--semantic_context_variant", choices=("aligned", "roll", "uniform"), default="aligned")
    group.add_argument("--semantic_hidden_dim", type=int, default=256)
    group.add_argument("--semantic_dropout", type=float, default=0.1)
    group.add_argument("--semantic_scale_init", type=float, default=1.0)
    group.add_argument("--semantic_detach_support", dest="semantic_detach_support", action="store_true", default=True)
    group.add_argument("--semantic_no_detach_support", dest="semantic_detach_support", action="store_false")
    group.add_argument("--semantic_evidence_source", choices=("native_pred_mask",), default="native_pred_mask")
    group.add_argument("--semantic_diagnostic_mode", action="store_true")
    group.add_argument("--init_from_native", type=str, default=None)
    group.add_argument("--semantic_scale_override", type=float, default=None)
    return parser


def _capture_runtime_semantic_overrides(argv):
    parser = argparse.ArgumentParser(add_help=False)
    # Do not install defaults here: values absent from the command line must
    # remain whatever the checkpoint's opt.json recorded.
    parser.add_argument("--semantic_variant", choices=("native", "static", "full"), default=argparse.SUPPRESS)
    parser.add_argument("--semantic_context_variant", choices=("aligned", "roll", "uniform"), default=argparse.SUPPRESS)
    parser.add_argument("--semantic_hidden_dim", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--semantic_dropout", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--semantic_scale_init", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--semantic_detach_support", dest="semantic_detach_support", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--semantic_no_detach_support", dest="semantic_detach_support", action="store_false", default=argparse.SUPPRESS)
    parser.add_argument("--semantic_evidence_source", choices=("native_pred_mask",), default=argparse.SUPPRESS)
    parser.add_argument("--semantic_diagnostic_mode", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--init_from_native", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--semantic_scale_override", type=float, default=argparse.SUPPRESS)
    return parser.parse_known_args(argv)[0]


class SemanticBaseOptions(BaseOptions):
    def initialize(self):
        super().initialize()
        add_semantic_args(self.parser)
        return self.parser


class SemanticTestOptions(TestOptions):
    def initialize(self):
        super().initialize()
        add_semantic_args(self.parser)
        return self.parser

    def parse(self, a_feat_dir=None):
        runtime = _capture_runtime_semantic_overrides(sys.argv[1:])
        opt = super().parse(a_feat_dir=a_feat_dir)
        # TestOptions restores the training opt.json. Reapply explicitly
        # requested intervention flags for counterfactual evaluation.
        for key, value in vars(runtime).items():
            if value is not None:
                setattr(opt, key, value)
        return opt
