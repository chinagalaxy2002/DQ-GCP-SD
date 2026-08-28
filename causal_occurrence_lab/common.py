"""Shared runtime helpers for the isolated Sim-DETR experiments.

The helpers in this file are deliberately thin adapters around the existing
Sim-DETR code.  They do not write to the production source tree and they do
not alter a checkpoint on disk.
"""

from __future__ import annotations

import copy
import json
import os
import random
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import torch


LAB_DIR = Path(__file__).resolve().parent
REPO_ROOT = LAB_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _install_torchvision_focal_fallback() -> None:
    """Provide the one torchvision op used by Sim-DETR when torchvision is absent.

    The current research environment may not install torchvision alongside its
    PyTorch build.  Sim-DETR only imports ``sigmoid_focal_loss`` from it, so an
    isolated, mathematically equivalent fallback keeps this lab runnable
    without changing the production model file or installing packages.
    """

    try:
        import torchvision  # noqa: F401
        return
    except Exception as exc:
        if "torchvision" in sys.modules:
            for name in list(sys.modules):
                if name == "torchvision" or name.startswith("torchvision."):
                    sys.modules.pop(name, None)
        if not isinstance(exc, (ImportError, ModuleNotFoundError, RuntimeError)):
            raise

    import torch.nn.functional as F

    def sigmoid_focal_loss(inputs, targets, alpha=-1, gamma=2, reduction="none"):
        probability = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = probability * targets + (1 - probability) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)
        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss
        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        return loss

    focal = types.ModuleType("torchvision.ops.focal_loss")
    focal.sigmoid_focal_loss = sigmoid_focal_loss
    ops = types.ModuleType("torchvision.ops")
    ops.focal_loss = focal
    torchvision = types.ModuleType("torchvision")
    torchvision.ops = ops
    sys.modules["torchvision"] = torchvision
    sys.modules["torchvision.ops"] = ops
    sys.modules["torchvision.ops.focal_loss"] = focal


_install_torchvision_focal_fallback()


def ensure_repo_on_path() -> Path:
    """Make imports work when a lab script is called by filename."""

    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return REPO_ROOT


def as_namespace(value: Any) -> SimpleNamespace:
    """Convert a saved argparse namespace/dict into a mutable namespace."""

    if isinstance(value, SimpleNamespace):
        return copy.deepcopy(value)
    if hasattr(value, "__dict__"):
        return SimpleNamespace(**copy.deepcopy(vars(value)))
    if isinstance(value, Mapping):
        return SimpleNamespace(**copy.deepcopy(dict(value)))
    raise TypeError(f"Cannot convert {type(value)!r} to an option namespace")


def load_checkpoint(path: os.PathLike[str] | str) -> Dict[str, Any]:
    """Load a Sim-DETR checkpoint on CPU with the explicit safe-load choice."""

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older PyTorch versions do not expose weights_only
        return torch.load(path, map_location="cpu")


def checkpoint_state(checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, Mapping):
        raise TypeError("Checkpoint model state must be a mapping")
    return dict(state)


def normalize_state_dict(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Remove only a DataParallel ``module.`` prefix, if present."""

    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }


def checkpoint_options(checkpoint: Mapping[str, Any]) -> SimpleNamespace:
    saved = checkpoint.get("opt", checkpoint.get("model_cfg"))
    if saved is None:
        raise KeyError(
            "Checkpoint has neither 'opt' nor 'model_cfg'. Pass explicit model "
            "overrides or use a checkpoint produced by Sim-DETR training."
        )
    return as_namespace(saved)


def resolve_device(requested: Optional[str | int], saved: Any = None) -> torch.device:
    """Resolve a human-friendly device value without touching global CUDA state."""

    value = requested if requested is not None else saved
    if value is None:
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    if isinstance(value, int):
        value = f"cuda:{value}" if value >= 0 else "cpu"
    elif isinstance(value, torch.device):
        if value.type == "cuda" and value.index is None:
            value = "cuda:0"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is unavailable")
    return device


def apply_option_overrides(
    opt: Any,
    *,
    device: Optional[str | int] = None,
    data_path: Optional[str] = None,
    split: Optional[str] = None,
    num_workers: Optional[int] = None,
    max_windows: Optional[int] = None,
    max_v_l: Optional[int] = None,
    eval_bsz: Optional[int] = None,
) -> SimpleNamespace:
    """Copy checkpoint options and apply safe analysis-time overrides."""

    out = as_namespace(opt)
    saved_device = getattr(out, "device", None)
    out.device = resolve_device(device, saved_device)
    if data_path is not None:
        out.eval_path = str(Path(data_path).expanduser().resolve())
    if split is not None:
        out.eval_split_name = split
    if num_workers is not None:
        out.num_workers = int(num_workers)
    if max_windows is not None:
        out.max_windows = int(max_windows)
    if max_v_l is not None:
        out.max_v_l = int(max_v_l)
    if eval_bsz is not None:
        out.eval_bsz = int(eval_bsz)

    # Fields used by older saved opts or by the DQ builder.
    defaults = {
        "a_feat_dir": None,
        "a_feat_dim": 0,
        "aux_loss": True,
        "contrastive_align_loss": False,
        "contrastive_align_loss_coef": 0.0,
        "dset_domain": None,
        "data_ratio": 1.0,
        "debug": False,
        "no_norm_vfeat": False,
        "no_norm_tfeat": False,
        "txt_drop_ratio": 0.0,
        "use_query_cgp": False,
        "query_cgp_num_basis": 16,
        "query_cgp_prompt_length": 6,
        "query_cgp_router_hidden_dim": 256,
        "query_cgp_frf_hidden_dim": 512,
        "query_cgp_temperature": 1.0,
        "query_cgp_beta": 0.05,
        "query_cgp_after_layer": 0,
        "query_cgp_binding_loss_coef": 0.2,
        "query_cgp_route_loss_coef": 0.01,
        "set_cost_mask": 6.0,
        "mask_loss_coef": 6.0,
        "iou_scores_loss_coef": 2.0,
        "temperature": 0.07,
        "saliency_margin": 0.2,
    }
    for key, value in defaults.items():
        if not hasattr(out, key):
            setattr(out, key, value)
    if not hasattr(out, "gpu_id"):
        out.gpu_id = out.device.index or 0 if out.device.type == "cuda" else -1
    if not hasattr(out, "eval_bsz"):
        out.eval_bsz = 100
    return out


def is_dq_state(state: Mapping[str, Any]) -> bool:
    return any(
        (key[7:] if key.startswith("module.") else key).startswith("query_cgp.")
        for key in state
    )


def build_model_for_mode(
    checkpoint_path: os.PathLike[str] | str,
    mode: str,
    *,
    device: Optional[str | int] = None,
    option_overrides: Optional[Mapping[str, Any]] = None,
):
    """Build and load one of the four Phase-1 checkpoint modes.

    ``dq_stripped`` constructs vanilla Sim-DETR and loads all common weights
    after removing exactly the ``query_cgp.*`` tensors.  This is intentionally
    strict: an architecture mismatch must fail loudly instead of producing an
    apparently valid comparison.
    """

    ensure_repo_on_path()
    if mode not in {"baseline", "dq_active", "dq_beta_zero", "dq_stripped"}:
        raise ValueError(f"Unknown checkpoint mode: {mode}")

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = load_checkpoint(checkpoint_path)
    state = normalize_state_dict(checkpoint_state(checkpoint))
    opt = apply_option_overrides(
        checkpoint_options(checkpoint), device=device,
        **(dict(option_overrides or {})),
    )
    opt.use_query_cgp = mode in {"dq_active", "dq_beta_zero"}

    if mode in {"dq_active", "dq_beta_zero"}:
        from sim_detr.dq_cgp.model_builder import build_model_dq_cgp

        model, criterion = build_model_dq_cgp(opt)
        model.load_state_dict(state, strict=True)
        if mode == "dq_beta_zero":
            model.query_cgp.set_beta(0.0)
    else:
        from sim_detr.model import build_model

        model, criterion = build_model(opt)
        if mode == "dq_stripped":
            state = {key: value for key, value in state.items()
                     if not key.startswith("query_cgp.")}
        elif is_dq_state(state):
            raise RuntimeError(
                f"Mode {mode!r} received a DQ checkpoint. Use dq_stripped to "
                "remove query_cgp.* explicitly."
            )
        model.load_state_dict(state, strict=True)

    model.to(opt.device)
    criterion.to(opt.device)
    model.eval()
    criterion.eval()
    return model, criterion, opt, checkpoint


def build_dataset(opt: Any, data_path: str, *, load_labels: bool = True):
    """Build the native QVHighlights dataset with checkpoint-compatible args."""

    ensure_repo_on_path()
    from sim_detr.start_end_dataset import StartEndDataset
    from sim_detr.start_end_dataset_audio import StartEndDataset_audio

    kwargs = dict(
        dset_name=opt.dset_name,
        data_path=str(Path(data_path).expanduser().resolve()),
        v_feat_dirs=opt.v_feat_dirs,
        q_feat_dir=opt.t_feat_dir,
        q_feat_type="last_hidden_state",
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        ctx_mode=opt.ctx_mode,
        data_ratio=getattr(opt, "data_ratio", 1.0),
        normalize_v=not getattr(opt, "no_norm_vfeat", False),
        normalize_t=not getattr(opt, "no_norm_tfeat", False),
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        span_loss_type=opt.span_loss_type,
        txt_drop_ratio=0.0,
        dset_domain=getattr(opt, "dset_domain", None),
    )
    if hasattr(opt, "a_feat_dir") and opt.a_feat_dir is not None:
        kwargs["a_feat_dir"] = opt.a_feat_dir
        dataset = StartEndDataset_audio(**kwargs)
    else:
        dataset = StartEndDataset(**kwargs)
    if hasattr(dataset, "load_labels"):
        dataset.load_labels = bool(load_labels)
    return dataset


def collate_and_prepare(batch, opt: Any):
    ensure_repo_on_path()
    if getattr(opt, "a_feat_dir", None) is None:
        from sim_detr.start_end_dataset import start_end_collate, prepare_batch_inputs
        return prepare_batch_inputs(
            start_end_collate(batch)[1], opt.device,
            non_blocking=getattr(opt, "pin_memory", False),
        )
    from sim_detr.start_end_dataset_audio import (
        start_end_collate_audio, prepare_batch_inputs_audio,
    )
    return prepare_batch_inputs_audio(
        start_end_collate_audio(batch)[1], opt.device,
        non_blocking=getattr(opt, "pin_memory", False),
    )


def layer_outputs(outputs: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    """Return D1...D4 (or all available decoder layers) in order."""

    aux = list(outputs.get("aux_outputs", []))
    if not aux:
        return [outputs]
    return aux + [outputs]


def tensor_to_numpy(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if isinstance(value, Mapping):
        return {key: tensor_to_numpy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_to_numpy(item) for item in value]
    return value


def jsonable(value: Any) -> Any:
    """Convert tensors/numpy/scalars recursively for JSON manifests."""

    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, torch.device):
        return str(value)
    return value


def save_json(path: os.PathLike[str] | str, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, indent=2, ensure_ascii=False)


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def discover_dq_checkpoint(root: Optional[os.PathLike[str] | str] = None) -> Path:
    """Resolve the intended release V3 checkpoint, or fail with candidates."""

    root_path = Path(root or REPO_ROOT / "results_dq_cgp")
    candidates = sorted(root_path.rglob("model_best.ckpt"))
    preferred = [
        path for path in candidates
        if "beta_0p050_bind_0p20_route_0p010_center" in str(path)
    ]
    if preferred:
        return preferred[-1]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No model_best.ckpt found below {root_path}. Pass --checkpoint."
        )
    raise RuntimeError(
        "More than one DQ checkpoint exists and no preferred V3 checkpoint was "
        "found; pass --checkpoint explicitly:\n" +
        "\n".join(str(path) for path in candidates[:30])
    )


def default_split_path(split: str) -> Path:
    if split not in {"train", "val", "test"}:
        raise ValueError(split)
    candidates = [
        REPO_ROOT / "data" / f"highlight_{split}_with_gt.jsonl",
        REPO_ROOT / "data" / f"highlight_{split}_release.jsonl",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Could not find a QVHighlights split. Tried:\n" +
        "\n".join(str(path) for path in candidates)
    )


def checkpoint_manifest(path: os.PathLike[str] | str, checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    opt = checkpoint.get("opt", checkpoint.get("model_cfg"))
    if hasattr(opt, "__dict__"):
        opt = vars(opt)
    return {
        "checkpoint": str(Path(path).expanduser().resolve()),
        "epoch": checkpoint.get("epoch"),
        "metric": checkpoint.get("metric", checkpoint.get("best_metric")),
        "options": jsonable(opt or {}),
        "has_query_cgp": is_dq_state(checkpoint_state(checkpoint)),
    }


__all__ = [
    "LAB_DIR", "REPO_ROOT", "apply_option_overrides", "as_namespace",
    "build_dataset", "build_model_for_mode", "checkpoint_manifest",
    "checkpoint_options", "checkpoint_state", "collate_and_prepare",
    "default_split_path", "discover_dq_checkpoint", "ensure_repo_on_path",
    "is_dq_state", "jsonable", "layer_outputs", "load_checkpoint",
    "normalize_state_dict", "resolve_device", "save_json", "set_seed",
    "tensor_to_numpy",
]
