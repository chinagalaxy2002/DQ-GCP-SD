#!/usr/bin/env bash
# ============================================================================
# train_tied_all_stage.sh - flexible Sim-DETR + tied all-interlayer DQ-CGP utility
#
# The canonical from-scratch effectiveness experiment is locked down in:
#   scripts/train_tied_all_stage_scratch.sh
#
# Use this flexible entry only for resume, smoke tests, or explicitly labelled
# fine-tuning experiments.
#
#   To warm-start from a baseline checkpoint:
#   bash sim_detr/dq_cgp/tied_all_stage/scripts/train_tied_all_stage.sh \
#       --init_from results/baseline/model.ckpt
#
#   Extra args are forwarded to the module entry point.
#
# Architecture:  D1 -> shared DQ -> D2 -> same DQ -> D3 -> same DQ -> D4
# Features:      Official Sim-DETR SlowFast + CLIP-B/32 + TEF / R2-Tuning CLIP
# Dataset:       QVHighlights
# ============================================================================

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${repo_root}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
python_bin="${PYTHON_BIN:-/home/guoxiangyu/miniconda3/envs/sim_detr/bin/python}"

# ---------------------------------------------------------------------------
# Dataset / split
# ---------------------------------------------------------------------------
dset_name=hl
ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_type=clip
results_root=results_dq_cgp/tied_all_stage_flexible
exp_id=sim_detr_tied_all_stage_dq_cgp

train_path=data/highlight_train_release.jsonl
eval_path=data/highlight_val_release.jsonl
eval_split_name=val

# ---------------------------------------------------------------------------
# Feature paths  (same as the official Sim-DETR train.sh – do NOT change)
# ---------------------------------------------------------------------------
feat_root=${repo_root}/data

v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"slowfast"* ]]; then
  v_feat_dirs+=(${feat_root}/slowfast_features)
  (( v_feat_dim += 2304 ))
fi
if [[ ${v_feat_types} == *"clip"* ]]; then
  v_feat_dirs+=(${feat_root}/clip_b32_vid_k4)
  (( v_feat_dim += 3072 ))
fi

if [[ ${t_feat_type} == "clip" ]]; then
  t_feat_dir=${feat_root}/clip_b32_txt_k4
  t_feat_dim=2048
else
  echo "Wrong arg for t_feat_type." && exit 1
fi

# ---------------------------------------------------------------------------
# Sim-DETR backbone hyperparameters  (MUST stay identical to baseline)
# ---------------------------------------------------------------------------
bsz=32
lr_drop=100
lr=0.0001
n_epoch=200
lw_saliency=1.0
seed=2017
VTC_loss_coef=0.3
CTC_loss_coef=0.5
label_loss_coef=4
dec_layers=4
enc_layers=2

# ---------------------------------------------------------------------------
# Parameter-tied all-interlayer DQ-CGP hyperparameters
# ---------------------------------------------------------------------------
use_query_cgp=true          # master switch
query_cgp_num_basis=16
query_cgp_prompt_length=6
query_cgp_router_hidden_dim=256
query_cgp_frf_hidden_dim=512
query_cgp_temperature=1.0
query_cgp_beta=0.05
query_cgp_after_layer=0        # first call:  D1 → DQ → D2
query_cgp_middle_after_layer=1 # second call: D2 → the same DQ → D3
query_cgp_late_after_layer=2   # third call:  D3 → the same DQ → D4
query_cgp_binding_loss_coef=0.2
query_cgp_route_loss_coef=0.01

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
"${python_bin}" -m sim_detr.dq_cgp.tied_all_stage.train_dq_cgp \
  --seed ${seed} \
  --label_loss_coef ${label_loss_coef} \
  --VTC_loss_coef ${VTC_loss_coef} \
  --CTC_loss_coef ${CTC_loss_coef} \
  --dset_name ${dset_name} \
  --ctx_mode ${ctx_mode} \
  --train_path ${train_path} \
  --eval_path ${eval_path} \
  --eval_split_name ${eval_split_name} \
  --v_feat_dirs ${v_feat_dirs[@]} \
  --v_feat_dim ${v_feat_dim} \
  --t_feat_dir ${t_feat_dir} \
  --t_feat_dim ${t_feat_dim} \
  --bsz ${bsz} \
  --results_root ${results_root} \
  --exp_id ${exp_id} \
  --lr ${lr} \
  --n_epoch ${n_epoch} \
  --lw_saliency ${lw_saliency} \
  --lr_drop ${lr_drop} \
  --dec_layers ${dec_layers} \
  --enc_layers ${enc_layers} \
  --use_query_cgp \
  --query_cgp_num_basis ${query_cgp_num_basis} \
  --query_cgp_prompt_length ${query_cgp_prompt_length} \
  --query_cgp_router_hidden_dim ${query_cgp_router_hidden_dim} \
  --query_cgp_frf_hidden_dim ${query_cgp_frf_hidden_dim} \
  --query_cgp_temperature ${query_cgp_temperature} \
  --query_cgp_beta ${query_cgp_beta} \
  --query_cgp_after_layer ${query_cgp_after_layer} \
  --query_cgp_middle_after_layer ${query_cgp_middle_after_layer} \
  --query_cgp_late_after_layer ${query_cgp_late_after_layer} \
  --query_cgp_binding_loss_coef ${query_cgp_binding_loss_coef} \
  --query_cgp_route_loss_coef ${query_cgp_route_loss_coef} \
  "$@"
