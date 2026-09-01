"""Official full-range MR evaluation split by ground-truth occurrence count."""

from __future__ import annotations

from standalone_eval.eval import compute_mr_ap, compute_mr_r1


SUBSETS = ("all", "single-occurrence", "multi-occurrence")


def occurrence_subset_rows(ground_truth, subset, max_windows=None):
    if subset not in SUBSETS:
        raise ValueError(f"Unknown occurrence subset: {subset}")
    selected = []
    for row in ground_truth:
        windows = row.get("relevant_windows", [])
        if max_windows is not None:
            windows = windows[:max_windows]
        count = len(windows)
        if subset == "all" or (subset == "single-occurrence" and count <= 1) or (
            subset == "multi-occurrence" and count >= 2
        ):
            selected.append(row)
    return selected


def filter_submission(submission, ground_truth):
    qids = {row["qid"] for row in ground_truth}
    return [row for row in submission if row["qid"] in qids]


def full_range_metrics(submission, ground_truth, num_workers=8, chunksize=50):
    if not ground_truth:
        return {"num_records": 0, "brief": {}}
    mr_ap = compute_mr_ap(
        submission, ground_truth, num_workers=num_workers, chunksize=chunksize
    )
    mr_r1 = compute_mr_r1(submission, ground_truth)
    brief = {
        "MR-full-mAP": mr_ap["average"],
        "MR-full-mAP@0.5": mr_ap["0.5"],
        "MR-full-mAP@0.75": mr_ap["0.75"],
        "MR-full-R1@0.5": mr_r1["0.5"],
        "MR-full-R1@0.7": mr_r1["0.7"],
    }
    return {
        "num_records": len(ground_truth),
        "brief": brief,
        "full": {"MR-mAP": mr_ap, "MR-R1": mr_r1},
    }


def stratified_occurrence_metrics(
    submission, ground_truth, max_windows=None, num_workers=8, chunksize=50
):
    results = {}
    for subset in SUBSETS:
        rows = occurrence_subset_rows(ground_truth, subset, max_windows=max_windows)
        predictions = filter_submission(submission, rows)
        results[subset] = full_range_metrics(
            predictions, rows, num_workers=num_workers, chunksize=chunksize
        )
    return results
