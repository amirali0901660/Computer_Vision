"""Diagnose which disparity candidates are favored by each matching cue.

Run from the project root: python code/utils/check_costs.py
This diagnostic uses ground truth ONLY for scoring, never to build cost volumes.
It also reads the saved final disparities; no graph-cut rerun is needed.
"""

import sys
from pathlib import Path

import numpy as np

# Allow running this diagnostic directly from code/utils/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import (CANDIDATES, LAMBDA_C, LAMBDA_M, NOISE_SEED,
                  PROJECT, RANGES, SIGMA, read_pfm)
from matching import focal_matching, stereo_costs
from refocus import load_views, refocus
from symmetry import in_focus_cost, occlusion_probability, symmetry_costs


SCENE = "dino"
VIEWS = 9
NOISE = 10


def evaluate(name, cost, true_label, valid):
    """Check label accuracy and whether the true depth beats distant labels."""
    best_label = np.argmin(cost, axis=0)
    close = np.abs(best_label - true_label) <= 1

    true_cost = np.take_along_axis(cost, true_label[None], axis=0)[0]
    label_numbers = np.arange(len(cost))[:, None, None]
    far = np.abs(label_numbers - true_label[None]) > 1
    wrong_cost = np.min(np.where(far, cost, np.inf), axis=0)
    margin = wrong_cost - true_cost

    print(f"{name}:")
    print(f"  Best label within one step: {100 * np.mean(close[valid]):.2f}%")
    print(f"  True depth beats distant alternatives: {100 * np.mean((margin > 0)[valid]):.2f}%")
    print(f"  Median cost margin: {float(np.median(margin[valid])):.6f}")


def evaluate_disparity(predicted, ground_truth, valid, step):
    """Return MSE, one-step accuracy and high-percentile absolute errors."""
    error = np.abs(predicted[valid] - ground_truth[valid])
    return (float(np.mean(error ** 2)),
            float(100 * np.mean(error <= step)),
            float(np.percentile(error, 90)),
            float(np.percentile(error, 99)))


def run():
    """Compute initial cost volumes and compare with saved final disparity maps."""
    folder = PROJECT / "data" / SCENE
    low, high = RANGES[SCENE]
    candidates = np.linspace(low, high, CANDIDATES).astype(np.float32)
    step = float(candidates[1] - candidates[0])
    offsets = max(1, int(round((high - low) / (5 * step))))
    focal_labels = low + np.arange(-offsets, CANDIDATES + offsets) * step

    views = load_views(folder, view_count=VIEWS)
    if NOISE:
        rng = np.random.default_rng(NOISE_SEED)
        views = np.clip(views + rng.normal(0, NOISE / 255, views.shape),
                        0, 1).astype(np.float32)
    center = views[VIEWS // 2, VIEWS // 2]

    print("Computing initial stereo and focal-stack costs...")
    stereo = stereo_costs(views, candidates)
    stack = refocus(views, focal_labels)
    normal, anti = symmetry_costs(stack, candidates, offsets, SIGMA)
    matching = focal_matching(stack, center, candidates, offsets, view_count=VIEWS)
    beta = occlusion_probability(normal, matching)
    focus = in_focus_cost(normal, anti, beta)

    gt = read_pfm(folder / "gt_disp_lowres.pfm")
    valid = np.isfinite(gt) & (gt >= low) & (gt <= high)
    valid[:8] = False
    valid[-8:] = False
    valid[:, :8] = False
    valid[:, -8:] = False
    true_label = np.abs(candidates[:, None, None] - gt[None]).argmin(axis=0)

    costs = {
        "stereo": LAMBDA_C * stereo,
        "symmetry": LAMBDA_C * stereo + focus,
        "full": LAMBDA_C * stereo + focus + LAMBDA_M * matching,
    }
    print(f"Valid pixels: {int(valid.sum())}")
    for name, cost in [("Normal symmetry only", normal),
                       ("Occlusion-aware symmetry only", focus),
                       ("Initial focal matching only", matching),
                       *costs.items()]:
        evaluate(name, cost, true_label, valid)

    print("\nInitial vs. final disparity:")
    for name, cost in costs.items():
        path = (PROJECT / "results" / "paper_alg1" / SCENE /
                f"v{VIEWS}_noise{NOISE}" / name / "disparity.npy")
        if not path.exists():
            print(f"{name}: final disparity not found at {path}")
            continue

        # For 'full', this is the initial cost BEFORE iterative refinement.
        initial = candidates[np.argmin(cost, axis=0)]
        final = np.load(path)
        for stage, predicted in [("initial", initial), ("final", final)]:
            mse, accuracy, p90, p99 = evaluate_disparity(predicted, gt, valid, step)
            print(f"{name} {stage}: MSE={mse:.6f}, within step={accuracy:.2f}%, "
                  f"p90={p90:.4f}, p99={p99:.4f}")


if __name__ == "__main__":
    run()
