"""Recover a disparity map from light-field camera images using Lin et al.

Choose the scene and experiment at the top, then run: python code/main.py
Ground truth is loaded only after prediction to evaluate the result.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from refocus import load_views, refocus
from symmetry import symmetry_costs, occlusion_probability, in_focus_cost
from matching import stereo_costs, focal_matching
from optimization import edge_weights, energy, graph_cuts, multiscale_graph_cuts


# Change these four settings for each experiment.
SCENE = "dino"                 # "dino" or "boxes"
VIEWS = 9                      # 9x9, 5x5, or 3x3 center camera views
NOISE = 10                     # Gaussian noise standard deviation: NOISE / 255
METHOD = "full"               # "stereo", "symmetry", or "full"

# Numerical settings chosen for this implementation.
CANDIDATES = 41                # Number of possible disparity values per pixel
CYCLES = 5                     # Maximum outer focal-matching refinements
MIN_ENERGY_UPDATE = 1e-3       # Stop when one optimization brings little improvement
GRAPH_CUT_CYCLES = 10          # Maximum passes through all label pairs per scale
GRAPH_CUT_MAX_PAIRS = None     # None means all label pairs
MULTISCALE = True              # Optimize at 1/4, 1/2, then full resolution

# Eq. 13 weights from the paper's "Multi-scale" configuration (Table 1).
LAMBDA_M = 0.4                 # Focal-matching weight
LAMBDA_C = 0.9                 # Stereo weight
LAMBDA_R = 0.08                # Spatial smoothness weight
SIGMA = 0.05                   # Robust-penalty scale for focal symmetry

# The disparity ranges are specific to our HCI scenes, not the paper's scenes.
RANGES = {"dino": (-1.9, 1.9), "boxes": (-2.2, 1.4)}
PROJECT = Path(__file__).resolve().parents[1]
NOISE_SEED = 2026


def read_pfm(path):
    """Read the HCI dataset's single-channel PFM ground-truth disparity."""
    with open(path, "rb") as file:
        assert file.readline().strip() == b"Pf"
        size_line = file.readline().strip()
        while size_line.startswith(b"#"):
            size_line = file.readline().strip()

        width, height = map(int, size_line.split())
        scale = float(file.readline())
        data_type = "<f4" if scale < 0 else ">f4"
        disparity = np.fromfile(file, dtype=data_type).reshape(height, width)
        return np.flipud(disparity) * abs(scale)


def mse(prediction, ground_truth):
    """Compute disparity mean squared error without invalid or border pixels."""
    valid_pixels = np.isfinite(ground_truth) & np.isfinite(prediction)
    valid_pixels[:8] = False
    valid_pixels[-8:] = False
    valid_pixels[:, :8] = False
    valid_pixels[:, -8:] = False
    return float(np.mean((prediction[valid_pixels] - ground_truth[valid_pixels]) ** 2))


def save_comparison(path, center_image, ground_truth, before_cuts,
                    final_disparity, minimum_disparity, maximum_disparity):
    """Save the input, ground truth, and disparity before and after graph cuts."""
    figure, axes = plt.subplots(1, 4, figsize=(16, 4))
    images = [center_image[..., ::-1], ground_truth, before_cuts, final_disparity]
    titles = ["Central view", "Ground truth", "Before graph cuts", "After graph cuts"]

    for index, (axis, image, title) in enumerate(zip(axes, images, titles)):
        if index == 0:
            axis.imshow(image)
        else:
            axis.imshow(image, cmap="turbo", vmin=minimum_disparity,
                        vmax=maximum_disparity)
        axis.set_title(title)
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def run():
    """Compute depth costs, minimize Eq. 13, then evaluate and save the map.

    Stereo: Eq. 11 only. Symmetry: Eqs. 6-8 and 12 plus stereo.
    Full: both of those plus iterative focal matching (Eqs. 9-10, Algorithm 1).
    """
    scene_folder = PROJECT / "data" / SCENE
    result_folder = PROJECT / "results" / "paper_alg1" / SCENE / f"v{VIEWS}_noise{NOISE}" / METHOD
    result_folder.mkdir(parents=True, exist_ok=True)

    minimum_disparity, maximum_disparity = RANGES[SCENE]
    disparity_candidates = np.linspace(
        minimum_disparity, maximum_disparity, CANDIDATES
    ).astype(np.float32)
    disparity_step = float(disparity_candidates[1] - disparity_candidates[0])

    # Extend the focal stack on both ends so each candidate has a full window.
    # The paper sets the maximum focal shift to 1/5 of the disparity range.
    focal_offsets = max(1, int(round(
        (maximum_disparity - minimum_disparity) / (5 * disparity_step)
    )))
    focal_disparities = minimum_disparity + np.arange(
        -focal_offsets, CANDIDATES + focal_offsets
    ) * disparity_step

    print(f"{SCENE} | {VIEWS}x{VIEWS} cameras | noise {NOISE}/255 | {METHOD}")
    views = load_views(scene_folder, view_count=VIEWS)
    if NOISE:
        random_generator = np.random.default_rng(NOISE_SEED)
        added_noise = random_generator.normal(0, NOISE / 255, views.shape)
        views = np.clip(views + added_noise, 0, 1).astype(np.float32)
    center_image = views[VIEWS // 2, VIEWS // 2]

    # Eq. 11: estimate disparity by matching colors and gradients across views.
    print("Computing stereo costs...")
    stereo = stereo_costs(views, disparity_candidates)

    if METHOD != "stereo":
        print("Computing focal-stack costs...")
        focal_stack = refocus(views, focal_disparities)
        normal, antisymmetric = symmetry_costs(
            focal_stack, disparity_candidates, focal_offsets, SIGMA
        )
        initial_matching = focal_matching(
            focal_stack, center_image, disparity_candidates, focal_offsets,
            view_count=VIEWS
        )
        occlusion = occlusion_probability(normal, initial_matching)
        focus_cost = in_focus_cost(normal, antisymmetric, occlusion)

    # Eq. 13: combine whichever costs belong to the selected method.
    data_cost = LAMBDA_C * stereo
    if METHOD == "symmetry":
        data_cost = data_cost + focus_cost
    elif METHOD == "full":
        data_cost = data_cost + focus_cost + LAMBDA_M * initial_matching

    initial_disparity = disparity_candidates[np.argmin(data_cost, axis=0)]
    before_graph_cuts = initial_disparity.copy()

    # Algorithm 1 starts at zero disparity (the closest available label).
    zero_label = int(np.argmin(np.abs(disparity_candidates)))
    disparity_labels = np.full(center_image.shape[:2], zero_label, dtype=np.int32)
    disparity = disparity_candidates[disparity_labels]
    optimizer = multiscale_graph_cuts if MULTISCALE else graph_cuts

    print("Optimizing disparity...")
    if METHOD == "full":
        horizontal_weights, vertical_weights = edge_weights(center_image)

        for iteration in range(CYCLES):
            # Use the CURRENT disparity to predict new focal profiles.
            refined_matching = focal_matching(
                focal_stack, center_image, disparity_candidates, focal_offsets,
                depth=disparity, view_count=VIEWS
            )
            data_cost = focus_cost + LAMBDA_M * refined_matching + LAMBDA_C * stereo
            before_graph_cuts = disparity_candidates[np.argmin(data_cost, axis=0)]

            previous_energy = energy(
                data_cost, disparity_labels, disparity_candidates,
                horizontal_weights, vertical_weights, LAMBDA_R
            )
            disparity, disparity_labels = optimizer(
                data_cost, disparity_candidates, center_image, LAMBDA_R,
                cycles=GRAPH_CUT_CYCLES, max_pairs=GRAPH_CUT_MAX_PAIRS,
                initial=disparity_labels
            )
            updated_energy = energy(
                data_cost, disparity_labels, disparity_candidates,
                horizontal_weights, vertical_weights, LAMBDA_R
            )
            improvement = previous_energy - updated_energy
            print(f"  Iteration {iteration + 1}: energy improvement {improvement:.3f}")

            # Matching costs change across outer iterations, so this is only
            # a practical stopping check, not proof of global convergence.
            if improvement < MIN_ENERGY_UPDATE:
                break
    else:
        # Stereo/symmetry have fixed costs and need only one optimizer call.
        disparity, disparity_labels = optimizer(
            data_cost, disparity_candidates, center_image, LAMBDA_R,
            cycles=GRAPH_CUT_CYCLES, max_pairs=GRAPH_CUT_MAX_PAIRS,
            initial=disparity_labels
        )

    # Ground truth is never used before this point.
    ground_truth = read_pfm(scene_folder / "gt_disp_lowres.pfm")
    stereo_disparity = disparity_candidates[np.argmin(stereo, axis=0)]
    scores = {
        "stereo_unregularized": mse(stereo_disparity, ground_truth),
        "initial": mse(initial_disparity, ground_truth),
        "before_graph_cuts": mse(before_graph_cuts, ground_truth),
        "graph_cuts": mse(disparity, ground_truth),
    }
    for name, score in scores.items():
        print(f"  {name}: {score:.6f}")

    np.save(result_folder / "disparity.npy", disparity)
    with open(result_folder / "metrics.txt", "w") as file:
        for name, score in scores.items():
            file.write(f"{name}: {score:.6f}\n")

    save_comparison(
        result_folder / "comparison.png", center_image, ground_truth,
        before_graph_cuts, disparity, minimum_disparity, maximum_disparity
    )
    print(f"Saved to {result_folder}")


if __name__ == "__main__":
    run()
