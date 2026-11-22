"""Optimize the disparity map using graph cuts (paper Eq. 13).

The energy has two parts:
  1. A data cost: how well a disparity explains each pixel.
  2. A smoothness cost: how differently neighboring pixels are labeled.

An alpha-beta swap allows pixels with either of two labels to change between
those labels. We use swaps because the squared disparity difference in Eq. 13
is not a metric, so the usual alpha-expansion guarantees do not apply.
"""

import cv2
import numpy as np


def edge_weights(image, smooth_sigma=0.05):
    """Measure color differences between neighboring pixels (Eq. 13).

    These are the paper's printed rho(color difference) weights. They increase
    across strong image edges, unlike more common edge-aware smoothness weights.
    We preserve that formula for the reproduction.
    """
    horizontal_difference = np.sum((image[:, 1:] - image[:, :-1]) ** 2, axis=2)
    vertical_difference = np.sum((image[1:] - image[:-1]) ** 2, axis=2)

    horizontal_weights = 1 - np.exp(-horizontal_difference / (2 * smooth_sigma ** 2))
    vertical_weights = 1 - np.exp(-vertical_difference / (2 * smooth_sigma ** 2))
    return horizontal_weights, vertical_weights


def energy(cost, labels, disparities, horizontal_weights, vertical_weights, smoothness_weight):
    """Calculate the total data cost plus neighbor smoothness cost (Eq. 13)."""
    height, width = labels.shape
    pixel_rows = np.arange(height)[:, None]
    pixel_columns = np.arange(width)[None, :]
    data_energy = cost[labels, pixel_rows, pixel_columns].sum()

    disparity_map = disparities[labels]
    horizontal_energy = np.sum(horizontal_weights * np.diff(disparity_map, axis=1) ** 2)
    vertical_energy = np.sum(vertical_weights * np.diff(disparity_map, axis=0) ** 2)
    return float(data_energy + smoothness_weight * (horizontal_energy + vertical_energy))


def graph_cuts(cost, candidates, image, weight=0.05, cycles=2,
               initial=None, max_pairs=None):
    """Minimize Eq. 13 using repeated alpha-beta swaps.

    During a swap between labels a and b, only pixels currently labeled a or b
    may change. Neighbors with other labels stay fixed. Their smoothness costs
    are added to the data cost of each movable pixel.

    cycles is the maximum number of passes over label pairs. max_pairs=None
    considers all pairs. These are numerical search settings, not new costs.
    """
    try:
        import maxflow
    except ImportError as exc:
        raise RuntimeError("Install PyMaxflow: python -m pip install PyMaxflow") from exc

    if initial is None:
        labels = np.argmin(cost, axis=0).astype(np.int32)
    else:
        labels = initial.copy().astype(np.int32)

    height, width = labels.shape
    horizontal_weights, vertical_weights = edge_weights(image)

    # PyMaxflow uses a 3x3 stencil to specify rightward and downward edges.
    right_structure = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]], np.int8)
    down_structure = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]], np.int8)

    number_of_labels = len(candidates)
    label_pairs = [(a, b) for a in range(number_of_labels)
                   for b in range(a + 1, number_of_labels)]
    if max_pairs is not None:
        label_pairs.sort(key=lambda pair: (pair[1] - pair[0], pair[0]))
        label_pairs = label_pairs[:max_pairs]

    for _ in range(cycles):
        changed_pixels = 0
        current_energy = energy(cost, labels, candidates,
                                horizontal_weights, vertical_weights, weight)

        for label_a, label_b in label_pairs:
            movable = (labels == label_a) | (labels == label_b)
            number_of_movable = int(movable.sum())
            if number_of_movable == 0:
                continue

            fixed = ~movable
            current_disparity = candidates[labels]
            disparity_a = float(candidates[label_a])
            disparity_b = float(candidates[label_b])
            cost_a = cost[label_a].copy()
            cost_b = cost[label_b].copy()

            # For a movable pixel next to a fixed one, add the boundary cost
            # to each choice (a or b). Four directions are handled explicitly.

            # Fixed neighbor on the right.
            fixed_right = movable[:, :-1] & fixed[:, 1:]
            right_penalty = weight * horizontal_weights * fixed_right
            cost_a[:, :-1] += right_penalty * (disparity_a - current_disparity[:, 1:]) ** 2
            cost_b[:, :-1] += right_penalty * (disparity_b - current_disparity[:, 1:]) ** 2

            # Fixed neighbor on the left.
            fixed_left = movable[:, 1:] & fixed[:, :-1]
            left_penalty = weight * horizontal_weights * fixed_left
            cost_a[:, 1:] += left_penalty * (disparity_a - current_disparity[:, :-1]) ** 2
            cost_b[:, 1:] += left_penalty * (disparity_b - current_disparity[:, :-1]) ** 2

            # Fixed neighbor below.
            fixed_below = movable[:-1] & fixed[1:]
            below_penalty = weight * vertical_weights * fixed_below
            cost_a[:-1] += below_penalty * (disparity_a - current_disparity[1:]) ** 2
            cost_b[:-1] += below_penalty * (disparity_b - current_disparity[1:]) ** 2

            # Fixed neighbor above.
            fixed_above = movable[1:] & fixed[:-1]
            above_penalty = weight * vertical_weights * fixed_above
            cost_a[1:] += above_penalty * (disparity_a - current_disparity[:-1]) ** 2
            cost_b[1:] += above_penalty * (disparity_b - current_disparity[:-1]) ** 2

            # Each movable pixel becomes a graph node. Non-movable pixels
            # are represented by -1 in the grid used by PyMaxflow.
            graph = maxflow.Graph[float](number_of_movable, 2 * number_of_movable)
            nodes = graph.add_nodes(number_of_movable)
            node_grid = np.full((height, width), -1, np.int64)
            node_grid[movable] = nodes

            # Source = label a, sink = label b. The terminal costs use
            # opposite positions because a cut pays for the cut edge.
            graph.add_grid_tedges(nodes, cost_b[movable], cost_a[movable])

            # Movable neighbors have a disagreement cost if one chooses a
            # and the other chooses b. Fixed neighbors were handled above.
            pairwise_cost = weight * (disparity_a - disparity_b) ** 2
            right_edge_weights = np.zeros((height, width), np.float32)
            down_edge_weights = np.zeros((height, width), np.float32)
            right_edge_weights[:, :-1] = horizontal_weights * pairwise_cost
            down_edge_weights[:-1] = vertical_weights * pairwise_cost

            graph.add_grid_edges(node_grid, weights=right_edge_weights,
                                 structure=right_structure, symmetric=True)
            graph.add_grid_edges(node_grid, weights=down_edge_weights,
                                 structure=down_structure, symmetric=True)
            graph.maxflow()

            is_sink = graph.get_grid_segments(nodes)
            proposed_labels = labels.copy()
            proposed_labels[movable] = np.where(is_sink, label_b, label_a)

            # Only accept a move when it does not increase the full energy.
            proposed_energy = energy(cost, proposed_labels, candidates,
                                     horizontal_weights, vertical_weights, weight)
            if proposed_energy <= current_energy + 1e-6:
                changed_pixels += int(np.count_nonzero(labels != proposed_labels))
                labels = proposed_labels
                current_energy = proposed_energy

        if changed_pixels == 0:
            break

    return candidates[labels], labels


def multiscale_graph_cuts(cost, candidates, image, weight=0.05,
                          cycles=2, max_pairs=None,
                          scales=(0.25, 0.5, 1.0), initial=None):
    """Apply graph cuts at low, middle, then full image resolution.

    A coarse disparity estimate is enlarged to initialize the next scale.
    The 1/4, 1/2, full-resolution schedule is our implementation choice.
    """
    height, width = image.shape[:2]
    labels = None if initial is None else initial.copy()

    for scale in scales:
        scaled_width = max(2, round(width * scale))
        scaled_height = max(2, round(height * scale))
        new_size = (scaled_width, scaled_height)

        cost_with_channels_last = np.moveaxis(cost, 0, -1)
        small_cost = cv2.resize(cost_with_channels_last, new_size,
                                interpolation=cv2.INTER_AREA)
        small_cost = np.moveaxis(small_cost, -1, 0)
        small_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

        if labels is not None:
            labels = cv2.resize(labels, new_size, interpolation=cv2.INTER_NEAREST)

        _, labels = graph_cuts(small_cost, candidates, small_image, weight,
                               cycles=cycles, initial=labels, max_pairs=max_pairs)

    return candidates[labels], labels
