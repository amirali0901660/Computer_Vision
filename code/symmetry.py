"""Focal-stack symmetry and occlusion cues (paper Eqs. 6-8 and 12)."""

import numpy as np


def robust(difference, sigma=0.05):
    """Robust difference: rho(v) = 1 - exp(-||v||² / (2 sigma²)).

    The last axis contains the image channels. Large differences are bounded
    rather than growing without limit.
    """
    squared_difference = np.sum(difference ** 2, axis=-1)
    return 1 - np.exp(-squared_difference / (2 * sigma ** 2))


def symmetry_costs(focal_stack, disparity_candidates, offsets, sigma=0.05):
    """Compute symmetry and antisymmetry costs for each candidate (Eqs. 6-7).

    Compare focal slices equally far before and after a candidate disparity.
    The integral in the paper is approximated by an average over focal offsets.
    A smaller cost suggests that the candidate is near the correct focus.
    """
    _, height, width, _ = focal_stack.shape
    symmetry = np.zeros((len(disparity_candidates), height, width), np.float32)
    antisymmetry = np.zeros_like(symmetry)

    for candidate_index in range(len(disparity_candidates)):
        middle_slice = candidate_index + offsets

        for offset in range(1, offsets + 1):
            slice_before = focal_stack[middle_slice - offset]
            slice_after = focal_stack[middle_slice + offset]
            center_slice = focal_stack[middle_slice]

            symmetry[candidate_index] += robust(slice_before - slice_after, sigma) / offsets
            antisymmetry[candidate_index] += robust(
                slice_before + slice_after - 2 * center_slice, sigma
            ) / offsets

    return symmetry, antisymmetry


def occlusion_probability(symmetry, initial_matching):
    """Estimate how likely each pixel is to be occluded (Eq. 12).

    Combine three indicators: high minimum symmetry cost, high initial
    matching cost, and high variation in the symmetry cost across disparities.
    Each is scaled by its 90th percentile, as described in the paper.
    """
    minimum_symmetry = symmetry.min(axis=0)
    minimum_matching = initial_matching.min(axis=0)
    symmetry_variance = symmetry.var(axis=0)

    occlusion = np.ones(symmetry.shape[1:], np.float32)
    for measurement in (minimum_symmetry, minimum_matching, symmetry_variance):
        scale = max(float(np.percentile(measurement, 90)), 1e-6)
        occlusion *= 1 - np.exp(-measurement ** 2 / (2 * scale ** 2))

    return occlusion


def in_focus_cost(symmetry, antisymmetry, occlusion):
    """Blend symmetry and antisymmetry using occlusion probability (Eq. 8)."""
    lower_cost = np.minimum(symmetry, antisymmetry)
    return (1 - occlusion)[None] * symmetry + occlusion[None] * lower_cost
