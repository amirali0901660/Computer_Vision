"""Compute depth costs from stereo views and focal-stack matching.

Stereo compares colors and gradients across camera views (paper Eq. 11).
Focal matching compares the observed stack with one rendered from a guessed
scene disparity (paper Eqs. 9-10).
"""

import cv2
import numpy as np

from symmetry import robust


def stereo_costs(views, disparities, sigma_color=0.05, sigma_grad=0.05):
    """Measure color and gradient agreement for every candidate disparity.

    All views are aligned to the central camera. A good disparity makes the
    corresponding colors and image gradients similar (Eq. 11, 50% each).
    """
    camera_rows, camera_cols, height, width, _ = views.shape
    middle_row = camera_rows // 2
    middle_col = camera_cols // 2
    center_image = views[middle_row, middle_col]

    # Gradients are computed once because they do not depend on disparity.
    center_gray = cv2.cvtColor(center_image, cv2.COLOR_BGR2GRAY)
    center_gradients = []
    for x_order, y_order in ((1, 0), (0, 1)):
        gradient = cv2.Sobel(center_gray, cv2.CV_32F, x_order, y_order, ksize=3) / 8
        center_gradients.append(gradient)

    view_gradients = []
    for camera_row in range(camera_rows):
        row_gradients = []
        for camera_col in range(camera_cols):
            gray_image = cv2.cvtColor(views[camera_row, camera_col], cv2.COLOR_BGR2GRAY)
            gradients = []
            for x_order, y_order in ((1, 0), (0, 1)):
                gradient = cv2.Sobel(gray_image, cv2.CV_32F, x_order, y_order, ksize=3) / 8
                gradients.append(gradient)
            row_gradients.append(gradients)
        view_gradients.append(row_gradients)

    pixel_y, pixel_x = np.mgrid[:height, :width].astype(np.float32)
    stereo_cost = np.zeros((len(disparities), height, width), np.float32)
    total_views = camera_rows * camera_cols

    for candidate_index, disparity in enumerate(disparities):
        for camera_row in range(camera_rows):
            for camera_col in range(camera_cols):
                if camera_row == middle_row and camera_col == middle_col:
                    continue  # The center image matches itself exactly.

                camera_x = camera_col - middle_col
                camera_y = camera_row - middle_row
                sample_x = (pixel_x - camera_x * disparity).astype(np.float32)
                sample_y = (pixel_y - camera_y * disparity).astype(np.float32)

                shifted_image = cv2.remap(
                    views[camera_row, camera_col], sample_x, sample_y,
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
                )
                color_cost = robust(shifted_image - center_image, sigma_color)

                gradient_differences = []
                for direction in range(2):
                    gradient = cv2.remap(
                        view_gradients[camera_row][camera_col][direction],
                        sample_x, sample_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE
                    )
                    gradient_differences.append(gradient - center_gradients[direction])

                squared_gradient_difference = (
                    gradient_differences[0] ** 2 + gradient_differences[1] ** 2
                )
                gradient_cost = 1 - np.exp(
                    -squared_gradient_difference / (2 * sigma_grad ** 2)
                )
                stereo_cost[candidate_index] += (color_cost + gradient_cost) / (2 * total_views)

    return stereo_cost


def focal_matching(focal_stack, center_image, disparities, offsets,
                   depth=None, sigma=0.05, view_count=9, depth_iterations=3):
    """Compare observed and rendered local focal profiles (Eqs. 9-10).

    For initialization, depth=None assumes equal neighboring disparities.
    Later, the current disparity map changes the sampled center-view locations.
    Three fixed-point updates approximate the implicit location in Eq. 9; this
    is our implementation choice, not a parameter given by the paper.
    """
    height, width = center_image.shape[:2]
    pixel_y, pixel_x = np.mgrid[:height, :width].astype(np.float32)
    valid_pixel_mask = np.ones((height, width), np.float32)
    middle_camera = view_count // 2
    disparity_step = float(disparities[1] - disparities[0])
    matching_cost = np.zeros((len(disparities), height, width), np.float32)

    for offset_index in range(offsets + 1):
        focus_shift = offset_index * disparity_step
        rendered_image = np.zeros((height, width, 3), np.float32)
        sample_weights = np.zeros((height, width), np.float32)

        # Predict a focal image by sampling the center camera at shifted points.
        for camera_row in range(view_count):
            for camera_col in range(view_count):
                camera_x = camera_col - middle_camera
                camera_y = camera_row - middle_camera
                sample_x = (pixel_x - camera_x * focus_shift).astype(np.float32)
                sample_y = (pixel_y - camera_y * focus_shift).astype(np.float32)

                if depth is not None:
                    for _ in range(depth_iterations):
                        sampled_depth = cv2.remap(
                            depth, sample_x, sample_y, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE
                        )
                        relative_depth = depth - sampled_depth + focus_shift
                        sample_x = (pixel_x - camera_x * relative_depth).astype(np.float32)
                        sample_y = (pixel_y - camera_y * relative_depth).astype(np.float32)

                shifted_center = cv2.remap(
                    center_image, sample_x, sample_y, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT
                )
                shifted_weights = cv2.remap(
                    valid_pixel_mask, sample_x, sample_y, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT
                )
                rendered_image += shifted_center
                sample_weights += shifted_weights

        rendered_image /= np.maximum(sample_weights[..., None], 1e-6)

        # Each candidate is compared to the observed stack at the same shift.
        for candidate_index in range(len(disparities)):
            observed_image = focal_stack[candidate_index + offsets + offset_index]
            difference = rendered_image - observed_image
            matching_cost[candidate_index] += robust(difference, sigma) / (offsets + 1)

    return matching_cost
