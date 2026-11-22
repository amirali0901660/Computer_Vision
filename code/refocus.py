"""Read light-field camera views and create a focal stack.

Refocusing (shift-and-average) aligns the camera images at a chosen disparity.
A point at that disparity stays sharp after averaging the shifted images.
"""

from pathlib import Path

import cv2
import numpy as np


def load_views(folder, view_count=9):
    """Load the central 9x9, 5x5, or 3x3 set of camera views.

    The HCI dataset stores 81 images numbered 000 to 080. The returned array
    has shape (camera rows, camera columns, height, width, BGR channels).
    Pixel values are converted from 0-255 integers to 0-1 floats.
    """
    camera_images = []

    for camera_number in range(81):
        image_path = Path(folder) / f"input_Cam{camera_number:03d}.png"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        camera_images.append(image.astype(np.float32) / 255)

    all_views = np.stack(camera_images).reshape(9, 9, *camera_images[0].shape)
    first_view = (9 - view_count) // 2
    last_view = first_view + view_count
    return all_views[first_view:last_view, first_view:last_view]


def refocus(views, focal_disparities):
    """Shift all camera views to each focal disparity, then average them.

    Only valid image samples contribute to the average at the borders.
    The shift signs follow the HCI disparity convention used by stereo.
    """
    camera_rows, camera_cols, height, width, _ = views.shape
    pixel_y, pixel_x = np.mgrid[:height, :width].astype(np.float32)
    valid_pixel_mask = np.ones((height, width), np.float32)
    middle_row = camera_rows // 2
    middle_col = camera_cols // 2
    focal_stack = []

    for disparity in focal_disparities:
        refocused_image = np.zeros((height, width, 3), np.float32)
        sample_weights = np.zeros((height, width), np.float32)

        for camera_row in range(camera_rows):
            for camera_col in range(camera_cols):
                camera_x = camera_col - middle_col
                camera_y = camera_row - middle_row
                sample_x = (pixel_x - camera_x * disparity).astype(np.float32)
                sample_y = (pixel_y - camera_y * disparity).astype(np.float32)

                shifted_image = cv2.remap(
                    views[camera_row, camera_col], sample_x, sample_y,
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
                )
                shifted_weights = cv2.remap(
                    valid_pixel_mask, sample_x, sample_y,
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
                )
                refocused_image += shifted_image
                sample_weights += shifted_weights

        refocused_image /= np.maximum(sample_weights[..., None], 1e-6)
        focal_stack.append(refocused_image)

    return np.stack(focal_stack)
