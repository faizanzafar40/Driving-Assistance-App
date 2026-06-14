"""Smoke test that pins the lane-detection pipeline's behaviour on the test frame.

This runs the whole Edge Drawing pipeline headless (no OpenCV windows) so I can
confirm it executes end to end and still produces edges after any refactor.
"""

import cv2
import numpy as np

from driving_assistance import lane_detection as ld


def load_test_image():
    image = cv2.imread(str(ld.DEFAULT_IMAGE), cv2.IMREAD_GRAYSCALE)
    assert image is not None, f"missing test image at {ld.DEFAULT_IMAGE}"
    return image


def test_pipeline_runs_and_returns_all_stages():
    image = load_test_image()
    stages = ld.detect_lane_edges(image)

    expected = {
        "raw_gradient",
        "gradient_map",
        "direction_map",
        "edge_area_image",
        "anchors",
        "edges",
    }
    assert expected == set(stages)

    rows, cols = image.shape
    assert stages["gradient_map"].shape == (rows, cols)
    assert stages["edges"].shape == (rows, cols)


def test_direction_map_only_uses_known_labels():
    image = load_test_image()
    direction_map = ld.build_direction_map(cv2.GaussianBlur(image, (5, 5), 0))
    assert set(np.unique(direction_map)).issubset({ld.HORIZONTAL, ld.VERTICAL})


def test_pipeline_marks_edges():
    image = load_test_image()
    stages = ld.detect_lane_edges(image)
    # Linking anchors should leave a non-trivial set of edge pixels behind.
    assert np.count_nonzero(stages["edges"] == ld.EDGE) > 0
