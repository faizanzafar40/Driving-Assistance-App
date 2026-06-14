"""Lane-marking detection with the Edge Drawing (ED) algorithm.

I wanted to detect lane edges without leaning on a black-box edge detector, so I
implemented Edge Drawing by hand. The pipeline is:

1. Blur the grayscale frame to suppress noise.
2. Build a gradient-magnitude map (Sobel) and a coarse horizontal/vertical
   direction map from the x/y derivatives.
3. Threshold the gradient map into an "edge area" mask.
4. Pick *anchors* on a sparse scan grid: pixels that are clear local gradient
   maxima along their dominant direction.
5. Starting from each anchor, walk along the ridge of high gradient values to
   connect anchors into continuous edge segments.

Run as a script to see each stage rendered in an OpenCV window; import
``detect_lane_edges`` to get the intermediate maps back as arrays.
"""

from pathlib import Path

import cv2
import numpy as np

# Gradient delta a pixel must beat its neighbours by to qualify as an anchor.
# Vertical edges use the larger threshold; horizontal edges use the smaller one.
ANCHOR_THRESH = 14
ANCHOR_DELTA = 8

# Only every SCAN_INTERVAL-th pixel (in both axes) is tested as a candidate anchor.
SCAN_INTERVAL = 5

# Minimum gradient magnitude for a pixel to count as part of an edge area.
GRADIENT_THRESHOLD = 36

# Direction-map labels and the value used to mark a confirmed edge pixel.
HORIZONTAL = 0
VERTICAL = 90
EDGE = 255

# The test frame is 192x259; these guard the edge-walking loops from running off
# the bottom-right of that image.
ROW_BOUND = 190
COL_BOUND = 256

# Default frame I developed the algorithm against.
DEFAULT_IMAGE = Path(__file__).resolve().parents[2] / "assets" / "test.jpg"


def sobel_x(image):
    """First-order horizontal derivative (Sobel)."""
    return cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)


def sobel_y(image):
    """First-order vertical derivative (Sobel)."""
    return cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)


def gradient_magnitude(image):
    """Combine the absolute x and y Sobel derivatives into a gradient-magnitude map."""
    dxabs = cv2.convertScaleAbs(sobel_x(image))
    dyabs = cv2.convertScaleAbs(sobel_y(image))
    return cv2.addWeighted(dxabs, 0.5, dyabs, 0.5, 0)


def is_anchor(x, y, G, D, anchor_thresh):
    """Return 1 if pixel (x, y) is a local gradient maximum along its direction."""
    if D[x, y] == 0:
        if (G[x, y] - G[x, y - 1] >= ANCHOR_DELTA) and (G[x, y] - G[x, y + 1] >= ANCHOR_DELTA):
            return 1
    else:
        if (G[x, y] - G[x - 1, y] >= anchor_thresh) and (G[x, y] - G[x + 1, y] >= anchor_thresh):
            return 1
    return 0


def go_left(x, y, G, D, E, counter):
    """Walk left from an anchor, marking the ridge of maximum gradient as edge pixels."""
    while counter == 1:
        E[x, y] = EDGE  # Mark this pixel as an edgel
        # Look at 3 neighbours to the left & pick the one with the max gradient value
        if G[x - 1, y - 1] > G[x - 1, y] and G[x - 1, y - 1] > G[x - 1, y + 1]:
            x = x - 1
            y = y - 1  # Up-Left
        elif G[x - 1, y + 1] > G[x - 1, y] and G[x - 1, y + 1] > G[x - 1, y - 1]:
            x = x - 1
            y = y + 1  # Down-Left
        else:
            x = x - 1

        if E[x, y] != EDGE:
            counter = 0

    while G[x, y] > 0 and D[x, y] == HORIZONTAL and E[x, y] != EDGE:
        E[x, y] = EDGE  # Mark this pixel as an edgel
        if G[x - 1, y - 1] > G[x - 1, y] and G[x - 1, y - 1] > G[x - 1, y + 1]:
            x = x - 1
            y = y - 1  # Up-Left
        elif G[x - 1, y + 1] > G[x - 1, y] and G[x - 1, y + 1] > G[x - 1, y - 1]:
            x = x - 1
            y = y + 1  # Down-Left
        else:
            x = x - 1  # Straight-Left
        if x > ROW_BOUND or y > COL_BOUND:
            break


def go_right(x, y, G, D, E, counter):
    """Walk right from an anchor, marking the ridge of maximum gradient as edge pixels."""
    while counter == 1:
        E[x, y] = EDGE  # Mark this pixel as an edgel
        if G[x - 1, y - 1] > G[x - 1, y] and G[x - 1, y - 1] > G[x - 1, y + 1]:
            x = x - 1
            y = y - 1  # Up-Left
        elif G[x - 1, y + 1] > G[x - 1, y] and G[x - 1, y + 1] > G[x - 1, y - 1]:
            x = x - 1
            y = y + 1  # Down-Left
        else:
            x = x - 1

        if E[x, y] != EDGE:
            counter = 0

    while G[x, y] > 0 and D[x, y] == HORIZONTAL and E[x, y] != EDGE:
        E[x, y] = EDGE  # Mark this pixel as an edgel
        # Look at 3 neighbours to the right & pick the one with the max gradient value
        if G[x + 1, y - 1] > G[x + 1, y] and G[x + 1, y - 1] > G[x + 1, y + 1]:
            x = x + 1
            y = y - 1
        elif G[x + 1, y] > G[x + 1, y - 1] and G[x + 1, y] > G[x + 1, y + 1]:
            x = x + 1
            y = y
        else:
            x = x + 1
            y = y + 1
        if x > ROW_BOUND or y > COL_BOUND:
            break


def go_up(x, y, G, D, E, counter):
    """Walk up from an anchor, marking the ridge of maximum gradient as edge pixels."""
    while counter == 1:
        E[x, y] = EDGE  # Mark this pixel as an edgel
        if G[x - 1, y - 1] > G[x - 1, y] and G[x - 1, y - 1] > G[x - 1, y + 1]:
            x = x - 1
            y = y - 1  # Up-Left
        elif G[x - 1, y + 1] > G[x - 1, y] and G[x - 1, y + 1] > G[x - 1, y - 1]:
            x = x - 1
            y = y + 1  # Down-Left
        else:
            x = x - 1

        if E[x, y] != EDGE:
            counter = 0

    while G[x, y] > 0 and D[x, y] == VERTICAL and E[x, y] != EDGE:
        E[x, y] = EDGE  # Mark this pixel as an edgel
        if G[x - 1, y - 1] > G[x, y - 1] and G[x - 1, y - 1] > G[x + 1, y - 1]:
            x = x - 1
            y = y - 1  # Up-Left
        elif G[x, y - 1] > G[x - 1, y - 1] and G[x, y - 1] > G[x + 1, y - 1]:
            x = x
            y = y - 1  # Down-Left
        else:
            x = x + 1
            y = y - 1  # Straight-Left
            if x > ROW_BOUND or y > COL_BOUND:
                break


def go_down(x, y, G, D, E, counter):
    """Walk down from an anchor, marking the ridge of maximum gradient as edge pixels."""
    while counter == 1:
        E[x, y] = EDGE  # Mark this pixel as an edgel
        if G[x - 1, y - 1] > G[x - 1, y] and G[x - 1, y - 1] > G[x - 1, y + 1]:
            x = x - 1
            y = y - 1  # Up-Left
        elif G[x - 1, y + 1] > G[x - 1, y] and G[x - 1, y + 1] > G[x - 1, y - 1]:
            x = x - 1
            y = y + 1  # Down-Left
        else:
            x = x - 1

        if E[x, y] != EDGE:
            counter = 0

    while G[x, y] > 0 and D[x, y] == VERTICAL and E[x, y] != EDGE:
        E[x, y] = EDGE  # Mark this pixel as an edgel
        if G[x - 1, y + 1] > G[x, y + 1] and G[x - 1, y + 1] > G[x + 1, y + 1]:
            x = x - 1
            y = y + 1  # Up-Left
        elif G[x, y + 1] > G[x - 1, y + 1] and G[x, y + 1] > G[x + 1, y + 1]:
            x = x
            y = y + 1  # Down-Left
        else:
            x = x + 1
            y = y + 1  # Straight-Left
        if x > ROW_BOUND or y > COL_BOUND:
            break


def build_direction_map(blurred):
    """Label each pixel HORIZONTAL or VERTICAL based on which Sobel derivative dominates."""
    gx = sobel_x(blurred)
    gy = sobel_y(blurred)
    rows, cols = gx.shape
    direction_map = np.zeros((rows, cols, 1), np.uint8)
    for i in range(rows):
        for j in range(cols):
            if gx[i, j] > gy[i, j]:
                direction_map[i, j, 0] = HORIZONTAL
            else:
                direction_map[i, j, 0] = VERTICAL
    return direction_map


def build_edge_area_image(gradient_map):
    """Threshold the gradient map into a binary edge-area mask."""
    rows, cols = gradient_map.shape
    edge_area_image = np.zeros((rows, cols, 1), np.uint8)
    for i in range(rows - 1):
        for j in range(cols - 1):
            if gradient_map[i, j] < GRADIENT_THRESHOLD:
                edge_area_image[i, j, 0] = 0
            else:
                edge_area_image[i, j, 0] = EDGE
    return edge_area_image


def find_anchors(gradient_map, direction_map):
    """Scan the gradient map on a sparse grid and flag anchor pixels."""
    rows, cols = gradient_map.shape
    anchor_image = np.zeros((rows, cols, 1), np.uint8)
    for i in range(rows - 1):
        for j in range(cols - 1):
            if i % SCAN_INTERVAL == 0 and j % SCAN_INTERVAL == 0:
                if is_anchor(i, j, gradient_map, direction_map, ANCHOR_THRESH) == 1:
                    anchor_image[i, j, 0] = EDGE
                else:
                    anchor_image[i, j, 0] = 0
            else:
                anchor_image[i, j, 0] = 0
    return anchor_image


def link_anchors(anchor_image, gradient_map, direction_map):
    """Connect anchors into edge segments by walking along the gradient ridge.

    Mutates ``anchor_image`` in place, marking linked edge pixels with EDGE.
    """
    rows, cols = gradient_map.shape
    rowsa = rows - 3
    colsa = cols - 3
    for i in range(rowsa):
        for j in range(colsa):
            if i == 0 or j == 0:
                continue
            if anchor_image[i, j] == EDGE:
                if direction_map[i, j] == HORIZONTAL:
                    go_left(i, j, gradient_map, direction_map, anchor_image, 1)
                    go_right(i, j, gradient_map, direction_map, anchor_image, 1)
                else:
                    go_up(i, j, gradient_map, direction_map, anchor_image, 1)
                    go_down(i, j, gradient_map, direction_map, anchor_image, 1)
    return anchor_image


def detect_lane_edges(image):
    """Run the full Edge Drawing pipeline on a grayscale frame.

    Returns the intermediate maps so callers (and tests) can inspect each stage
    without needing a display.
    """
    blur = cv2.GaussianBlur(image, (5, 5), 0)

    raw_gradient = gradient_magnitude(blur)
    kernel = np.ones((3, 3), np.uint8)
    gradient_map = cv2.erode(raw_gradient, kernel, iterations=1)

    direction_map = build_direction_map(blur)
    edge_area_image = build_edge_area_image(gradient_map)

    anchors = find_anchors(gradient_map, direction_map).squeeze()
    anchors_after_scan = anchors.copy()
    edges = link_anchors(anchors, gradient_map, direction_map)

    return {
        "raw_gradient": raw_gradient,
        "gradient_map": gradient_map,
        "direction_map": direction_map,
        "edge_area_image": edge_area_image,
        "anchors": anchors_after_scan,
        "edges": edges,
    }


def main(image_path=DEFAULT_IMAGE):
    """Run the pipeline on a frame and show each stage in an OpenCV window."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    stages = detect_lane_edges(image)

    cv2.imshow("gradient-map", stages["raw_gradient"])
    cv2.imshow("thresholded-gradient map : Edge Area Image", stages["edge_area_image"])
    cv2.imshow("After step 3", stages["anchors"])
    cv2.imshow("After step 4", stages["edges"])

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
