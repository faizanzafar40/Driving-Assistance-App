"""Traffic-light detection from a live camera feed.

I detect red and green signals by colour rather than by shape first: convert each
frame to HSV, threshold the red and green ranges, clean the masks up with a
morphological close and a blur, then run a Hough circle transform to find the
round lamp. Circles in the plausible lamp-radius range are drawn on the frame and
labelled STOP (red) or GO (green).

The colour ranges are tuned for the lighting I tested under; expect to retune
``lower_*`` / ``upper_*`` for a different camera or environment.
"""

import cv2
import numpy as np

# HSV thresholds for each signal colour.
LOWER_RED = np.array([30, 0, 240])
UPPER_RED = np.array([110, 4, 255])
LOWER_GREEN = np.array([60, 5, 240])
UPPER_GREEN = np.array([100, 20, 255])

# Only accept Hough circles whose radius falls in this (exclusive) range, which
# filters out specular highlights and large background blobs.
MIN_RADIUS = 10
MAX_RADIUS = 30


def signal_mask(hsv, lower, upper):
    """Build a cleaned-up binary mask for one signal colour."""
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return cv2.GaussianBlur(closed, (15, 15), 0)


def find_circles(mask):
    """Run the Hough circle transform on a colour mask."""
    return cv2.HoughCircles(
        mask, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0
    )


def draw_signals(frame, circles, overlay_text, radius_label):
    """Draw and label any detected circles whose radius is in the accepted range."""
    if circles is None:
        return

    # Convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")
    for x, y, r in circles:
        if MIN_RADIUS < r < MAX_RADIUS:
            # Draw the circle in the output frame, then a marker at its centre
            cv2.circle(frame, (x, y), r, (255, 0, 0), 2, 8, 0)
            cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            print("Column Number: ")
            print(x)
            print("Row Number: ")
            print(y)
            print(radius_label)
            print(r)
            cv2.putText(frame, overlay_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 155)
            cv2.imshow("gray", frame)


def main(camera_index=0):
    """Capture from the camera and annotate detected traffic signals until 'q'/Esc."""
    cap = cv2.VideoCapture(camera_index)

    while True:
        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        red_mask = signal_mask(hsv, LOWER_RED, UPPER_RED)
        green_mask = signal_mask(hsv, LOWER_GREEN, UPPER_GREEN)

        draw_signals(frame, find_circles(red_mask), "Red Light->STOP", "Radius: ")
        draw_signals(frame, find_circles(green_mask), "Green Light->GO", "Radius is: ")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        cv2.imshow("frame", frame)

        if (cv2.waitKey(5) & 0xFF) == 27:
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
