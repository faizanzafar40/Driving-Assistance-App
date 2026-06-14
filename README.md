# Driving Assistance App

A small computer-vision project that detects **lane markings** and **traffic
lights** from camera input using OpenCV. I built it to get hands-on with classic
image-processing techniques — gradient-based edge detection and colour/shape-based
object detection — without relying on a pre-trained model.

It has two independent modules:

- **Lane detection** — my from-scratch implementation of the Edge Drawing (ED)
  algorithm. It blurs the frame, builds a Sobel gradient-magnitude map and a
  horizontal/vertical direction map, picks *anchor* pixels that are strong local
  gradient maxima, and then walks along the gradient ridge from each anchor to
  link them into continuous edge segments.
- **Traffic-light detection** — converts each camera frame to HSV, thresholds the
  red and green ranges, cleans the masks with a morphological close and a blur,
  then uses a Hough circle transform to locate the lamp. Detected signals are
  drawn on the frame and labelled `STOP` (red) or `GO` (green).

## Tech stack

- Python 3.9+
- [OpenCV](https://opencv.org/) (`opencv-python`) for image processing
- [NumPy](https://numpy.org/) for array math

## Features

- Edge Drawing lane-marking detection on a still image, with each pipeline stage
  rendered in its own window (gradient map, edge-area mask, anchors, linked edges).
- Real-time red/green traffic-light detection from a webcam, with on-frame labels.
- Importable functions (`detect_lane_edges`) so the pipeline can be run headless
  and inspected stage by stage.

## Demo

A short walkthrough is in [`docs/demo_video.webm`](docs/demo_video.webm), and the
slides I presented are in [`docs/Presentation.pptx`](docs/Presentation.pptx).

## Prerequisites

- Python 3.9 or newer
- A webcam (only for the traffic-light module)

## Installation

```bash
git clone https://github.com/faizanzafar40/Driving-Assistance-App.git
cd Driving-Assistance-App
pip install -r requirements.txt
```

## Running it

The source uses a `src/` layout, so either install the package once in editable
mode:

```bash
pip install -e .
```

…and then use the console scripts:

```bash
lane-detection     # runs the lane pipeline on assets/test.jpg
traffic-lights     # opens the webcam and detects traffic signals
```

…or run the modules directly without installing by pointing Python at `src/`:

```bash
# macOS / Linux
PYTHONPATH=src python -m driving_assistance.lane_detection
PYTHONPATH=src python -m driving_assistance.traffic_lights

# Windows (PowerShell)
$env:PYTHONPATH = "src"; python -m driving_assistance.lane_detection
```

The lane module opens OpenCV windows showing each stage — press any key (with a
window focused) to close them. The traffic-light module reads from camera index
`0`; press `q` or `Esc` to quit.

## Running the tests

```bash
pip install pytest
pytest
```

The tests run the lane-detection pipeline headless on `assets/test.jpg` and check
that it executes end to end and still produces edges.

## Project structure

```
.
├── src/driving_assistance/
│   ├── lane_detection.py     # Edge Drawing lane-marking detection
│   └── traffic_lights.py     # HSV + Hough-circle traffic-light detection
├── tests/
│   └── test_lane_detection.py
├── assets/
│   └── test.jpg              # sample frame for the lane pipeline
├── docs/
│   ├── demo_video.webm
│   └── Presentation.pptx
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

## Context / what I learned

This started as a digital-image-processing project, and implementing Edge Drawing
by hand taught me a lot about how gradient direction, non-maximum suppression, and
edge linking actually work under the hood — things I'd previously only used through
a single `Canny` call. The traffic-light side was a good lesson in how sensitive
colour thresholding is to lighting: the HSV ranges here are tuned to the conditions
I tested under and need retuning for a different camera or environment.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file.
