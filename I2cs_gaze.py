"""
Wrapper around L2CS-Net so calibration/validation scripts don't need
to know anything about the Pipeline/GazeResultContainer API directly.

Setup (not pip-installable under the name "l2cs" — that name on PyPI
is an unrelated package):

    pip install git+https://github.com/edavalosanaya/L2CS-Net.git@main
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

Then download L2CSNet_gaze360.pkl from the repo's README (Google Drive
link under "Demo") into models/L2CSNet_gaze360.pkl — the weights are
not distributed via pip.
"""

import torch
from pathlib import Path

try:
    from l2cs import Pipeline
except ImportError as e:
    raise ImportError(
        "l2cs package not found. Install with:\n"
        "  pip install git+https://github.com/edavalosanaya/L2CS-Net.git@main"
    ) from e


FEATURE_VERSION = "l2cs_yawpitch_v1"

DEFAULT_WEIGHTS = Path("models") / "L2CSNet_gaze360.pkl"


class L2CSGazeEstimator:
    """
    Wraps L2CS-Net's Pipeline. Call estimate(frame) per webcam frame;
    returns [yaw, pitch] in radians for the highest-confidence detected
    face, or None if no face was detected.
    """

    def __init__(self, weights_path=DEFAULT_WEIGHTS, arch="ResNet50", device="cpu",
                 min_confidence=0.5):
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"L2CS-Net weights not found at {weights_path}. Download "
                "L2CSNet_gaze360.pkl from the repo README (Demo section, "
                "Google Drive link) and place it there."
            )

        self.min_confidence = min_confidence
        self.pipeline = Pipeline(
            weights=weights_path,
            arch=arch,
            device=torch.device(device),
        )

    def estimate(self, frame_bgr):
        """frame_bgr: a raw OpenCV BGR frame (same as cap.read() gives you)."""
        results = self.pipeline.step(frame_bgr)

        if results is None or len(results.bboxes) == 0:
            return None

        # If multiple faces are detected, use the highest-confidence one.
        best_idx = int(results.scores.argmax())
        if results.scores[best_idx] < self.min_confidence:
            return None

        yaw = float(results.yaw[best_idx])
        pitch = float(results.pitch[best_idx])
        return [yaw, pitch]


def feature_names():
    return ["yaw", "pitch"]
