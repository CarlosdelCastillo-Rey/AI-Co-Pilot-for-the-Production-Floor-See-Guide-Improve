from __future__ import annotations
import numpy as np
import joblib
from pathlib import Path
from collections import defaultdict, deque


class LiveFeatureProcessor:
    """
    Core layer for the Digital Twin.
    Adapting Feature Engineering mathematical pipeline
    to ingest state updates frame-by-frame from the live camera stream.
    """

    def __init__(self, window_size: int = 150):
        self.history = defaultdict(lambda: deque(maxlen=window_size))
        self.ROI_LIMITS = {"x_min": 100, "x_max": 800, "y_min": 150, "y_max": 950}

        model_path = Path(__file__).resolve().parents[2] / "baseline_tree.joblib"

        if model_path.is_file():
            try:
                self.model = joblib.load(model_path)
                self.has_model = True
            except Exception:
                self.has_model = False
                self.model = None
        else:
            self.has_model = False
            self.model = None

    def process_frame_track(self, track_id: int, bbox: list[float], current_fps: float) -> dict:
        """
        Receives real-time tracking metrics from YOLOv8 + DeepSORT
        and extracts the exact mathematical features, and appends live ML Predictions for the current frame.
        """
        l, t, r, b = bbox
        w = r - l
        h = b - t
        centroid_x = l + (w / 2.0)
        centroid_y = t + (h / 2.0)

        frame_duration_sec = 1.0 / max(current_fps, 1.0)

        if w <= 0 or h <= 0:
            if len(self.history[track_id]) > 0:
                last_valid = self.history[track_id][-1]
                w, h = last_valid["w"], last_valid["h"]
            else:
                w, h = 1.0, 1.0

        spatial_aspect_ratio = float(w) / float(h + 1e-5)

        if len(self.history[track_id]) > 0:
            prev = self.history[track_id][-1]
            distance_pixels = np.sqrt(
                (centroid_x - prev["cx"]) ** 2 + (centroid_y - prev["cy"]) ** 2
            )
            mov_intensity = distance_pixels / frame_duration_sec
        else:
            mov_intensity = 0.0

        camera_proximity_index = mov_intensity / (float(w * h) + 1e-5)
        lighting_stability_index = frame_duration_sec / (spatial_aspect_ratio + 1e-5)

        roi_assembly_zone = 0
        if (
            self.ROI_LIMITS["x_min"] <= centroid_x <= self.ROI_LIMITS["x_max"]
            and self.ROI_LIMITS["y_min"] <= centroid_y <= self.ROI_LIMITS["y_max"]
        ):
            roi_assembly_zone = 1
        roi_transit_zone = 1 if roi_assembly_zone == 0 else 0

        tool_proximity_proxy = np.abs(mov_intensity) * (lighting_stability_index + 0.2)

        current_state = {
            "w": w,
            "h": h,
            "cx": centroid_x,
            "cy": centroid_y,
            "mov_intensity": mov_intensity,
            "proximity": camera_proximity_index,
        }
        self.history[track_id].append(current_state)

        intensities = [state["mov_intensity"] for state in self.history[track_id]]
        mean_intensity = np.mean(intensities)
        std_intensity = np.std(intensities) + 1e-5
        z_mov_intensity = (mov_intensity - mean_intensity) / std_intensity

        predicted_action = "INITIALIZING"
        prediction_confidence = 0.0

        if self.has_model and self.model is not None:
            input_vector = np.array(
                [
                    [
                        spatial_aspect_ratio,
                        z_mov_intensity,
                        camera_proximity_index,
                        lighting_stability_index,
                        tool_proximity_proxy,
                    ]
                ]
            )

            try:
                pred_class = self.model.predict(input_vector)[0]
                pred_proba = self.model.predict_proba(input_vector)[0]

                predicted_action = str(pred_class)
                prediction_confidence = float(np.max(pred_proba))
            except Exception:
                predicted_action = "ERROR_IN_INFERENCE"

        return {
            "track_id": track_id,
            "spatial_aspect_ratio": round(spatial_aspect_ratio, 4),
            "mov_intensity_z": round(z_mov_intensity, 4),
            "camera_proximity_index": round(camera_proximity_index, 6),
            "lighting_stability_index": round(lighting_stability_index, 4),
            "tool_proximity_proxy": round(tool_proximity_proxy, 4),
            "roi_assembly_zone": roi_assembly_zone,
            "roi_transit_zone": roi_transit_zone,
            "predicted_action": predicted_action,
            "prediction_confidence": round(prediction_confidence, 4),
        }
