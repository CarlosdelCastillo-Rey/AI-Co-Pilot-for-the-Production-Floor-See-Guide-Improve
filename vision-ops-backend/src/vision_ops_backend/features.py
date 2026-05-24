from __future__ import annotations
import numpy as np
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

    def process_frame_track(self, track_id: int, bbox: list[float], current_fps: float) -> dict:
        """
        Receives real-time tracking metrics from YOLOv8 + DeepSORT
        and extracts the exact mathematical features.
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

        tool_proximity_proxy = mov_intensity * (lighting_stability_index + 1e-5)

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

        return {
            "track_id": track_id,
            "spatial_aspect_ratio": round(spatial_aspect_ratio, 4),
            "mov_intensity_z": round(z_mov_intensity, 4),
            "camera_proximity_index": round(camera_proximity_index, 6),
            "lighting_stability_index": round(lighting_stability_index, 4),
            "tool_proximity_proxy": round(tool_proximity_proxy, 4),
            "roi_assembly_zone": roi_assembly_zone,
            "roi_transit_zone": roi_transit_zone,
        }
