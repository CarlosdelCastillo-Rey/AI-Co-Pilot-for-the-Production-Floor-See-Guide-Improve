"""Paths for /api/vision artifacts and InHARD / pipeline outputs."""

from __future__ import annotations

from pathlib import Path

CAM_ASSEMBLY = "cam-01"
CAM_WAREHOUSE = "cam-02"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def vision_data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "vision"


def camera_artifact_dir(camera_id: str) -> Path:
    return vision_data_dir() / camera_id


def last_probe_path(camera_id: str) -> Path:
    return camera_artifact_dir(camera_id) / "last_probe.json"


def heatmap_path(camera_id: str) -> Path:
    return camera_artifact_dir(camera_id) / "heatmap.png"


def heatmap_overlay_path(camera_id: str) -> Path:
    return camera_artifact_dir(camera_id) / "heatmap_overlay.jpg"


def still_path(camera_id: str) -> Path:
    return camera_artifact_dir(camera_id) / "still.jpg"


def preview_path(camera_id: str) -> Path:
    return camera_artifact_dir(camera_id) / "preview.jpg"


def embedding_path(camera_id: str) -> Path:
    return camera_artifact_dir(camera_id) / "embedding.npy"


def baseline_embedding_path(camera_id: str) -> Path:
    return camera_artifact_dir(camera_id) / "baseline_embedding.npy"


def inhard_rgb_dir() -> Path:
    return (
        repo_root()
        / "data_sample"
        / "InHARD-master"
        / "01-InHARD"
        / "Segmented"
        / "RGBSegmented"
    )


def segments_manifest_path() -> Path:
    return repo_root() / "outputs" / "02_segments" / "manifest.json"


def outputs_vjepa_dir() -> Path:
    return repo_root() / "outputs" / "08_vjepa"


def find_first_mp4(root: Path | None = None, max_depth: int = 6) -> Path | None:
    if root is None:
        root = inhard_rgb_dir()
    if not root.is_dir():
        return None

    def _walk(path: Path, depth: int) -> Path | None:
        if depth > max_depth:
            return None
        try:
            entries = sorted(path.iterdir(), key=lambda p: p.name)
        except OSError:
            return None
        for entry in entries:
            if entry.is_file() and entry.suffix.lower() == ".mp4":
                return entry
        for entry in entries:
            if entry.is_dir():
                found = _walk(entry, depth + 1)
                if found is not None:
                    return found
        return None

    return _walk(root, 0)


MOCK_STILL_IMAGES: dict[str, str] = {
    CAM_ASSEMBLY: (
        "https://lh3.googleusercontent.com/aida-public/AB6AXuCtdXg1qgVaATzDFV4GlsmN6CkUoyf1Z5phhagAyhKszH_SM-XO_97YtvK6_rhFO1EC5ny-HEVEIP1Wz2oRu_LYR5IOJVdWCFu0csqXHHopNJFR5fD0-ooCwFJKB6q8aDm0yPLzbPKtGYY7AQThGRta6LJSy3krV7Ze8hd6UnLyT7J6eiI11S6664PLbZ9IWYxq4SeOpkEwSm2g-eCVrZOwtq7YjtLw8HV8C_23jAB7xWqoV3X1prHnLVBcL0GHSMS0ayxsVG6QrqY"
    ),
    CAM_WAREHOUSE: (
        "https://lh3.googleusercontent.com/aida-public/AB6AXuD0WaxmzB30i4UvnXB1kC5UAjuno45jZ0-lYANMKEPRwQpqZH639_Ac7yPq9EJwxynwUc8jWfLtP6TuMgSHCc4R8QV2j8GXrckY0OSBfzsbliQXwp7qGaM_dgRn_CJ_-YN2M84FIR_4mTkNuSeOUqYZv-zFb-PGGtCtruaN4-mtsBu_sa6AvIDb6JnHCMjexxBix8FdInNk_8IbvGQsjiq1a0uDuXJABCY-cv8XDEYCM9YPSBwMnKCs_vAm8Ksn_xYUDxWv9393I"
    ),
}
