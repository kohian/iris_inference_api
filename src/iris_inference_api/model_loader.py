from pathlib import Path

import joblib


def load_model(path: str):
    """
    Load a model from:
    - local file path
    - GCS path (gs://...)
    """
    if path.startswith("gs://"):
        import gcsfs

        fs = gcsfs.GCSFileSystem()
        with fs.open(path, "rb") as f:
            return joblib.load(f)

    local_path = Path(path)

    if not local_path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    return joblib.load(local_path)