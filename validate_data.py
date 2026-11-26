"""Validate the released, already prepared forecast arrays and origin indices."""
from pathlib import Path
import numpy as np


def validate_prepared(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        required = {"data", "normalized_data", "scaler_mean", "scaler_std", "split_counts"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"Missing prepared arrays: {sorted(missing)}")
        raw = archive["data"]
        normalized = archive["normalized_data"]
        mean, std = archive["scaler_mean"], archive["scaler_std"]
        bounds = archive["split_counts"]
        if raw.ndim != 2 or raw.dtype != np.float32 or not np.isfinite(raw).all():
            raise ValueError("data must be finite float32 [time, channels]")
        if normalized.shape != raw.shape or normalized.dtype != np.float32 or not np.isfinite(normalized).all():
            raise ValueError("normalized_data must be finite float32 with the same shape")
        if bounds.shape != (3,) or bounds.dtype.kind not in "iu":
            raise ValueError("split_counts must contain three integer absolute end offsets")
        train_end, val_end, total = map(int, bounds)
        if not 96 < train_end < val_end < total == len(raw):
            raise ValueError("Invalid chronological split boundaries")
        if mean.shape != (raw.shape[1],) or std.shape != mean.shape:
            raise ValueError("Scaler shape does not match channel count")
        expected_mean = raw[:train_end].mean(axis=0, dtype=np.float64).astype(np.float32)
        expected_std = raw[:train_end].std(axis=0, dtype=np.float64).astype(np.float32)
        expected_std[expected_std < 1e-6] = 1.0
        if not np.array_equal(mean, expected_mean) or not np.array_equal(std, expected_std):
            raise ValueError("Scaler is not the original train-only float64-to-float32 estimator")
        if not np.array_equal(normalized, (raw - mean) / std):
            raise ValueError("Prepared normalized_data differs from the original standardization")
        counts = {}
        for horizon in (96, 192, 336, 720):
            for name, start, end in (("train", 96, train_end),
                                     ("validation", train_end, val_end),
                                     ("test", val_end, total)):
                key = f"{name}_origins_h{horizon}"
                if key not in archive:
                    raise ValueError(f"Missing origins: {key}")
                expected = np.arange(start, end - horizon + 1, dtype=np.int64)
                if not len(expected) or not np.array_equal(archive[key], expected):
                    raise ValueError(f"Origin boundary/count mismatch: {key}")
                counts[key] = len(expected)
        return {"status": "PASS", "path": str(path), "shape": list(raw.shape),
                "split_counts": bounds.tolist(), "length": 96,
                "horizons": [96, 192, 336, 720], "origin_counts": counts,
                "scaler": "Training prefix only, float64 population mean/std cast to float32",
                "targets": "All channels and all target elements, stride one"}
