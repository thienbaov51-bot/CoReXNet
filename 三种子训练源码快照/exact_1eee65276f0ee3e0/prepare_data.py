"""Download/audit real forecasting datasets and freeze leakage-safe array splits.

Linux usage: python experiments/prepare_data.py --raw-dir raw_data --out-dir data
The output NPZ has raw float32 ``data[T,C]`` and cumulative integer boundaries
``split_counts=[train_end,val_end,T]``. Scaling belongs to the training runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ETT_REV = "1d16c8f4f943005d613b5bc962e9eeb06058cf07"
TSLIB_REV = "2b66e59ee19dac8f6f19fb5d4997f289fdfea357"
ASTGNN_REV = "9c2e19b98c4cedf1f35214d8789685b6381b3aad"
DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Electricity", "Traffic", "Weather", "PEMS03", "PEMS04", "PEMS08"]
SPECS = {
    "ETTh1": dict(filename="ETTh1.csv", channels=7, rows=17420, minutes=60, sha256="f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066"),
    "ETTh2": dict(filename="ETTh2.csv", channels=7, rows=17420, minutes=60, sha256="a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b"),
    "ETTm1": dict(filename="ETTm1.csv", channels=7, rows=69680, minutes=15, sha256="6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e"),
    "ETTm2": dict(filename="ETTm2.csv", channels=7, rows=69680, minutes=15, sha256="db973ca252c6410a30d0469b13d696cf919648d0f3fd588c60f03fdbdbadd1fd"),
    "Electricity": dict(filename="electricity.csv", channels=321, rows=26304, minutes=60, sha256="7e45845d54c5219bad0ae6bc1b5316cf8ff9cead5d33fa998a5a51c2e4a497ad"),
    "Traffic": dict(filename="traffic.csv", channels=862, rows=17544, minutes=60, sha256="cb06463d56fa17d87f47027cd9389ceae82a69eddee51cdb61480e120dab0b16"),
    "Weather": dict(filename="weather.csv", channels=21, rows=52696, minutes=10, sha256="34ee981d07313e51da2a50bb600072c8ae4a69cb4b0651f4cb93a069d7a2ba63"),
    "PEMS03": dict(filename="PEMS03.npz", channels=358, rows=26208, minutes=5, sha256=None),
    "PEMS04": dict(filename="PEMS04.npz", channels=307, rows=16992, minutes=5, sha256="95a3c9b720fffdb85f0330d09bfab41b0b3cad0ca86c0d7d5f3accacb4ac999a"),
    "PEMS08": dict(filename="PEMS08.npz", channels=170, rows=17856, minutes=5, sha256=None),
}


def source_info(name: str) -> dict:
    spec = SPECS[name]
    if name.startswith("ETT"):
        return dict(source_repository="https://github.com/zhouhaoyi/ETDataset", source_revision=ETT_REV,
                    source_url=f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/{ETT_REV}/ETT-small/{spec['filename']}",
                    original_provider="ETDataset / Informer authors")
    if name.startswith("PEMS"):
        return dict(source_repository="https://github.com/guoshnBJTU/ASTGNN", source_revision=ASTGNN_REV,
                    source_url=f"https://raw.githubusercontent.com/guoshnBJTU/ASTGNN/{ASTGNN_REV}/data/{name}/{spec['filename']}",
                    original_provider="Caltrans Performance Measurement System (PeMS)",
                    selected_signal="traffic flow; raw feature channel 0")
    folder = name.lower()
    return dict(source_repository="https://huggingface.co/datasets/thuml/Time-Series-Library", source_revision=TSLIB_REV,
                source_url=f"https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/{TSLIB_REV}/{folder}/{spec['filename']}",
                original_provider={"Electricity": "UCI ElectricityLoadDiagrams20112014 (standard hourly 321-client forecasting subset)",
                                   "Traffic": "California Department of Transportation (standard 862-sensor hourly forecasting subset)",
                                   "Weather": "Max Planck Institute for Biogeochemistry Jena weather station (standard 2020 forecasting subset)"}[name])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def split_boundaries(name: str, rows: int) -> list[int]:
    if name.startswith("ETT"):
        multiple = 4 if name.startswith("ETTm") else 1
        return [12 * 30 * 24 * multiple, 16 * 30 * 24 * multiple, 20 * 30 * 24 * multiple]
    if name.startswith("PEMS"):
        # Standard chronological 60/20/20; test is the remaining rows.
        train = int(rows * 0.6)
        return [train, train + int(rows * 0.2), rows]
    # Match the commonly used Dataset_Custom rounding exactly.
    train, test = int(rows * 0.7), int(rows * 0.2)
    return [train, rows - test, rows]


def forecast_origins(boundaries, split: str, seq_len=96, pred_len=96) -> np.ndarray:
    """Return starts of forecast targets; x=data[o-L:o], y=data[o:o+H]."""
    idx = {"train": 0, "val": 1, "test": 2}[split]
    lower = 0 if idx == 0 else int(boundaries[idx - 1])
    upper = int(boundaries[idx])
    return np.arange(max(seq_len, lower), upper - pred_len + 1, dtype=np.int64)


def train_scaler(data: np.ndarray, train_end: int) -> tuple[np.ndarray, np.ndarray]:
    """Train-only population moments; keep constant channels finite."""
    mean = data[:train_end].mean(axis=0, dtype=np.float64)
    std = data[:train_end].std(axis=0, dtype=np.float64)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def fetch_one(name: str, raw_dir: Path, reuse_map: dict, offline: bool) -> Path:
    dst = raw_dir / SPECS[name]["filename"]
    alternate = raw_dir / (name + Path(SPECS[name]["filename"]).suffix)
    if not dst.exists() and alternate.is_file():
        dst = alternate
    if not dst.exists():
        if name in reuse_map:
            src = Path(reuse_map[name])
            if not src.is_file():
                raise FileNotFoundError(f"Reuse source missing: {name}: {src}")
            shutil.copyfile(src, dst)
        elif offline:
            raise FileNotFoundError(f"No raw data available for {name}: {dst}")
        else:
            part = dst.with_suffix(dst.suffix + ".part")
            request = urllib.request.Request(source_info(name)["source_url"], headers={"User-Agent": "research-data-audit/1.0"})
            with urllib.request.urlopen(request, timeout=180) as response, part.open("wb") as out:
                shutil.copyfileobj(response, out, length=1024 * 1024)
            part.replace(dst)
    digest = sha256(dst)
    expected = SPECS[name]["sha256"]
    if expected is not None and digest != expected:
        raise ValueError(f"Raw SHA256 mismatch for {name}; expected {expected}, actual {digest}")
    return dst


def read_raw(name: str, path: Path) -> tuple[np.ndarray, dict]:
    spec = SPECS[name]
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as z:
            array = z["data"]
        shape = list(array.shape)
        if array.ndim != 3:
            raise ValueError(f"Expected raw PeMS data[T,C,F], got {shape}")
        data = np.ascontiguousarray(array[:, :, 0], dtype=np.float32)
        info = dict(raw_shape=shape, signal_channel=0, timestamps_present=False,
                    date_start={"PEMS03": "2018-09-01T00:00:00", "PEMS04": "2018-01-01T00:00:00", "PEMS08": "2016-07-01T00:00:00"}[name],
                    timestamp_origin="benchmark dataset description; no timestamp column in NPZ")
    else:
        df = pd.read_csv(path)
        if "date" not in df.columns:
            raise ValueError(f"Date header absent in {path}")
        timestamps = pd.to_datetime(df["date"], errors="raise")
        if not timestamps.is_monotonic_increasing:
            raise ValueError(f"Timestamps are not chronological: {name}")
        channels = [c for c in df.columns if c != "date"]
        data = np.ascontiguousarray(df[channels].to_numpy(dtype=np.float32))
        differences = timestamps.diff().dropna().dt.total_seconds()
        info = dict(raw_shape=list(data.shape), channel_names=channels, timestamps_present=True,
                    date_start=timestamps.iloc[0].isoformat(), date_end=timestamps.iloc[-1].isoformat(),
                    duplicate_timestamps_count=int(timestamps.duplicated().sum()),
                    timestamp_processing="original row order preserved; no resampling or removal of source duplicates",
                    timestamp_delta_seconds_counts={str(k): int(v) for k, v in differences.value_counts().sort_index().items()})
    if data.shape != (spec["rows"], spec["channels"]):
        raise ValueError(f"Unexpected dimensions for {name}: {data.shape}, expected {(spec['rows'], spec['channels'])}")
    if not np.isfinite(data).all():
        raise ValueError(f"Non-finite values found in {name}; no silent imputation is allowed")
    return data, info


def prepare_one(name: str, raw_dir: Path, out_dir: Path, reuse_map: dict, offline: bool,
                seq_len: int, pred_len: int) -> dict:
    path = fetch_one(name, raw_dir, reuse_map, offline)
    full_data, info = read_raw(name, path)
    boundaries = split_boundaries(name, len(full_data))
    if boundaries[-1] > len(full_data):
        raise ValueError(f"Insufficient rows for fixed standard split: {name}")
    data = full_data[:boundaries[-1]]
    mean, std = train_scaler(data, boundaries[0])
    metadata = dict(dataset=name, raw_file=path.name, raw_size_bytes=path.stat().st_size,
                    raw_sha256=sha256(path), prepared_file=f"{name}.npz", dtype="float32",
                    original_rows=len(full_data), used_rows=len(data), channels=data.shape[1],
                    unused_tail_rows=len(full_data)-len(data), frequency_minutes=SPECS[name]["minutes"],
                    split_counts=boundaries, split_counts_semantics="cumulative exclusive row boundaries: train_end, val_end, used_T",
                    split_row_counts=[boundaries[0], boundaries[1]-boundaries[0], boundaries[2]-boundaries[1]],
                    seq_len=seq_len, pred_len=pred_len, seed=2026,
                    scaler="runner fits channelwise population mean/std on training rows only; prepared data remain raw",
                    metric="unmasked MSE and MAE averaged over all forecast windows, horizons, and channels in train-standardized units",
                    window_stride=1, history_overlap=True, target_crosses_split=False,
                    imputation="none; finite-value audit required", zero_values="preserved as in original benchmark files",
                    zero_fraction=float(np.mean(data == 0)),
                    window_counts={s: len(forecast_origins(boundaries, s, seq_len, pred_len)) for s in ("train", "val", "test")},
                    pems_protocol_note="96-to-96 common forecasting experiment; does not reproduce published 12-to-12 short-term PeMS metrics" if name.startswith("PEMS") else None,
                    **source_info(name), **info)
    if min(metadata["window_counts"].values()) < 1:
        raise ValueError(f"No windows for split: {name}: {metadata['window_counts']}")
    target = out_dir / f"{name}.npz"
    # Metadata string uses NumPy Unicode; loading never requires pickle.
    np.savez_compressed(target, data=data, split_counts=np.asarray(boundaries, dtype=np.int64),
                        train_mean=mean, train_std=std, metadata_json=json.dumps(metadata, ensure_ascii=False))
    metadata["prepared_size_bytes"] = target.stat().st_size
    metadata["prepared_sha256"] = sha256(target)
    (out_dir / f"{name}.metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: metadata[k] for k in ("dataset", "original_rows", "used_rows", "channels", "split_counts", "window_counts", "raw_sha256")}), flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("raw_data"))
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--reuse-map", type=Path, help="JSON object mapping dataset names to existing public raw files")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=96)
    args = parser.parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    reuse = json.loads(args.reuse_map.read_text(encoding="utf-8-sig")) if args.reuse_map else {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda name: prepare_one(name, args.raw_dir, args.out_dir, reuse, args.offline,
                                                        args.seq_len, args.pred_len), args.datasets))
    report = dict(protocol_version="fifth-paper-data-v1", generated_utc=datetime.now(timezone.utc).isoformat(),
                  selected_before_model_outcomes=True, dataset_order=args.datasets, datasets=results)
    (args.out_dir / "metadata.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
