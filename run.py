"""Run the published CoReXNet recipe with the original scientific trainer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
DATASETS = ("Electricity", "PEMS03", "PEMS04", "PEMS08", "Solar", "AEMO5")
HORIZONS = (96, 192, 336, 720)
SEEDS = (2026, 2027, 2028)


def experiment_config(dataset, horizon, seed, train_only=False):
    config = json.loads((HERE / "configs" / "corexnet.json").read_text(encoding="utf-8"))
    config.update(dataset=dataset, horizon=int(horizon), seed=int(seed),
                  phase="full", evaluate_test=not train_only,
                  run_id=f"CoReXNet_{dataset}_L96_H{horizon}_S{seed}")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--horizon", choices=HORIZONS, type=int, default=96)
    parser.add_argument("--seed", choices=SEEDS, type=int, default=2026)
    parser.add_argument("--data-root", type=Path, default=HERE / "data" / "processed")
    parser.add_argument("--output-root", type=Path, default=HERE / "runs")
    parser.add_argument("--train-only", action="store_true",
                        help="Select the checkpoint using validation and skip test evaluation.")
    parser.add_argument("--show-config", action="store_true")
    parser.add_argument("--check-data", action="store_true",
                        help="Validate prepared arrays and split boundaries without importing PyTorch.")
    args = parser.parse_args()
    config = experiment_config(args.dataset, args.horizon, args.seed, args.train_only)
    if args.show_config:
        print(json.dumps(config, indent=2))
        return
    if args.check_data:
        from validate_data import validate_prepared
        result = validate_prepared(args.data_root / f"{args.dataset}.npz")
        print(json.dumps(result, indent=2))
        return
    data_root = args.data_root.resolve()
    if not (data_root / f"{args.dataset}.npz").is_file():
        raise SystemExit(f"Missing prepared dataset: {data_root / (args.dataset + '.npz')}")
    # The paper recipe uses CUDA BF16 with TF32. Model and trainer are unchanged.
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("The paper training command requires a CUDA GPU. Run smoke_test.py for the CPU model check.")
    output = args.output_root.resolve() / config["run_id"]
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "requested_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != config:
            raise SystemExit("Output directory already belongs to a different configuration. Select another output root.")
    else:
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    command = [sys.executable, str(HERE / "original_core" / "train.py"),
               "--config", str(config_path), "--output", str(output),
               "--data-root", str(data_root)]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
