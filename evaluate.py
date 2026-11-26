"""Evaluate a saved validation-selected CoReXNet checkpoint without training."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=HERE / "data" / "processed")
    args = parser.parse_args()
    folder = args.run.resolve()
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    config = metrics["config"]
    if "test" in metrics:
        print(json.dumps({"status": "ALREADY_EVALUATED", "test": metrics["test"]}, indent=2))
        return
    if (folder / "test_metrics.json").exists():
        print((folder / "test_metrics.json").read_text(encoding="utf-8"))
        return
    sys.path.insert(0, str(HERE / "original_core"))
    import torch
    import train
    from models import create_model
    if not torch.cuda.is_available():
        raise SystemExit("The original scientific evaluator requires CUDA.")
    path = args.data_root.resolve() / f"{config['dataset']}.npz"
    data_hash = file_hash(path)
    if metrics["data_sha256"] != data_hash:
        raise SystemExit("Dataset bytes differ from the training input.")
    for name, expected in metrics["code_sha256"].items():
        if file_hash(HERE / "original_core" / name) != expected:
            raise SystemExit(f"Scientific source changed after training: {name}")
    train.seed_all(config["seed"])
    windows = train.Windows(path, config["length"], config["horizon"], "cuda", pilot=False)
    model = create_model(config, windows.channels).cuda()
    checkpoint = folder / "best.pt"
    model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True), strict=True)
    result = train.evaluate(model, windows, "test", config["batch_size"] * 2, config["amp"])
    payload = {"status": "COMPLETE_TEST", "config": config, "test": result,
               "checkpoint_sha256": file_hash(checkpoint), "data_sha256": data_hash,
               "selection": "Existing full-validation-MSE-selected checkpoint; no optimizer"}
    with (folder / "test_metrics.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
