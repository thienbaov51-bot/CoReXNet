"""CPU-only synthetic checks for the published model and window boundaries."""
from pathlib import Path
import json
import sys
import tempfile
import numpy as np

HERE = Path(__file__).resolve().parent


def data_check():
    from validate_data import validate_prepared
    rng = np.random.default_rng(2026)
    raw = rng.normal(size=(5000, 4)).astype(np.float32)
    raw[:, -1] = 3.0
    bounds = np.array([3000, 4000, 5000], dtype=np.int64)
    mean = raw[:bounds[0]].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = raw[:bounds[0]].std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    payload = dict(data=raw, normalized_data=(raw - mean) / std,
                   scaler_mean=mean, scaler_std=std, split_counts=bounds)
    for horizon in (96, 192, 336, 720):
        for name, start, end in (("train", 96, 3000), ("validation", 3000, 4000), ("test", 4000, 5000)):
            payload[f"{name}_origins_h{horizon}"] = np.arange(start, end-horizon+1, dtype=np.int64)
    with tempfile.TemporaryDirectory(prefix="corexnet_synthetic_") as folder:
        file = Path(folder) / "synthetic.npz"
        np.savez_compressed(file, **payload)
        result = validate_prepared(file)
        # The first validation forecast uses observed history before the boundary,
        # but every validation target starts at or after train_end.
        assert payload["validation_origins_h96"][0] == 3000
        assert payload["validation_origins_h96"][-1] + 96 == 4000
        assert payload["test_origins_h720"][-1] + 720 == 5000
        payload["scaler_mean"] = mean.copy()
        payload["scaler_mean"][0] += 1
        np.savez_compressed(file, **payload)
        try:
            validate_prepared(file)
        except ValueError:
            pass
        else:
            raise AssertionError("Corrupt scaler was accepted")
    return {"status": result["status"], "checked_origin_arrays": len(result["origin_counts"]),
            "constant_channel_guard": True, "bad_scaler_rejected": True}


def model_check():
    import torch
    sys.path.insert(0, str(HERE / "original_core"))
    from models import create_model
    config = json.loads((HERE / "configs/corexnet.json").read_text(encoding="utf-8"))
    # Small synthetic dimensions exercise the same unchanged module code.
    config.update(dim=16, ffn_dim=32, layers=2, horizon=12)
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    model = create_model(config, 7).cpu()
    history = torch.randn(2, 96, 7)
    prediction = model(history)
    assert prediction.shape == (2, 12, 7) and torch.isfinite(prediction).all()
    prediction.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    with torch.no_grad():
        constant = model(torch.ones_like(history))
        assert torch.isfinite(constant).all()
    return {"status": "PASS", "device": "cpu", "output_shape": list(prediction.shape),
            "finite_backward": True, "constant_history_finite": True}


if __name__ == "__main__":
    result = {"scope": "Synthetic code checks only", "data": data_check()}
    if "--data-only" not in sys.argv:
        result["model"] = model_check()
    print(json.dumps(result, indent=2))
