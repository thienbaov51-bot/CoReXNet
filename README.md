# CoReXNet

Code and data for **CoReXNet: Multivariate Time Series Forecasting via Common Response Reconstruction and Exchange**.

## Quick start

Python 3.10+ and CUDA-enabled PyTorch are required for training.

```bash
pip install -r requirements.txt
python download_data.py
python run.py --dataset PEMS04 --horizon 96 --seed 2026
```

The command trains CoReXNet, selects the checkpoint by validation MSE and evaluates it on the test set. Checkpoints and results are saved in `runs/`.

## Data

[Download all six preprocessed datasets](https://github.com/thienbaov51-bot/CoReXNet/releases/tag/data-v1.0.0) · [Data format and splits](data/README.md)

The archive includes standardized arrays, normalization statistics, chronological splits and forecasting indices for Electricity, PEMS03, PEMS04, PEMS08, Solar and AEMO5.

## Experiments

- Input length: 96
- Forecast horizons: 96, 192, 336, 720
- Random seeds: 2026, 2027, 2028
- Settings: `configs/corexnet.json` and `configs/baselines.json`
- Model and training code: `original_core/`
- Reported per-seed results: `results/`

To run another setting, change `--dataset`, `--horizon` and `--seed` in the command above. [Dataset and baseline sources](THIRD_PARTY_NOTICES.md).
