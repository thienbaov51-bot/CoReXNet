# CoReXNet

Code and reproduction entry points for **CoReXNet: Multivariate Time Series Forecasting via Common Response Reconstruction and Exchange**.

CoReXNet uses response loadings estimated from the current historical window for both forecast reconstruction and cross-channel residual exchange. The implementation contains Common Response Reconstruction (CRR), the Nonlinear Response Dictionary (NRD), and Amplitude-guided Response Exchange (ARE).

## Quick start

Install Python 3.10 or newer, NumPy, and a CUDA-compatible PyTorch build. Then install the requirements and check the implementation.

```bash
pip install -r requirements.txt
python smoke_test.py
```

Run `python download_data.py` to download and verify the [prepared-data release](https://github.com/thienbaov51-bot/CoReXNet/releases/tag/data-v1.0.0) into `data/processed/`. Each file contains the prepared multivariate series, chronological split boundaries, train-only normalization statistics, standardized series, and forecast-origin indices for all four horizons. No manual splitting or preprocessing is required.

```bash
python run.py --dataset PEMS04 --horizon 96 --check-data
python run.py --dataset PEMS04 --horizon 96 --seed 2026
```

The training command selects the checkpoint by full-validation MSE and then evaluates that checkpoint on the full test split. Outputs are stored under `runs/CoReXNet_PEMS04_L96_H96_S2026/`, including `best.pt`, `history.json`, `config.json`, and `metrics.json`.

```bash
# Inspect the fixed recipe without loading PyTorch or a dataset.
python run.py --dataset Solar --horizon 720 --seed 2028 --show-config

# Train and select the checkpoint using validation, leaving test unevaluated.
python run.py --dataset AEMO5 --horizon 192 --seed 2027 --train-only

# Evaluate that saved validation-selected checkpoint without further training.
python evaluate.py --run runs/CoReXNet_AEMO5_L96_H192_S2027

# Reproduce the four horizons and three reported seeds for one dataset.
for seed in 2026 2027 2028; do
  for horizon in 96 192 336 720; do
    python run.py --dataset PEMS04 --horizon "$horizon" --seed "$seed"
  done
done
```

The six dataset IDs are `Electricity`, `PEMS03`, `PEMS04`, `PEMS08`, `Solar`, and `AEMO5`. The history length is 96 and forecast horizons are 96, 192, 336, and 720. Training uses CUDA, batch size 64, at most 30 epochs, patience 7, AdamW with learning rate 0.001 and weight decay 0.0001, cosine decay to 5% of the initial learning rate, gradient clipping at 1.0, and BF16 with TF32. The reported seeds are 2026, 2027, and 2028. CPU support in `smoke_test.py` checks the model's forward and backward operations on synthetic histories.

## Data format and evaluation

Each `data/processed/<dataset>.npz` contains the following arrays.

| Key | Content |
| --- | --- |
| `data` | Prepared float32 multivariate series with shape `[T, C]` |
| `normalized_data` | Float32 standardized series |
| `scaler_mean`, `scaler_std` | Training-prefix statistics |
| `split_counts` | Absolute end offsets `[train_end, validation_end, T]` |
| `train_origins_h96`, `validation_origins_h96`, `test_origins_h96` | Forecast origins for horizon 96 |
| Corresponding keys ending in `h192`, `h336`, `h720` | Forecast origins for the other horizons |

For origin `o`, the input is `series[o-96:o]` and the target is `series[o:o+H]`. Validation and test inputs may use the immediately preceding observed history. Their forecast targets remain within their own chronological split. Windows have stride one, and all legal windows and target elements contribute to MSE and MAE. Results are calculated in training-standardized units. The scale is the population standard deviation computed in float64 on the training prefix and cast once to float32. Values below `1e-6` are replaced with one.

The unchanged original trainer reads `data`, computes the same train-only standardization, and never reads `normalized_data` as raw input. The released `normalized_data` and origin arrays are also available for direct use with another loader. `validate_data.py` verifies their exact agreement with the original arithmetic and boundaries.

## Implementation map

| Computation | Original scientific source |
| --- | --- |
| Shared temporal predictor and window normalization | `original_core/models.py` |
| CRR and zero-input forecast correction | `original_core/i1_common_contrast.py` |
| Common trajectory and signed response estimation | `original_core/i1_context_v2.py` |
| NRD and response coefficient | `original_core/i2_curvature.py` |
| Float64 dictionary geometry and ridge fitting | `original_core/i2_curvature_stable.py` |
| ARE with response-magnitude routing | `original_core/i3_response_residual_exchange.py` |
| Seed, optimizer, training, validation, and test loop | `original_core/train.py` |

The `original_core` files are byte-preserved copies of the scientific implementation. The historical internal model ID `I3_response_residual_exchange` refers to CoReXNet. The canonical fixed recipe is in `configs/corexnet.json`. `configs/recipe_72_source_mapping.json` links all 72 CoReXNet main-experiment settings to their original run IDs and configuration hashes. `SCIENTIFIC_SOURCE_MANIFEST.json` records the scientific-source SHA256 hashes.

## Reported results

`results/` contains the original per-seed values for main experiments, ablations, hyperparameter sensitivity, efficiency measurements, and the fixed visualization windows. The main results cover six datasets, eight models, four horizons, and three seeds. Ablation and main-experiment metrics in the manuscript use test-set results. Hyperparameter sensitivity uses validation results. The original baseline source snapshots, baseline recipes, and their accompanying third-party notices are retained in the repository's source archive.

Pretrained checkpoints are not included in this release. Running the original trainer creates a new validation-selected checkpoint. The archived CSV files preserve the reported experimental values and source identities.

## Source data and provenance

See [data/README.md](data/README.md) for all six exact chronological splits and the prepared-array interface. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for dataset attribution and original baseline-source licences. The data-build workflow independently verifies source, array and scaler identities and tests the original model on CPU before publishing the dataset release.
