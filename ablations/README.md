# Ablation experiments

The seven variants use PEMS03, PEMS04 and PEMS08 with L=96, H=96 and seeds 2026, 2027 and 2028.

| Variant | Source directory | Configuration directory |
| --- | --- | --- |
| CoReXNet | `src/are` | `configs/full` |
| Signed routing | `src/are` | `configs/signed_routing` |
| Without ARE | `src/nrd` | `configs/without_are` |
| Adaptive hinge | `src/adaptive_hinge` | `configs/adaptive_hinge` |
| Without NRD and ARE | `src/crr` | `configs/without_nrd_are` |
| PCA | `src/pca` | `configs/pca` |
| B0 | `src/b0` | `configs/b0` |

Run from the repository root after installing the requirements and downloading the prepared data.

```bash
python ablations/src/are/train.py --config ablations/configs/full/PEMS04/seed2026.json --output runs/ablation_full_PEMS04_seed2026 --data-root data/processed
python ablations/src/nrd/train.py --config ablations/configs/without_are/PEMS04/seed2027.json --output runs/ablation_without_are_PEMS04_seed2027 --data-root data/processed
```

Choose the source directory and matching configuration from the table. Each command trains the variant, selects its checkpoint by full-validation MSE, and reports test-set MSE and MAE. Results and checkpoints are saved in the output directory. Training uses CUDA.
