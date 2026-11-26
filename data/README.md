# Prepared benchmark data

Download all six fully prepared datasets from the [data-v1.0.0 release](https://github.com/thienbaov51-bot/CoReXNet/releases/tag/data-v1.0.0), or run:

```bash
python download_data.py
python run.py --dataset PEMS04 --horizon 96 --check-data
```

The single archive contains the six NPZ arrays, dataset metadata, a data manifest and SHA256 checksums. The downloader verifies every member before extracting it into `data/processed/`.

| Dataset | Rows | Channels | TRAIN end | Validation end | TEST end |
| --- | ---: | ---: | ---: | ---: | ---: |
| AEMO5 | 87648 | 5 | 61353 | 70118 | 87648 |
| Electricity | 26304 | 321 | 18412 | 21044 | 26304 |
| PEMS03 | 26208 | 358 | 15724 | 20966 | 26208 |
| PEMS04 | 16992 | 307 | 10195 | 13593 | 16992 |
| PEMS08 | 17856 | 170 | 10713 | 14284 | 17856 |
| Solar | 52560 | 137 | 36792 | 42048 | 52560 |

Boundaries are zero-based cumulative exclusive end offsets. Each NPZ contains `data`, `normalized_data`, `scaler_mean`, `scaler_std`, `split_counts`, and all twelve forecast-origin arrays for the three splits and four horizons. For origin `o`, read `normalized_data[o-96:o]` as history and `normalized_data[o:o+H]` as the target. All normalization and indexing have already been performed.

The experiment arrays are recovered from pinned source files and checked against full-panel hashes, original TRAIN-scaler hashes and the actual chronological boundaries. The release includes provenance and source-specific terms in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

To independently rebuild the same prepared inputs:

```bash
pip install numpy==1.26.4 pandas==2.2.2 requests==2.32.3
python scripts/rebuild_public_data.py --output-dir build/data
```

The published benchmark source for Electricity is the pinned Time-Series-Library CSV. Solar retains the LSTNet source matrix and its genuine zero values. PEMS uses channel zero from the pinned ASTGNN benchmark files. AEMO5 is the five-state 2010–2014 selection from the Monash Australian electricity demand archive. Exact source URLs and SHA256 hashes are in `scripts/dataset_specs.json`.
