# Preprocessed data

[Download](https://github.com/thienbaov51-bot/CoReXNet/releases/tag/data-v1.0.0), or run `python download_data.py`. Files are placed in `data/processed/`.

| Dataset | Rows | Channels | Train end | Validation end | Test end |
| --- | ---: | ---: | ---: | ---: | ---: |
| Electricity | 26304 | 321 | 18412 | 21044 | 26304 |
| PEMS03 | 26208 | 358 | 15724 | 20966 | 26208 |
| PEMS04 | 16992 | 307 | 10195 | 13593 | 16992 |
| PEMS08 | 17856 | 170 | 10713 | 14284 | 17856 |
| Solar | 52560 | 137 | 36792 | 42048 | 52560 |
| AEMO5 | 87648 | 5 | 61353 | 70118 | 87648 |

Boundaries are cumulative exclusive row offsets. Each NPZ contains:

- `data`: float32 series, shape `[T, C]`
- `normalized_data`: series standardized with training-only statistics
- `scaler_mean`, `scaler_std`: training mean and population standard deviation
- `split_counts`: the three boundaries listed above
- `{train,validation,test}_origins_h{96,192,336,720}`: forecast-origin indices

For origin `o`, the history is `normalized_data[o-96:o]` and the target is `normalized_data[o:o+H]`. Evaluation uses all stride-one windows and channels, with unmasked MSE and MAE in standardized units.

Preprocessing scripts and source URLs are in `scripts/`. [Source attribution](../THIRD_PARTY_NOTICES.md).
