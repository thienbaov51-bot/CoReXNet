# Data and software sources

## Data

| Dataset | Source | Terms |
| --- | --- | --- |
| Electricity | [UCI](https://doi.org/10.24432/C58C86), [Time-Series-Library benchmark](https://huggingface.co/datasets/thuml/Time-Series-Library) | CC BY 4.0 |
| Solar | [LSTNet](https://github.com/laiguokun/multivariate-time-series-data), [Monash archive](https://doi.org/10.5281/zenodo.4656144) | CC BY 4.0 |
| AEMO5 | [Monash Australian electricity demand archive](https://doi.org/10.5281/zenodo.4659727) | CC BY 4.0 |
| PEMS03, PEMS04, PEMS08 | [Caltrans PeMS](https://pems.dot.ca.gov/), [ASTGNN benchmark](https://github.com/guoshnBJTU/ASTGNN) | [Caltrans source terms](https://dot.ca.gov/conditions-of-use) |

Electricity uses the hourly 321-channel benchmark. Solar uses the 137-channel Alabama benchmark. AEMO5 contains the five-state 2010–2014 subset. PEMS uses traffic-flow channel 0. Exact source revisions and checksums are provided in `scripts/dataset_specs.json`.

## Baselines

[DLinear](https://github.com/cure-lab/LTSF-Linear) · [SparseTSF](https://github.com/lss-1138/SparseTSF) · [ModernTCN](https://github.com/luodhhh/ModernTCN) · [MixLinear](https://github.com/aitianma/MixLinear) · [PhaseFormer](https://github.com/neumyor/PhaseFormer_TSL) · [RootPurgeLinear](https://github.com/Wangzzzzzzzz/RootPurge) · [ITSMixer](https://github.com/JZX100II/ITSMixer)

Baseline settings are in `configs/baselines.json`. Each upstream project retains its own licence and copyright notices.
