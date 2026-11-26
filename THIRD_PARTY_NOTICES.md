# Third-party data and software notices

The data sources and third-party implementations retain their upstream rights and attribution. Each separately supplied third-party licence continues to apply to its corresponding material. Data and software notices are recorded separately below.

## Benchmark data

| Released benchmark | Original source and benchmark packaging | Rights and attribution |
| --- | --- | --- |
| Electricity | Artur Trindade, *ElectricityLoadDiagrams20112014*, UCI Machine Learning Repository, 2015, [doi:10.24432/C58C86](https://doi.org/10.24432/C58C86). The experiment uses the standard 321-client hourly forecasting subset distributed by [Time-Series-Library](https://huggingface.co/datasets/thuml/Time-Series-Library). | The [UCI dataset record](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014) specifies [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Credit the original creator and identify the benchmark subsetting, chronological splitting and TRAIN-fitted normalization when reusing these prepared files. |
| Solar | DOE/NREL solar power data for integration studies, Alabama 2006 benchmark with 137 series sampled every ten minutes, distributed by [Lai's LSTNet benchmark data repository](https://github.com/laiguokun/multivariate-time-series-data). The same benchmark is archived by Godahewa and colleagues as [Solar Dataset (10 Minutes Observations), doi:10.5281/zenodo.4656144](https://doi.org/10.5281/zenodo.4656144). | The Monash Zenodo dataset record specifies [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The matrix in that archive was independently parsed and verified to be bitwise equal, after the experiment's float32 conversion, to the actual Solar matrix. Credit the original NREL source, Lai's benchmark packaging and the Monash archive. |
| AEMO5 | Australian Energy Market Operator, distributed by Godahewa and colleagues through the Monash archive as [Australian Electricity Demand Dataset, doi:10.5281/zenodo.4659727](https://doi.org/10.5281/zenodo.4659727). The released experiment subset contains the five aligned series for 2010–2014. | The Zenodo record specifies [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Credit the original provider and the Monash dataset creators, and identify the subset and transformations. |
| PEMS03, PEMS04 and PEMS08 | [Caltrans Performance Measurement System](https://dot.ca.gov/programs/traffic-operations/mpr/pems-source), with the research benchmark arrays distributed by [ASTGNN](https://github.com/guoshnBJTU/ASTGNN). The experiment selects traffic-flow feature channel 0. Exact source revisions and checksums appear in the data manifest. | Governed by the original source terms. The [Caltrans Conditions of Use, Ownership section](https://dot.ca.gov/conditions-of-use) generally permits lawful copying and distribution of its public information, with exceptions for separately identified third-party materials. ASTGNN does not expose a separate dataset-specific SPDX licence for these benchmark arrays in the inspected release. Preserve this provenance and the source notices when redistributing the prepared arrays. |

Prepared data include chronological splits and standardization fitted on the training portion. These transformations are documented in the accompanying manifest. The supplied benchmark arrays are attributed to their original providers.

### Dataset citations

- Trindade, A. (2015). *ElectricityLoadDiagrams20112014*. UCI Machine Learning Repository. https://doi.org/10.24432/C58C86
- Godahewa, R., Bergmeir, C., Webb, G. I., Hyndman, R. J. and Montero-Manso, P. *Solar Dataset (10 Minutes Observations)*. Zenodo. https://doi.org/10.5281/zenodo.4656144
- Godahewa, R., Bergmeir, C., Webb, G. I., Hyndman, R. J. and Montero-Manso, P. (2021). *Australian Electricity Demand Dataset*. Zenodo. https://doi.org/10.5281/zenodo.4659727
- Godahewa, R., Bergmeir, C., Webb, G. I., Hyndman, R. J. and Montero-Manso, P. (2021). *Monash Time Series Forecasting Archive*. NeurIPS Datasets and Benchmarks. https://forecastingdata.org/
- Caltrans. *Performance Measurement System*. https://pems.dot.ca.gov/
- Guo, S. and collaborators. *ASTGNN research benchmark release*. https://github.com/guoshnBJTU/ASTGNN
- Lai, G., Chang, W.-C., Yang, Y. and Liu, H. (2018). *Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks*. SIGIR. https://github.com/laiguokun/LSTNet

## Baseline software provenance

The source snapshots used for the comparisons retain the following upstream identities. Where files from a project are included, preserve the corresponding licence and copyright notices alongside them.

| Component | Upstream project | Licence identified in the supplied snapshot |
| --- | --- | --- |
| DLinear | [cure-lab/LTSF-Linear](https://github.com/cure-lab/LTSF-Linear) | Apache License 2.0, original DLinear copyright notice retained |
| SparseTSF | [lss-1138/SparseTSF](https://github.com/lss-1138/SparseTSF) | Apache License 2.0 |
| ModernTCN | [luodhhh/ModernTCN](https://github.com/luodhhh/ModernTCN) | MIT, copyright 2024 luodhhh |
| PhaseFormer | [neumyor/PhaseFormer_TSL](https://github.com/neumyor/PhaseFormer_TSL) | MIT, including the retained THUML notice |
| ITSMixer | [JZX100II/ITSMixer](https://github.com/JZX100II/ITSMixer) | MIT, copyright 2026 Arian Lotfi |
| MixLinear | [aitianma/MixLinear](https://github.com/aitianma/MixLinear) | No explicit project licence was located in the inspected snapshot or repository root. Original upstream rights and notices are retained. |
| RootPurgeLinear | [Wangzzzzzzzz/RootPurge](https://github.com/Wangzzzzzzzz/RootPurge) | No explicit project licence was located in the inspected snapshot or repository root. Original upstream rights and notices are retained. |

Package dependencies such as NumPy and PyTorch retain their own licences. Their use does not change the licensing of the benchmark data or of other included source files.

Sources checked on 26 November 2026.
