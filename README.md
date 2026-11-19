# CoReXNet

CoReXNet: Multivariate Time Series Forecasting via Common Response Reconstruction and Exchange

## 源码版本

2026-10-23 留存的真实代码、三种子源码快照与固定任务配置。

## 实验入口

- `模型与基线源码`
- `三种子训练源码快照`
- `种子2026配置`
- `新增2027与2028配置`
- `测试与效率测量/test_common.py`

按数据集、预测步长、方法和种子定位训练配置。主实验覆盖 Electricity、PEMS03、PEMS04、PEMS08、Solar、AEMO5。三个 PEMS 数据集上的模块消融使用 96 步预测。新增两种子的训练器在对应快照的 three_seed_train.py 中。

## 输入与运行条件

需恢复原版时序输入、训练窗口和数据划分。原项目使用多个独立适配目录，其中 AEMO5 和 Solar 有各自的训练适配器。

主模型前向位于 模型与基线源码/方法核心/主实验OURS/i3_response_residual_exchange.py。训练器的命令为 python train.py --config 实际配置.json --output 新输出目录 --data-root 数据目录。CRR、NRD 和 ARE 对应的最终分支分别为 mean_residual、fixed_hinge 配合 no_subtraction、absolute_response 配合 width32。
