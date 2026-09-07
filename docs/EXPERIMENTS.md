# 实验覆盖与产物

所有路径相对于项目根目录；配置为 UTF-8 YAML，`extends` 相对于当前 YAML 解析，数据/输出路径相对于运行目录解析。

| 论文实验 | 配置/入口 | 数据与输出 |
|---|---|---|
| ISIC17 定量、S/B/L | `configs/main/isic17_*.yaml`，12 个配置 | ISIC17；`test_256.json` |
| ISIC18 定量、S/B/L | `configs/main/isic18_*.yaml`，15 个配置 | ISIC18；`test_256.json` |
| 息肉定量、S/B/L | `configs/main/polyp_*.yaml`，9 个配置 | 联合训练，五测试集单独报告 |
| BFSS 与 SS2D | `configs/ablation/isic17_ss2d.yaml`，对照 main B | ISIC17；指标、profile |
| 去除 SG | `configs/ablation/isic17_no_sg.yaml`，对照 main B | ISIC17；指标、spectral 可视化 |
| AB/MFFB 四组合 | `configs/ablation/{isic17,isic18}_{no_bridge,ab_only,mffb_only}.yaml`，完整组合复用 main B | 两个 ISIC 数据集；指标、bridge 可视化 |
| CNN/ViT/BSSS 基础块 | `configs/ablation/isic17_{cnn,vit}.yaml`，BSSS 对照 main B | ISIC17；指标、saliency 可视化 |
| 参数量、算量与性能 | `profile --config ...` | 输入 1×3×256×256；profile JSON |
| ERF | `visualize --kind erf`，主模型与四种 SSM 基线 | 固定样本，初始化/训练权重的中心像素输入梯度 |
| SG 输出 | `visualize --kind spectral` | 最后两级解码器，逐 block 平均通道特征 |
| skip 输出 | `visualize --kind bridge` | 四个尺度的跳连输出 |
| 定性分割 | `visualize --kind segmentation`，`predict` | 图像、GT、概率、预测；原尺寸叠图 |
| 基础块热图 | `visualize --kind saliency` | 输入梯度显著性，预测区域平均 logit 为目标 |
| 多模型定性对比 | `compare --checkpoints ...` | 固定样本、统一列顺序的预测/热图/ERF 对比 |
| 参数/算量—DSC 散点图 | `plot --summary ... --profiles ... --dataset ...` | 真实评估报告与对应 profile 的散点图 |

主实验 36 配置，消融新增 10 配置。默认 42/43/44 三个种子，共 138 个训练任务；完整桥和 BSSS 对照复用主实验，避免重复训练同配置。SANet 在 ISIC18 表中出现的同名行对应一个网络实现。

```bash
# 仅列出任务，不启动训练
python -m mamba_mis experiments --suite all > jobs.json
# 显式执行时按顺序训练、保存最佳权重、再独立测试
python -m mamba_mis experiments --suite all --execute
# 单个消融
python -m mamba_mis train --config configs/ablation/isic17_no_sg.yaml
```

默认批量入口按单设备逐任务运行；多卡任务可使用 jobs.json 中的配置与覆盖参数配合 torchrun。各任务的输出目录唯一；已有 `last.pt` 不会被新训练覆盖。

性能报告列明算量未覆盖项。卷积/线性为乘加各计 1，SG 单列实数 FFT 估计，SSM 算术和 exponential 次数分列。跨模型比较优先同时看实测延迟与显存；`estimated_counted_gflops` 是已统计运算的估计，不是完整硬件计数。

可视化默认测试集中按 dataset/id 排序选前 8 项；可用 `--ids ISIC17/样本名` 固定样本。保存 selection.json 和原始 npz，热图保留说明中的目标和共享色阶；显著性图属于梯度解释，不是网络内部注意力权重。
