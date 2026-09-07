# Mamba-MIS

Visual Bi-Directional State Space Model with Fusion Bridge and Spectral Gate Integration for Medical Image Segmentation

Mamba-MIS 提供 S/B/L 三种模型规模，通过双向融合选择性扫描、频域门控和注意力多尺度特征融合桥进行医学图像分割。仓库包含模型网络、训练与评估流程、13 个对比基线、消融实验及可视化工具，默认输入为 256×256 RGB 图像，输出为二值分割图。

## Ubuntu 安装

目标：Ubuntu 22.04/24.04、Python 3.11、NVIDIA GPU。GPU 构建使用 CUDA Toolkit 12.8（包含 nvcc）、对应驱动、C++ 编译器和 Python 开发头文件；`CUDA_HOME` 指向 Toolkit。PyTorch 2.8.0、torchvision 0.23.0、Triton 3.4.0 固定搭配；基线扫描扩展从源码针对当前 PyTorch 构建。

```bash
cd Mamba-MIS
bash scripts/install_ubuntu.sh
source .venv/bin/activate
python -m mamba_mis models
```

脚本安装依赖并导出 `installed-ubuntu.lock.txt`，不会下载数据或训练。CPU 检查可独立安装 CPU torch/torchvision，再 `pip install -e '.[test]'`；CPU 参考扫描主要用于小尺寸检查。

## 本地数据

[数据获取入口](docs/DATA.md)。无需把数据传入工程；可在 YAML 中设置任意本地根目录。CSV 的 image/mask 路径相对于 `data.root`。

ISIC 每个集合采用 `images/`、`masks/`，支持 `train/val/test` 明确目录，或全部样本放在根目录 `images/masks` 后按指定数量划分。原发布包若将测试集命名为 val，应通过显式清单标为 test；不要直接当作本工程验证集。

```text
data/ISIC17/train/images/xxx.png
data/ISIC17/train/masks/xxx_segmentation.png
data/ISIC17/test/images/yyy.png
data/ISIC17/test/masks/yyy_segmentation.png

data/polyp/TrainDataset/images/...
data/polyp/TrainDataset/masks/...
data/polyp/TestDataset/Kvasir/images/...
data/polyp/TestDataset/Kvasir/masks/...
# 同时准备 CVC-ClinicDB、CVC-ColonDB、CVC-300、ETIS-LaribPolypDB
```

也支持息肉数据的 `<数据集>/train/{images,masks}`、`<数据集>/test/{images,masks}` 目录。

```bash
python -m mamba_mis prepare --dataset isic17 --root data/ISIC17 --output data/isic17.csv
python -m mamba_mis prepare --dataset isic18 --root data/ISIC18 --output data/isic18.csv
python -m mamba_mis prepare --dataset polyp --root data/polyp --output data/polyp.csv
# 已有明确划分：仅校验并规范化，不重新划分测试集
python -m mamba_mis prepare --dataset isic17 --root /datasets/ISIC17 \
  --manifest /datasets/split.csv --output data/isic17.csv
```

清单列为 `id,dataset,image,mask,split,group`；split 取 train/val/test，group 可空。有患者/病灶标识时通过清单或 `--groups groups.csv` 提供（列 dataset,id,group），划分会保留完整分组。若没有独立 val，从训练池固定留出约 10%；同一清单供所有种子共用。图像/标签配对、尺寸、重复路径、跨集合相同图像和分组重叠会被检查。ISIC 自动全量划分要求分别有 2150/2694 张图，否则提供自己的明确清单。

## 训练、恢复、测试

```bash
# 单卡：micro-batch 4，累积到有效 batch 32
python -m mamba_mis train --config configs/main/isic17_mamba_mis_b.yaml
# 双卡：自动调整累积步数，有效 batch 仍为 32
torchrun --standalone --nproc_per_node=2 -m mamba_mis train \
  --config configs/main/isic17_mamba_mis_b.yaml
# 更改种子/路径
python -m mamba_mis train --config configs/main/isic17_mamba_mis_b.yaml \
  --set train.seed=43 --set output=runs/main/isic17_mamba_mis_b/seed43
# 从完整 epoch 边界恢复，使用与原运行一致的配置和 world size
python -m mamba_mis train --config configs/main/isic17_mamba_mis_b.yaml \
  --resume runs/main/isic17_mamba_mis_b/seed42/last.pt
# 独立测试；同一 checkpoint 同时覆盖息肉的五个测试集
python -m mamba_mis evaluate \
  --checkpoint runs/main/isic17_mamba_mis_b/seed42/best.pt --output runs/test_256.json
# 原图尺寸评估单独生成报告
python -m mamba_mis evaluate \
  --checkpoint runs/main/isic17_mamba_mis_b/seed42/best.pt --original --output runs/test_original.json
```

默认 300 epochs、AdamW、初始 lr=1e-3、weight decay=.01、Dice+BCE、CosineAnnealingLR(T_max=50,eta_min=1e-5)、BF16。只根据验证集 macro Dice 选最佳模型；训练入口不运行测试集。`last.pt` 保存优化器、调度器、AMP 和各 rank 随机状态，`best.pt` 保存最佳验证 checkpoint。训练输出还有 config.json、normalization.json、history.jsonl。

`--set train.amp=no` 禁用混合精度，`--set train.deterministic=true` 为主模型选择确定性的 CPU/PyTorch 扫描后端（也可在 CUDA 上运行，速度较低）。基线原始 CUDA 扩展的确定性行为由该扩展决定。Micro-batch 会影响带 BatchNorm 的基线统计，比较时保持一致。

默认不加载预训练。需要时显式覆盖 YAML：

```yaml
pretrained:
  path: /weights/encoder.pt
  key: model                 # 权重字典内部键；空字符串表示整个字典
  target: model.vmunet       # named_modules() 中的目标模块；空字符串表示整网
  strict: true
```

## 批量实验与分析

```bash
python -m mamba_mis experiments --suite all > jobs.json
python -m mamba_mis experiments --suite all --execute
python -m mamba_mis summarize --inputs runs/seed42_test.json runs/seed43_test.json runs/seed44_test.json \
  --output runs/summary.json
python -m mamba_mis profile --config configs/main/isic17_mamba_mis_b.yaml --output runs/profile_b.json
python -m mamba_mis visualize --checkpoint runs/main/isic17_mamba_mis_b/seed42/best.pt \
  --kind spectral --output runs/figures/spectral
python -m mamba_mis predict --checkpoint runs/main/isic17_mamba_mis_b/seed42/best.pt \
  --inputs /images/example.png --output runs/predictions
python -m mamba_mis compare --checkpoints /runs/model_a/best.pt /runs/model_b/best.pt \
  --kind segmentation --output runs/comparison
python -m mamba_mis plot --summary runs/summary.json --profiles runs/profile_b.json \
  --dataset ISIC17 --output runs/complexity_dice.png
```

`visualize --kind` 支持 segmentation、erf、spectral、bridge、saliency；`--ids dataset/id` 固定样本。`compare` 支持 segmentation、saliency、erf。图像同时保存原始数值与样本清单。复杂度图只使用实际生成的报告。

默认 46 个配置、三种子共 138 个任务；定量、消融和每张分析图的对应命令见 [实验配置](docs/EXPERIMENTS.md)。网络结构、指标定义、扫描后端及恢复条件见 [模型文档](docs/ARCHITECTURE.md)。第三方组件及版本见 [依赖与致谢](docs/SOURCES.md) 与 SOURCE_MANIFEST.json。

## 验证

```bash
python -m pytest -m 'not slow' -q
# Linux GPU：包含 Triton 梯度对齐、基线算子对齐和 256×256 反向
python -m pytest -m cuda -q
```

测试仅使用合成图像、人工掩码和随机张量。当前完成情况与未执行的 Ubuntu/GPU 检查见 [验证记录](docs/VALIDATION.md)。
