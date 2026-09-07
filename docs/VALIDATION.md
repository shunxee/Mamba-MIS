# 验证记录

日期：2026-09-08。

## 已执行

测试环境为项目独立虚拟环境：Windows、Python 3.12、PyTorch 2.8.0+cpu、torchvision 0.23.0+cpu，以及 requirements.txt 中固定的直接依赖。这里的 CPU 测试不代表 Ubuntu CUDA 构建已经验证。

命令：

```text
python -m pytest -q --tb=short -p no:cacheprovider
40 passed, 24 skipped, 19 warnings in 69.47s
```

通过项目包括：

- Mamba-MIS S/B/L 和全部 13 基线的 64×64 实际前向，输出维度及有限值检查。
- 七种消融的合成输入反向，所有参与参数的梯度存在且有限；矩形输入前向。
- ZOH 的 float64 数值梯度检查、标量解析解、近零 A 稳定性；横纵及反向扫描的坐标还原。
- SG 恒等频率滤波的数值对齐、复数参数梯度。
- 手工掩码指标、逐图/全局聚合、空掩码、极端 logits 损失。
- 本地图像/标签配对、跨划分重复检测、分组保留、训练集归一化独立性、同步增强和无重复评估采样。
- 纯合成数据上的训练流程、epoch 边界中断恢复：恢复运行与连续运行最终权重逐项精确一致，调度器状态一致。
- 梯度累积在不足完整 micro-batch 时与完整 batch 更新一致（测试模型不含 BatchNorm）。
- 合成 checkpoint 的标准/原尺寸评估、预测掩码和叠图、汇总、profile、复杂度散点图、多模型对比。
- SG/桥接/显著性/ERF 图像导出及原始数组输出。
- 36 个主实验配置、10 个消融配置及 138 个种子任务唯一性。
- 22 个 vendor 源码记录的交付哈希匹配，Ubuntu 安装脚本 LF 行结束符。

其他检查：Python 文件通过 AST 语法解析；editable 安装成功，`mis models` 正确列出全部 16 个模型；CLI 帮助入口可调用；`pip check` 返回无依赖冲突。第三方组件文件通过 SHA256 校验。

19 条提示来自 Matplotlib/pyparsing/NumPy 的弃用提示，以及 UNetV2 默认从头初始化时的提示，不影响本轮测试结果。

## 尚未执行

24 个有明确条件的测试被跳过：

- 4 个 Triton ZOH 前向/反向一致性测试，涵盖跨 32-token 块边界。
- 2 个基线 CUDA selective_scan 的分组/非分组参考一致性测试。
- 1 个Mamba-1 与上游模块的权重及输出一致性测试。
- 3 个 Mamba-MIS S/B/L 的 256×256 CUDA/BF16 反向测试。
- 13 个基线的 256×256 CUDA 前向测试。
- 1 个 Linux 双进程 Gloo 指标归约测试。当前 Windows CPU wheel 的 Gloo 设备不可用，测试限定为目标 Linux 环境执行。

Ubuntu Python 3.11 下的安装、CUDA 12.8 扩展构建、Triton kernel 编译、真实多卡 DDP、GPU 显存和训练速度尚未实测。源码和对应测试已提供，运行命令如下：

```bash
bash scripts/install_ubuntu.sh
source .venv/bin/activate
python -m pytest -m 'not slow' -q
python -m pytest -m cuda -q
```

## 测试数据

自动化测试使用随机张量、人工掩码和小型合成图像，检查数值运算、训练流程及输出文件。测试临时文件不包含在源码包中。
