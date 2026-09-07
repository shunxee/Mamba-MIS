# 依赖与致谢

Mamba-MIS 使用 PyTorch 进行模型训练，Triton 提供 GPU 扫描算子，timm 和 torchvision 提供视觉模型组件，NumPy、Pillow 和 Matplotlib 用于数据处理及可视化。

## 对比模型

| 名称 | 模型组件 | 默认配置 |
|---|---|---|
| unet | U_Net | 输入 3 通道，输出 1 通道 |
| attunet | AttU_Net | 注意力 U-Net |
| unetv2 | UNetV2 | PVTv2-B2 骨干 |
| malunet | MALUNet | 轻量注意力网络 |
| vmunet | VMUNet | 基础宽度 96，深度 [2,2,9,2] |
| vmunetv2 | VMUNetV2 | mid_channel=48 |
| hvmunet | H_vmunet | 高阶视觉状态空间模型 |
| ulvmunet | UltraLight_VM_UNet | 并行视觉 Mamba |
| unetpp | UNetPlusPlus | base_ch=32 |
| utnetv2 | UTNetV2 | base_chan=32 |
| transfuse | TransFuse_S | ResNet34 + DeiT-small，depth=8 |
| sanet | SANet | Res2Net50 骨干 |
| pranet | PraNet | Res2Net50 骨干 |

感谢以下开源项目提供的模型与计算组件：

- https://github.com/yhygao/CBIM-Medical-Image-Segmentation
- https://github.com/Rayicer/TransFuse
- https://github.com/weijun-arc/SANet
- https://github.com/DengPingFan/PraNet
- https://github.com/state-spaces/mamba

组件版本、源文件哈希和仓库文件哈希记录于 SOURCE_MANIFEST.json。模型源码包含于 models/vendor，使用时无需另行检出上游仓库。基线统一通过 BaselineAdapter 返回 logits 与可选辅助输出。

第三方源码版权头保留，已有 LICENSE 随对应目录保存，各组件遵循其原有许可证。
