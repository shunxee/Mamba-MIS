# 数据获取

- ISIC17 / ISIC18： [VM-UNet 官方数据说明](https://github.com/JCruan519/VM-UNet#1-prepare-the-dataset)。该发布格式中命名为 `val` 的外部保留集应在清单中记为 `test`；本工程再从训练池划分验证集。
- 息肉五数据集： [PraNet 官方数据说明](https://github.com/DengPingFan/PraNet#trainingtesting)。使用其明确的训练/测试划分，保留实际文件数量。

数据由用户自行准备；工程没有下载器，不包含数据。README 的数据获取入口统一指向本页。
