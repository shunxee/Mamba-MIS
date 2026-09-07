# 模型结构与计算

Mamba-MIS 由编码器、解码器和注意力多尺度特征融合桥组成。BSSS 模块结合选择性状态空间扫描与频域门控，建模空间和频域特征。下文介绍各阶段尺寸、模块运算及训练设置。

## 尺寸与连接

对于 256×256 输入，patch embedding 为 Conv2d(3,16,4,stride=4) 加 LayerNorm。编码器每级输出作为跳连，分辨率与通道依次为 64²×16、32²×32、16²×64、8²×128。前三个编码阶段后执行四邻域拼接、LayerNorm 和线性通道压缩。

解码器为 8²×128、16²×64、32²×32、64²×16、256²×8；末次 patch expansion 倍率为 4，将完整分辨率特征送入第五阶段。四个桥接输出分别与前四个解码阶段相加，最后经 1×1 convolution 输出单通道 logits。

S/B/L 均使用基础宽度 16；深度见配置。空间尺寸必须是 32 的倍数；网络本身支持矩形输入，默认数据流程使用正方形 256。频域参数按配置分辨率创建，不能直接用另一输入尺寸加载同一权重。

## BSSS 与扫描

输入通道分为 `[0:floor(C/3)]`、`[floor(C/3):floor(2C/3)]`、剩余通道；不会丢弃余数。各分支 LayerNorm 后投影到 C。第一分支为 SiLU 门控，第二分支为 depthwise 3×3 convolution、SiLU 和 BFSS，第三分支为 SG、LayerNorm 和 SiLU。第二、三分支拼接后投影至 C，与门控相乘，输出投影后加输入残差。

BFSS 为行优先和列优先两条独立 S6；列扫描结果先恢复轴顺序再拼接，FFFN 为 LayerNorm(2C)→Linear(2C,C)→SiLU→Linear(C,C)。SS2D 消融使用行/列与各自反向共四路，恢复原坐标后求和；保留同一 S6 离散化，隔离扫描/融合方式的改变。

S6 的 A 为 `-exp(A_log)`，A_log 初始化为 log(1…N)，D 初始化为 1，N=32；B、C 为输入的线性投影，Δ 为逐通道 Linear 后 softplus，偏置初始化 -4。初始状态为零。

采用 `a_bar=exp(ΔA)`、`b_bar=Δ*expm1(ΔA)/(ΔA)*B`，接近零使用解析级数。CPU 后端可用 float64 做梯度检查；通常状态为 float32。Triton 后端按 batch/channel 并行、state 向量化，每 32 token 保存边界状态；反向在块内重计算状态，显存不存储全部时刻的隐状态。B/C 与参数梯度跨通道/样本进行 atomic reduction。严格确定性模式使用 PyTorch 后端。

基线的 Mamba-1 运算保留原始 `delta*B` 输入离散化；不会与主模型的 ZOH 算子混用。GPU 调用固定版本 mamba-ssm selective_scan，CPU 提供同参数的参考运算。UltraLight 的 Mamba-1 模块包含原始因果卷积、门控和投影，保留 Δ 特殊偏置初始化。

## SG 与桥接

SG 使用 ortho 归一化的 rfft2/irfft2；复数权重用两个实数存储，初始正态分布标准差 .02，偏置为零。半谱表示由 irfft2 返回实值张量；参数统计按两个实标量计数。主干 FFT 禁用 autocast，输出转回输入 dtype。

AB 的计算顺序为：空间注意力→通道注意力→残差。空间分支由通道均值/最大值、7×7 convolution、BFSS 和 sigmoid 构成。通道分支使用平均/最大池化、共享 Linear、BFSS、SiLU、Linear，求和后 sigmoid；压缩宽度至少为 1。

MFFB 统一压缩到 32 通道，高分辨率来源使用 adaptive average pooling，低分辨率来源使用 bilinear interpolation（align_corners=False）；每个源/目标对有独立 3×3 depthwise convolution。四路乘积在 FP32 中累计，投影至目标通道。

去除 SG 时保留门控和扫描分支及输入残差，删除 SG 分支参数并将融合投影输入缩为 C。AB-only、MFFB-only、无桥分别只保留相应组件；无桥使用同尺度恒等跳连。CNN 消融为 3×3 convolution、BatchNorm、ReLU；ViT 消融为 pre-LN 全局多头注意力和 GELU MLP，4 heads、MLP ratio=4。全分辨率全局注意力计算量很大，配置保留这一运算定义。

## 训练与评估

有效 batch size 是跨设备、跨累积步数的样本数；BatchNorm 统计仍对应实际 micro-batch。默认 micro-batch=4，单卡累积 8 步、双卡累积 4 步。末尾不足一组时按实际样本数加权。DDP 训练采样可能补齐少量样本；评估 sampler 不补齐、不重复，BN buffers 从 rank 0 同步后由各 rank 独立前向。

Dice 按每张图计算后平均；BCE 在全部像素上平均；主损失两项等权。辅助输出默认权重为零，可显式设置 `train.aux_weight`。所有基线默认从头初始化，构造时不下载权重。预训练映射由配置中的 path/key/target/strict 指定，记录 SHA256 及不匹配键。

CosineAnnealingLR 在每个 epoch 结束调用一次，T_max=50 且共 300 epochs；遵循 PyTorch 的余弦函数行为，经过最小点后学习率会回升。这不是自动重启调度器，也不是单次 300-epoch 余弦衰减。

统计以 0…1 小数存储。foreground_iou 只计算前景；miou_fg_bg 是前景和背景 IoU 的平均；macro 是逐图平均，global 来自所有图像的 TP/FP/FN/TN 总和。预测与真值均空时 Dice/前景 IoU 为 1；无负类时 specificity、无正类时 sensitivity 定义为 1。不存在空测试集成绩。训练选权重只使用验证 macro Dice。

默认 256×256 评估；original 模式将 logits 双线性恢复至原 mask 分辨率，再 sigmoid 和阈值化。未启用 TTA、按样本 min/max 拉伸或预测后处理。均值和样本标准差仅在相同模型配置、训练协议、划分哈希、阈值、分辨率的不同种子间汇总；单次结果的标准差为 null。

断点恢复支持 epoch 边界，保存模型、优化器、调度器、AMP scaler、每个 rank 的随机数状态、归一化和完整配置。同一 world size、训练配置、模型配置及清单下恢复。完整 checkpoint 包含 Python/NumPy 状态，只加载自己生成或可信来源的文件。
