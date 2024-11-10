from nndet.arch.encoder.swimTransformer import SwinTransformerBlock3D, RetinaUNetWithSwin3D
from nndet.arch.blocks.basic import AbstractBlock

# 初始化encoder
encoder = Encoder(
    conv=lambda: nn.Conv3d,  # 使用3D卷积
    conv_kernels=[3, 3, 3],
    strides=[2, 2, 2],
    block_cls=AbstractBlock,  # 使用自定义的卷积块
    in_channels=1,  # 单通道输入
    start_channels=32,  # 输出通道开始为32
    use_swin_in_last_stage=True  # 启用Swin Transformer
)

# 初始化decoder（假设你有Decoder类）
decoder = Decoder(...)

# 使用Swin3D的RetinaUnet网络
model = RetinaUNetWithSwin3D(
    encoder=encoder,
    decoder=decoder,
    dim=256,  # Swin Transformer的特征维度
    num_heads=8,  # 注意力头的数量
    window_size=(7, 7, 7),  # Swin Transformer的窗口大小
    shift_size=(3, 3, 3)  # 窗口的偏移大小
)

# 输入一个3D张量，例如B=1, C=1, D=128, H=128, W=128
input_tensor = torch.randn(1, 1, 128, 128, 128)
output = model(input_tensor)
print(output.shape)
