import torch
import torch.nn as nn
from nndet.arch.encoder.modular import Encoder
from timm.models.swin_transformer import SwinTransformer

class ResNetSwinEncoder(Encoder):
    def __init__(self, **kwargs):
        super(ResNetSwinEncoder, self).__init__(**kwargs)
        # 定义ResNet部分（已由父类Encoder处理）
        # 添加Swin Transformer模块
        self.swin_transformer = SwinTransformer(pretrained=True)
        # 根据需要调整输入/输出通道数
        resnet_output_channels = self.get_channels()[-1]
        swin_input_channels = 1024  # Swin Transformer的输入通道数，视具体模型而定

        if resnet_output_channels != swin_input_channels:
            self.conv_adjust = nn.Conv2d(resnet_output_channels, swin_input_channels, kernel_size=1)
        else:
            self.conv_adjust = nn.Identity()

    def forward(self, x):
        # 使用父类的forward方法，得到ResNet的输出
        x = super().forward(x)
        # x的形状为List[Tensor]，取最后一层的输出
        x = x[-1]
        # 调整通道数
        x = self.conv_adjust(x)
        # Swin Transformer的前向传播
        x = self.swin_transformer(x)
        return [x]  # 返回列表，以与原始编码器的输出格式一致
