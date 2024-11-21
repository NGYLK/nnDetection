import torch 
import torch.nn as nn
from typing import Callable, Tuple, Sequence, Union, List, Optional
from timm.models.swin_transformer import swin_base_patch4_window7_224_in22k as SwinTransformer
from nndet.arch.encoder.abstract import AbstractEncoder
from nndet.arch.blocks.basic import AbstractBlock
from nndet.arch.encoder.SE import SEBlock
from nndet.arch.encoder.DropBlock3D import DropBlock3D  # 适用于3D数据
from nndet.arch.encoder.SelfAttention import SelfAttention3D  # 导入自注意力模块
from nndet.arch.encoder.swimTransformer import SwinTransformer3D
import torch.nn.functional as F  # 加入插值函数
import logging

# 配置日志级别和格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

__all__ = ["Encoder"]

class SelfAttentionFusion(nn.Module):
    def __init__(self, conv_channels, transformer_channels, fused_channels):
        super(SelfAttentionFusion, self).__init__()
        # 投影卷积特征和Transformer特征到相同的维度
        self.conv_proj = nn.Conv3d(conv_channels, fused_channels, kernel_size=1)
        self.transformer_proj = nn.Conv3d(transformer_channels, fused_channels, kernel_size=1)
        
        # 自注意力模块
        self.self_attention = SelfAttention3D(fused_channels * 2)
        
        # 残差连接，如果卷积通道数和融合通道数不同，则添加1x1卷积调整维度
        self.residual_conv = nn.Conv3d(conv_channels, fused_channels, kernel_size=1) if conv_channels != fused_channels else nn.Identity()
    
    def forward(self, conv_feat, transformer_feat):
        # 投影到相同维度
        conv_feat_proj = self.conv_proj(conv_feat)
        transformer_feat_proj = self.transformer_proj(transformer_feat)
        
        # 拼接特征
        combined = torch.cat([conv_feat_proj, transformer_feat_proj], dim=1)  # (B, 2*C, D, H, W)
        
        # 使用SelfAttention3D进行融合
        fused_feat = self.self_attention(combined)  # (B, 2*C, D, H, W)
        
        # 分离卷积和Transformer特征
        conv_fused, transformer_fused = torch.split(fused_feat, conv_feat_proj.size(1), dim=1)
        
        # 加权融合
        fused_feat = conv_fused + transformer_fused
        
        # 残差连接，将原始卷积输出加回到融合输出上
        residual = self.residual_conv(conv_feat)  # 原始卷积特征经过调整后添加
        fused_feat = fused_feat + residual  # 加上残差
        
        return fused_feat

class Encoder(AbstractEncoder):
    def __init__(self,
                 conv: Callable[[], nn.Module],
                 conv_kernels: Sequence[Union[Tuple[int], int]],
                 strides: Sequence[Union[Tuple[int], int]],
                 block_cls: AbstractBlock,
                 in_channels: int,
                 start_channels: int,
                 stage_kwargs: Sequence[dict] = None,
                 out_stages: Sequence[int] = None,
                 max_channels: int = None,
                 first_block_cls: Optional[AbstractBlock] = None,
                 se_reduction_ratio: int = 16,
                 drop_prob: float = 0.2,
                 block_size: int = 5):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.num_stages = len(conv_kernels)
        self.dim = conv.dim
        if stage_kwargs is None:
            stage_kwargs = [{}] * self.num_stages
        elif isinstance(stage_kwargs, dict):
            stage_kwargs = [stage_kwargs] * self.num_stages
        assert len(stage_kwargs) == len(conv_kernels)

        if out_stages is None:
            self.out_stages = list(range(self.num_stages))
        else:
            self.out_stages = out_stages
        if first_block_cls is None:
            first_block_cls = block_cls

        stages = []
        self.out_channels = []
        in_ch = in_channels
        if isinstance(strides[0], int):
            strides = [tuple([s] * self.dim) for s in strides]
        self.strides = strides

        # 定义并行的Swin Transformer分支
        self.use_transformer = True
        if self.use_transformer:
            self.transformer = SwinTransformer3D(
                in_chans=in_channels,
                embed_dim=start_channels,
                window_size=(4,8,8),
                patch_size=(2,4,4),
                depths=[4,6,8],
                num_heads=[4, 8, 16],
                mlp_ratio=4.,
                qkv_bias=True,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.2,
                norm_layer=nn.LayerNorm
            )
            transformer_out_channels = start_channels * (2 ** (len([4,4,4]) - 1))

        for stage_id in range(self.num_stages):
            current_in_channels = in_ch
            if stage_id == 0:
                _block = first_block_cls(
                    conv=conv,
                    in_channels=current_in_channels,
                    out_channels=start_channels,
                    conv_kernel=conv_kernels[stage_id],
                    stride=None,
                    max_out_channels=max_channels,
                    **stage_kwargs[stage_id],
                )
            else:
                _block = block_cls(
                    conv=conv,
                    in_channels=current_in_channels,
                    out_channels=None,
                    conv_kernel=conv_kernels[stage_id],
                    stride=strides[stage_id - 1],
                    max_out_channels=max_channels,
                    **stage_kwargs[stage_id],
                )
            in_ch = _block.get_output_channels()
            self.out_channels.append(in_ch)

            # 在卷积块后添加 SE 模块和 DropBlock
            se = SEBlock(in_ch, reduction_ratio=se_reduction_ratio)
            drop_block = DropBlock3D(drop_prob=drop_prob, block_size=block_size)

            stages.append(nn.Sequential(_block, se, drop_block))

        self.stages = torch.nn.ModuleList(stages)

        # SelfAttentionFusion 初始化
        if self.use_transformer:
            self.self_attention_fusion = SelfAttentionFusion(
                conv_channels=in_ch,
                transformer_channels=transformer_out_channels,
                fused_channels=in_ch
            )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = []

        # Transformer分支的特征
        if self.use_transformer:
            transformer_feat = self.transformer(x)

        for stage_id, module in enumerate(self.stages):
            x = module(x)

            # 在最后一层进行特征融合
            if self.use_transformer and stage_id == self.num_stages - 1:
                # 调整 transformer_feat 的尺寸以匹配 x
                transformer_feat_resized = F.interpolate(transformer_feat, size=x.shape[2:], mode='trilinear', align_corners=False)
                # 使用 SelfAttentionFusion 进行融合
                x = self.self_attention_fusion(x, transformer_feat_resized)

            if stage_id in self.out_stages:
                outputs.append(x)

        return outputs

    def get_channels(self) -> List[int]:
        """
        计算每个返回特征图的通道数
        """
        out_channels = []
        for stage_id in range(self.num_stages):
            if stage_id in self.out_stages:
                # 如果使用了融合，通道数需要调整
                if self.use_transformer and stage_id == self.num_stages - 1:
                    out_channels.append(self.out_channels[stage_id])  # 假设 SelfAttentionFusion 保持通道数不变
                else:
                    out_channels.append(self.out_channels[stage_id])
        return out_channels

    def get_strides(self) -> List[List[int]]:
        """
        计算每个输出特征图相对于输入大小的步幅
        """
        out_strides = []
        for stage_id in range(self.num_stages):
            if stage_id == 0:
                out_strides.append([1] * self.dim)
            else:
                new_stride = [prev_stride * pool_size for prev_stride, pool_size
                              in zip(out_strides[stage_id - 1], self.strides[stage_id - 1])]
                out_strides.append(new_stride)
        return out_strides
