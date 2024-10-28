"""
Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import torch.nn as nn
from typing import Callable, Tuple, Sequence, Union, List, Optional
#swim
from timm.models.swin_transformer import swin_base_patch4_window7_224_in22k as SwinTransformer
from nndet.arch.encoder.abstract import AbstractEncoder
from nndet.arch.blocks.basic import AbstractBlock
from nndet.arch.encoder.SE import SEBlock
from nndet.arch.encoder.DropBlock3D import DropBlock3D  # 适用于3D数据
from nndet.arch.encoder.SelfAttention import SelfAttention3D  # 导入自注意力模块

__all__ = ["Encoder"]

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

        for stage_id in range(self.num_stages):
            current_in_channels = in_ch

            # 选择卷积块的种类
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

            # 如果是最后一层，则添加 Self-Attention
            if stage_id == self.num_stages - 1:
                self_attention = SelfAttention3D(in_ch)
                stages.append(nn.Sequential(_block, se, drop_block, self_attention))
            else:
                stages.append(nn.Sequential(_block, se, drop_block))

        self.stages = torch.nn.ModuleList(stages)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = []
        for stage_id, module in enumerate(self.stages):
            x = module(x)
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

