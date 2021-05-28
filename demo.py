#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/28 11:31
# @Author  : yangqiang
# @File    : demo.py
import torch

from graphs.swin_transformer import SwinTransformer


model = SwinTransformer(img_size=224,
                        patch_size=4,
                        in_chans=3,
                        num_classes=1000,
                        embed_dim=96,
                        depths=[2, 2, 6, 2],
                        num_heads=[3, 6, 12, 24],
                        window_size=7,
                        mlp_ratio=4.,
                        qkv_bias=True,
                        qk_scale=None,
                        drop_rate=0.0,
                        drop_path_rate=0.2,
                        ape=False,
                        patch_norm=True,
                        use_checkpoint=False)
# print(str(model))

model.load_state_dict(torch.load("swin_tiny_patch4_window7_224.pth", map_location='cpu')['model'])
