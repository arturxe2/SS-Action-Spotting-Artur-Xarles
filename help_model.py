# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 13:16:20 2022

@author: artur
"""

import torch
import torchvision
import torchvision.transforms as T
from vit_pytorch import ViT, SimpleViT, CrossFormer, SepViT

from torchvision.models import swin_t, Swin_S_Weights

def mem(model):
    mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs
    return mem / 1000000000

model = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

print('ViT:' + str(mem(model)))

model = swin_t()


print('Swin Tiny:' + str(mem(model)))

v = SimpleViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048
)

print('Simple ViT:' + str(mem(v)))

model = CrossFormer(
    num_classes = 1000,                # number of output classes
    dim = (64, 128, 256, 512),         # dimension at each stage
    depth = (2, 2, 8, 2),              # depth of transformer at each stage
    global_window_size = (8, 4, 2, 1), # global window sizes at each stage
    local_window_size = 7,             # local window size (can be customized for each stage, but in paper, held constant at 7 for all stages)
)


print('CrossFormer:' + str(mem(model)))

transform = T.Resize((224,224))
model.head = torch.nn.Identity()
#model.classifier = torch.nn.Identity()
print(model)

v = SepViT(
    num_classes = 1000,
    dim = 32,               # dimensions of first stage, which doubles every stage (32, 64, 128, 256) for SepViT-Lite
    dim_head = 32,          # attention head dimension
    heads = (1, 2, 4, 8),   # number of heads per stage
    depth = (1, 2, 6, 2),   # number of transformer blocks per stage
    window_size = 7,        # window size of DSS Attention block
    dropout = 0.1           # dropout
)

print('SepViT:' + str(mem(v)))

image = torch.randn([31, 3, 224, 398])
#images = transform(image)

print(image.shape)
#image = image.view(-1, 3, 224, 398)
output = model(image)
print(output.shape)
