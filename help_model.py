# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 13:16:20 2022

@author: artur
"""

import torch
import torchvision
import torchvision.transforms as T
from vit_pytorch import ViT

from torchvision.models import swin_s, Swin_S_Weights

model = ViT()

mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
mem = mem_params + mem_bufs

print(mem)

transform = T.Resize((224,224))
model.head = torch.nn.Identity()
#model.classifier = torch.nn.Identity()
print(model)

image = torch.randn([31, 3, 224, 398])
#images = transform(image)

print(image.shape)
#image = image.view(-1, 3, 224, 398)
output = model(image)
print(output.shape)
