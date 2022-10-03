# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 13:16:20 2022

@author: artur
"""

import torch

from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights

model = mvit_v2_s(MViT_V2_S_Weights)

#model.classifier = torch.nn.Identity()
print(model)

image = torch.randn([10, 31, 3, 224, 398])
image = image.view(-1, 3, 224, 398)
output = model(image)
print(output.view(10, 31, -1).shape)
