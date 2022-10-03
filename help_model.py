# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 13:16:20 2022

@author: artur
"""

import torch

from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

model = mobilenet_v3_small(MobileNet_V3_Small_Weights)

model.classifier = torch.nn.Identity()
#print(model)

image = torch.randn([10, 31, 3, 224, 398])
image = image.view(-1, 3, 224, 398)
output = model(image)
print(output.view(10, 31, -1).shape)
