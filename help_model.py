# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 13:16:20 2022

@author: artur
"""

import torch
from torchvision.models import mobilenet_v3_small

model = mobilenet_v3_small()
print(model)

image = torch.randn([10, 224, 398, 3])
model(image)