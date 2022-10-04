# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 13:16:20 2022

@author: artur
"""

import torch
import torchvision

from torchvision.models import vit_b_16

model = vit_b_16()

#model.classifier = torch.nn.Identity()
print(model)

image = torch.randn([31, 3, 224, 224])
#image = image.view(-1, 3, 224, 398)
output = model(image)
print(output.shape)
