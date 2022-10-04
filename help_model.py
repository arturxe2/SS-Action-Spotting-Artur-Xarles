# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 13:16:20 2022

@author: artur
"""

import torch
import torchvision
import torchvision.transforms as T

from torchvision.models import swin_t, Swin_T_Weights

model = swin_t(weights = Swin_T_Weights)

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
