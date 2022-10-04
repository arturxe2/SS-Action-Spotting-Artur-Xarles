# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 13:16:20 2022

@author: artur
"""

import torch
import torchvision
import torchvision.transforms as T

from torchvision.models import vit_b_16

model = vit_b_16()
transform = T.Resize((224,224))
model.heads = torch.nn.Identity()
#model.classifier = torch.nn.Identity()
print(model)

image = torch.randn([31, 3, 224, 398])
images = transform(image)

print(images.shape)
#image = image.view(-1, 3, 224, 398)
output = model(images)
print(output.shape)
