# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 14:20:47 2022

@author: arturxe
"""
import torch


class CLIP_loss(torch.nn.Module):
    def __init__(self):
        super(CLIP_loss, self).__init__()
        #self.weights1 = weights1
        
    def forward(self, logits):
        labels = torch.arange(0, logits.shape[0])
        loss = torch.nn.CrossEntropyLoss()
        loss1 = loss(logits, labels, axis = 0)
        loss2 = loss(logits, labels, axis = 1)
        
        return (loss1 + loss2) / 2