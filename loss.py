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
        labels = torch.arange(0, logits.shape[0]).cuda()
        loss1 = torch.nn.functional.cross_entropy(logits, labels)
        loss2 = torch.nn.functional.cross_entropy(torch.transpose(logits, 0, 1), labels)
        
        return (loss1 + loss2) / 2