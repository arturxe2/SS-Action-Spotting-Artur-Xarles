# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 12:20:25 2022

@author: artur
"""

import math
import torch
import torch.nn as nn


#Model class
class Model(nn.Module):
    def __init__(self, weights=None, d=512, chunk_size=10, framerate=2, model="SSModel"):
        """
        INPUT: two Tensors of shape (batch_size,chunk_size*framerate,feature_size)
        OUTPUTS: two Tensors of shape (batch_size,d)
        """

        super(Model, self).__init__()

        self.d = d
        self.chunk_size = chunk_size
        self.framerate = framerate
        self.model = model
        
        self.conv1V = nn.Conv1d(8576, d, 1, stride=1, bias=False)
        self.conv1A = nn.Conv1d(128, d, 1, stride=1, bias=False)
        
        encoder_layerV = nn.TransformerEncoderLayer(d_model = d, nhead = 8)
        self.encoderV = nn.TransformerEncoder(encoder_layerV, 2)
        encoder_layerA = nn.TransformerEncoderLayer(d_model = d, nhead = 8)
        self.encoderA = nn.TransformerEncoder(encoder_layerA, 1)
        
        self.clasV = nn.Parameter(d)
        self.clasA = nn.Parameter(d)
        
        self.relu = nn.ReLU()

        self.load_weights(weights=weights)

    def load_weights(self, weights=None):
        if(weights is not None):
            print("=> loading checkpoint '{}'".format(weights))
            checkpoint = torch.load(weights)
            self.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(weights, checkpoint['epoch']))
    #def forward(self, inputs):
    def forward(self, inputsV, inputsA):
        
        inputsV = inputsV.permute((0, 2, 1)) #(B x n_features x chunk_size * framerate)
        inputsA = inputsA.permute((0, 2, 1)) #(B x n_features x chunk_size * framerate)
        
        #Reduce dimensionality
        inputsV = self.relu(self.conv1V(inputsV)) #(B x d x chunk_size * framerate)
        inputsA = self.relu(self.conv1A(inputsA)) #(B x d x chunk_size * framerate)
        
        inputsV = inputsV.permute((0, 2, 1)) #(B x chunk_size * framerate x d)
        inputsA = inputsA.permute((0, 2, 1)) #(B x chunk_size * framerate x d)
        
        #Add class token
        inputsV = torch.cat((self.clasV, inputsV), dim=1) #(B x (chunk_size * framerate) + 1 x d)
        inputsA = torch.cat((self.clasA, inputsA), dim=1) #(B x (chunk_size * framerate) + 1 x d)
        
        #Transformer encoders
        inputsV = self.encoderV(inputsV) #(B x (chunk_size * framerate) + 1 x d)
        inputsA = self.encoderA(inputsA) #(B x (chunk_size * framerate) + 1 x d)
        
        #Extract class token
        classV = inputsV[:, 0, :] #(B x 1 x d)
        classA = inputsA[:, 0, :] #(B x 1 x d)
            
        return classV, classA
