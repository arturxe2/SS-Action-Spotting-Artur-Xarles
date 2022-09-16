# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 12:20:25 2022

@author: arturxe
"""

import math
import torch
import torch.nn as nn
import numpy as np
import copy


def mask_tokens(features, mask_token, p_mask = 0.20):
    n_B, n_T, d = features.shape
    ids = []
    for b in range(n_B):
        n_masks = max(1, (np.random.uniform(size = n_T) < p_mask).sum())
        id_masks = np.random.choice(np.arange(n_T), n_masks)
        ids.append(id_masks.tolist())
        for id_mask in id_masks:
            option = np.random.uniform()
            if option < 0.8:
                features[b, id_mask, :] = mask_token
            elif option < 0.9:
                change_token = np.random.choice(np.arange(n_T), 1)
                features[b, id_mask, :] = features[b, change_token, :]
    return features, ids

#Model class
class Model(nn.Module):
    def __init__(self, weights=None, num_classes = 17, d=512, chunk_size=10, framerate=2, p_mask = 0.15, model="SSModel"):
        """
        INPUT: two Tensors of shape (batch_size,chunk_size*framerate,feature_size)
        OUTPUTS: two Tensors of shape (batch_size,d)
        """

        super(Model, self).__init__()
        
        self.num_classes = num_classes
        self.d = d
        self.chunk_size = chunk_size
        self.framerate = framerate
        self.p_mask = p_mask
        self.model = model
        
        #Self-supervised layers / parameters
        self.conv1V = nn.Conv1d(8576, d, 1, stride=1, bias=False)
        self.conv1A = nn.Conv1d(128, d, 1, stride=1, bias=False)
        self.conv1Vmask = nn.Conv1d(8576, d, 1, stride=1, bias=False)
        self.conv1Amask = nn.Conv1d(128, d, 1, stride=1, bias=False)
        
        #Masked tokens
        self.mask_tokenV = nn.Parameter(torch.randn(d))
        self.mask_tokenA = nn.Parameter(torch.randn(d))
        
        encoder_layerV = nn.TransformerEncoderLayer(d_model = d, nhead = 8)
        self.encoderV = nn.TransformerEncoder(encoder_layerV, 2)
        self.encoderVmask = copy.deepcopy(self.encoderV)
        encoder_layerA = nn.TransformerEncoderLayer(d_model = d, nhead = 8)
        self.encoderA = nn.TransformerEncoder(encoder_layerA, 1)
        self.encoderAmask = copy.deepcopy(self.encoderA)
        
        
        
        self.clasV = nn.Parameter(torch.randn(d))
        self.clasA = nn.Parameter(torch.randn(d))
        self.clasVmask = copy.deepcopy(self.clasV)
        self.clasAmask = copy.deepcopy(self.clasA)
        
        #Not gradient in these layers
        self.encoderVmask.requires_grad_(False)
        self.encoderAmask.requires_grad_(False)
        self.clasVmask.requires_grad_(False)
        self.clasAmask.requires_grad_(False)
        
        #Spotting layers
        self.fc = nn.Linear(d, self.num_classes+1)
        encoder_layerM = nn.TransformerEncoderLayer(d_model = d, nhead = 8)
        self.encoderM = nn.TransformerEncoder(encoder_layerM, 2)
        
        self.clasM = nn.Parameter(torch.randn(d))
        
        #General functions
        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()

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
        
        inputsV = inputsV.float() #(B x chunk_size*framerate x n_features)
        inputsA = inputsA.float()
        

        #Masked features
        inputsVmask = torch.clone(inputsV) 
        inputsAmask = torch.clone(inputsA)
        
        inputsVmask, ids_maskV = mask_tokens(inputsVmask, self.mask_tokenV, 0.2)
        inputsAmask, ids_maskA = mask_tokens(inputsAmask, self.mask_tokenA, 0.2)
        
        #Permutation
        inputsV = inputsV.permute((0, 2, 1)) #(B x n_features x chunk_size * framerate)
        inputsA = inputsA.permute((0, 2, 1)) #(B x n_features x chunk_size * framerate)
        inputsVmask = inputsVmask.permute((0, 2, 1)) #(B x n_features x chunk_size * framerate)
        inputsAmask = inputsAmask.permute((0, 2, 1)) #(B x n_features x chunk_size * framerate)
        
        #Reduce dimensionality
        inputsV = self.relu(self.conv1V(inputsV)) #(B x d x chunk_size * framerate)
        inputsA = self.relu(self.conv1A(inputsA)) #(B x d x chunk_size * framerate)
        inputsVmask = self.relu(self.conv1Vmask(inputsVmask)) #(B x d x chunk_size * framerate)
        inputsAmask = self.relu(self.conv1Amask(inputsAmask)) #(B x d x chunk_size * framerate)
        
        #Permutation
        inputsV = inputsV.permute((0, 2, 1)) #(B x chunk_size * framerate x d)
        inputsA = inputsA.permute((0, 2, 1)) #(B x chunk_size * framerate x d)
        inputsVmask = inputsVmask.permute((0, 2, 1)) #(B x chunk_size * framerate x d)
        inputsAmask = inputsAmask.permute((0, 2, 1)) #(B x chunk_size * framerate x d)
        
        #Class token to size [B x 1 x d]
        clasV = torch.unsqueeze(self.clasV.repeat(inputsV.shape[0], 1), dim=1)
        clasA = torch.unsqueeze(self.clasA.repeat(inputsA.shape[0], 1), dim=1)
        clasVmask = torch.unsqueeze(self.clasVmask.repeat(inputsVmask.shape[0], 1), dim=1)
        clasAmask = torch.unsqueeze(self.clasAmask.repeat(inputsAmask.shape[0], 1), dim=1)
        
        #Add class token
        inputsV = torch.cat((clasV, inputsV), dim=1) #(B x (chunk_size * framerate) + 1 x d)
        inputsA = torch.cat((clasA, inputsA), dim=1) #(B x (chunk_size * framerate) + 1 x d)
        inputsVmask = torch.cat((clasVmask, inputsVmask), dim=1)
        inputsAmask = torch.cat((clasAmask, inputsAmask), dim=1)
        
        #Transformer encoders
        inputsV = self.encoderV(inputsV) #(B x (chunk_size * framerate) + 1 x d)
        inputsA = self.encoderA(inputsA) #(B x (chunk_size * framerate) + 1 x d)
        inputsVmask = self.encoderVmask(inputsVmask)
        inputsAmask = self.encoderAmask(inputsAmask)
        
        #Extract class token
        classV = torch.squeeze(inputsV[:, 0, :]) #(B x d)
        classA = torch.squeeze(inputsA[:, 0, :]) #(B x d)
        classVmask = torch.squeeze(inputsVmask[:, 0, :])
        classAmask = torch.squeeze(inputsAmask[:, 0, :])
        
        embeddingsV = inputsV[:, 1:, :] #(B x (chunk_size * framerate) x d)
        embeddingsA = inputsA[:, 1:, :] #(B x (chunk_size * framerate) x d)
        #embeddingsVmask
        
        embeddings = torch.cat((embeddingsV, embeddingsA), dim=1) #(B x 2*(chunk_size * framerate) x d)
        
        #Class token to size [B x 1 x d]
        clasM = torch.unsqueeze(self.clasM.repeat(embeddings.shape[0], 1), dim=1) 
        
        embeddings = torch.cat((clasM, embeddings), dim=1) #(B x 1 + 2*(chunk_size * framerate) x d)
        
        embeddings = self.encoderM(embeddings) #(B x 1 + 2*(chunk_size * framerate) x d)
        
        classM = torch.squeeze(embeddings[:, 0, :]) #(B x d)
        
        outputs = self.sigm(self.fc(classM))
        #logits = torch.mm(classV, torch.transpose(classA, 0, 1)) * torch.exp(self.temperature)
            
        return classV, classA, outputs
