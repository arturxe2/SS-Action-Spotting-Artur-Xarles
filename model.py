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
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights, swin_t, Swin_T_Weights
import torchvision.transforms as T
from vit_pytorch.mobile_vit import MobileViT

'''
def mask_tokens(features, mask_token, p_mask = 0.20, consecutive_tokens = False, n_consecutive = 5):
    n_B, n_T, d = features.shape
    
    R = torch.rand([n_B, n_T])
    random_token = features[torch.randint(0, n_B, (1,)), torch.randint(0, n_T, (1,)), :]
    
      
    M1 = R < (p_mask * 0.8)
    M2 = (R >= (p_mask * 0.8)) & (R < (p_mask * 0.9))
    M3 = (R >= (p_mask * 0.9)) & (R < p_mask)
    
    features[M1] = mask_token
    features[M2] = random_token
    
    M = (M1 | M2) | M3
    
    return features, M
'''

class mask_tokens(nn.Module):
    
    def __init__(self, mask_token, p_mask = 0.20, consecutive_tokens = False, n_consecutive = 5):
        super().__init__()
        self.mask_token = mask_token
        self.p_mask = p_mask
        self.consecutive_tokens = consecutive_tokens
        self.n_consecutive = n_consecutive
        
        if self.consecutive_tokens:
            self.maxpool = torch.nn.MaxPool1d(n_consecutive, stride=1, padding=n_consecutive//2)
            #Aprox
            self.aux_p_mask = self.p_mask / self.n_consecutive
            
    def forward(self, features: torch.Tensor):
        n_B, n_T, d = features.shape
        
        R = torch.rand([n_B, n_T])
        random_token = features[torch.randint(0, n_B, (1,)), torch.randint(0, n_T, (1,)), :]
        
        if self.consecutive_tokens:
            R = self.maxpool(R)
            M1 = R >= 1 - self.aux_p_mask * 0.8
            M2 = (R < (1 - self.aux_p_mask * 0.8)) & (R >= 1 - self.aux_p_mask * 0.9)
            M3 = (R < 1 - self.aux_p_mask * 0.9) & (R >= 1 - self.aux_p_mask)
        
        else:  
            M1 = R < (self.p_mask * 0.8)
            M2 = (R >= (self.p_mask * 0.8)) & (R < (self.p_mask * 0.9))
            M3 = (R >= (self.p_mask * 0.9)) & (R < self.p_mask)
        
        features[M1] = self.mask_token
        features[M2] = random_token
        
        M = (M1 | M2) | M3
        
        return features, M
    
class mask_frames(nn.Module):
    
    def __init__(self, mask_token, p_mask = 0.20, n_consecutive = [5, 51, 51], n_generations = 2000):
        super().__init__()
        self.mask_token = mask_token
        self.p_mask = p_mask
        self.npixels = np.prod(n_consecutive)
        self.n_consecutive = [n//2 for n in n_consecutive]
        self.n_generations = n_generations

        
            
    def forward(self, frames: torch.Tensor):
        n_B, n_T, H, W, C = frames.shape
        tnpixels = n_B * n_T * H * W
        
        R1 = torch.zeros([n_B, n_T, H, W], dtype=torch.bool)
        R2 = torch.zeros([n_B, n_T, H, W], dtype=torch.bool)
        R3 = torch.zeros([n_B, n_T, H, W], dtype=torch.bool)
        
        random_token = frames[torch.randint(0, n_B, (1,)), torch.randint(0, n_T, (1,)), torch.randint(0, H, (1,)), torch.randint(0, W, (1,)), :]
        
        b = torch.randint(0, n_B, (self.n_generations, ))
        t = torch.randint(0, n_T, (self.n_generations, ))
        h = torch.randint(0, H, (self.n_generations, ))
        w = torch.randint(0, W, (self.n_generations, ))
        
        n1 = int((tnpixels * self.p_mask * 0.8) // self.npixels)
        n2 = int((tnpixels * self.p_mask * 0.1) // self.npixels)
        
        for i in range(n1):
            R1[b[i], max(0, t[i]-self.n_consecutive[0]):min(t[i]+self.n_consecutive[0], n_T-1), 
              max(0, h[i]-self.n_consecutive[1]):min(h[i]+self.n_consecutive[1], H-1), 
              max(0, w[i]-self.n_consecutive[2]):min(w[i]+self.n_consecutive[2], W-1)] = True
            
        for i in range(n1, n1+n2):
            R2[b[i], max(0, t[i]-self.n_consecutive[0]):min(t[i]+self.n_consecutive[0], n_T-1), 
              max(0, h[i]-self.n_consecutive[1]):min(h[i]+self.n_consecutive[1], H-1), 
              max(0, w[i]-self.n_consecutive[2]):min(w[i]+self.n_consecutive[2], W-1)] = True

        for i in range(n1+n2, n1+n2+n2):
            R3[b[i], max(0, t[i]-self.n_consecutive[0]):min(t[i]+self.n_consecutive[0], n_T-1), 
              max(0, h[i]-self.n_consecutive[1]):min(h[i]+self.n_consecutive[1], H-1), 
              max(0, w[i]-self.n_consecutive[2]):min(w[i]+self.n_consecutive[2], W-1)] = True
            
        frames[R1] = self.mask_token
        frames[R2] = random_token
        
        M = (R1 | R2) | R3
        M = torch.max(M.view(n_B, n_T, -1), dim=2).values
        
        return frames, M
    
    
#Positional Encoding class
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 6000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor, add: float = 0.) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)] + add
        return self.dropout(x)

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
        
        #Positional encodding layer
        self.positional = PositionalEncoding(d)
        
        #Self-supervised layers / parameters
        self.conv1V = nn.Conv1d(8576, d, 1, stride=1, bias=False)
        self.conv1A = nn.Conv1d(128, d, 1, stride=1, bias=False)
        self.conv1Vmask = copy.deepcopy(self.conv1V)
        self.conv1Amask = copy.deepcopy(self.conv1A)
        
        
        #Masked tokens
        self.mask_tokenV = nn.Parameter(torch.randn(8576))
        self.mask_tokenA = nn.Parameter(torch.randn(128))
        
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
        
        #Mask predictors
        self.convMV = nn.Conv1d(d, d, 1, stride=1, bias=False)
        self.convMA = nn.Conv1d(d, d, 1, stride=1, bias=False)

        
        #Not gradient in these layers
        self.conv1V.requires_grad_(False)
        self.conv1A.requires_grad_(False)
        self.encoderV.requires_grad_(False)
        self.encoderA.requires_grad_(False)
        self.clasV.requires_grad_(False)
        self.clasA.requires_grad_(False)
        
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
        
        aux_inputsV = torch.clone(inputsV)
        aux_inputsA = torch.clone(inputsA)
        
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
        inputsV = inputsV.permute((1, 0, 2))
        inputsA = inputsA.permute((1, 0, 2))
        inputsVmask = inputsVmask.permute((1, 0, 2))
        inputsAmask = inputsAmask.permute((1, 0, 2))
        
        #Positional encoding
        inputsV = self.positional(inputsV)
        inputsA = self.positional(inputsA)
        inputsVmask = self.positional(inputsVmask)
        inputsAmask = self.positional(inputsAmask)
        
        inputsV = inputsV.permute((1, 0, 2))
        inputsA = inputsA.permute((1, 0, 2))
        inputsVmask = inputsVmask.permute((1, 0, 2))
        inputsAmask = inputsAmask.permute((1, 0, 2))
        
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
        embeddingsVmask = inputsVmask[:, 1:, :]
        embeddingsAmask = inputsAmask[:, 1:, :]
        
        #Mask predictions
        Vpreds = (self.convMV(embeddingsVmask.permute((0, 2, 1)))) #(B x d x (chunk_size * framerate))
        Apreds = (self.convMA(embeddingsAmask.permute((0, 2, 1)))) #(B x d x (chunk_size * framerate))
        
        Vreal = embeddingsV.permute((0, 2, 1))[:, :, ids_maskV]
        Vpreds = Vpreds[:, :, ids_maskV]
        Areal = embeddingsA.permute((0, 2, 1))[:, :, ids_maskA]
        Apreds = Apreds[:, :, ids_maskA]
        
        
        embeddings = torch.cat((aux_inputsV, aux_inputsA), dim=1) #(B x 2*(chunk_size * framerate) x d)
        
        #Class token to size [B x 1 x d]
        clasM = torch.unsqueeze(self.clasM.repeat(embeddings.shape[0], 1), dim=1) 
        
        embeddings = torch.cat((clasM, embeddings), dim=1) #(B x 1 + 2*(chunk_size * framerate) x d)
        
        embeddings = self.encoderM(embeddings) #(B x 1 + 2*(chunk_size * framerate) x d)
        
        classM = torch.squeeze(embeddings[:, 0, :]) #(B x d)
        
        outputs = self.sigm(self.fc(classM))
        #logits = torch.mm(classV, torch.transpose(classA, 0, 1)) * torch.exp(self.temperature)
            
        return classVmask, classAmask, Vreal, Vpreds, Areal, Apreds, outputs



#Model class
class Model2(nn.Module):
    def __init__(self, weights=None, num_classes = 17, d=512, chunk_size=10, framerate=2, p_mask = 0.15, model="SSModel"):
        """
        INPUT: two Tensors of shape (batch_size,chunk_size*framerate,feature_size)
        OUTPUTS: two Tensors of shape (batch_size,d)
        """

        super(Model2, self).__init__()
        
        #MODEL PARAMETERS
        
        self.num_classes = num_classes
        self.d = d
        self.chunk_size = chunk_size
        self.framerate = framerate
        self.p_mask = p_mask
        self.model = model
        
        
        #SS MODEL LAYERS
        
        #Convolutions (reduce dimensionality)
        self.conv1V = nn.Conv1d(8576, d, 1, stride=1, bias=False)
        self.conv1A = nn.Conv1d(128, d, 1, stride=1, bias=False)
        self.norm1 = nn.LayerNorm([self.chunk_size * self.framerate, d])
        self.norm2 = nn.LayerNorm([self.chunk_size * self.framerate + 1, d])
        self.norm3 = nn.LayerNorm([2 * self.chunk_size * self.framerate, d])
        
        #Masked tokens
        self.mask_tokenV = nn.Parameter(torch.randn(d))
        self.mask_tokenA = nn.Parameter(torch.randn(d))
        
        #Transformer Encoders
        encoder_layerV = nn.TransformerEncoderLayer(d_model = d, nhead = 8, batch_first=True)
        self.encoderVmask = nn.TransformerEncoder(encoder_layerV, 2)
        self.encoderV = copy.deepcopy(self.encoderVmask)
        self.clasVmask = nn.Parameter(torch.randn(d))
        self.clasV = copy.deepcopy(self.clasVmask)
        
        
        encoder_layerA = nn.TransformerEncoderLayer(d_model = d, nhead = 8, batch_first=True)
        self.encoderAmask = nn.TransformerEncoder(encoder_layerA, 1)
        self.encoderA = copy.deepcopy(self.encoderAmask)
        self.clasAmask = nn.Parameter(torch.randn(d))
        self.clasA = copy.deepcopy(self.clasVmask)
        
        #Positional embeddings
        self.posVmask = nn.Parameter(torch.randn([self.chunk_size * self.framerate, d]))
        self.posAmask = nn.Parameter(torch.randn([self.chunk_size * self.framerate, d]))
        self.posV = copy.deepcopy(self.posVmask)
        self.posA = copy.deepcopy(self.posAmask)
        
        #Mask predictors
        self.convMV1 = nn.Conv1d(d, 2*d, 1, stride=1, bias=True)
        self.convMV2 = nn.Conv1d(2*d, d, 1, stride=1, bias=True)
        self.convMA1 = nn.Conv1d(d, 2*d, 1, stride=1, bias=True)
        self.convMA2 = nn.Conv1d(2*d, d, 1, stride=1, bias=True)
        
        #Avoid gradient on momentum layers
        self.encoderV.requires_grad_(False)
        self.encoderA.requires_grad_(False)
        self.posV.requires_grad_(False)
        self.posA.requires_grad_(False)
        self.clasV.requires_grad_(False)
        self.clasA.requires_grad_(False)
        
        #Pooling layer
        self.pool_layerSS = nn.MaxPool1d(chunk_size * framerate, stride = 1)
        #self.pool_layerSS = nn.AvgPool1d(chunk_size * framerate, stride = 1)
        
        
        #AS MODEL LAYERS
        
        #Transformer Encoders
        encoder_layerM = nn.TransformerEncoderLayer(d_model = d, nhead = 8, batch_first=True)
        self.encoderM = nn.TransformerEncoder(encoder_layerM, 2)
        
        #Pooling layer
        self.pool_layerAS = nn.MaxPool1d(chunk_size * framerate * 2, stride = 1)
        
        #Linear layer
        self.fc1 = nn.Linear(d, 2*d)
        self.fc2 = nn.Linear(2*d, self.num_classes+1)
        
        
        #GENERAL LAYERS
        
        #Positional encodding layer (not used for the moment)
        self.positional = PositionalEncoding(d)
        
        #General functions
        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()
        
        #Load weights parameter
        self.load_weights(weights=weights)
        
        
        
    def load_weights(self, weights=None):
        if(weights is not None):
            print("=> loading checkpoint '{}'".format(weights))
            checkpoint = torch.load(weights)
            self.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(weights, checkpoint['epoch']))

    def forward(self, inputsV, inputsA, inference = False):
        
        
        #INPUTS TO FLOAT
        inputsV = inputsV.float() #(B x chunk_size*framerate x n_features)
        inputsA = inputsA.float() #(B x chunk_size*framerate x n_features)
        
        #PERMUTATION
        inputsV = inputsV.permute((0, 2, 1)) #(B x n_features x chunk_size*framerate)
        inputsA = inputsA.permute((0, 2, 1)) #(B x n_features x chunk_size*framerate)
        
        #REDUCE DIMENSIONALITY
        inputsV = self.relu(self.conv1V(inputsV)) #(B x d x chunk_size*framerate)
        inputsA = self.relu(self.conv1A(inputsA)) #(B x d x chunk_size*framerate)
        
        #PERMUTATION
        inputsV = inputsV.permute((0, 2, 1)) #(B x chunk_size*framerate x d)
        inputsA = inputsA.permute((0, 2, 1)) #(B x chunk_size*framerate x d)
        
        #COPY OF FEATURES (FOR MASKED ONES)
        inputsVmask = torch.clone(inputsV) #(B x chunk_size*framerate x d)
        inputsAmask = torch.clone(inputsA) #(B x chunk_size*framerate x d)
        
        #GET MASKING OF FEATURES
        if not inference:
            inputsVmask, MV = mask_tokens(inputsVmask, self.mask_tokenV, self.p_mask) #(B x chunk_size*framerate x d)
            inputsAmask, MA = mask_tokens(inputsAmask, self.mask_tokenA, self.p_mask) #(B x chunk_size*framerate x d)
            
        else:
            MV = torch.rand([inputsVmask.shape[0], inputsVmask.shape[1]]) < 0.05
            MA = torch.rand([inputsAmask.shape[0], inputsAmask.shape[1]]) < 0.05
        
        #LAYER NORMALIZATION
        inputsV = self.norm1(inputsV) #(B x chunk_size*framerate x d)
        inputsA = self.norm1(inputsA) #(B x chunk_size*framerate x d)
        inputsVmask = self.norm1(inputsVmask) #(B x chunk_size*framerate x d)
        inputsAmask = self.norm1(inputsAmask) #(B x chunk_size*framerate x d)
        
        #POSITIONAL ENCODING
        inputsV = inputsV + self.posV #(B x chunk_size*framerate x d)
        inputsA = inputsA + self.posA #(B x chunk_size*framerate x d)
        inputsVmask = inputsVmask + self.posVmask #(B x chunk_size*framerate x d)
        inputsAmask = inputsAmask + self.posAmask #(B x chunk_size*framerate x d)
        
        #ADDING CLASS TOKEN
        clasV = torch.unsqueeze(self.clasV.repeat(inputsV.shape[0], 1), dim=1) #(B x 1 x d)
        inputsV = torch.cat((clasV, inputsV), dim=1) #(B x chunk_size*framerate + 1 x d)
        clasVmask = torch.unsqueeze(self.clasVmask.repeat(inputsVmask.shape[0], 1), dim=1) #(B x 1 x d)
        inputsVmask = torch.cat((clasVmask, inputsVmask), dim=1) #(B x chunk_size*framerate + 1 x d)
        clasA = torch.unsqueeze(self.clasA.repeat(inputsA.shape[0], 1), dim=1) #(B x 1 x d)
        inputsA = torch.cat((clasA, inputsA), dim=1) #(B x chunk_size*framerate + 1 x d)
        clasAmask = torch.unsqueeze(self.clasAmask.repeat(inputsAmask.shape[0], 1), dim=1) #(B x 1 x d)
        inputsAmask = torch.cat((clasAmask, inputsAmask), dim=1) #(B x chunk_size*framerate + 1 x d)
        
        #TRANSFORMER ENCODER
        inputsV = self.encoderV(inputsV) #(B x chunk_size * framerate +1 x d)
        inputsA = self.encoderA(inputsA) #(B x chunk_size * framerate +1 x d)
        inputsVmask = self.encoderVmask(inputsVmask) #(B x chunk_size * framerate +1 x d)
        inputsAmask = self.encoderAmask(inputsAmask) #(B x chunk_size * framerate +1 x d)
        
        #LAYER NORMALIZATION
        inputsV = self.norm2(inputsV) #(B x chunk_size * framerate +1 x d)
        inputsA = self.norm2(inputsA) #(B x chunk_size * framerate +1 x d)
        inputsVmask = self.norm2(inputsVmask) #(B x chunk_size * framerate +1 x d)
        inputsAmask = self.norm2(inputsAmask) #(B x chunk_size * framerate +1 x d)
        
        #POOLING TO GET EMBEDDING FOR VISUAL AND AUDIO REPRESENTATIONS (INSTEAD OF CLASS TOKEN)
        #embeddingV = self.pool_layerSS(aux_inputsVmask).squeeze(-1) #(B x d)
        #embeddingA = self.pool_layerSS(aux_inputsAmask).squeeze(-1) #(B x d)
        embeddingV = inputsVmask[:, 0, :]
        embeddingA = inputsAmask[:, 0, :]
        
        #NOT CLASS TOKENS
        inputsV = inputsV[:, 1:, :]
        inputsA = inputsA[:, 1:, :]
        inputsVmask = inputsVmask[:, 1:, :]
        inputsAmask = inputsAmask[:, 1:, :]
        
        #PERMUTATION
        aux_inputsVmask = inputsVmask.permute((0, 2, 1)) #(B x d x chunk_size*framerate +1)
        aux_inputsAmask = inputsAmask.permute((0, 2, 1)) #(B x d x chunk_size*framerate +1)
        
        
        #PREDICTION OF MASK TOKENS
        Vpreds = self.convMV2(self.relu(self.convMV1(aux_inputsVmask))).permute((0, 2, 1)) #(B x chunk_size * framerate x d)
        Apreds = self.convMA2(self.relu(self.convMA1(aux_inputsAmask))).permute((0, 2, 1)) #(B x chunk_size * framerate x d)
        
        #GET MASKED IDS
        realV = inputsV[MV] #(n_maskV x d)
        realA = inputsA[MA] #(n_maskA x d)
        predsV = Vpreds[MV] #(n_maskV x d)
        predsA = Apreds[MA] #(n_maskA x d)
        
        #CONCATENATION OF VISUAL AND AUDIO EVOLVED FEATURES (MASK PART)
        embeddings = torch.cat((inputsVmask, inputsAmask), dim=1) #(B x 2*(chunk_size * framerate) x d)
        
        #MULTIMODAL TRANSFORMER ENCODER
        embeddings = self.encoderM(embeddings) #(B x 2*(chunk_size * framerate) x d)

        
        #LAYER NORMALIZATION
        embeddings = self.norm3(embeddings) #(B x 2*(chunk_size * framerate) x d)
        
        #PERMUTATION
        embeddings = embeddings.permute((0, 2, 1)) #(B x d x 2*(chunk_size*framerate))
        
        #POOLING (INSTEAD OF CLASS TOKEN)
        embeddings = self.pool_layerAS(embeddings).squeeze(-1) #(B x d)
        
        #FC AND SIGMOID TO MAKE PREDICTIONS        
        outputs = self.sigm(self.fc2(self.relu(self.fc1(embeddings))))
        
        if realV.shape[0] == 0:
            realV = inputsV[1:2, 1:2, :]
            predsV = Vpreds[1:2, 1:2, :]

            
        return embeddingV, embeddingA, realV, predsV, realA, predsA, outputs


#Model class
class ModelAS(nn.Module):
    def __init__(self, weights=None, num_classes = 17, d=512, chunk_size=10, framerate=2, p_mask = 0.15, model="SSModel"):
        """
        INPUT: two Tensors of shape (batch_size,chunk_size*framerate,feature_size)
        OUTPUTS: two Tensors of shape (batch_size,d)
        """

        super(ModelAS, self).__init__()
        
        #MODEL PARAMETERS
        
        self.num_classes = num_classes
        self.d = d
        self.chunk_size = chunk_size
        self.framerate = framerate
        self.p_mask = p_mask
        self.model = model
        
        
        #SS MODEL LAYERS
        
        #Convolutions (reduce dimensionality)
        self.conv1V = nn.Conv1d(8576, d, 1, stride=1, bias=False)
        self.conv1A = nn.Conv1d(128, d, 1, stride=1, bias=False)
        self.norm1 = nn.LayerNorm(d)
        
        #Positional embeddings
        self.pos = nn.Parameter(torch.randn([self.chunk_size * self.framerate, d]))
        
        #AS MODEL LAYERS
        
        #Unimodal Transformer Encoder
        encoder_layerV = nn.TransformerEncoderLayer(d_model = d, nhead = 8, dim_feedforward = 2048, batch_first=True)
        self.encoderV = nn.TransformerEncoder(encoder_layerV, 2)
        encoder_layerA = nn.TransformerEncoderLayer(d_model = d, nhead = 8, dim_feedforward = 2048, batch_first=True)
        self.encoderA = nn.TransformerEncoder(encoder_layerA, 1)
        
        
        #Multimodal Transformer Encoders
        encoder_layerM = nn.TransformerEncoderLayer(d_model = d, nhead = 8, dim_feedforward = 2048, batch_first=True)
        self.encoderM = nn.TransformerEncoder(encoder_layerM, 2)

        
        #Pooling layer
        self.pool_layerAS = nn.MaxPool1d(chunk_size * framerate * 2, stride = 1)
        
        #Linear layer
        self.fc1 = nn.Linear(d, 2*d)
        self.fc2 = nn.Linear(2*d, self.num_classes+1)
        
        
        #GENERAL LAYERS
        
        #Positional encodding layer (not used for the moment)
        self.positional = PositionalEncoding(d)
        
        #General functions
        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()
        
        #Load weights parameter
        self.load_weights(weights=weights)
        
        
        
    def load_weights(self, weights=None):
        if(weights is not None):
            print("=> loading checkpoint '{}'".format(weights))
            checkpoint = torch.load(weights)
            self.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(weights, checkpoint['epoch']))

    def forward(self, inputsV, inputsA, inference = False):
        
        
        #INPUTS TO FLOAT
        inputsV = inputsV.float() #(B x chunk_size*framerate x n_features)
        inputsA = inputsA.float() #(B x chunk_size*framerate x n_features)
        
        
        #PERMUTATION
        inputsV = inputsV.permute((0, 2, 1)) #(B x n_features x chunk_size * framerate)
        inputsA = inputsA.permute((0, 2, 1)) #(B x n_features x chunk_size * framerate)

        
        #REDUCE DIMENSIONALITY
        inputsV = self.relu(self.conv1V(inputsV)) #(B x d x chunk_size * framerate)
        inputsA = self.relu(self.conv1A(inputsA)) #(B x d x chunk_size * framerate)
        
        #PERMUTATION
        inputsV = inputsV.permute((0, 2, 1)) #(B x chunk_size * framerate x d)
        inputsA = inputsA.permute((0, 2, 1)) #(B x chunk_size * framerate x d)

        
        #LAYER NORMALIZATION
        inputsV = self.norm1(inputsV) #(B x chunk_size * framerate x d)
        inputsA = self.norm1(inputsA) #(B x chunk_size * framerate x d)
        
        #UNIMODAL TRANSFORMER ENCODERS
        inputsV = self.encoderV(inputsV) #(B x chunk_size * framerate x d)
        inputsA = self.encoderA(inputsA) #(B x chunk_size * framerate x d)
        
        #CONCATENATION OF VISUAL AND AUDIO EVOLVED FEATURES (MASK PART)
        embeddings = torch.cat((inputsV, inputsA), dim=1) #(B x 2*(chunk_size * framerate) x d)
        
        #MULTIMODAL TRANSFORMER ENCODER
        embeddings = self.encoderM(embeddings) #(B x 2*(chunk_size * framerate) x d)
        #embeddings = self.encoderM2(embeddings) #(B x 2*(chunk_size * framerate) x d)
        
        #LAYER NORMALIZATION
        embeddings = self.norm1(embeddings)
        
        #PERMUTATION
        embeddings = embeddings.permute((0, 2, 1)) #(B x d x 2*(chunk_size*framerate))
        
        #POOLING (INSTEAD OF CLASS TOKEN)
        embeddings = self.pool_layerAS(embeddings).squeeze(-1) #(B x d)
        
        #FC AND SIGMOID TO MAKE PREDICTIONS        
        outputs = self.sigm(self.fc2(self.relu(self.fc1(embeddings))))

            
        return outputs, outputs, outputs, outputs, outputs, outputs, outputs
    
    
#Model class
class ModelFrames(nn.Module):
    def __init__(self, weights=None, num_classes = 17, d=512, chunk_size=5, framerate=2, p_mask = 0.15, 
                 framestride = 4, model="SSModel", backbone = 'mobilenet', masking = 'token'):
        """
        INPUT: two Tensors of shape (batch_size,chunk_size*framerate,feature_size)
        OUTPUTS: two Tensors of shape (batch_size,d)
        """

        super(ModelFrames, self).__init__()
        
        #MODEL PARAMETERS
        
        self.num_classes = num_classes
        self.d = d
        self.chunk_size = chunk_size
        self.framerate = framerate
        self.p_mask = p_mask
        self.model = model
        self.framestride = framestride
        self.backbone = backbone
        self.masking = masking
        
        #MODEL BACKBONES (VISUAL ONES)
        if backbone == 'mobilenet':
            self.mobilenet = mobilenet_v3_small(MobileNet_V3_Small_Weights)
            self.mobilenet.classifier = torch.nn.Identity()
            self.conv1V = nn.Conv1d(576, d, 1, stride=1, bias=False)
        elif backbone == 'vit':
            self.vit = MobileViT(
                image_size = (256, 512),
                dims = [96, 120, 144],
                channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384],
                num_classes = 512
            )
            #self.vit.head = torch.nn.Identity()
            self.conv1V = nn.Conv1d(512, d, 1, stride=1, bias=False)
            self.transform = T.Resize((256,256))
        
        
        #SS MODEL LAYERS
        
        #Convolutions (reduce dimensionality)

        self.conv1A = nn.Conv1d(128, d, 1, stride=1, bias=False)
        self.norm1V = nn.LayerNorm([chunk_size * 25 // self.framestride, d])
        self.norm1A = nn.LayerNorm([self.chunk_size * self.framerate, d])
        self.norm2V = nn.LayerNorm([chunk_size * 25 // self.framestride + 1, d])
        self.norm2A = nn.LayerNorm([self.chunk_size * self.framerate + 1, d])
        self.norm3 = nn.LayerNorm([chunk_size * 25 // self.framestride + (self.chunk_size * self.framerate), d])
        
        #Masked tokens
        self.mask_tokenA = nn.Parameter(torch.randn(d))
        self.maskingA = mask_tokens(self.mask_tokenA, p_mask=0.2, consecutive_tokens=True)
        if masking == 'token':
            self.mask_tokenV = nn.Parameter(torch.randn(d))
            self.maskingV = mask_tokens(self.mask_tokenV, p_mask=0.2, consecutive_tokens=True)
            
        elif masking == 'frame':
            self.mask_tokenV = nn.Parameter(torch.randn(1))
            self.maskingV = mask_frames(self.mask_tokenV, p_mask=0.2)
        
        #Transformer Encoders
        encoder_layerV = nn.TransformerEncoderLayer(d_model = d, nhead = 8, batch_first=True)
        self.encoderVmask = nn.TransformerEncoder(encoder_layerV, 2)
        self.encoderV = copy.deepcopy(self.encoderVmask)
        self.clasVmask = nn.Parameter(torch.randn(d))
        self.clasV = copy.deepcopy(self.clasVmask)
        
        
        encoder_layerA = nn.TransformerEncoderLayer(d_model = d, nhead = 8, batch_first=True)
        self.encoderAmask = nn.TransformerEncoder(encoder_layerA, 1)
        self.encoderA = copy.deepcopy(self.encoderAmask)
        self.clasAmask = nn.Parameter(torch.randn(d))
        self.clasA = copy.deepcopy(self.clasVmask)
        
        #Positional embeddings
        self.posVmask = nn.Parameter(torch.randn([self.chunk_size * 25 // self.framestride, d]))
        self.posAmask = nn.Parameter(torch.randn([self.chunk_size * self.framerate, d]))
        self.posV = copy.deepcopy(self.posVmask)
        self.posA = copy.deepcopy(self.posAmask)
        
        #Mask predictors
        self.convMV1 = nn.Conv1d(d, 2*d, 1, stride=1, bias=True)
        self.convMV2 = nn.Conv1d(2*d, d, 1, stride=1, bias=True)
        self.convMA1 = nn.Conv1d(d, 2*d, 1, stride=1, bias=True)
        self.convMA2 = nn.Conv1d(2*d, d, 1, stride=1, bias=True)
        
        #Avoid gradient on momentum layers
        self.encoderV.requires_grad_(False)
        self.encoderA.requires_grad_(False)
        self.posV.requires_grad_(False)
        self.posA.requires_grad_(False)
        self.clasV.requires_grad_(False)
        self.clasA.requires_grad_(False)
        
        #Pooling layer
        self.pool_layerSS = nn.MaxPool1d(chunk_size * framerate, stride = 1)
        #self.pool_layerSS = nn.AvgPool1d(chunk_size * framerate, stride = 1)
        
        
        #AS MODEL LAYERS
        
        #Transformer Encoders
        encoder_layerM = nn.TransformerEncoderLayer(d_model = d, nhead = 8, batch_first=True)
        self.encoderM = nn.TransformerEncoder(encoder_layerM, 2)
        
        #Pooling layer
        self.pool_layerAS = nn.MaxPool1d(chunk_size * 25 // self.framestride + (self.chunk_size * self.framerate), stride = 1)
        
        #Linear layer
        self.fc1 = nn.Linear(d, 2*d)
        self.fc2 = nn.Linear(2*d, self.num_classes+1)
        
        
        #GENERAL LAYERS
        
        #Positional encodding layer (not used for the moment)
        self.positional = PositionalEncoding(d)
        
        #General functions
        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()
        
        #Load weights parameter
        self.load_weights(weights=weights)
        
        
        
    def load_weights(self, weights=None):
        if(weights is not None):
            print("=> loading checkpoint '{}'".format(weights))
            checkpoint = torch.load(weights)
            self.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(weights, checkpoint['epoch']))

    def forward(self, inputsV, inputsA, inference = False):
        
        
        #INPUTS TO FLOAT
        inputsV = inputsV.float() #(B x n_frames x H x W x C)
        inputsA = inputsA.float() #(B x chunk_size*framerate x n_features)
        images_shape = inputsV.shape
        
        #DIFFERENT MASKING STRATEGIES (VIDEO)
        
        #FRAME MASKING STRATEGY
        if self.masking == 'frame':
        
        #MASKING OF FRAMES
            if not inference:
                inputsVmask, MV = self.maskingV(inputsV) #(B x n_frames x H x W x C), (B x n_frames)
                inputsVmask = inputsVmask.permute((0, 1, 4, 2, 3)) #(B x n_frames x C x H x W)
                inputsVmask = inputsVmask.view(-1, images_shape[4], images_shape[2], images_shape[3]) #(B*n_frames x C x H x W)
                
            inputsV = inputsV.view(-1, images_shape[4], images_shape[2], images_shape[3]) #(B*n_frames x C x H x W)
            
            #BACKBONE FRAME FEATURE EXTRACTION
            if self.backbone == 'mobilenet':
                inputsV = self.mobilenet(inputsV) #(B*n_frames x n_features)
                if not inference:
                    inputsVmask = self.mobilenet(inputsVmask) #(B*n_frames x n_features)
            elif self.backbone == 'vit':
                inputsV = self.transform(inputsV) #(B*n_frames x C x H2 x W2)
                inputsV = self.vit(inputsV) #(B*n_frames x n_features)
                if not inference:
                    inputsVmask = self.transform(inputsVmask) #(B*n_frames x C x H2 x W2)
                    inputsVmask = self.vit(inputsVmask) #(B*n_frames x n_features)        
                    
            #PFFN TO REDUCE DIMENSIONALITY
            inputsV = inputsV.view(images_shape[0], images_shape[1], -1) #(B x n_frames x n_features)
            inputsV = inputsV.permute((0, 2, 1)) #(B x n_features x n_frames)
            inputsV = self.relu(self.conv1V(inputsV)) #(B x d x n_frames)
            inputsV = inputsV.permute((0, 2, 1)) #(B x n_frames x d)
            inputsV = self.norm1V(inputsV) #(B x n_frames x d)
            
            #PFFN TO REDUCE DIMENSIONALITY (MASKED PART)
            if not inference:
                inputsVmask = inputsVmask.view(images_shape[0], images_shape[1], -1) #(B x n_frames x n_features)
                inputsVmask = inputsVmask.permute((0, 2, 1)) #(B x n_features x n_frames)
                inputsVmask = self.relu(self.conv1V(inputsVmask)) #(B x d x n_frames)
                inputsVmask = inputsVmask.permute((0, 2, 1)) #(B x n_frames x d)
                inputsVmask = self.norm1V(inputsVmask) #(B x n_frames x d)
                
            #IF NOT TRAINING
            if inference:
                inputsVmask = torch.clone(inputsV) #(B x n_frames x d)
            
        #TOKEN MASKING STRATEGY
        elif self.masking == 'token':
            
            #RESHAPE
            inputsV = inputsV.permute((0, 1, 4, 2, 3)) #(B x n_frames x C x H x W)
            inputsV = inputsV.view(-1, images_shape[4], images_shape[2], images_shape[3]) #(n x H x W x C)
            
            #BACKBONE FRAME FEATURE EXTRACTION
            if self.backbone == 'mobilenet':
                inputsV = self.mobilenet(inputsV) #(n x 576)
            elif self.backbone == 'vit':
                inputsV = self.transform(inputsV)
                inputsV = self.vit(inputsV)
            
            #PFFN TO REDUCE DIMENSIONALITY 
            inputsV = inputsV.view(images_shape[0], images_shape[1], -1) #(B x n_frames x n_features(576))
            inputsV = inputsV.permute((0, 2, 1)) #(B x n_features x chunk_size*framerate)
            inputsV = self.relu(self.conv1V(inputsV)) #(B x d x chunk_size*framerate)
            inputsV = inputsV.permute((0, 2, 1)) #(B x chunk_size*framerate x d)

            #CREATE MASKING COPY
            inputsVmask = torch.clone(inputsV) #(B x chunk_size*framerate x d)
            
            #MASKING OF TOKENS
            if (not inference):
                inputsVmask, MV = self.maskingV(inputsVmask) #(B x chunk_size*framerate x d)
                
                
            #LAYER NORMALIZATION
            inputsV = self.norm1V(inputsV) #(B x chunk_size*framerate x d)
            inputsVmask = self.norm1V(inputsVmask) #(B x chunk_size*framerate x d)
            
            
        #AUDIO FEATURES
        
        #PFFN TO REDUCE DIMENSIONALITY
        inputsA = inputsA.permute((0, 2, 1)) #(B x n_features x chunk_size*framerate)
        inputsA = self.relu(self.conv1A(inputsA)) #(B x d x chunk_size*framerate)
        inputsA = inputsA.permute((0, 2, 1)) #(B x chunk_size*framerate x d)
        
        #CREATE MASKING COPY
        inputsAmask = torch.clone(inputsA) #(B x chunk_size*framerate x d)
        
        #MASKING OF TOKENS
        if not inference:
            inputsAmask, MA = self.maskingA(inputsAmask) #(B x chunk_size*framerate x d)
        
        #LAYER NORMALIZATION
        inputsA = self.norm1A(inputsA) #(B x chunk_size*framerate x d)
        inputsAmask = self.norm1A(inputsAmask) #(B x chunk_size*framerate x d)
            
        
        #NOT TRAINING PROCEDURE 
        if inference:
            MV = torch.rand([inputsVmask.shape[0], inputsVmask.shape[1]]) < 0.05
            MA = torch.rand([inputsAmask.shape[0], inputsAmask.shape[1]]) < 0.05
        
        
        
        #POSITIONAL ENCODING
        inputsV = inputsV + self.posV #(B x chunk_size*framerate x d)
        inputsA = inputsA + self.posA #(B x chunk_size*framerate x d)
        inputsVmask = inputsVmask + self.posVmask #(B x chunk_size*framerate x d)
        inputsAmask = inputsAmask + self.posAmask #(B x chunk_size*framerate x d)
        
        #ADDING CLASS TOKEN
        clasV = torch.unsqueeze(self.clasV.repeat(inputsV.shape[0], 1), dim=1) #(B x 1 x d)
        inputsV = torch.cat((clasV, inputsV), dim=1) #(B x chunk_size*framerate + 1 x d)
        del clasV
        clasVmask = torch.unsqueeze(self.clasVmask.repeat(inputsVmask.shape[0], 1), dim=1) #(B x 1 x d)
        inputsVmask = torch.cat((clasVmask, inputsVmask), dim=1) #(B x chunk_size*framerate + 1 x d)
        del clasVmask
        clasA = torch.unsqueeze(self.clasA.repeat(inputsA.shape[0], 1), dim=1) #(B x 1 x d)
        inputsA = torch.cat((clasA, inputsA), dim=1) #(B x chunk_size*framerate + 1 x d)
        del clasA
        clasAmask = torch.unsqueeze(self.clasAmask.repeat(inputsAmask.shape[0], 1), dim=1) #(B x 1 x d)
        inputsAmask = torch.cat((clasAmask, inputsAmask), dim=1) #(B x chunk_size*framerate + 1 x d)
        del clasAmask
        
        #TRANSFORMER ENCODER
        inputsV = self.encoderV(inputsV) #(B x chunk_size * framerate +1 x d)
        inputsA = self.encoderA(inputsA) #(B x chunk_size * framerate +1 x d)
        inputsVmask = self.encoderVmask(inputsVmask) #(B x chunk_size * framerate +1 x d)
        inputsAmask = self.encoderAmask(inputsAmask) #(B x chunk_size * framerate +1 x d)
        
        #LAYER NORMALIZATION
        inputsV = self.norm2V(inputsV) #(B x chunk_size * framerate +1 x d)
        inputsA = self.norm2A(inputsA) #(B x chunk_size * framerate +1 x d)
        inputsVmask = self.norm2V(inputsVmask) #(B x chunk_size * framerate +1 x d)
        inputsAmask = self.norm2A(inputsAmask) #(B x chunk_size * framerate +1 x d)
        
        #POOLING TO GET EMBEDDING FOR VISUAL AND AUDIO REPRESENTATIONS (INSTEAD OF CLASS TOKEN)
        #embeddingV = self.pool_layerSS(aux_inputsVmask).squeeze(-1) #(B x d)
        #embeddingA = self.pool_layerSS(aux_inputsAmask).squeeze(-1) #(B x d)
        embeddingV = inputsVmask[:, 0, :]
        embeddingA = inputsAmask[:, 0, :]
        
        #NOT CLASS TOKENS
        inputsV = inputsV[:, 1:, :]
        inputsA = inputsA[:, 1:, :]
        inputsVmask = inputsVmask[:, 1:, :]
        inputsAmask = inputsAmask[:, 1:, :]
        
        #PREDICTION OF MASK TOKENS
        Vpreds = self.convMV2(self.relu(self.convMV1(inputsVmask.permute((0, 2, 1))))).permute((0, 2, 1)) #(B x chunk_size * framerate x d)
        Apreds = self.convMA2(self.relu(self.convMA1(inputsAmask.permute((0, 2, 1))))).permute((0, 2, 1)) #(B x chunk_size * framerate x d)
        
        #GET MASKED IDS
        realV = inputsV[MV] #(n_maskV x d)
        realA = inputsA[MA] #(n_maskA x d)
        predsV = Vpreds[MV] #(n_maskV x d)
        predsA = Apreds[MA] #(n_maskA x d)
        
        #In case there is no masking in one batch
        if realV.shape[0] == 0:
            realV = inputsV[1:3, 0, :]
            predsV = Vpreds[1:3, 0, :]
            
        if realA.shape[0] == 0:
            realA = inputsA[1:3, 0, :]
            predsA = Apreds[1:3, 0, :]
        
        del MV
        del MA
        del inputsV
        del inputsA
        
        #CONCATENATION OF VISUAL AND AUDIO EVOLVED FEATURES (MASK PART)
        embeddings = torch.cat((inputsVmask, inputsAmask), dim=1) #(B x 2*(chunk_size * framerate) x d)
        
        del inputsVmask
        del inputsAmask
        
        
        #MULTIMODAL TRANSFORMER ENCODER
        embeddings = self.encoderM(embeddings) #(B x 2*(chunk_size * framerate) x d)

        
        #LAYER NORMALIZATION
        embeddings = self.norm3(embeddings) #(B x 2*(chunk_size * framerate) x d)
        
        #PERMUTATION
        embeddings = embeddings.permute((0, 2, 1)) #(B x d x 2*(chunk_size*framerate))
        
        #POOLING (INSTEAD OF CLASS TOKEN)
        embeddings = self.pool_layerAS(embeddings).squeeze(-1) #(B x d)
        
        #FC AND SIGMOID TO MAKE PREDICTIONS        
        outputs = self.sigm(self.fc2(self.relu(self.fc1(embeddings))))
        
        del embeddings
        
        
        

            
        return embeddingV, embeddingA, realV, predsV, realA, predsA, outputs


class ModelFramesAudio(nn.Module):
    def __init__(self, weights=None, num_classes = 17, d=512, chunk_size=5, framerate=2, p_mask = 0.15, 
                 framestride = 4, model="SSModel", backbone = 'mobilenet', masking = 'token',
                 K = 64):
        """
        INPUT: two Tensors of shape (batch_size,chunk_size*framerate,feature_size)
        OUTPUTS: two Tensors of shape (batch_size,d)
        """

        super(ModelFramesAudio, self).__init__()
        
        #MODEL PARAMETERS
        
        self.num_classes = num_classes
        self.d = d
        self.chunk_size = chunk_size
        self.framerate = framerate
        self.p_mask = p_mask
        self.model = model
        self.framestride = framestride
        self.backbone = backbone
        self.masking = masking
        self.K = K

        #QUEUING DEQUEUING PARAMETERS
        self.register_buffer('queueV', torch.randn(self.K, d))
        self.queueV = nn.functional.normalize(self.queueV, dim=0)

        self.register_buffer('queueA', torch.randn(self.K, d))
        self.queueA = nn.functional.normalize(self.queueA, dim=0)
        
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        
        #MODEL BACKBONES (VISUAL ONES)
        if backbone == 'mobilenet':
            #Video mobilenet
            self.mobilenet = mobilenet_v3_small(MobileNet_V3_Small_Weights)
            self.mobilenet.classifier = torch.nn.Identity()
            self.conv1V = nn.Conv1d(576, d, 1, stride=1, bias=False)

            #Audio mobilenet
            self.mobilenetA = mobilenet_v3_small(MobileNet_V3_Small_Weights)
            self.mobilenetA.classifier = torch.nn.Identity()
            self.mobilenetA.features[0][0] = nn.Conv2d(1, self.mobilenetA.features[0][0].out_channels,
                                kernel_size=self.mobilenetA.features[0][0].kernel_size[0],
                                stride=self.mobilenetA.features[0][0].stride[0],
                                padding=self.mobilenetA.features[0][0].padding[0])
            self.conv1A = nn.Conv1d(576, d, 1, stride=1, bias=False)
        elif backbone == 'vit':
            self.vit = MobileViT(
                image_size = (256, 512),
                dims = [96, 120, 144],
                channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384],
                num_classes = 512
            )
            #self.vit.head = torch.nn.Identity()
            self.conv1V = nn.Conv1d(512, d, 1, stride=1, bias=False)
            self.transform = T.Resize((256,256))
        
        
        #SS MODEL LAYERS
        
        #Convolutions (reduce dimensionality)

        #self.conv1A = nn.Conv1d(128, d, 1, stride=1, bias=False)
        self.norm1V = nn.LayerNorm([chunk_size * 25 // self.framestride, d])
        self.norm1A = nn.LayerNorm([self.chunk_size * self.framerate, d])
        self.norm2V = nn.LayerNorm([chunk_size * 25 // self.framestride + 1, d])
        self.norm2A = nn.LayerNorm([self.chunk_size * self.framerate + 1, d])
        self.norm3 = nn.LayerNorm([chunk_size * 25 // self.framestride + (self.chunk_size * self.framerate), d])
        
        #Masked tokens
        self.mask_tokenA = nn.Parameter(torch.randn(d))
        self.maskingA = mask_tokens(self.mask_tokenA, p_mask=self.p_mask, consecutive_tokens=True)
        if masking == 'token':
            self.mask_tokenV = nn.Parameter(torch.randn(d))
            self.maskingV = mask_tokens(self.mask_tokenV, self.p_mask, consecutive_tokens=True)
            
        elif masking == 'frame':
            self.mask_tokenV = nn.Parameter(torch.randn(1))
            self.maskingV = mask_frames(self.mask_tokenV, p_mask=self.p_mask)
        
        #Transformer Encoders
        encoder_layerV = nn.TransformerEncoderLayer(d_model = d, nhead = 8, batch_first=True)
        self.encoderVmask = nn.TransformerEncoder(encoder_layerV, 2)
        self.encoderV = copy.deepcopy(self.encoderVmask)
        self.clasVmask = nn.Parameter(torch.randn(d))
        self.clasV = copy.deepcopy(self.clasVmask)
        
        
        encoder_layerA = nn.TransformerEncoderLayer(d_model = d, nhead = 8, batch_first=True)
        self.encoderAmask = nn.TransformerEncoder(encoder_layerA, 1)
        self.encoderA = copy.deepcopy(self.encoderAmask)
        self.clasAmask = nn.Parameter(torch.randn(d))
        self.clasA = copy.deepcopy(self.clasVmask)
        
        #Positional embeddings
        self.posVmask = nn.Parameter(torch.randn([self.chunk_size * 25 // self.framestride, d]))
        self.posAmask = nn.Parameter(torch.randn([self.chunk_size * self.framerate, d]))
        self.posV = copy.deepcopy(self.posVmask)
        self.posA = copy.deepcopy(self.posAmask)
        
        #Mask predictors
        self.convMV1 = nn.Conv1d(d, 2*d, 1, stride=1, bias=True)
        self.convMV2 = nn.Conv1d(2*d, d, 1, stride=1, bias=True)
        self.convMA1 = nn.Conv1d(d, 2*d, 1, stride=1, bias=True)
        self.convMA2 = nn.Conv1d(2*d, d, 1, stride=1, bias=True)
        
        #Avoid gradient on momentum layers
        self.encoderV.requires_grad_(False)
        self.encoderA.requires_grad_(False)
        self.posV.requires_grad_(False)
        self.posA.requires_grad_(False)
        self.clasV.requires_grad_(False)
        self.clasA.requires_grad_(False)
        
        #Pooling layer
        self.pool_layerSS = nn.MaxPool1d(chunk_size * framerate, stride = 1)
        #self.pool_layerSS = nn.AvgPool1d(chunk_size * framerate, stride = 1)
        
        
        #AS MODEL LAYERS
        
        #Transformer Encoders
        encoder_layerM = nn.TransformerEncoderLayer(d_model = d, nhead = 8, batch_first=True)
        self.encoderM = nn.TransformerEncoder(encoder_layerM, 2)
        
        #Pooling layer
        self.pool_layerAS = nn.MaxPool1d(chunk_size * 25 // self.framestride + (self.chunk_size * self.framerate), stride = 1)
        
        #Linear layer
        self.fc1 = nn.Linear(d, 2*d)
        self.fc2 = nn.Linear(2*d, self.num_classes+1)
        
        
        #GENERAL LAYERS
        
        #Positional encodding layer (not used for the moment)
        self.positional = PositionalEncoding(d)
        
        #General functions
        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()
        
        #Load weights parameter
        self.load_weights(weights=weights)
        
        
        
    def load_weights(self, weights=None):
        if(weights is not None):
            print("=> loading checkpoint '{}'".format(weights))
            checkpoint = torch.load(weights)
            self.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(weights, checkpoint['epoch']))

    def enqueue_dequeue(self, keysV, keysA):
        # gather keys before updating queue
        current_batch = keysV.shape[0]

        ptr = int(self.queue_ptr)

        batch_size = keysV.shape[0]

        ptr = int(self.queue_ptr)

        if (ptr + batch_size) <= self.K:
            self.queueV[ptr:ptr + batch_size, :] = keysV
            self.queueA[ptr:ptr + batch_size, :] = keysA

            ptr = (ptr + batch_size) % self.K

        else:
            self.queueV[ptr:self.K, :] = keysV[0:(self.K - ptr), :]
            self.queueV[0:(ptr + batch_size - self.K), :] = keysV[(self.K - ptr):, :]

            self.queueA[ptr:self.K, :] = keysA[0:(self.K - ptr), :]
            self.queueA[0:(ptr + batch_size - self.K), :] = keysA[(self.K - ptr):, :]

            ptr = ptr + batch_size - self.K

        self.queue_ptr[0] = ptr

    def forward(self, inputsV, inputsA, inference = False):
        
        
        #INPUTS TO FLOAT
        inputsV = inputsV.float() #(B x n_frames x H x W x C)
        inputsA = inputsA.float() #(B x n_frames x H x W)
        inputsA = inputsA.unsqueeze(-1) #(B x n_frames x H x W x C(1))
        images_shape = inputsV.shape
        audio_shape = inputsA.shape

        
        not_audio = ((inputsA == 0).float().mean(axis=[1, 2, 3]) != 1).squeeze()
        inputsA[~not_audio] += 1e-05
        
        #DIFFERENT MASKING STRATEGIES (VIDEO)
        
        #FRAME MASKING STRATEGY
        if self.masking == 'frame':
        
        #MASKING OF FRAMES
            if not inference:
                inputsVmask, MV = self.maskingV(inputsV) #(B x n_frames x H x W x C), (B x n_frames)
                inputsVmask = inputsVmask.permute((0, 1, 4, 2, 3)) #(B x n_frames x C x H x W)
                inputsVmask = inputsVmask.view(-1, images_shape[4], images_shape[2], images_shape[3]) #(B*n_frames x C x H x W)
                
            inputsV = inputsV.view(-1, images_shape[4], images_shape[2], images_shape[3]) #(B*n_frames x C x H x W)
            
            #BACKBONE FRAME FEATURE EXTRACTION
            if self.backbone == 'mobilenet':
                inputsV = self.mobilenet(inputsV) #(B*n_frames x n_features)
                if not inference:
                    inputsVmask = self.mobilenet(inputsVmask) #(B*n_frames x n_features)
            elif self.backbone == 'vit':
                inputsV = self.transform(inputsV) #(B*n_frames x C x H2 x W2)
                inputsV = self.vit(inputsV) #(B*n_frames x n_features)
                if not inference:
                    inputsVmask = self.transform(inputsVmask) #(B*n_frames x C x H2 x W2)
                    inputsVmask = self.vit(inputsVmask) #(B*n_frames x n_features)        
                    
            #PFFN TO REDUCE DIMENSIONALITY
            inputsV = inputsV.view(images_shape[0], images_shape[1], -1) #(B x n_frames x n_features)
            inputsV = inputsV.permute((0, 2, 1)) #(B x n_features x n_frames)
            inputsV = self.relu(self.conv1V(inputsV)) #(B x d x n_frames)
            inputsV = inputsV.permute((0, 2, 1)) #(B x n_frames x d)
            inputsV = self.norm1V(inputsV) #(B x n_frames x d)
            
            #PFFN TO REDUCE DIMENSIONALITY (MASKED PART)
            if not inference:
                inputsVmask = inputsVmask.view(images_shape[0], images_shape[1], -1) #(B x n_frames x n_features)
                inputsVmask = inputsVmask.permute((0, 2, 1)) #(B x n_features x n_frames)
                inputsVmask = self.relu(self.conv1V(inputsVmask)) #(B x d x n_frames)
                inputsVmask = inputsVmask.permute((0, 2, 1)) #(B x n_frames x d)
                inputsVmask = self.norm1V(inputsVmask) #(B x n_frames x d)
                
            #IF NOT TRAINING
            if inference:
                inputsVmask = torch.clone(inputsV) #(B x n_frames x d)
            
        #TOKEN MASKING STRATEGY
        elif self.masking == 'token':
            
            #RESHAPE
            inputsV = inputsV.permute((0, 1, 4, 2, 3)) #(B x n_frames x C x H x W)
            inputsV = inputsV.view(-1, images_shape[4], images_shape[2], images_shape[3]) #(n x C x H x W)

            inputsA = inputsA.permute((0, 1, 4, 2, 3)) #(B x n_frames x C(1) x H x W)
            inputsA = inputsA.view(-1, audio_shape[4], audio_shape[2], audio_shape[3]) #(n x C x H x W)
            
            #BACKBONE FRAME FEATURE EXTRACTION
            if self.backbone == 'mobilenet':
                inputsV = self.mobilenet(inputsV) #(n x 576)
                inputsA = self.mobilenetA(inputsA) #(n x 576)
            elif self.backbone == 'vit':
                inputsV = self.transform(inputsV)
                inputsV = self.vit(inputsV)
            
            #PFFN TO REDUCE DIMENSIONALITY 
            inputsV = inputsV.view(images_shape[0], images_shape[1], -1) #(B x n_frames x n_features(576))
            inputsV = inputsV.permute((0, 2, 1)) #(B x n_features x chunk_size*framerate)
            inputsV = self.relu(self.conv1V(inputsV)) #(B x d x chunk_size*framerate)
            inputsV = inputsV.permute((0, 2, 1)) #(B x chunk_size*framerate x d)

            inputsA = inputsA.view(audio_shape[0], audio_shape[1], -1)
            inputsA = inputsA.permute((0, 2, 1)) #(B x n_features x chunk_size*framerate)
            inputsA = self.relu(self.conv1A(inputsA)) #(B x d x chunk_size*framerate)
            inputsA = inputsA.permute((0, 2, 1)) #(B x chunk_size*framerate x d)

            #CREATE MASKING COPY
            inputsVmask = torch.clone(inputsV) #(B x chunk_size*framerate x d)
            inputsAmask = torch.clone(inputsA) #(B x chunk_size*framerate x d)
            
            #MASKING OF TOKENS
            if (not inference):
                inputsVmask, MV = self.maskingV(inputsVmask) #(B x chunk_size*framerate x d)
                inputsAmask, MV = self.maskingA(inputsAmask) #(B x nframes x d)
                
                
            #LAYER NORMALIZATION
            inputsV = self.norm1V(inputsV) #(B x chunk_size*framerate x d)
            inputsVmask = self.norm1V(inputsVmask) #(B x chunk_size*framerate x d)
            inputsA = self.norm1A(inputsA) #(B x chunk_size*framerate x d)
            inputsAmask = self.norm1A(inputsAmask) #(B x chunk_size*framerate x d)
        
        #NOT TRAINING PROCEDURE 
        if inference:
            MV = torch.rand([inputsVmask.shape[0], inputsVmask.shape[1]]) < 0.05
            MA = torch.rand([inputsAmask.shape[0], inputsAmask.shape[1]]) < 0.05
        
        
        
        #POSITIONAL ENCODING
        inputsV = inputsV + self.posV #(B x chunk_size*framerate x d)
        inputsA = inputsA + self.posA #(B x chunk_size*framerate x d)
        inputsVmask = inputsVmask + self.posVmask #(B x chunk_size*framerate x d)
        inputsAmask = inputsAmask + self.posAmask #(B x chunk_size*framerate x d)
        
        #ADDING CLASS TOKEN
        clasV = torch.unsqueeze(self.clasV.repeat(inputsV.shape[0], 1), dim=1) #(B x 1 x d)
        inputsV = torch.cat((clasV, inputsV), dim=1) #(B x chunk_size*framerate + 1 x d)

        clasVmask = torch.unsqueeze(self.clasVmask.repeat(inputsVmask.shape[0], 1), dim=1) #(B x 1 x d)
        inputsVmask = torch.cat((clasVmask, inputsVmask), dim=1) #(B x chunk_size*framerate + 1 x d)

        clasA = torch.unsqueeze(self.clasA.repeat(inputsA.shape[0], 1), dim=1) #(B x 1 x d)
        inputsA = torch.cat((clasA, inputsA), dim=1) #(B x chunk_size*framerate + 1 x d)

        clasAmask = torch.unsqueeze(self.clasAmask.repeat(inputsAmask.shape[0], 1), dim=1) #(B x 1 x d)
        inputsAmask = torch.cat((clasAmask, inputsAmask), dim=1) #(B x chunk_size*framerate + 1 x d)

        
        #TRANSFORMER ENCODER
        inputsV = self.encoderV(inputsV) #(B x chunk_size * framerate +1 x d)
        inputsA = self.encoderA(inputsA) #(B x chunk_size * framerate +1 x d)
        inputsVmask = self.encoderVmask(inputsVmask) #(B x chunk_size * framerate +1 x d)
        inputsAmask = self.encoderAmask(inputsAmask) #(B x chunk_size * framerate +1 x d)
        
        #LAYER NORMALIZATION
        inputsV = self.norm2V(inputsV) #(B x chunk_size * framerate +1 x d)
        inputsA = self.norm2A(inputsA) #(B x chunk_size * framerate +1 x d)
        inputsVmask = self.norm2V(inputsVmask) #(B x chunk_size * framerate +1 x d)
        inputsAmask = self.norm2A(inputsAmask) #(B x chunk_size * framerate +1 x d)
        
        #POOLING TO GET EMBEDDING FOR VISUAL AND AUDIO REPRESENTATIONS (INSTEAD OF CLASS TOKEN)
        #embeddingV = self.pool_layerSS(aux_inputsVmask).squeeze(-1) #(B x d)
        #embeddingA = self.pool_layerSS(aux_inputsAmask).squeeze(-1) #(B x d)
        embeddingV = inputsVmask[:, 0, :]
        embeddingA = inputsAmask[:, 0, :]

        embeddingV = embeddingV[not_audio, :]
        embeddingA = embeddingA[not_audio, :]

        negV = self.queueV.clone().detach()
        negA = self.queueA.clone().detach()

        self.enqueue_dequeue(embeddingV, embeddingA)
        
        #NOT CLASS TOKENS
        inputsV = inputsV[:, 1:, :]
        inputsA = inputsA[:, 1:, :]
        inputsVmask = inputsVmask[:, 1:, :]
        inputsAmask = inputsAmask[:, 1:, :]
        
        #PREDICTION OF MASK TOKENS
        Vpreds = self.convMV2(self.relu(self.convMV1(inputsVmask.permute((0, 2, 1))))).permute((0, 2, 1)) #(B x chunk_size * framerate x d)
        Apreds = self.convMA2(self.relu(self.convMA1(inputsAmask.permute((0, 2, 1))))).permute((0, 2, 1)) #(B x chunk_size * framerate x d)
        
        #GET MASKED IDS
        MA = MA.cuda()
        not_audio = not_audio.expand(MA.shape[1], MA.shape[0]).T.cuda()
        
        realV = inputsV[MV] #(n_maskV x d)
        realA = inputsA[MA & not_audio] #(n_maskA x d)
        predsV = Vpreds[MV] #(n_maskV x d)
        predsA = Apreds[MA & not_audio] #(n_maskA x d)
        
        #CONCATENATION OF VISUAL AND AUDIO EVOLVED FEATURES (MASK PART)
        embeddings = torch.cat((inputsVmask, inputsAmask), dim=1) #(B x 2*(chunk_size * framerate) x d)        
        
        #MULTIMODAL TRANSFORMER ENCODER
        embeddings = self.encoderM(embeddings) #(B x 2*(chunk_size * framerate) x d)

        
        #LAYER NORMALIZATION
        embeddings = self.norm3(embeddings) #(B x 2*(chunk_size * framerate) x d)
        
        #PERMUTATION
        embeddings = embeddings.permute((0, 2, 1)) #(B x d x 2*(chunk_size*framerate))
        
        #POOLING (INSTEAD OF CLASS TOKEN)
        embeddings = self.pool_layerAS(embeddings).squeeze(-1) #(B x d)
        
        #FC AND SIGMOID TO MAKE PREDICTIONS        
        outputs = self.sigm(self.fc2(self.relu(self.fc1(embeddings))))
        
        
        

            
        return embeddingV, embeddingA, negV, negA, realV, predsV, realA, predsA, outputs