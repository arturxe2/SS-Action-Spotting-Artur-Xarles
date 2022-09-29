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
    
    R = torch.rand([n_B, n_T])
    random_token = features[torch.randint(0, n_B, (1,)), torch.randint(0, n_T, (1,)), :]
    M1 = R < (p_mask * 0.8)
    M2 = (R >= (p_mask * 0.8)) & (R < (p_mask * 0.9))
    M3 = (R >= (p_mask * 0.9)) & (R < p_mask)
    
    features[M1] = mask_token
    features[M2] = random_token
    
    M = (M1 | M2) | M3
    
    return features, M

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
        self.norm2 = nn.LayerNorm([2 * self.chunk_size * self.framerate, d])
        
        #Masked tokens
        self.mask_tokenV = nn.Parameter(torch.randn(d))
        self.mask_tokenA = nn.Parameter(torch.randn(d))
        
        #Transformer Encoders
        encoder_layerV = nn.TransformerEncoderLayer(d_model = d, nhead = 8, batch_first=True)
        self.encoderVmask = nn.TransformerEncoder(encoder_layerV, 2)
        self.encoderV = copy.deepcopy(self.encoderVmask)
        
        
        encoder_layerA = nn.TransformerEncoderLayer(d_model = d, nhead = 8, batch_first=True)
        self.encoderAmask = nn.TransformerEncoder(encoder_layerA, 1)
        self.encoderA = copy.deepcopy(self.encoderAmask)
        
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
        
        #TRANSFORMER ENCODER
        inputsV = self.encoderV(inputsV) #(B x chunk_size * framerate x d)
        inputsA = self.encoderA(inputsA) #(B x chunk_size * framerate x d)
        inputsVmask = self.encoderVmask(inputsVmask) #(B x chunk_size * framerate x d)
        inputsAmask = self.encoderAmask(inputsAmask) #(B x chunk_size * framerate x d)
        
        #LAYER NORMALIZATION
        inputsV = self.norm1(inputsV) #(B x chunk_size * framerate x d)
        inputsA = self.norm1(inputsA) #(B x chunk_size * framerate x d)
        inputsVmask = self.norm1(inputsVmask) #(B x chunk_size * framerate x d)
        inputsAmask = self.norm1(inputsAmask) #(B x chunk_size * framerate x d)
        
        #PERMUTATION
        aux_inputsVmask = inputsVmask.permute((0, 2, 1)) #(B x d x chunk_size*framerate)
        aux_inputsAmask = inputsAmask.permute((0, 2, 1)) #(B x d x chunk_size*framerate)
        
        #POOLING TO GET EMBEDDING FOR VISUAL AND AUDIO REPRESENTATIONS (INSTEAD OF CLASS TOKEN)
        embeddingV = self.pool_layerSS(aux_inputsVmask).squeeze(-1) #(B x d)
        embeddingA = self.pool_layerSS(aux_inputsAmask).squeeze(-1) #(B x d)
        
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
        embeddings = self.norm2(embeddings) #(B x 2*(chunk_size * framerate) x d)
        
        #PERMUTATION
        embeddings = embeddings.permute((0, 2, 1)) #(B x d x 2*(chunk_size*framerate))
        
        #POOLING (INSTEAD OF CLASS TOKEN)
        embeddings = self.pool_layerAS(embeddings).squeeze(-1) #(B x d)
        
        #FC AND SIGMOID TO MAKE PREDICTIONS        
        outputs = self.sigm(self.fc2(self.relu(self.fc1(embeddings))))

            
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
