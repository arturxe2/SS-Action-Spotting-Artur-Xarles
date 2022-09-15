# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 10:49:15 2022

@author: arturxe
"""
import torch
from torch.utils.data import Dataset
from SoccerNet.Downloader import getListGames
from tqdm import tqdm
import numpy as np
import logging
import os


#Function to extract features of a clip
def feats2clip(feats, stride, clip_length, padding = "replicate_last", off=0):
    if padding =="zeropad":
        print("beforepadding", feats.shape)
        pad = feats.shape[0] - int(feats.shape[0]/stride)*stride
        print("pad need to be", clip_length-pad)
        m = torch.nn.ZeroPad2d((0, 0, clip_length-pad, 0))
        feats = m(feats)
        print("afterpadding", feats.shape)
        # nn.ZeroPad2d(2)

    idx = torch.arange(start=0, end=feats.shape[0]-1, step=stride)
    idxs = []
    for i in torch.arange(-off, clip_length-off):
    # for i in torch.arange(0, clip_length):
        idxs.append(idx+i)
    idx = torch.stack(idxs, dim=1)

    if padding=="replicate_last":
        idx = idx.clamp(0, feats.shape[0]-1)
        # Not replicate last, but take the clip closest to the end of the video
        # idx[-1] = torch.arange(clip_length)+feats.shape[0]-clip_length
    # print(idx)
    return feats[idx,...]



class SoccerNetClips(Dataset):
    def __init__(self, path_baidu = '/data-local/data3-ssd/axesparraguera', 
                 path_audio = '/data-local/data3-ssd/axesparraguera',  
                 features_baidu = 'baidu_soccer_embeddings_2fps.npy',
                 features_audio = 'audio_embeddings_2fps.npy', 
                 split=["train"], framerate=2, chunk_size=20):

        self.listGames = getListGames(split)
        self.chunk_size = chunk_size

        logging.info("Pre-compute clips")
        
        self.game_featsV = list()
        self.game_featsA = list()


        stride = self.chunk_size #// 2
        for game in tqdm(self.listGames):
            
            feat_half1V = np.load(os.path.join(path_baidu, game, "1_" + features_baidu))
            feat_half1V = feat_half1V.reshape(-1, feat_half1V.shape[-1])
            feat_half1A = np.load(os.path.join(path_audio, game, "1_" + features_audio))
            feat_half1A = feat_half1A.reshape(-1, feat_half1A.shape[-1])
            feat_half2V = np.load(os.path.join(path_baidu, game, "2_" + features_baidu))
            feat_half2V = feat_half2V.reshape(-1, feat_half2V.shape[-1])
            feat_half2A = np.load(os.path.join(path_audio, game, "2_" + features_audio))
            feat_half2A = feat_half2A.reshape(-1, feat_half2A.shape[-1])
                
            #Check same size Visual and Audio features
            
            #Visual features bigger than audio features
            if feat_half1V.shape[0] > feat_half1A.shape[0]:
                feat_half1A_aux = np.zeros((feat_half1V.shape[0], feat_half1A.shape[1]))
                feat_half1A_aux[:feat_half1A.shape[0]] = feat_half1A
                feat_half1A_aux[feat_half1A.shape[0]:] = feat_half1A[feat_half1A.shape[0]-1]
                feat_half1A = feat_half1A_aux
                
            if feat_half2V.shape[0] > feat_half2A.shape[0]:
                feat_half2A_aux = np.zeros((feat_half2V.shape[0], feat_half2A.shape[1]))
                feat_half2A_aux[:feat_half2A.shape[0]] = feat_half2A
                feat_half2A_aux[feat_half2A.shape[0]:] = feat_half2A[feat_half2A.shape[0]-1]
                feat_half2A = feat_half2A_aux
                
            #Audio features bigger than visual features
            if feat_half1A.shape[0] > feat_half1V.shape[0]:
                feat_half1V_aux = np.zeros((feat_half1A.shape[0], feat_half1V.shape[1]))
                feat_half1V_aux[:feat_half1V.shape[0]] = feat_half1V
                feat_half1V_aux[feat_half1V.shape[0]:] = feat_half1V[feat_half1V.shape[0]-1]
                feat_half1V = feat_half1V_aux
                
            if feat_half2A.shape[0] > feat_half2V.shape[0]:
                feat_half2V_aux = np.zeros((feat_half2A.shape[0], feat_half2V.shape[1]))
                feat_half2V_aux[:feat_half2V.shape[0]] = feat_half2V
                feat_half2V_aux[feat_half2V.shape[0]:] = feat_half2V[feat_half2V.shape[0]-1]
                feat_half2V = feat_half2V_aux                    

                
            #Generate clips from features
            feat_half1V = feats2clip(torch.from_numpy(feat_half1V), stride=stride, clip_length=self.chunk_size) 
            feat_half1A = feats2clip(torch.from_numpy(feat_half1A), stride=stride, clip_length=self.chunk_size) 
            feat_half2V = feats2clip(torch.from_numpy(feat_half2V), stride=stride, clip_length=self.chunk_size) 
            feat_half2A = feats2clip(torch.from_numpy(feat_half2A), stride=stride, clip_length=self.chunk_size) 

            #Append visual and audio features of all games
            self.game_featsV.append(feat_half1V)
            self.game_featsA.append(feat_half1A)
            self.game_featsV.append(feat_half2V)
            self.game_featsA.append(feat_half2A)
                        
        #Concatenate features
        self.game_featsV = np.concatenate(self.game_featsV)
        self.game_featsA = np.concatenate(self.game_featsA)




    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            clip_feat (np.array): clip of features.
            clip_labels (np.array): clip of labels for the segmentation.
            clip_targets (np.array): clip of targets for the spotting.
        """
        return self.game_featsV[index,:,:], self.game_featsA[index,:,:]

    def __len__(self):

        return len(self.game_featsV)

a = SoccerNetClips()
print(a.__len__())