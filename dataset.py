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
from SoccerNet.Evaluation.utils import AverageMeter, EVENT_DICTIONARY_V2, INVERSE_EVENT_DICTIONARY_V2
import json
import pickle
import skimage.io as io
import math
import cv2
from joblib import Parallel, delayed
import time



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

    return feats[idx,...]


#SoccerNet clips in ram
class SoccerNetClips(Dataset):
    def __init__(self, path_baidu = '/data-local/data3-ssd/axesparraguera', 
                 path_audio = '/data-local/data3-ssd/axesparraguera',  
                 path_labels = "/data-net/datasets/SoccerNetv2/ResNET_TF2",
                 features_baidu = 'baidu_soccer_embeddings_2fps.npy',
                 features_audio = 'audio_embeddings_2fps.npy', 
                 split=["train"], framerate=2, chunk_size=20):

        self.listGames = getListGames(split)
        self.chunk_size = chunk_size
        
        self.dict_event = EVENT_DICTIONARY_V2
        self.num_classes = 17
        self.labels="Labels-v2.json"

        logging.info("Pre-compute clips")
        
        self.game_featsV = list()
        self.game_featsA = list()
        self.game_labels = list()


        stride = self.chunk_size

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

            
            
            # Load labels
            labels = json.load(open(os.path.join(path_labels, game, self.labels)))


            label_half1 = np.zeros((feat_half1V.shape[0], self.num_classes+1))
            label_half1[:,0]=1 # those are BG classes
            label_half2 = np.zeros((feat_half2V.shape[0], self.num_classes+1))
            label_half2[:,0]=1 # those are BG classes
            
            for annotation in labels["annotations"]:

                time = annotation["gameTime"]
                event = annotation["label"]

                half = int(time[0])
                

                minutes = int(time[-5:-3])
                seconds = int(time[-2::])
                frame = framerate * ( seconds + 60 * minutes ) 

                if event not in self.dict_event:
                    continue
                label = self.dict_event[event]

                # if label outside temporal of view
                if half == 1 and frame//stride>=label_half1.shape[0]:
                    continue
                if half == 2 and frame//stride>=label_half2.shape[0]:
                    continue
                a = frame // stride
                
                if half == 1:
                    if self.chunk_size >= stride:
                        for i in range(self.chunk_size // stride):
                            label_half1[max(a - self.chunk_size // stride + 1 + i, 0)][0] = 0 # not BG anymore
                            label_half1[max(a - self.chunk_size // stride + 1 + i, 0)][label+1] = 1
                        #label_half1[max(a - self.chunk_size//stride + 1, 0) : (a + 1)][0] = 0 # not BG anymore
                        
                    else:
                        a2 = (frame - self.chunk_size) // stride
                        if a != a2:
                            label_half1[a][0] = 0
                            label_half1[a][label+1] = 1

                if half == 2:
                    if self.chunk_size >= stride:
                        for i in range(self.chunk_size // stride):
                            label_half2[max(a - self.chunk_size // stride + 1 + i, 0)][0] = 0 # not BG anymore
                            label_half2[max(a - self.chunk_size // stride + 1 + i, 0)][label+1] = 1 # that's my class
                            
                    else:
                        a2 = (frame - self.chunk_size) // stride
                        if a != a2:
                            label_half2[a][0] = 0
                            label_half2[a][label+1] = 1

            #Append visual and audio features of all games
            self.game_featsV.append(feat_half1V)
            self.game_featsA.append(feat_half1A)
            self.game_featsV.append(feat_half2V)
            self.game_featsA.append(feat_half2A)
            
            self.game_labels.append(label_half1)
            self.game_labels.append(label_half2)
                        
        #Concatenate features
        self.game_featsV = np.concatenate(self.game_featsV)
        self.game_featsA = np.concatenate(self.game_featsA)
        self.game_labels = np.concatenate(self.game_labels)




    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            clip_feat (np.array): clip of features.
            clip_labels (np.array): clip of labels for the segmentation.
            clip_targets (np.array): clip of targets for the spotting.
        """
        return self.game_featsV[index,:,:], self.game_featsA[index,:,:], self.game_labels[index, :]

    def __len__(self):

        return len(self.game_featsV)
    
    
class OnlineSoccerNetClips(Dataset):
    def __init__(self, path_baidu = '/data-local/data3-ssd/axesparraguera', 
                 path_audio = '/data-local/data3-ssd/axesparraguera',  
                 path_labels = "/data-net/datasets/SoccerNetv2/ResNET_TF2",
                 path_output = '/data-local/data1-hdd/axesparraguera/online_data',
                 features_baidu = 'baidu_soccer_embeddings_2fps.npy',
                 features_audio = 'audio_embeddings_2fps.npy', 
                 split=["train"], framerate=2, chunk_size=20, store = True):

        self.listGames = getListGames(split)
        self.chunk_size = chunk_size
        self.path_store = path_output
        
        self.dict_event = EVENT_DICTIONARY_V2
        self.num_classes = 17
        self.labels="Labels-v2.json"

        logging.info("Pre-compute clips")
        
        self.game_featsV = list()
        self.game_featsA = list()
        self.game_labels = list()
        stride = self.chunk_size
        
        if store:
            
            self.path_list = []
            self.n_samples = []
            
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
    
                
                
                # Load labels
                labels = json.load(open(os.path.join(path_labels, game, self.labels)))
    
    
                label_half1 = np.zeros((feat_half1V.shape[0], self.num_classes+1))
                label_half1[:,0]=1 # those are BG classes
                label_half2 = np.zeros((feat_half2V.shape[0], self.num_classes+1))
                label_half2[:,0]=1 # those are BG classes
                
                for annotation in labels["annotations"]:
    
                    time = annotation["gameTime"]
                    event = annotation["label"]
    
                    half = int(time[0])
                    
    
                    minutes = int(time[-5:-3])
                    seconds = int(time[-2::])
                    frame = framerate * ( seconds + 60 * minutes ) 
    
                    if event not in self.dict_event:
                        continue
                    label = self.dict_event[event]
    
                    # if label outside temporal of view
                    if half == 1 and frame//stride>=label_half1.shape[0]:
                        continue
                    if half == 2 and frame//stride>=label_half2.shape[0]:
                        continue
                    a = frame // stride
                    
                    if half == 1:
                        if self.chunk_size >= stride:
                            for i in range(self.chunk_size // stride):
                                label_half1[max(a - self.chunk_size // stride + 1 + i, 0)][0] = 0 # not BG anymore
                                label_half1[max(a - self.chunk_size // stride + 1 + i, 0)][label+1] = 1
                            #label_half1[max(a - self.chunk_size//stride + 1, 0) : (a + 1)][0] = 0 # not BG anymore
                            
                        else:
                            a2 = (frame - self.chunk_size) // stride
                            if a != a2:
                                label_half1[a][0] = 0
                                label_half1[a][label+1] = 1
    
                    if half == 2:
                        if self.chunk_size >= stride:
                            for i in range(self.chunk_size // stride):
                                label_half2[max(a - self.chunk_size // stride + 1 + i, 0)][0] = 0 # not BG anymore
                                label_half2[max(a - self.chunk_size // stride + 1 + i, 0)][label+1] = 1 # that's my class
                                
                        else:
                            a2 = (frame - self.chunk_size) // stride
                            if a != a2:
                                label_half2[a][0] = 0
                                label_half2[a][label+1] = 1
                                
                #Store path
                path = os.path.join(self.path_store, game)
                exists = os.path.exists(path)
                if not exists:
                    os.makedirs(path)
                
                #STORE HALF 1 FILES
                feat_half1V = feat_half1V.numpy()
                feat_half1A = feat_half1A.numpy()

                for i in range(feat_half1V.shape[0]):
                    np.save(path + '/half1_chunk' + str(i) + '_featuresV.npy', feat_half1V[i, :, :])
                    np.save(path + '/half1_chunk' + str(i) + '_featuresA.npy', feat_half1A[i, :, :])
                    np.save(path + '/half1_chunk' + str(i) + '_labels.npy', label_half1[i, :])
                    self.path_list.append(path + '/half1_chunk' + str(i) + '_')
                    
                #STORE HALF 2 FILES
                feat_half2V = feat_half2V.numpy()
                feat_half2A = feat_half2A.numpy()

                for i in range(feat_half2V.shape[0]):
                    np.save(path + '/half2_chunk' + str(i) + '_featuresV.npy', feat_half2V[i, :, :])
                    np.save(path + '/half2_chunk' + str(i) + '_featuresA.npy', feat_half2A[i, :, :])
                    np.save(path + '/half2_chunk' + str(i) + '_labels.npy', label_half2[i, :])
                    self.path_list.append(path + '/half2_chunk' + str(i) + '_')
    
                        
            #Concatenate features
            with open(self.path_store + '/chunk_list.pkl', 'wb') as f:
                pickle.dump(self.path_list, f)
                
        else:
            with open(self.path_store + '/chunk_list.pkl', 'rb') as f:
                self.path_list = pickle.load(f)




    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            clip_feat (np.array): clip of features.
            clip_labels (np.array): clip of labels for the segmentation.
            clip_targets (np.array): clip of targets for the spotting.
        """
        path = self.path_list[index]
        return torch.from_numpy(np.load(path + 'featuresV.npy')), torch.from_numpy(np.load(path + 'featuresA.npy')), np.load(path + 'labels.npy')


    def __len__(self):

        return(len(self.path_list))
    
    
    
#Class to generate the samples for the test part
class SoccerNetClipsTesting(Dataset):
    
    
    def __init__(self, path_baidu = '/data-local/data3-ssd/axesparraguera', 
                 path_audio = '/data-local/data3-ssd/axesparraguera',  
                 path_labels = "/data-net/datasets/SoccerNetv2/ResNET_TF2",
                 features_baidu = 'baidu_soccer_embeddings_2fps.npy',
                 features_audio = 'audio_embeddings_2fps.npy', 
                 split=["test"], framerate=2, chunk_size=20):
        
        self.path_baidu = path_baidu
        self.path_labels = path_labels
        self.path_audio = path_audio
        self.features_baidu = features_baidu
        self.features_audio = features_audio
        self.listGames = getListGames(split)
        self.chunk_size = chunk_size
        self.framerate = framerate
        self.split = split


        self.dict_event = EVENT_DICTIONARY_V2
        self.num_classes = 17
        self.labels="Labels-v2.json"

        logging.info("Checking/Download features and labels locally")
        #downloader = SoccerNetDownloader(path)
        #for s in split:
        #    if s == "challenge":
        #        downloader.downloadGames(files=[f"1_{self.features}", f"2_{self.features}"], split=[s], verbose=False)
        #    else:
        #        downloader.downloadGames(files=[self.labels, f"1_{self.features}", f"2_{self.features}"], split=[s], verbose=False)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            feat_half1 (np.array): features for the 1st half.
            feat_half2 (np.array): features for the 2nd half.
            label_half1 (np.array): labels (one-hot) for the 1st half.
            label_half2 (np.array): labels (one-hot) for the 2nd half.
        """
        
        

        featV_half1 = np.load(os.path.join(self.path_baidu, self.listGames[index], "1_" + self.features_baidu))
        featV_half1 = featV_half1.reshape(-1, featV_half1.shape[-1])    #for C3D non PCA
        featV_half2 = np.load(os.path.join(self.path_baidu, self.listGames[index], "2_" + self.features_baidu))
        featV_half2 = featV_half2.reshape(-1, featV_half2.shape[-1])    #for C3D non PCA
        featA_half1 = np.load(os.path.join(self.path_audio, self.listGames[index], "1_" + self.features_audio))
        featA_half1 = featA_half1.reshape(-1, featA_half1.shape[-1])    #for C3D non PCA
        featA_half2 = np.load(os.path.join(self.path_audio, self.listGames[index], "2_" + self.features_audio))
        featA_half2 = featA_half2.reshape(-1, featA_half2.shape[-1])    #for C3D non PCA


            
        label_half1 = np.zeros((featV_half1.shape[0], self.num_classes))
        label_half2 = np.zeros((featV_half2.shape[0], self.num_classes))
        
        # check if annoation exists
        if os.path.exists(os.path.join(self.path_labels, self.listGames[index], self.labels)):
        
            labels = json.load(open(os.path.join(self.path_labels, self.listGames[index], self.labels)))
            for annotation in labels["annotations"]:

                time = annotation["gameTime"]
                event = annotation["label"]

                half = int(time[0])

                minutes = int(time[-5:-3])
                seconds = int(time[-2::])
                frame = self.framerate * ( seconds + 60 * minutes ) 

                if event not in self.dict_event:
                    continue
                label = self.dict_event[event]
                


                value = 1
                if "visibility" in annotation.keys():
                    if annotation["visibility"] == "not shown":
                        value = -1

                if half == 1:
                    frame = min(frame, featV_half1.shape[0]-1)
                    label_half1[frame][label] = value
    
                if half == 2:
                    frame = min(frame, featV_half2.shape[0]-1)
                    label_half2[frame][label] = value
        
            
        #Check same size Visual and Audio features
        
        #Visual features bigger than audio features
        if featV_half1.shape[0] > featA_half1.shape[0]:
            featA_half1_aux = np.zeros((featV_half1.shape[0], featA_half1.shape[1]))
            featA_half1_aux[:featA_half1.shape[0]] = featA_half1
            featA_half1_aux[featA_half1.shape[0]:] = featA_half1[featA_half1.shape[0]-1]
            featA_half1 = featA_half1_aux
            
        if featV_half2.shape[0] > featA_half2.shape[0]:
            featA_half2_aux = np.zeros((featV_half2.shape[0], featA_half2.shape[1]))
            featA_half2_aux[:featA_half2.shape[0]] = featA_half2
            featA_half2_aux[featA_half2.shape[0]:] = featA_half2[featA_half2.shape[0]-1]
            featA_half2 = featA_half2_aux
            
        #Audio features bigger than visual features
        if featA_half1.shape[0] > featV_half1.shape[0]:
            featV_half1_aux = np.zeros((featA_half1.shape[0], featV_half1.shape[1]))
            featV_half1_aux[:featV_half1.shape[0]] = featV_half1
            featV_half1_aux[featV_half1.shape[0]:] = featV_half1[featV_half1.shape[0]-1]
            featV_half1 = featV_half1_aux
            
        if featA_half2.shape[0] > featV_half2.shape[0]:
            featV_half2_aux = np.zeros((featA_half2.shape[0], featV_half2.shape[1]))
            featV_half2_aux[:featV_half2.shape[0]] = featV_half2
            featV_half2_aux[featV_half2.shape[0]:] = featV_half2[featV_half2.shape[0]-1]
            featV_half2 = featV_half2_aux   
            
        featV_half1 = feats2clip(torch.from_numpy(featV_half1),
                                     stride=1, off=int(self.chunk_size/2),
                                     clip_length=self.chunk_size)
        featV_half2 = feats2clip(torch.from_numpy(featV_half2),
                                     stride=1, off=int(self.chunk_size/2),
                                     clip_length=self.chunk_size)
        featA_half1 = feats2clip(torch.from_numpy(featA_half1),
                                     stride=1, off=int(self.chunk_size/2),
                                     clip_length=self.chunk_size)
        featA_half2 = feats2clip(torch.from_numpy(featA_half2),
                                     stride=1, off=int(self.chunk_size/2),
                                     clip_length=self.chunk_size)
            
        if featV_half1.shape[0] != featA_half1.shape[0]:
            featA_half1 = featA_half1[:featV_half1.shape[0]]
        if featV_half2.shape[0] != featA_half2.shape[0]:
            featA_half2 = featA_half2[:featV_half2.shape[0]]
            
        return self.listGames[index], featV_half1, featA_half1, featV_half2, featA_half2, label_half1, label_half2

        
        

    def __len__(self):
        return len(self.listGames)
    
    
class OnlineSoccerNetFrames(Dataset):
    def __init__(self, path_frames = '/data-local/data1-hdd/axesparraguera/SoccerNetFrames', 
                 path_audio = '/data-local/data3-ssd/axesparraguera',  
                 path_labels = "/data-net/datasets/SoccerNetv2/ResNET_TF2",
                 path_store = '/data-local/data1-hdd/axesparraguera/SoccerNetFrames/trial',
                 features_audio = 'audio_embeddings_2fps.npy', 
                 split=["train"], framerate=2, chunk_size=5, framestride = 4, store = True):

        self.listGames = getListGames(split)
        self.chunk_size = chunk_size
        self.framestride = framestride
        self.framerate = framerate
        self.path_frames = path_frames
        self.path_store = path_store
        
        self.dict_event = EVENT_DICTIONARY_V2
        self.num_classes = 17
        self.labels="Labels-v2.json"

        logging.info("Pre-compute clips")
        
        self.game_frames = list()
        self.game_featsA = list()
        self.game_labels = list()
        stride = self.chunk_size
        
        if store:
            
            self.path_list = []
            self.frames_path = []
            z = 0
            for game in tqdm(self.listGames):
                z += 1
                if z == 2:
                    break
                if ('valid' in split) & (z >=50):
                    break
                
                feat_half1A = np.load(os.path.join(path_audio, game, "1_" + features_audio))
                feat_half1A = feat_half1A.reshape(-1, feat_half1A.shape[-1])
                feat_half2A = np.load(os.path.join(path_audio, game, "2_" + features_audio))
                feat_half2A = feat_half2A.reshape(-1, feat_half2A.shape[-1])
                
                i = 100000
                found1 = False
                found2 = False
                while i > 0:
                    ex1 = os.path.exists(os.path.join(self.path_frames, game, 'half1', 'frame ' + str(i) + '.jpg'))
                    ex2 = os.path.exists(os.path.join(self.path_frames, game, 'half2', 'frame ' + str(i) + '.jpg'))
                    
                    if (not found1) & ex1:
                        frames1 = i
                        found1 = True
                    if (not found2) & ex2:
                        frames2 = i
                        found2 = True
                    if found1 & found2:
                        break
                    i -= self.framestride
                    
    
                #Check same size Visual and Audio features

                #Visual frames bigger than audio features
                if math.ceil(frames1 // 25) * 2 > feat_half1A.shape[0]:
                    feat_half1A_aux = np.zeros((math.ceil(frames1 // 25) * 2, feat_half1A.shape[1]))
                    feat_half1A_aux[:feat_half1A.shape[0]] = feat_half1A
                    feat_half1A_aux[feat_half1A.shape[0]:] = feat_half1A[feat_half1A.shape[0]-1]
                    feat_half1A = feat_half1A_aux
                    
                if math.ceil(frames2 // 25) * 2 > feat_half2A.shape[0]:
                    feat_half2A_aux = np.zeros((math.ceil(frames2 // 25) * 2, feat_half2A.shape[1]))
                    feat_half2A_aux[:feat_half2A.shape[0]] = feat_half2A
                    feat_half2A_aux[feat_half2A.shape[0]:] = feat_half2A[feat_half2A.shape[0]-1]
                    feat_half2A = feat_half2A_aux
                
                #More audio than frames
                if math.ceil(frames1 // 25) * 2 < feat_half1A.shape[0]:
                    feat_half1A = feat_half1A[0:math.ceil(frames1 // 25) * 2, :]
                
                
                if math.ceil(frames2 // 25) * 2 < feat_half2A.shape[0]:
                    feat_half2A = feat_half2A[0:math.ceil(frames2 // 25) * 2, :]
      
                #Generate clips from features
                feat_half1A = feats2clip(torch.from_numpy(feat_half1A), stride=stride * self.framerate, clip_length=self.chunk_size*self.framerate)  
                feat_half2A = feats2clip(torch.from_numpy(feat_half2A), stride=stride * self.framerate, clip_length=self.chunk_size*self.framerate) 
    
                
                
                # Load labels
                labels = json.load(open(os.path.join(path_labels, game, self.labels)))
    
    
                label_half1 = np.zeros((feat_half1A.shape[0], self.num_classes+1))
                label_half1[:,0]=1 # those are BG classes
                label_half2 = np.zeros((feat_half2A.shape[0], self.num_classes+1))
                label_half2[:,0]=1 # those are BG classes
                
                for annotation in labels["annotations"]:
    
                    time = annotation["gameTime"]
                    event = annotation["label"]
    
                    half = int(time[0])
                    
    
                    minutes = int(time[-5:-3])
                    seconds = int(time[-2::])
                    frame = framerate * ( seconds + 60 * minutes ) 
    
                    if event not in self.dict_event:
                        continue
                    label = self.dict_event[event]
    
                    # if label outside temporal of view
                    if half == 1 and frame//stride>=label_half1.shape[0]:
                        continue
                    if half == 2 and frame//stride>=label_half2.shape[0]:
                        continue
                    a = frame // stride
                    
                    if half == 1:
                        if self.chunk_size >= stride:
                            for i in range(self.chunk_size // stride):
                                label_half1[max(a - self.chunk_size // stride + 1 + i, 0)][0] = 0 # not BG anymore
                                label_half1[max(a - self.chunk_size // stride + 1 + i, 0)][label+1] = 1
                            #label_half1[max(a - self.chunk_size//stride + 1, 0) : (a + 1)][0] = 0 # not BG anymore
                            
                        else:
                            a2 = (frame - self.chunk_size) // stride
                            if a != a2:
                                label_half1[a][0] = 0
                                label_half1[a][label+1] = 1
    
                    if half == 2:
                        if self.chunk_size >= stride:
                            for i in range(self.chunk_size // stride):
                                label_half2[max(a - self.chunk_size // stride + 1 + i, 0)][0] = 0 # not BG anymore
                                label_half2[max(a - self.chunk_size // stride + 1 + i, 0)][label+1] = 1 # that's my class
                                
                        else:
                            a2 = (frame - self.chunk_size) // stride
                            if a != a2:
                                label_half2[a][0] = 0
                                label_half2[a][label+1] = 1
                                
                #Store path
                path = os.path.join(self.path_store, game)
                exists = os.path.exists(path)
                if not exists:
                    os.makedirs(path)
                
                #STORE HALF 1 FILES
                self.frames_clip = self.chunk_size * 25 // self.framestride
                feat_half1A = feat_half1A.numpy()

                for i in range(feat_half1A.shape[0]):
                    np.save(path + '/half1_chunk' + str(i) + '_featuresA.npy', feat_half1A[i, :, :])
                    np.save(path + '/half1_chunk' + str(i) + '_labels.npy', label_half1[i, :])
                    self.path_list.append(path + '/half1_chunk' + str(i) + '_')
                    self.frames_path.append([os.path.join(self.path_frames, game, 'half1', 'frame ' + str(min(frames1, (i * 25 * stride) // 4 * 4 + j * 4)) + '.jpg') for j in range(self.frames_clip)])
                    #self.path_list2.append(os.path.join(self.path_frames, game, 'half1', 'frame ' ))
                                           
                #STORE HALF 2 FILES
                feat_half2A = feat_half2A.numpy()

                for i in range(feat_half2A.shape[0]):
                    np.save(path + '/half2_chunk' + str(i) + '_featuresA.npy', feat_half2A[i, :, :])
                    np.save(path + '/half2_chunk' + str(i) + '_labels.npy', label_half2[i, :])
                    self.path_list.append(path + '/half2_chunk' + str(i) + '_')
                    self.frames_path.append([os.path.join(self.path_frames, game, 'half2', 'frame ' + str(min(frames2, (i * 25 * stride) // 4 * 4 + j * 4)) + '.jpg') for j in range(self.frames_clip)])
                    #self.path_list2.append(os.path.join(self.path_frames, game, 'half2', 'frame ' ))
    
                        
            #Concatenate features
            with open(self.path_store + '/chunk_list.pkl', 'wb') as f:
                pickle.dump(self.path_list, f)
            with open(self.path_store + '/frames_path.pkl', 'wb') as f:
                pickle.dump(self.frames_path, f)

                
        else:
            with open(self.path_store + '/chunk_list.pkl', 'rb') as f:
                self.path_list = pickle.load(f)
            with open(self.path_store + '/frames_path.pkl', 'rb') as f:
                self.frames_path = pickle.load(f)


                
        self.frames_clip = self.chunk_size * 25 // self.framestride




    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            clip_feat (np.array): clip of features.
            clip_labels (np.array): clip of labels for the segmentation.
            clip_targets (np.array): clip of targets for the spotting.
        """
        path = self.path_list[index]
        frames_path = self.frames_path[index]
        #Most time expensive
        frames = [cv2.imread(path) for path in frames_path]



        return torch.tensor(np.array(frames)), torch.from_numpy(np.load(path + 'featuresA.npy')), np.load(path + 'labels.npy')


    def __len__(self):

        return(len(self.path_list))
