# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 10:37:14 2022

@author: artur
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


def check_features(path_baidu = '/data-local/data3-ssd/axesparraguera', 
                   path_baidu2 = '/data-net/datasets/SoccerNetv2/Baidu_features', 
             path_audio = '/data-local/data3-ssd/axesparraguera',  
             path_audio2 = '/data-local/data1-hdd/axesparraguera/vggish',
             path_labels = "/data-net/datasets/SoccerNetv2/ResNET_TF2",
             features_baidu = 'baidu_soccer_embeddings_2fps.npy',
             features_baidu2 = 'baidu_soccer_embeddings.npy', 
             features_audio = 'audio_embeddings_2fps.npy', 
             features_audio2 = 'featA2.npy',
             split=["train"], framerate=2, chunk_size=20):

    listGames = getListGames(split)
    
    dict_event = EVENT_DICTIONARY_V2
    num_classes = 17
    labels="Labels-v2.json"

    logging.info("Pre-compute clips")


    stride = chunk_size * 2
    for game in tqdm(listGames):
        
        feat_half1V = np.load(os.path.join(path_baidu, game, "1_" + features_baidu))
        feat_half1V2 = np.load(os.path.join(path_baidu2, game, "1_" + features_baidu2))
        
        feat_half1A = np.load(os.path.join(path_audio, game, "1_" + features_audio))
        feat_half1A2 = np.load(os.path.join(path_audio, game, "1_" + features_audio2))
        
        print('Baidu features:')
        print(feat_half1V[0:10, :])
        print(feat_half1V2[0:10, :])
        print('Audio features')
        print(feat_half1A[0:10, :])
        print(feat_half1A2[0:10, :])
        print(asdf)