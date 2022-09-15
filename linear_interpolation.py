# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 09:25:43 2022

@author: arturxe
"""

from SoccerNet.Downloader import getListGames
from tqdm import tqdm
import os
import numpy as np


def LinearInterpolation(path = '/data-net/datasets/SoccerNetv2/Baidu_features', 
                        output_path = "/data-local/data3-ssd/axesparraguera",
                        input_name = 'baidu_soccer_embeddings.npy', 
                        output_name = 'baidu_soccer_embeddings_2fps.npy', 
                        split = ['train', 'valid', 'test']):
    
    #Get list of games (also directories)
    listGames = getListGames(split)
    
    for game in tqdm(listGames):
        #Read half 1 and 2 features
        feat_half1 = np.load(os.path.join(path, game, '1_' + input_name))
        feat_half2 = np.load(os.path.join(path, game, '2_' + input_name))
        
        #Half 1 linear interpolation
        feat_half1_2fps = []
        for i in range(len(feat_half1) - 1):
            feat_aux = (feat_half1[i, :] + feat_half1[i+1, :]) / 2
            feat_half1_2fps.append(feat_half1[i, :])
            feat_half1_2fps.append(feat_aux)
            
        feat_half1_2fps.append(feat_half1[-1, :])
        feat_half1_2fps = np.array(feat_half1_2fps)
        
        #Half 2 linear interpolation
        feat_half2_2fps = []
        for i in range(len(feat_half2) - 1):
            feat_aux = (feat_half2[i, :] + feat_half2[i+1, :]) / 2
            feat_half2_2fps.append(feat_half2[i, :])
            feat_half2_2fps.append(feat_aux)
            
        feat_half2_2fps.append(feat_half2[-1, :])
        feat_half2_2fps = np.array(feat_half2_2fps)
        
        #Check if output path exists
        out_path = os.path.join(output_path, game)
        exists = os.path.exists(out_path)
        if not exists:
            os.makedirs(out_path)
        
        #Store new features at 2fps
        np.save(os.path.join(out_path, '1_' + output_name), feat_half1_2fps)
        np.save(os.path.join(out_path, '2_' + output_name), feat_half2_2fps)
        
        
    print('Saved 2fps features')
    
LinearInterpolation()