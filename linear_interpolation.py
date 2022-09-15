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
                        input_name = 'baidu_soccer_embeddings.npy', 
                        output_name = 'baidu_soccer_embeddings_2fps.npy', 
                        split = ['train', 'valid', 'test']):
    
    listGames = getListGames(split)
    
    for game in tqdm(listGames):
        feat_half1 = np.load(os.path.join(path, game, "1_" + input_name))
        print(feat_half1.shape)
        
    print(listGames)
    
LinearInterpolation()