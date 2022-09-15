# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 09:25:43 2022

@author: arturxe
"""

from SoccerNet.Downloader import getListGames
def LinearInterpolation(path = '/data-net/datasets/SoccerNetv2/Baidu_features', 
                        input_name = 'baidu_soccer_embeddings.npy', 
                        output_name = 'baidu_soccer_embeddings_2fps.npy', 
                        split = ['train', 'valid', 'test']):
    
    listGames = getListGames(split)
    print(listGames)
    
LinearInterpolation()