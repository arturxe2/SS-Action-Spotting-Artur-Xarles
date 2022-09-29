# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 12:23:03 2022

@author: artur
"""

from SoccerNet.Downloader import getListGames
import cv2
import os
from tqdm import tqdm

def vid2frame(vid_path = '/data-net/datasets/SoccerNetv2/videos_lowres',
              vid_name = '224p.mkv',
              store_path = '/data-local/data1-hdd/axesparraguera/SoccerNetFrames',
              frame_stride = 4,
              split = ['train', 'valid', 'test', 'challenge']):
    
    listGames = getListGames(split)
    n = 0
    for game in tqdm(listGames):
        n+=1
        if n > 3:
            break
        
        #HALF 1
        cam = cv2.VideoCapture(os.path.join(vid_path, game, "1_224p.mkv"))
        
        path = os.path.join(store_path, game, 'half1')
        
        if not os.path.exists(path):
            os.makedirs(path)
        
        currentframe = 0
        while(True):
              
            # reading from frame
            ret,frame = cam.read()
          
            if ret:
                # if video is still left continue creating images
                # save frame
                name = 'frame ' + str(currentframe) + '.jpg'
          
                # writing the extracted images
                cv2.imwrite(os.path.join(path, name), frame)
          
                # increasing counter so that it will
                # show how many frames are created
                currentframe += frame_stride # i.e. at 30 fps, this advances one second
                cam.set(1, currentframe)

            else:
                break
            
        
        #HALF 2
        cam = cv2.VideoCapture(os.path.join(vid_path, game, "2_224p.mkv"))
        
        path = os.path.join(store_path, game, 'half2')
        
        if not os.path.exists(path):
            os.makedirs(path)
        
        currentframe = 0
        while(True):
              
            # reading from frame
            ret,frame = cam.read()
          
            if ret:
                # if video is still left continue creating images
                # save frame
                name = 'frame ' + str(currentframe) + '.jpg'
          
                # writing the extracted images
                cv2.imwrite(os.path.join(path, name), frame)
          
                # increasing counter so that it will
                # show how many frames are created
                currentframe += frame_stride # i.e. at 30 fps, this advances one second
                cam.set(1, currentframe)

            else:
                break
            
vid2frame()