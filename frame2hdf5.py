from SoccerNet.Downloader import getListGames
from tqdm import tqdm
import numpy as np
import logging
import os
import json
import pickle
import skimage.io as io
import math
import cv2
import time
import h5py

def frame2hdf5(frame_path = '/data-local/data1-ssd/axesparraguera/SoccerNetFrames',
                framestride = 4, split = ['train']):

    listGames = getListGames(split)

    # Create hdf5 file
    f = h5py.File(os.path.join(frame_path, "SoccerNetFrames.hdf5"), 'w')


    z = 0
    for game in tqdm(listGames):
        z += 1
        if z == 2:
            break

        i = 100000
        found1 = False
        found2 = False
        while i > 0:
            ex1 = os.path.exists(os.path.join(frame_path, game, 'half1', 'frame ' + str(i) + '.jpg'))
            ex2 = os.path.exists(os.path.join(frame_path, game, 'half2', 'frame ' + str(i) + '.jpg'))
                    
            if (not found1) & ex1:
                frames1 = i
                found1 = True
            if (not found2) & ex2:
                frames2 = i
                found2 = True
            if found1 & found2:
                break
            i -= framestride

        n_frames1 = (frames1 // framestride) + 1
        n_frames2 = (frames2 // framestride) + 1
        n_frames1 = 1000

        dset = f.create_dataset(game + '1', (n_frames1, 224, 398, 3), dtype='uint8', compression = 9)
        images = []
        for i in tqdm(range(n_frames1)):
            images.append(cv2.imread(os.path.join(frame_path, game, 'half1', 'frame ' + str(i * framestride) + '.jpg')))
        dset[:, :, :, :] = images

        #dset = f.create_dataset(game + '2', (n_frames2, 224, 398, 3), dtype='uint8')
        #for i in tqdm(range(n_frames1)):
        #    dset[i, :, :, :] = cv2.imread(os.path.join(frame_path, game, 'half2', 'frame ' + str(i * framestride) + '.jpg'))
    
    print('Done')

#frame2hdf5()
frame_path = '/data-local/data1-ssd/axesparraguera/SoccerNetFrames'
framestride = 4
split = ['train']
listGames = getListGames(split)


game = listGames[0]

time0 = time.time()
f = h5py.File(os.path.join(frame_path, "SoccerNetFrames.hdf5"), 'r')       
a = f[game + '1'][0:224:2]
time1 = time.time()
print(time1 - time0)