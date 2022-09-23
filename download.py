# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 12:56:16 2022

@author: artur
"""

import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader
mySoccerNetDownloader=SoccerNetDownloader(LocalDirectory="/data-net/datasets/SoccerNetv2/videos_lowres")
mySoccerNetDownloader.password = 's0cc3rn3t'
mySoccerNetDownloader.downloadGames(files=["1_224p.mkv", "2_224p.mkv"], split=["train","valid","test","challenge"])