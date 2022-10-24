"""
Created on Thu Sep 29 12:23:03 2022

@author: artur
"""

from SoccerNet.Downloader import getListGames
import cv2
import os
from tqdm import tqdm
import moviepy.editor as mp
import librosa
from scipy.io import wavfile
import numpy as np

def vid2mel(vid_path = '/data-net/datasets/SoccerNetv2/videos_lowres',
            vid_name = '224p.mkv',
            store_path = '/data-local/data1-ssd/axesparraguera/SoccerNetAudio',
            frame_stride = 4,
            split = ['train', 'valid', 'test', 'challenge']):
    
    listGames = getListGames(split)

    for game in tqdm(listGames):
        path = os.path.join(store_path, game)
        if not os.path.exists(path):
            os.makedirs(path)

        #HALF 1
        print('Doing half 1...')

        try:
            ex1 = os.path.exists(os.path.join(path, "1_224p.wav"))
            ex2 = os.path.exists(os.path.join(path, "audio1.npy"))

            if ex2:
                mel_spect = np.load(os.path.join(path, 'audio1.npy'))
                print('audio features already extracted from this game and half1')
                print(mel_spect.shape)
                if mel_spect.shape[1] <= 4000:
                    print('1-NOT AUDIO FOR ALL THE VIDEO')
                    os.remove(os.path.join(path, 'audio1.npy'))
                if mel_spect.shape[1] <= 100:
                    print('2-REALLY NOT AUDIO FOR ALL THE VIDEO')
            
            elif ex1:
                print('not correct audio for this game')
                print(game)
                print('Retrying for the game')
                samplerate, data = wavfile.read(os.path.join(path, '1_224p.wav'))
                data, samplerate = librosa.load(os.path.join(path, '1_224p.wav'), sr = samplerate)
                mel_spect = librosa.feature.melspectrogram(y=data, sr=samplerate, hop_length= samplerate // 2, window='hann', n_mels=256)
                if mel_spect.shape[1] >= 4000:
                    np.save(os.path.join(path, 'audio1.npy'), mel_spect)
                    print('Correctly saved the file')
                    print(mel_spect.shape)

            else:
                my_clip = mp.VideoFileClip(os.path.join(vid_path, game, "1_224p.mkv"))
                my_clip.audio.write_audiofile(os.path.join(path, "1_224p.wav"))
                samplerate, data = wavfile.read(os.path.join(path, '1_224p.wav'))
                data, samplerate = librosa.load(os.path.join(path, '1_224p.wav'), sr = samplerate)
                mel_spect = librosa.feature.melspectrogram(y=data, sr=samplerate, hop_length= samplerate // 2, window='hann', n_mels=256)
                np.save(os.path.join(path, 'audio1.npy'), mel_spect)
                if mel_spect.shape[1] <= 4000:
                    print('1-NOT AUDIO FOR ALL THE VIDEO')
                    os.remove(os.path.join(path, 'audio1.npy'))
                if mel_spect.shape[1] <= 100:
                    print('2-REALLY NOT AUDIO FOR ALL THE VIDEO')
        except:
            print('Problems half 1 with game:')
            print(game)

        #HALF 2
        print('Doing half 2...')

        try:
            ex1 = os.path.exists(os.path.join(path, "2_224p.wav"))
            ex2 = os.path.exists(os.path.join(path, "audio2.npy"))

            if ex2:
                mel_spect = np.load(os.path.join(path, 'audio2.npy'))
                print('audio features already extracted from this game and half2')
                print(mel_spect.shape)
                if mel_spect.shape[1] <= 4000:
                    print('1-NOT AUDIO FOR ALL THE VIDEO')
                    os.remove(os.path.join(path, 'audio2.npy'))
                if mel_spect.shape[1] <= 100:
                    print('2-REALLY NOT AUDIO FOR ALL THE VIDEO')
            
            elif ex1:
                print('not correct audio for this game')
                print(game)
                print('Retrying for the game')
                samplerate, data = wavfile.read(os.path.join(path, '2_224p.wav'))
                data, samplerate = librosa.load(os.path.join(path, '2_224p.wav'), sr = samplerate)
                mel_spect = librosa.feature.melspectrogram(y=data, sr=samplerate, hop_length= samplerate // 2, window='hann', n_mels=256)
                if mel_spect.shape[1] >= 4000:
                    np.save(os.path.join(path, 'audio2.npy'), mel_spect)
                    print('Correctly saved the file')
                    print(mel_spect.shape)

            else:
                my_clip = mp.VideoFileClip(os.path.join(vid_path, game, "2_224p.mkv"))
                my_clip.audio.write_audiofile(os.path.join(path, "2_224p.wav"))
                samplerate, data = wavfile.read(os.path.join(path, '2_224p.wav'))
                data, samplerate = librosa.load(os.path.join(path, '2_224p.wav'), sr = samplerate)
                mel_spect = librosa.feature.melspectrogram(y=data, sr=samplerate, hop_length= samplerate // 2, window='hann', n_mels=256)
                np.save(os.path.join(path, 'audio2.npy'), mel_spect)
                if mel_spect.shape[1] <= 4000:
                    print('1-NOT AUDIO FOR ALL THE VIDEO')
                    os.remove(os.path.join(path, 'audio2.npy'))
                if mel_spect.shape[1] <= 100:
                    print('2-REALLY NOT AUDIO FOR ALL THE VIDEO')
        except:
            print('Problems half 2 with game:')
            print(game)

            
vid2mel(split = ['valid'])