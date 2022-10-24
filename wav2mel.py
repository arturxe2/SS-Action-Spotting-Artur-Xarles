from SoccerNet.Downloader import getListGames
import cv2
import os
from tqdm import tqdm
import moviepy.editor as mp
import librosa
from scipy.io import wavfile
import numpy as np

def vid2mel(wav_path = '/data-local/data1-ssd/axesparraguera/SoccerNetAudio',
            split = ['train', 'valid', 'test', 'challenge'], feat_sec = 100, n_mels = 128, eps=1e-05):
    
    listGames = getListGames(split)

    for game in tqdm(listGames):
        path = os.path.join(wav_path, game)
        if not os.path.exists(path):
            os.makedirs(path)

        #HALF 1
        print('Doing half 1...')

        ex1 = os.path.exists(os.path.join(path, '1_224p.wav'))

        if ex1:
            samplerate, data = wavfile.read(os.path.join(path, '1_224p.wav'))
            data, samplerate = librosa.load(os.path.join(path, '1_224p.wav'), sr = samplerate)
            mel_spect = librosa.feature.melspectrogram(y=data, sr=samplerate, hop_length= samplerate // feat_sec, window='hann', n_mels=n_mels)
            mel_shape = mel_spect.shape
            print(mel_shape)
            if (mel_spect == 0).mean() == 1:
                mel_spect += eps
            np.save(os.path.join(path, 'audio1.npy'), mel_spect)
        else:
            print(game)
            print('not wav file for this half')


        #HALF 2
        print('Doing half 2...')

        ex1 = os.path.exists(os.path.join(path, '2_224p.wav'))

        if ex1:
            samplerate, data = wavfile.read(os.path.join(path, '2_224p.wav'))
            data, samplerate = librosa.load(os.path.join(path, '2_224p.wav'), sr = samplerate)
            mel_spect = librosa.feature.melspectrogram(y=data, sr=samplerate, hop_length= samplerate // feat_sec, window='hann', n_mels=n_mels)
            print(mel_spect.shape)
            if (mel_spect == 0).mean() == 1:
                mel_spect += eps
            np.save(os.path.join(path, 'audio2.npy'), mel_spect)
        else:
            print(game)
            print('not wav file for this half')


            
vid2mel(split = ['challenge'])