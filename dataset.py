import librosa
import numpy as np
import os
import random
import madmom
import pickle

from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.audio.spectrogram import FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor
from madmom.processors import SequentialProcessor
from madmom.audio.signal import SignalProcessor, FramedSignalProcessor


# FPS = 100
# FFT_SIZE = 2048
# NUM_BANDS = 12

class PreProcessor(SequentialProcessor):
    def __init__(self, frame_size=None, num_bands=None, log=np.log, add=1e-6, sr=None, fps=None):
        # resample to a fixed sample rate in order to get always the same number of filter bins
        sig = SignalProcessor(num_channels=1, sample_rate=sr)
        # split audio signal in overlapping frames
        frames = FramedSignalProcessor(frame_size=frame_size, fps=fps)
        # compute STFT
        stft = ShortTimeFourierTransformProcessor()
        # filter the magnitudes
        filt = FilteredSpectrogramProcessor(num_bands=num_bands)
        # scale them logarithmically
        spec = LogarithmicSpectrogramProcessor(log=log, add=add)
        # instantiate a SequentialProcessor
        super(PreProcessor, self).__init__((sig, frames, stft, filt, spec, np.array))
        # safe fps as attribute (needed for quantization of events)
        self.fps = fps


if __name__ == '__main__':

    random.seed(10)

    path_to_audio = 'data/WAVS'
    path_to_annot = 'data/ANOTACE'

    audio_files = os.listdir(path_to_audio)
    annot_files = os.listdir(path_to_annot)

    sr_names = ['5', '11',  '22', '44']
    srs = [5500, 11000, 22050, 44100]
    frame_sizes = [256, 512, 1024, 2048]

    for i, (sr, frame_size, sr_name) in enumerate(zip(srs, frame_sizes, sr_names)):
        print(f'Processing dataset --datasets/dataset_{sr_name}-- with frame_size {frame_size} and sr {sr}')

        proc = PreProcessor(sr=sr, frame_size=frame_size, num_bands=12, fps=100)
        for k, audio_file in enumerate(audio_files):
            track = {}
            audio = librosa.load(os.path.join(path_to_audio, audio_file), sr=sr)
            signal = madmom.audio.Signal(*audio)
            # compute spectrogram
            logspec = proc(signal)
            # load beat annotations
            beats = np.loadtxt(os.path.join(path_to_annot, audio_file[:-4] + '.txt'))
            # insert track to dictionary
            track['x'] = logspec
            track['beats'] = beats

            with open(f'datasets/dataset_{sr_name}/{audio_file[:-4]}.pkl', 'wb') as f:
                pickle.dump(track, f)
            print('track', k + 1, 'out of', len(audio_files))
        print(f'Input shape: {np.shape(logspec)}')


## Additional info
# Bock 2020
# Training: SMC, Ballroom, Hainsworth, Beatles
# ---Missing: HJDB, Simac, Cuidado---
# Testing: GTZAN, ACM Mirum, GiantSteps

# Steinmetz 2021
# Training: Ballroom, Hainsworth, Beatles
# ---Missing: RWC Popular---
# Testing: GTZAN, SMC

# Bock splits
# Ballroom

# ballroom_splits = 'D:/ISMIR2020/splits/ballroom_8-fold_cv_dancestyle.folds'
# with open(ballroom_splits) as file:
#     br_folds = []
#     for line in file:
#         fname, fold = line.split('\t')
#         fname = fname.replace('ballroom', 'BRD')
#         fold.strip('\n')
#         br_folds.append([fname, int(fold)])
#
# # SMC
# smc_splits = 'D:/ISMIR2020/splits/smc_8-fold_cv_random.folds'
# with open(smc_splits) as file:
#     smc_folds = []
#     for line in file:
#         fname, fold = line.split('\t')
#         fname = fname.replace('smc', 'SMC')
#         fold.strip('\n')
#         smc_folds.append([fname, int(fold)])
#
# # Hainsworth
# hainsworth_splits = 'D:/ISMIR2020/splits/hainsworth_8-fold_cv_genre.folds'
# with open(hainsworth_splits) as file:
#     hainsworth_folds = []
#     for line in file:
#         fname, fold = line.split('\t')
#         fname = fname.replace('hainsworth', 'HW')
#         fold.strip('\n')
#         hainsworth_folds.append([fname, int(fold)])
