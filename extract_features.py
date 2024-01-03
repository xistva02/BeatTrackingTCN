from madmom.processors import SequentialProcessor
from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.audio.spectrogram import FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor

import numpy as np


FPS = 100
FFT_SIZE = 2048
NUM_BANDS = 12


class PreProcessor(SequentialProcessor):
    def __init__(self, frame_size=FFT_SIZE, num_bands=NUM_BANDS, log=np.log, add=1e-6, fps=FPS):
        # resample to a fixed sample rate in order to get always the same number of filter bins
        sig = SignalProcessor(num_channels=1, sample_rate=44100)
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


# if __name__ == '__main__':
#     # example of usage
#     pre_processor = PreProcessor()
#     audio = librosa.load('002_Tokyo_2.wav', sr=44100, mono=True)
#     s = madmom.audio.Signal(audio[0], sample_rate=audio[1])
#     x = pre_processor(s)
