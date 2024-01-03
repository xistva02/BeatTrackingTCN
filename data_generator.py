import tensorflow as tf
import pickle
import numpy as np
import madmom
import random
import os


def randomize_annotations(beats):
    for i in range(len(beats)):
        beats[i] = beats[i] + random.uniform(-0.07, 0.07)
    return beats


def widen_beat_targets(beats):
    for i in range(len(beats)):
        if i > 0:
            if beats[i] == 1.:
                beats[i - 1] = 0.5
        if i < len(beats) - 1:
            if beats[i] == 1.:
                beats[i + 1] = 0.5
    return beats


def cnn_pad(data, pad_frames):
    """Pad the data by repeating the first and last frame N times."""
    pad_start = np.repeat(data[:1], pad_frames, axis=0)
    pad_stop = np.repeat(data[-1:], pad_frames, axis=0)
    return np.concatenate((pad_start, data, pad_stop))


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataset_path, file_paths, train=True, shuffle=True, pad=True):
        self.indexes = np.arange(len(file_paths))
        self.dataset_path = dataset_path
        self.file_paths = file_paths
        self.shuffle = shuffle
        self.batch_size = 1
        self.fps = 100
        self.train = train
        self.pad = pad
        self.on_epoch_end()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        idx = self.indexes[index]
        file_path = self.file_paths[idx]
        with open(os.path.join(self.dataset_path, file_path), 'rb') as f:
            track = pickle.load(f)
        x = track['x']
        beats = track['beats']
        if self.train:
            beats = randomize_annotations(beats)
        beats = madmom.utils.quantize_events(beats, fps=self.fps, length=len(x))
        beats = widen_beat_targets(beats)
        if self.pad:
            x = cnn_pad(x, 2)
        return x[np.newaxis, ..., np.newaxis], beats[np.newaxis, ..., np.newaxis]

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.file_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)
