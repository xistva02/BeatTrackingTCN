from data_generator import DataGenerator
from sklearn.model_selection import train_test_split
from simple_tcn_model import create_simple_tcn
from madmom_model import create_2019_model
from bock_model_2020 import create_2020_model
from keras.callbacks import TensorBoard, CSVLogger
import os
import tensorflow as tf
import argparse
import numpy as np
import time

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def train(args):
    """ config """
    # model type
    model_type = args.model_type
    # version of residual block for simple TCN
    version = args.version
    # prediction from concat skips or from the TCN
    skip_output = args.skip_output
    # whether to load the madmom weights for the 2019 model
    load_weights = args.load_weights
    # if its trained on 44.1 kHz, 22.05 kHz, 11 kHz, or 5.5 kHz
    version_sr = args.version_sr

    dataset_path = f'./datasets/dataset_{version_sr}'
    files = os.listdir(dataset_path)

    # input changes with the sampling rate
    if version_sr == '16' or version_sr == '22':
        input_shape_size = 74
    elif version_sr == '11':
        input_shape_size = 62
    elif version_sr == '5':
        input_shape_size = 50
    else:
        input_shape_size = 81

    print(f'Model version: {model_type} and sr: {version_sr}')
    t0 = time.monotonic()

    if model_type == 'simple_tcn':
        model = create_simple_tcn(input_shape=(None, input_shape_size, 1), skip_output=skip_output, version=version)
        model_type += '_' + version
        if skip_output:
            model_type += '_skip'
            model_type += '_dilations'
        pad = False

    elif model_type == 'bock_2019':
        model = create_2019_model(input_shape=(None, input_shape_size, 1), load_weights=load_weights)
        if load_weights:
            model_type += '_madmom_weights'
        pad = True

    elif model_type == 'bock_2020':
        model = create_2020_model(input_shape=(None, input_shape_size, 1))
        pad = True

    files_train, files_rest = train_test_split(files, test_size=0.2, random_state=42)
    files_val, files_test = train_test_split(files_rest, test_size=0.5, random_state=42)
    # np.save(f'eval_data/{model_type}_sr{version_sr}_testData.npy', files_test)

    train_gen = DataGenerator(dataset_path=dataset_path, file_paths=files_train, train=True, shuffle=True, pad=pad)
    val_gen = DataGenerator(dataset_path=dataset_path, file_paths=files_val, train=False, shuffle=False, pad=pad)

    # learning_rate = 0.005
    # clipnorm = 0.5
    # optimizer = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate, clipnorm=clipnorm)
    # optimizer = tfa.optimizers.Lookahead(optimizer, sync_period=5)

    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer,
                  loss=['binary_crossentropy'])

    csv_path = f'./data/new_csv_logs/{model_type}_{version_sr}.csv'

    if version_sr:
        tensorboard_path = f'./tensorboard/{model_type}_{version_sr}'
    else:
        tensorboard_path = f'./tensorboard/{model_type}'
        print("...not using 'version_sr'...")

    tensorboard = TensorBoard(log_dir=tensorboard_path,
                              profile_batch=0)
    csv_logger = CSVLogger(filename=csv_path)

    # rest of the callbacks
    # model checkpointing
    if version_sr:
        mc = tf.keras.callbacks.ModelCheckpoint(f'./trained_models/{model_type}_{version_sr}.h5',
                                                monitor='val_loss',
                                                save_best_only=True)
    else:
        mc = tf.keras.callbacks.ModelCheckpoint(f'./trained_models/{model_type}.h5',
                                                monitor='val_loss',
                                                save_best_only=True)
    # learn rate scheduler
    lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=10,
        verbose=1,
        mode='auto',
        min_delta=1e-3,
        cooldown=0,
        min_lr=1e-7)

    # early stopping
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=20, verbose=1)

    # save the characteristics of the model
    # plot_model(model, to_file=f'figs_models/model_{model_type}_{version_sr}.pdf', show_shapes=True, show_layer_names=True)
    # np.save(f'models_summary/model_{model_type}_{version_sr}.npy', model.summary())

    # fit the model
    model.fit(train_gen,
              validation_data=val_gen,
              epochs=150,
              verbose=1,
              callbacks=[tensorboard, csv_logger, mc, lr, es])

    t1 = time.monotonic()
    result_time = t1 - t0
    print(f'Process time: {result_time} seconds')
    np.savetxt(f'data/computational_time/time_{model_type}_{version_sr}.txt', [round(result_time, 2)], fmt='%1.4f')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='beat tracking traning')
    parser.add_argument('--model_type', type=str, default='')
    parser.add_argument('--version', type=str, default='')
    parser.add_argument('--version_sr', type=str, default='')
    parser.add_argument('--skip_output', type=int, default=1)
    parser.add_argument('--load_weights', type=int, default=1)
    args_, _ = parser.parse_known_args()
    train(args_)
