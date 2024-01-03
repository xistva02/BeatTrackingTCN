import numpy as np
import mir_eval
from data_generator import DataGenerator
from keras.models import load_model
from madmom.features.beats import DBNBeatTrackingProcessor
import pandas as pd
import glob
import os


def evaluate_beat_positions(reference_beats, estimated_beats, f_measure_threshold=0.07):
    reference_beats = mir_eval.io.load_events(reference_beats)
    estimated_beats = mir_eval.io.load_events(estimated_beats)
    reference_beats = mir_eval.beat.trim_beats(reference_beats)
    estimated_beats = mir_eval.beat.trim_beats(estimated_beats)
    scores = mir_eval.beat.evaluate(reference_beats, estimated_beats, f_measure_threshold=f_measure_threshold)
    return scores


if __name__ == '__main__':

    path_to_data = 'data/test_data.npy'
    test_data = np.load(path_to_data)
    model_paths = glob.glob('trained_models/*.h5')

    # this is experimental and was changed multiple times, check if the versions and trained models are correct
    pad = False  # False for bock models
    pad2 = True  # True for TCN models
    # True if 44.1 kHz version, otherwise False for 22.05 kHz
    model_versions = [False, True, False, True, False, True, False, True, False, True, False, True, False, True]
    pad_versions = [pad, pad, pad, pad, pad, pad, pad2, pad2, pad2, pad2, pad2, pad2, pad2, pad2]

    models = {}

    for i, model in enumerate(model_paths):
        if '_16.h5' in model:
            model_version = '16'
        elif '_22.h5' in model:
            model_version = '22'
        elif '_11.h5' in model:
            model_version = '11'
        elif '_5.h5' in model:
            model_version = '5'
        else:
            model_version = '44'

        if 'simple_tcn' in model:
            pad = False
        else:
            pad = True
        models[i] = {}
        models[i]['path'] = model
        models[i]['model_versions'] = model_version
        models[i]['pad'] = pad

    for i, m in enumerate(range(len(models))):

        model_path = models[m]['path']
        model_name = os.path.basename(model_path)[:-3]
        model_version = models[m]['model_versions']
        pad_version = models[m]['pad']

        # only if datasets are available
        if model_version == '16':
            dataset_path = './datasets/dataset_16'
        elif model_version == '22':
            dataset_path = './datasets/dataset_22'
        elif model_version == '11':
            dataset_path = './datasets/dataset_11'
        elif model_version == '5':
            dataset_path = './datasets/dataset_5'
        else:
            dataset_path = './datasets/dataset_44'
        print(f'Model name: {model_name}')
        print(f'Path to dataset: {dataset_path}')

        all_predictions = []
        proc = DBNBeatTrackingProcessor(fps=100)
        train_gen = DataGenerator(dataset_path=dataset_path, file_paths=test_data, train=False, shuffle=False,
                                  pad=pad_version)

        # all_gt = []
        # for data in test_data:
        #     print(f'data/NEW_ANOTACE/{data[:-4]}.txt')
        #     gt = np.loadtxt(f'data/NEW_ANOTACE/{data[:-4]}.txt').tolist()
        #     all_gt.append(gt)
        # np.save('eval_data/models/gt.npy', all_gt)

        if not os.path.exists(f'eval_data/models_new/{model_name}_pred.npy'):
            for train_data in train_gen:
                model = load_model(model_path)
                x = train_data[0]
                beats = train_data[1]
                act = model.predict(x)
                pred = proc(act.reshape(act.shape[1], ))
                # print(f'Prediction: ' + str(pred))
                all_predictions.append(pred)
            np.save(f'eval_data/models_new/{model_name}_pred.npy', all_predictions)
            estimated_beats_all = np.load(f'eval_data/models_new/{model_name}_pred.npy', allow_pickle=True)
        else:
            estimated_beats_all = np.load(f'eval_data/models_new/{model_name}_pred.npy', allow_pickle=True)

        reference_beats_all = np.load('eval_data/models_new/gt.npy', allow_pickle=True)
        all_scores = []
        for estimated_beats, reference_beats in zip(estimated_beats_all, reference_beats_all):
            # estimated_beats = np.array(re.split("\s+", estimated_beats.replace('[','').replace(']','')), dtype=float)
            reference_beats = np.array(reference_beats)
            reference_beats = mir_eval.beat.trim_beats(reference_beats)
            estimated_beats = mir_eval.beat.trim_beats(estimated_beats)
            scores = mir_eval.beat.evaluate(reference_beats, estimated_beats, f_measure_threshold=0.07)
            all_scores.append(scores)

        df = pd.DataFrame(all_scores)
        df_mean = df.mean(axis=0)
        xlsx_file = f'eval_data/results_new/results_{model_name}.xlsx'
        df_mean.to_excel(xlsx_file, index=True)
        df_modelName = [f'{model_name}']
        df_out = pd.DataFrame(df_mean.copy())
        df_out.columns = [model_name]

        print(df_mean.head())

        if 'df_output' in locals():
            df_output = pd.concat([df_output, df_out], axis=1)
        else:
            df_output = df_out.copy()

    xlsx_final_file = f'data/eval_data/_results_all.xlsx'
    df_output.to_excel(xlsx_final_file, index=True)
