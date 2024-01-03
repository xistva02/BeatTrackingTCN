# BeatTrackingTCN

## Introduction:

This repository contains supplementary material for the paper "Beat Tracking: Is 44.1 kHz Really Needed?" from the EEICT 2023 student conference, where it won 1st place in the Audio, Speech and Language Processing category. The paper is available on the conference [website](https://www.eeict.cz/download) or here as '_Istvanek_Miklanek_EEICT2023.pdf'.

Keep in mind that some of the code is taken from [[1]](https://tempobeatdownbeat.github.io/tutorial/intro.html) and we strongly encourage you to check it first.

This study retrains some of the state-of-the-art Temporal Convolutional Network (TCN) models for the beat tracking task with various sampling rates and evaluates the training phase and detection accuracy on well-known data. It also reimplements some models from the [madmom](https://github.com/CPJKU/madmom) module.
It provides a bit of insight (most of it is expected and obvious) into the models. We share code and all trained models mentioned in the article plus additional ones (such as 16-kHz models and simple_tcn trained with 50 fps temporal resolution).
We further used the 'simple_tcn_dp_skip_dilations_22_fps50' model for the synchronization pipeline in [[2]](https://ieeexplore.ieee.org/document/10335098), [[3]](https://ismir2023program.ismir.net/lbd_322.html), and the [MemoVision software](https://github.com/stepanmk/memovision).

## Run:

First, install the requirements:

```
pip install -r requirements.txt
```

To reproduce the results, you need audio data (_wav_ files from Ballroom, Hainsworth, GTZAN, SMC, Beatles) in 'data/WAVS' and corresponding annotations in 'data/ANOTACE'. Then, preprocess the dataset:

```
python dataset.py
```

and run the training:

```
python train_models.py
```

## Additional information:

If you have all audio and annotation files (see the paper for details), you should be able to reproduce the results (and evaluate them using evaluation.py) with the scripts provided. 

If you have any questions, feel free to contact me at matej.istvanek@vut.cz.

## Acknowledgment:

If you wish to use information, code, or models from the article and this repository, please, cite the original paper:

```
@inproceedings{Istvanek_EEICT_Beat_2023,
    author      = {Matěj Ištvánek and Štěpán Miklánek},
    title       = {Beat Tracking: Is 44.1 kHz Really Needed?},
    booktitle   = {Proceedings of the conference STUDENT EEICT 2023 Selected Papers},
    address     = {Brno, Czech Republic},
    year        = {2023},
    pages       = {227--231},
    isbn        = {978-80-214-6154-3}}
```

## Sources:

[1] M. E. P. Davies, S. Böck, and M. Fuentes, _Tempo, Beat and Downbeat Estimation_. https://tempobeatdownbeat.github.io/tutorial/intro.html, November 2021, (accessed on 27 March 2023).

[2] M. Ištvánek, Š. Miklánek, K. H. Mühlová, L. Spurný, and Z. Smékal, “Application of computational methods for comparative music analysis,” in _2023 4th International Symposium on the Internet of Sounds_, 2023, pp. 1-6. [Online]. Available: https://ieeexplore.ieee.org/document/10335098

[3] M. Ištvánek and Š. Miklánek, “Memovision: a tool for feature selection and visualization of performance data,” in _Extended Abstracts for the Late-Breaking Demo Session of the 24th International Society for Music Information Retrieval Conference (ISMIR)_, Milan, Italy, 2023, p. 3. [Online]. Available: https://ismir2023program.ismir.net/lbd_322.html
