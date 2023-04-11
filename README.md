# SNAC : Speaker-normalized Affine Coupling Layer in Flow-based Architecture for Zero-Shot Multi-Speaker Text-to-Speech

This is the unofficial pytorch implementation of [SNAC](https://arxiv.org/abs/2211.16866)
We built up our codes based on [VITS](https://github.com/jaywalnut310/vits)

0. [VCTK]((https://paperswithcode.com/dataset/vctk)) dataset is used.
1. [LibriTTS]((https://research.google/tools/datasets/libri-tts/)) dataset (train-clean-100 and train-clean-360) is also supported.
2. This is the implementation of ```Proposed + REF + FLOW``` in the paper.

|Text Encoder|Duration Predictor|Flow|Vocoder|
|-----|-----|-----|-----|
|None|Input addition|SNAC|None|

## Prerequisites
0. Clone this repository.
0. Install python requirements. Please refer [requirements.txt](requirements.txt)
    1. You may need to install espeak first: `apt-get install espeak`
0. Download datasets
    1. Download and extract the VCTK dataset, and downsample wav files to 22050 Hz. Then rename or create a link to the dataset folder: `ln -s /path/to/VCTK-Corpus/downsampled_wavs DUMMY3`
    1. For LibriTTS dataset, downsample wav files to 22050 Hz and link to the dataset folder: `ln -s /path/to/LibriTTS DUMMY2`
0. Build Monotonic Alignment Search and run preprocessing if you use your own datasets.
```sh
# Cython-version Monotonoic Alignment Search
cd monotonic_align
python setup.py build_ext --inplace
```
## Training Exmaple
```sh
python train.py -c configs/vctk_base.json -m vctk_base
```

## Inference Example
See [inference.ipynb](inference.ipynb)
