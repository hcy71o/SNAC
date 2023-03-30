# %matplotlib inline
import matplotlib.pyplot as plt
# import IPython.display as ipd

import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence, cleaned_text_to_sequence
from utils import load_wav_to_torch
from mel_processing import spectrogram_torch
from scipy.io.wavfile import write


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

hps = utils.get_hparams_from_file("./configs/vctk_base.json")

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model).cuda()
_ = net_g.eval()

_ = utils.load_checkpoint("./logs/vctk/G_200000.pth", net_g, None)

ref_path = "ref_wav/p335_149.wav"

stn_tst = get_text("Speaker conditional convolutional neural networks is applied.", hps)
with torch.no_grad():
    x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
    ref_audio, _ = load_wav_to_torch(ref_path)
    ref_audio_norm = ref_audio / 32768.0
    ref_audio_norm = ref_audio_norm.unsqueeze(0)
    spec = spectrogram_torch(ref_audio_norm, hps.data.filter_length, hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length, center=False)
    spec = torch.squeeze(spec, 0).cuda()
    
    audio = net_g.infer(x_tst, x_tst_lengths, spec.unsqueeze(0), noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
write('gen.wav',22050,(audio*32767.0).astype(np.int16))
# ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate, normalize=False))