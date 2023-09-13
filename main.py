import jukemirlib
import torch as t
import os
import csv

import torch.cuda
import torch
from tqdm import tqdm
global VQVAE, TOP_PRIOR, DEVICE, x_cond, y_cond
VQVAE = None
TOP_PRIOR = None
x_cond = None
y_cond = None
CACHE_DIR = "/home/marta/jku/jukebox_extractor/models/"
DEVICE = "cuda" if t.cuda.is_available() else "cpu"
# DEVICE = "cpu"
T = 8192
JUKEBOX_SAMPLE_RATE = 44100
# 1048576 found in original paper, last page
CTX_WINDOW_LENGTH = 1048576
CSV_OUT_PATH = "/home/marta/jku/jukebox_extractor/id_jukebox.csv"
# if on google cloud, this one is better
# REMOTE_PREFIX = "https://storage.googleapis.com/jukebox-weights/"

# for stability, original endpoint is this one
REMOTE_PREFIX = "https://openaipublic.azureedge.net/jukebox/models/5b/"

# header = ['id'] + [str(i).zfill(4) for i in range(reps[36].shape[0])]
header = ['id'] + [str(i).zfill(4) for i in range(4800)]

wav_dir = "/home/marta/jku/jukebox_extractor/wav/wav/"
with open(CSV_OUT_PATH, "w") as csv_out:
    writer = csv.writer(csv_out, delimiter='\t', quotechar='|')
    writer.writerow(header)
    for root, dirs, files in os.walk(wav_dir):
        for file in tqdm(files):
            id = file[:-4]
            fpath = os.path.join(root, file)
            audio = jukemirlib.load_audio(fpath)
            reps = jukemirlib.extract(audio, layers=[36], meanpool=True)[36].tolist()
            line = [id] + reps
            writer.writerow(line)
#%%c