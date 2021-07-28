# prepro.py
# author: Diego Magdaleno

# Python 3.7
# Tensorflow 2.4.0


import os
import tqdm
import numpy as np
from utils import load_spectrograms
from data_load import load_data


# Load data.
fpaths, _, _ = load_data() # list

for fpath in tqdm.tqdm(fpaths):
	fname, mel, mag = load_spectrograms(fpath)
	if not os.path.exists("mels"):
		os.mkdir("mels")
	if not os.path.exists("mags"):
		os.mkdir("mags")
	
	np.save("mels/{}".format(fname.replace("wav", "npy")), mel)
	np.save("mags/{}".format(fname.replace("wav", "npy")), mag)