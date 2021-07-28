# hyperparams.py
# author: Diego Magdaleno
# Copy of the hyperparams.py file in the Tensorflow implementation of
# dc_tts.
# Python 3.7
# Tensorflow 2.4.0

class Hyperparams:
	'''Hyper parameters'''
	# pipeline
	prepro = True  # if True, run `python prepro.py` first before running `python train.py`.
	
	# signal processing
	sr = 22050  # Sampling rate.
	n_fft = 2048  # fft points (samples)
	frame_shift = 0.0125  # seconds
	frame_length = 0.05  # seconds
	hop_length = int(sr * frame_shift)  # samples. =276.
	win_length = int(sr * frame_length)  # samples. =1102.
	n_mels = 80  # Number of Mel banks to generate
	#power = 1.5  # Exponent for amplifying the predicted magnitude
	power = 1.2  # Exponent for amplifying the predicted magnitude
	n_iter = 50  # Number of inversion iterations
	preemphasis = .97 # or None
	max_db = 100
	ref_db = 20

	# Model
	#r = 4 # Reduction factor. Do not change this.
	r = 5 # Reduction factor. Do not change this. Paper => 2, 3, 5
	#dropout_rate = 0.05
	dropout_rate = 0.5
	embed_size = 256 # alias = E
	encoder_num_banks = 16
	decoder_num_banks = 8
	num_highwaynet_blocks = 4

	# data
	data = "./data/private/voice/LJSpeech-1.1"
	# data = "./data/private/voice/kate"
	test_data = 'harvard_sentences.txt'
	vocab = "PE abcdefghijklmnopqrstuvwxyz'.?" # P: Padding, E: EOS.
	max_N = 180 # Maximum number of characters.
	max_T = 210 # Maximum number of mel frames.
	max_duration = 10.0

	# training scheme
	lr = 0.001 # Initial learning rate.
	logdir = "logdir/LJ01"
	sampledir = 'samples'
	#B = 32 # batch size
	#B = 16
	#num_iterations = 2000000
	batch_size = 32