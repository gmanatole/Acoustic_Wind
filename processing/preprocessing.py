import numpy as np
from config import literature
import soundfile as sf
from scipy.fft import fft, fftfreq, fftshift
import scipy.signal as sg
import pandas as pd

def pad_signal(fn, start, stop, fs):
	sig = np.full((int(fs*(stop-start)), 1), np.nan)
	raw_sig = sf.read(fn, start=int(start*fs), stop=int(stop*fs), always_2d=True)[0]
	sig[:len(raw_sig)] = raw_sig
	return sig
	
	
class Load_Data():

	
	def __init__(self, timestamps, *, method = None, batch_size = 8):
		'''
		Parameters
		----------
		timestamps : array
			array containing timestamps in audio file where signal will be extracted.
		method : str, optional
			Method to compute SPL from literture to use for loading. The default is None.
		batch_size : int, optional
			Batch size of output. Number of extracted signals to be grouped together. The default is 8.

		Returns
		-------
		Generator containing the batched filenames, timestamps, samplerates, and signals.

		'''
		self.fns = timestamps['fn']
		self.fs = timestamps['fs']
		self.ts = timestamps['time']
		self.batch_size = batch_size
		self.config = literature[method]

	def __len__(self):
		return len(self.fns)//self.batch_size+1
		
	def __iter__(self):
		'''
		Loops through the signals that are extracted from the original audio file and returns a generator containing batched filenames, timestamps, samplerates, and signals. 
		'''
		for i in range(0, len(self.fns), self.batch_size):
			yield (self.fns[i:i + self.batch_size], self.ts[i:i + self.batch_size], self.fs[i:i + self.batch_size], 
		  np.hstack([pad_signal(self.fns[j], start=self.ts[j], stop=self.ts[j]+self.config['preprocessing']['duration'], fs=self.fs[j]) for j in range(i, min(len(self.fns), i+self.batch_size))]).T)

		  #np.hstack([sf.read(self.fns[j], start=int(self.timestamps[j]*self.fs[j]), stop=int((self.timestamps[j]+self.config['preprocessing']['duration'])*self.fs[j]), always_2d=True)[0] for j in range(i, min(len(self.fns), i+self.batch_size))]).T)




def compute_spl(sig, fs, fcs, * , device, params):
	'''
	Parameters
	----------
	sig : array
		2D array (L, N) containing L batched audio signals of length N.
	fs : float
		Original samplerate of audio signal, may be resampled to samplerate of desired method.
	fcs : List
		Frequencies at which to compute the SPL.
	device : dic
		Dictionary containing device parameters for calibration. Accessible in config/device.toml .
	params : dic
		Dictionary containing method parameters for preprocessing. Accessible in config/literature.toml .

	Returns
	-------
	array
		Array of size (L, len(fcs)) containing the SPL values for each signal of the batch at the desired frequencies.
	'''
	
	# Fetch hydrophone calibration and preprocessing values in toml files
	fs = params['metadata']['samplerate']
	sig =  sg.resample(sig, int(params['preprocessing']['duration']*fs), axis=1)
	sensitivity, gain, Nbits, Vadc = device['sensitivity'], device['gain'], device['Nbits'], device['Vadc']
	winname, winsize = params['preprocessing']['window'], int(params['preprocessing']['sample_length']*fs)
	overlap, sample_number = params['preprocessing']['overlap'], params['preprocessing']['sample_number']
	assert int(params['preprocessing']['duration']) > int(params['preprocessing']['sample_length']), "Sample duration must be larger than sample length (window size)"
	
	# Get time_pos of subsignal beginning position through number of samples or overlap
	if sample_number != "" :
		time_pos = np.linspace(0, sig.shape[1]-winsize, sample_number).astype(int)
	else :
		overlap = winsize-int(overlap*winsize)
		time_pos = list(range(0, sig.shape[1]-winsize+1, overlap))
			
	# Get arrays and parameters for windowing
	if winname == 'hanning':
		window = np.hanning(winsize)
		alpha, overlap, B = np.sum(window)/winsize, int(winsize-overlap*winsize), 1.5
		
	# Compute FFT, power and SPL for every subsignal 
	spl = np.zeros((len(sig), len(fcs), len(time_pos)))
	for i, iteration in enumerate(time_pos):
		sig_win = sig[:, iteration:iteration+winsize] * window / alpha
		X = fftshift(fft(sig_win), axes = (1,))
		freq_X = np.tile(fftshift(fftfreq(winsize, 1/fs)),(sig.shape[0],1))
		X = X[(freq_X > 0) & (freq_X < fs/2)].reshape(sig.shape[0], -1)
		freq_X = freq_X[(freq_X > 0) & (freq_X < fs/2)].reshape(sig.shape[0], -1)
		P = np.abs(X/sig_win.shape[1])**2
		Pss = 2*P
		for j, fc in enumerate(fcs):
			ind_lower = np.argmin(np.abs(freq_X - fc * 10 ** (-1 / 20)), axis=1)
			ind_upper = np.argmin(np.abs(freq_X - fc * 10 ** (1 / 20)), axis=1)+1
			spl[:, j, i] = 10*np.log10((1/((1e-6)**2)) * np.sum(Pss[:, ind_lower[0]:ind_upper[0]], axis = 1) / B) 
			- (sensitivity + gain + 20*np.log10(1/Vadc) + 20*np.log10(2**(Nbits-1)))
			
	return np.mean(spl, axis = 2)



