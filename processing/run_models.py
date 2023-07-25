import numpy as np
import scipy.signal as sg
import soundfile as sf
import pandas as pd
import utils as u
from tqdm import tqdm
from scipy.fft import fft, fftfreq, fftshift


class Weather_Estimation():

	def __init__(self, welch):
		assert len(welch) == 2, 'Welch should also contain frequency array'
		self.welch = welch

	def get_spl(self):    #Method found online based on PSD
		#welch = self.welch(window='hann', nperseg=None, noverlap=None, nfft=None)
		df = self.welch[0][1]-self.welch[0][0]
		self.spl_scale = self.welch[0]
		self.spl = 10*np.log10(self.welch[1]*df/(1e-6**2))

	def wind_pensieri(self):
		spl = inter_spl(self.spl, self.spl_scale, 8000)
		if spl < 38:
			return 0.1458*spl-3.146
		else:
			return 0.044642*spl**2-3.2917*spl+63.016

	def wind_lemon_evans(self):
		spl4 = inter_spl(self.spl, self.spl_scale, 4300)
		spl8 = inter_spl(self.spl, self.spl_scale, 8000)
		spl14 = inter_spl(self.spl, self.spl_scale, 14500)
		return {'Lemon4300':10**(0.783*spl4/20-26.9/20), 'Evans4300':10**(0.831*spl4/20-29.9/20), 'Evans8000':10**(0.838*spl8/20-26.6/20),
			'Lemon14500':10**(0.805*spl14/20-21.3/20), 'Evans14500':10**(0.868*spl14/20-24.3/20)}

	@classmethod
	def from_fn(cls, path):
		if path[-3:] == 'WAV' or path[-3:] == 'wav':
			welch = np.load(path[:-4]+'_5mn.npy')
		else :
			welch = np.load(path)
		return cls(welch)

	@classmethod
	def from_sig(cls, path = None, sig = None, fs = None, start = 0, stop = None):
		if path != None:
			inst = Signal_Analysis.from_fn(path, start = start, stop = stop)
		elif sig.any() != None and fs != None :
			inst = Signal_Analysis(sig, fs)
		inst.get_welch()
		return cls(inst.welch)
