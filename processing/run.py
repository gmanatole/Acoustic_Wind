import numpy as np
from config import literature, device
from models import empirical
from processing.preprocessing import Load_Data, compute_spl
from PyQt5.QtCore import QObject, pyqtSignal, QRunnable, pyqtSlot
import os
import pandas as pd

class WorkerSignals(QObject):
	finished = pyqtSignal()
	error = pyqtSignal(tuple)
	results = pyqtSignal(object)


class Worker(QRunnable):

	def __init__(self, timestamps, *, method = 'Hildebrand', batch_size = 8):
		'''
		Parameters
		----------
		timestamps : dict
			Dictionary created by get_timestamps with fn, time and fs of event to extract.
		method : str, optional
			Method from literature (or as defined by user) used to estimate wind speed. The default is Hildebrand
		batch_size : int, optional
			Number of signals that are analyzed simultaneously. The default is 8.
		'''
		super(Worker, self).__init__()
		self.method = method  #method name
		self.batch_size = batch_size
		self.config = literature[method]   #parameters linked to method
		self.wind_speed = []
		self.sound_pressure_level = []
		self.timestamps = timestamps
		self.ts = self.timestamps['time']
		self.function = empirical.get[self.config['model']]  #get function linked to method
		parameters = list(self.config['parameters'].values())
		parameters.insert(0, self.config['metadata']['frequency'])    # Add frequency to parameters as it might be used by model
		self.parameters = parameters[-self.function.__code__.co_argcount+1:]   #Remove frequency if it is not used by method
		self.instrument = device['hydrophone']
		self.signals = WorkerSignals()

	@pyqtSlot()

	def run(self):
		if not os.path.exists('data/noise_levels.csv') :
			inst = Load_Data(self.timestamps, method = self.method, batch_size = self.batch_size)
			for batch in inst :
				fn, ts, fs, sig = batch
				ts, fs = np.array(ts), np.array(fs)
				spl = compute_spl(sig, fs, np.array(self.config['metadata']['frequency']).reshape(1,-1)[0], device = self.instrument, params = self.config)
				self.sound_pressure_level.extend(spl)
			self.timestamps['spl'] = np.array(self.sound_pressure_level).reshape(-1)
			pd.DataFrame(self.timestamps).to_csv('data/noise_levels.csv')
		else :
			self.timestamps = pd.read_csv('data/noise_levels.csv')
			self.sound_pressure_level = self.timestamps['spl']
			self.ts = self.timestamps['time']
		self.wind_speed.extend(np.array(self.function(np.array(self.timestamps['spl']).reshape(-1), *self.parameters)).flatten().tolist())
		self.signals.finished.emit()

