from config import literature, device
from models import empirical
from utils import get_files, get_timestamps
from processing.preprocessing import Load_Data, compute_spl
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import *
import sys
import matplotlib.pyplot as plt

path = '/home6/grosmaan/Documents/codes/test_wavs'
method = 'Pensieri'


class WorkerSignals(QObject):
	finished = pyqtSignal()
	error = pyqtSignal(tuple)
	results = pyqtSignal(object)


class Worker(QRunnable):

	def __init__(self, timestamps, *, method = None, batch_size = 8):

		super(Worker, self).__init__()
		self.method = method
		self.batch_size = batch_size
		self.config = literature[method]
		self.wind_speed = []
		self.sound_pressure_level = []
		self.timestamps = timestamps
		self.ts = self.timestamps['time']
		self.function = empirical.get[self.config['model']]
		parameters = list(self.config['parameters'].values())
		parameters.insert(0, self.config['metadata']['frequency'])
		self.parameters = parameters[-self.function.__code__.co_argcount+1:]
		self.instrument = device['hydrophone']
		self.signals = WorkerSignals()

	@pyqtSlot()
	def run(self):
		inst = Load_Data(self.timestamps, method = self.method, batch_size = self.batch_size)
		for batch in inst :
			fn, ts, fs, sig = batch
			ts, fs = np.array(ts), np.array(fs)
			spl = compute_spl(sig, fs, np.array(self.config['metadata']['frequency']).reshape(1,-1)[0], device = self.instrument, params = self.config)
			self.wind_speed.extend(np.array(self.function(spl, *self.parameters)).flatten().tolist())
			self.sound_pressure_level.extend(spl)
		self.signals.finished.emit()

if __name__ == '__main__':


	fns = [get_files(path)[1]]
	timestamps = get_timestamps(fns, 60)
	inst = Load_Data(timestamps, method = method, batch_size = 5)
	ws, all_spl, all_time = [], [], []
	for batch in inst :
		print('ok')
		fn, timestamps, fs, sig = batch
		timestamps, fs = np.array(timestamps), np.array(fs)
		spl = compute_spl(sig, fs, [8000], device = device['hydrophone'], params = literature[method])
		ws.extend(empirical.get[literature[method]['model']]
			(spl, *literature[method]['parameters'].values()))
		all_spl.extend(spl)
		all_time.extend(timestamps)

