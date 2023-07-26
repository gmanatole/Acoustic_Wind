import glob, os
import soundfile as sf
import numpy as np
from config import literature, device
from models import empirical
from processing.preprocessing import Load_Data, compute_spl
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import *
import sys
import matplotlib.pyplot as plt
import toml

def get_files(path):
	'''
	Get all wav or mp3 files in directory linked to given path 
	'''
	return [elem for elem in glob.glob(os.path.join(path, '*')) if elem.endswith(('wav', 'mp3'))]

def get_timestamps(fns, dt):
	'''
	This function looks at available wav files and extracts timestamps for wind speed estimation.

	Parameters
	----------
	fns : List
		List of path of audio files from which to extract signals.
	dt : float or int
		Timestep between two successive extracted signals.

	Returns
	-------
	timestamps : dic
		Dictionary containing filename, samplerate and timestamp of signals to extract.

	'''
	
	timestamps = {'fn':[], 'time':[], 'fs':[]}
	for fn in fns :
		duration = sf.info(fn).duration
		fs = sf.info(fn).samplerate
		timestamps['fn'].extend([fn] * (int(duration / dt)+1))
		timestamps['fs'].extend([fs] * (int(duration / dt)+1))
		timestamps['time'].extend(np.arange(0, duration, dt))
	return timestamps
 

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



def read_toml_file(file_path):
    with open(file_path, "r") as file:
        return toml.load(file)
def modify_toml_data(existing_data, new_data):
    existing_data.update(new_data)
    return existing_data
def write_toml_file(file_path, data):
    with open(file_path, "w") as file:
        toml.dump(data, file)

def append_to_toml(file_path, new_data):
    existing_data = read_toml_file(file_path)
    modified_data = modify_toml_data(existing_data, new_data)
    write_toml_file(file_path, modified_data)



