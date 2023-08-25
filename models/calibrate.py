import numpy as np
import os
import pandas as pd
from config import literature, device
from models import empirical
from processing.preprocessing import Load_Data, compute_spl
from PyQt5.QtCore import QObject, pyqtSignal, QRunnable, pyqtSlot
from utils import beaufort
import scipy.interpolate as inter
from sklearn.model_selection import StratifiedKFold
from utils import read_toml_file, write_toml_file

class WorkerSignals(QObject):
	finished = pyqtSignal()
	error = pyqtSignal(tuple)
	results = pyqtSignal(object)


class Calibrate(QRunnable):

	def __init__(self, timestamps, timestep, wind_path, *, method = 'Hildebrand', batch_size = 8):
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
		super(Calibrate, self).__init__()
		self.method = method  #method name
		self.batch_size = batch_size
		self.timestep = timestep
		self.config = literature[method]   #parameters linked to method
		self.wind_path = wind_path
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
		'''
		Parameters
		----------
		n_splits : Number of folds for the split. Results are averaged. The default is 5.
		bounds : Optional bounds on fit parameters. The default is False.
		'''
		already_computed = False
		if os.path.exists('data/noise_levels.csv') and os.path.exists('app/analysis_params.toml'):
			temp_timestamps = pd.read_csv('data/noise_levels.csv')
			temp_params = read_toml_file('app/analysis_params.toml')
			if (set(temp_timestamps['fn']) == set(self.timestamps['fn'])) and (self.method == temp_params['analysis_params']['method']) and (self.timestep == temp_params['analysis_params']['timestep']) :
				already_computed = True
				self.sound_pressure_level = temp_timestamps['spl']
				self.signals.finished.emit()
		if not already_computed :
			inst = Load_Data(self.timestamps, method = self.method, batch_size = self.batch_size)
			for batch in inst :
				fn, ts, fs, sig = batch
				ts, fs = np.array(ts), np.array(fs)
				spl = compute_spl(sig, fs, np.array(self.config['metadata']['frequency']).reshape(1,-1)[0], device = self.instrument, params = self.config)
				self.sound_pressure_level.extend(spl)
			self.timestamps['spl'] =  self.sound_pressure_level
			self.signals.finished.emit()
			pd.DataFrame(self.timestamps).to_csv('data/noise_levels.csv')
			analysis_params = {"analysis_params" : {"timestep":self.timestep, "method":self.method}}
			write_toml_file('app/analysis_params.toml', analysis_params)


		stamps_data = np.load(os.path.join(self.wind_path, 'stamps.npz'), allow_pickle = True)
		u10_data = np.load(os.path.join(self.wind_path, 'u10_era.npy')).mean(axis = (1,2))
		v10_data = np.load(os.path.join(self.wind_path, 'v10_era.npy')).mean(axis = (1,2))
		wind_data = np.sqrt(u10_data**2 + v10_data**2)
		interp = inter.interp1d(stamps_data['timestamps_epoch'], wind_data)
		dataset = pd.DataFrame()
		dataset['wind'] = interp(self.ts)
		dataset['spl'] = self.sound_pressure_level
		dataset['wind_classes'] = dataset.wind.apply(beaufort)
		dataset = dataset[dataset.spl < np.quantile(dataset.spl, 0.95)]
		popt_tot = []
		skf = StratifiedKFold(n_splits=n_splits)
		for i, (train_index, test_index) in enumerate(skf.split(dataset.spl, dataset.wind_classes)):
			self.trainset = dataset.iloc[train_index]
			assert np.isnan(self.trainset[self.freq]).sum() != len(self.trainset), 'Array contains only NaNs'
			self.testset = dataset.iloc[test_index]
			if not bounds:
				popt, popv = curve_fit(self.function, self.trainset.spl.to_numpy(), self.trainset.wind.to_numpy(), maxfev = 2000)
			else :
				popt, popv = curve_fit(self.function, self.trainset.spl.to_numpy(), self.trainset.wind.to_numpy(),
							  bounds = bounds, maxfev = 5000)
			popt_tot.append(popt)
			popv_tot.append(popv)
		popt = np.mean(popt_tot, axis=0)
		file_path = "config/literature.toml"
		data = literature[self.method]
		data['parameters'] = dict(zip(string.ascii_lowercase, popt))
		new_data = {f'User_{self.method}' : data}
		append_to_toml(file_path, new_data)  #process is detailed in utils




