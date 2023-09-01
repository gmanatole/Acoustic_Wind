import numpy as np
import os
import string
import pandas as pd
from config import literature, device
from models import empirical
from processing.preprocessing import Load_Data, compute_spl
from PyQt5.QtCore import QObject, pyqtSignal, QRunnable, pyqtSlot
from utils import beaufort
import scipy.interpolate as inter
from sklearn.model_selection import StratifiedKFold
from utils import read_toml_file, write_toml_file, append_to_toml, get_yyyymmdd_date, get_datetime_date
from scipy.optimize import curve_fit

class WorkerSignals(QObject):
	finished = pyqtSignal()
	error = pyqtSignal(tuple)
	results = pyqtSignal(object)


class Calibrate(QRunnable):

	def __init__(self, timestamps, timestep, timeformat, timeformat_df, wind_path, *, method = 'Hildebrand', batch_size = 8):
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
		self.timeformat = timeformat
		self.timeformat_df = timeformat_df
		self.method = method  #method name
		self.batch_size = batch_size
		self.timestep = timestep
		self.config = literature[method]   #parameters linked to method
		self.wind_path = wind_path
		self.wind_speed = []
		self.sound_pressure_level = []
		self.timestamps = pd.DataFrame(timestamps)
		self.ts = self.timestamps['time']
		self.function = empirical.get[self.config['model']]  #get function linked to method
		self.parameters = list(self.config['parameters'].values())
		self.frequency = self.config['metadata']['frequency']
		'''
		parameters.insert(0, self.config['metadata']['frequency'])    # Add frequency to parameters as it might be used by model
		self.parameters = parameters[-self.function.__code__.co_argcount+1:]   #Remove frequency if it is not used by method
		'''
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
		if not already_computed :
			inst = Load_Data(self.timestamps, method = self.method, batch_size = self.batch_size)
			for batch in inst :
				fn, ts, fs, sig = batch
				ts, fs = np.array(ts), np.array(fs)
				spl = compute_spl(sig, fs, np.array(self.frequency).reshape(1,-1)[0], device = self.instrument, params = self.config)
				self.sound_pressure_level.extend(spl)
			self.timestamps['spl'] =  np.array(self.sound_pressure_level).reshape(-1)
			pd.DataFrame(self.timestamps).to_csv('data/noise_levels.csv', index=False)
			analysis_params = {"analysis_params" : {"timestep":self.timestep, "method":self.method}}
			write_toml_file('app/analysis_params.toml', analysis_params)

		stamps_data = np.load(os.path.join(self.wind_path, 'stamps.npz'), allow_pickle = True)
		u10_data = np.load(os.path.join(self.wind_path, 'u10_era.npy')) #.mean(axis = (1,2))
		v10_data = np.load(os.path.join(self.wind_path, 'v10_era.npy')) #.mean(axis = (1,2))
		wind_data = np.sqrt(u10_data**2 + v10_data**2)

		interp = inter.RegularGridInterpolator((stamps_data['timestamps_epoch'], stamps_data['latitude'], stamps_data['longitude']), wind_data)
		self.timestamps['fn'] = self.timestamps['fn'].apply(lambda x : x.split('/')[-1])
		if self.timeformat == 'yyyymmdd-HHMMSS':
			self.timeformat_df.iloc[:,1] = self.timeformat_df.iloc[:,1].apply(get_yyyymmdd_date)
			self.timeformat = 'epoch'
		if self.timeformat == 'Python datetime':
			self.timeformat_df.iloc[:,1] = self.timeformat_df.iloc[:,1].apply(get_datetime_date)
			self.timeformat = 'epoch'
		ts_epoch = [elem + self.timeformat_df.iloc[:,1][self.timeformat_df.iloc[:,0] == self.timestamps['fn'][i]].item() for i, elem in enumerate(self.ts)]
		self.timestamps['ts_epoch'] = ts_epoch

		temp_timeformat_df = self.timeformat_df.copy().iloc[-1]
		temp_timeformat_df[1] += self.timestep
		self.timeformat_df.loc[len(self.timeformat_df)] = temp_timeformat_df
		lat_interp = inter.interp1d(self.timeformat_df.iloc[:,1], self.timeformat_df.iloc[:,2])
		lon_interp = inter.interp1d(self.timeformat_df.iloc[:,1], self.timeformat_df.iloc[:,3])
		self.timestamps['lat'] = lat_interp(self.timestamps.ts_epoch)
		self.timestamps['lon'] = lon_interp(self.timestamps.ts_epoch)

		dataset = pd.DataFrame()
		dataset['wind'] = interp(np.stack((self.timestamps.ts_epoch, self.timestamps.lat, self.timestamps.lon), axis = 1))
		dataset['spl'] = self.sound_pressure_level
		dataset['wind_classes'] = dataset.wind.apply(beaufort)
		dataset = dataset[dataset.spl < np.quantile(dataset.spl, 0.95)]
		
		popt_tot = []
		skf = StratifiedKFold(n_splits=5)
		for i, (train_index, test_index) in enumerate(skf.split(dataset.spl, dataset.wind_classes)):
			self.trainset = dataset.iloc[train_index]
			assert np.isnan(self.trainset.spl).sum() != len(self.trainset), 'Array contains only NaNs'
			self.testset = dataset.iloc[test_index]
			popt, popv = curve_fit(self.function, self.trainset.spl.to_numpy(), self.trainset.wind.to_numpy(), maxfev = 5000)
			popt_tot.append(popt)
		self.popt = np.mean(popt_tot, axis=0)
		self.dataset = dataset
		self.dataset.to_csv('dataset')
		self.parameters.insert(0, self.frequency)    # Add frequency to parameters as it might be used by model
		self.parameters = self.parameters[-self.function.__code__.co_argcount+1:]   #Remove frequency if it is not used by method
		file_path = "config/literature.toml"
		data = literature[self.method]
		data['parameters'] = dict(zip(string.ascii_lowercase, self.popt.astype(float)))
		new_data = {f'User_{self.method}' : data}
		append_to_toml(file_path, new_data)  #process is detailed in utils
		self.signals.finished.emit()




