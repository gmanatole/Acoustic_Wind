import glob, os
import soundfile as sf
import numpy as np

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
  