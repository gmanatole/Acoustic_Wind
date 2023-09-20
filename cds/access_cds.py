import cdsapi
from datetime import date, datetime
import os
import calendar
from tqdm import tqdm
from netCDF4 import Dataset
import numpy as np
import pandas as pd

get_epoch_time = lambda x : calendar.timegm(x.timetuple()) if isinstance(x, datetime) else x

def make_windows_cds_file(key, udi):
    cdsapirc_path = os.path.expanduser("~/.cdsapirc")
    
    try:
        os.remove(cdsapirc_path)
    except FileNotFoundError:
        pass

    with open(cdsapirc_path, "w") as file:
        file.write("url: https://cds.climate.copernicus.eu/api/v2\n")
        file.write("key: {}:{}\n".format(udi, key))

def make_cds_file(key, udi):
	try :
 	   os.remove(os.path.join(os.path.expanduser("~"), '.cdsapirc'))
	except FileNotFoundError :
	    pass

	cmd1 = "echo url: https://cds.climate.copernicus.eu/api/v2 >> ~/.cdsapirc"
	cmd2 = "echo key: {}:{} >> ~/.cdsapirc".format(udi, key)
	os.system(cmd1)
	os.system(cmd2)

def return_cdsapi(filename, key, variable, year, month, day, time, area):

  filename = 'data/' + filename + '.nc'

  c = cdsapi.Client()
  
  r = c.retrieve('reanalysis-era5-single-levels',
            {
              'product_type' : 'reanalysis',
              'variable' : variable,
              'year' : year,
              'month' : month,
              'day' : day,
              'time' : time,
              'area' : area,
              'format' : 'netcdf',
              'grid':[0.25, 0.25],
            },
            filename,
            )
  r.download(filename)



def format_nc(filename):

  downloaded_cds = 'data/' + filename + '.nc'
  fh = Dataset(downloaded_cds, mode='r')

  variables = list(fh.variables.keys())[3:]
  single_levels = np.zeros((len(variables), fh.dimensions.get('time').size, fh.dimensions.get('latitude').size, fh.dimensions.get('longitude').size))

  lon = fh.variables['longitude'][:]
  lat = fh.variables['latitude'][:]
  time_bis = fh.variables['time'][:]
  for i in range(len(variables)):
    single_levels[i] = fh.variables[variables[i]][:]

  time_units = fh.variables['time'].units

  def transf_temps(time) :
    time_str = float(time)/24 + date.toordinal(date(1900,1,1))
    return date.fromordinal(int(time_str))

  time_ = list(time_bis)

  data = pd.DataFrame()
  data['time'] = time_
  data['time'] = data['time'].apply(lambda x : transf_temps(x))
  hours = np.array(time_)%24
  dates = np.array([datetime(elem.year, elem.month, elem.day, hour) for elem, hour in list(zip(data['time'], hours))])

  return dates, lon, lat, single_levels, variables



def save_results(dates, lat, lon, single_levels, variables, filename):
	for i in range(len(variables)):
		np.save('data/' + variables[i]+'_'+filename, np.ma.filled(single_levels[i], fill_value=float('nan')), allow_pickle = True)

	np.savez('data/stamps', timestamps = np.array(dates), timestamps_epoch = np.array(list(map(get_epoch_time, np.array(dates)))), latitude = np.array(lat), longitude = np.array(lon))
#	np.save('timestamps.npy', np.array(dates))
#	np.save('latitude.npy', np.array(lat))
#	np.save('longitude.npy', np.array(lon))
#	np.save('timestamps_epoch.npy', np.array(list(map(get_epoch_time, np.array(dates)))))
#	stamps = np.zeros((len(dates), len(lat), len(lon), 3), dtype=object)
#	for i in range(len(dates)):
#		for j in range(len(lat)):
#			for k in range(len(lon)):
#				stamps[i,j,k] = [dates[i], lat[j], lon[k]]
#	np.save('stamps_'+filename, stamps, allow_pickle = True)




def download_era_data(df1, filename, key, variable, year, month, day, time, area) :
  return_cdsapi(filename, key, variable, year, month, day, time, area)
  with tqdm(total = 100) as pbar :
    pbar.update(33)
    dates, lon, lat, single_levels, variables = format_nc(filename)
    pbar.update(33)
    test = save_results(dates, lat, lon, single_levels, variables, filename)
    pbar.update(33)
  return test

