import sys
import string
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtCore import Qt as QtDisp
import matplotlib
import pandas as pd
matplotlib.use('Qt5Agg')
from pyqtgraph import *
import pyqtgraph as pg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from config import literature, device
from models import empirical
from utils import get_files, get_timestamps, append_to_toml, format_plot_axis, write_toml_file
from processing.run import Worker
from models.calibrate import Calibrate
from processing.preprocessing import Load_Data, compute_spl
import numpy as np
from cds.access_cds import make_cds_file, download_era_data

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

'''def check_fns(fns):
	if os.path.exists('app/files.json') :
		with open('file_names.json', 'r') as file:
			file_names = json.load(file)
		crit = set(fns) == set(file_names)
		if crit:
			return crit
	with open('app/files.json', 'w') as file:
		json.dump(fns, file)
	return crit'''


class Fenetre(QWidget):
	
	'''
	Class that create GUI. Multiple tabs are created.
	Tab 1 : Main interface to load acoustic data, chose method and get wind speed estimation.
	Tab 2 : Used to enter the user's own method and parameters for wind speed estimation.
	'''
	
	def __init__(self):

		super(QWidget, self).__init__()
		self.layout = QVBoxLayout(self)
		self.setWindowTitle("ARAMIS")
		self.setWindowIcon(QtGui.QIcon('app/logo.png'))
		#instantiate threadpool for worker class
		self.threadpool = QThreadPool()

		# CREATE TABS
		self.tabs = QTabWidget()
		self.tab1 = QWidget()
		self.tab2 = QWidget()
		self.tab3 = QWidget()
		self.tab4 = QWidget()
		self.tabs.addTab(self.tab1,"Estimate Weather")
		self.tabs.addTab(self.tab3,"Download ERA wind speed")
		self.tabs.addTab(self.tab2,"Define user parameters")
		self.tabs.addTab(self.tab4,"Calibrate model to data")
		self.tabs.resize(300,200)


		####################### TAB 1 #############################

		#PREPARE PLOT
		self.tab1.layout = QGridLayout()	
		self.wind_plot = PlotWidget()		
		self.tab1.layout.addWidget(self.wind_plot, 4, 2)

		#PREPARE TABLE
		self.spl_table = QTableWidget()
		self.spl_table.setColumnCount(3)
		self.spl_table.setRowCount(1)
		self.spl_table.setItem(0, 0, QTableWidgetItem('Timestamp'))
		self.spl_table.setItem(0, 1, QTableWidgetItem('Wind speed (m/s)'))
		self.spl_table.setItem(0, 2, QTableWidgetItem('SPL (db re 1 uPa)'))
		self.tab1.layout.addWidget(self.spl_table, 4, 1)

		#METHOD SELECTION
		self.dropdown_est = QComboBox(self)
		for key in literature.keys():
			self.dropdown_est.addItem(key)
		self.dropdown_est.addItem('User Defined')
		self.method_est = 'Default'
		self.dropdown_est.currentIndexChanged.connect(self.handle_selection_change_est)
		self.dropdown_est_label = QLabel('No method selected')
		self.tab1.layout.addWidget(self.dropdown_est_label, 0, 1)
		self.tab1.layout.addWidget(self.dropdown_est, 0, 0)

		#BATCH SIZE AND TIME STEP
		self.timestep_est_label = QLabel('Please enter audio timestep (in seconds)')
		self.tab1.layout.addWidget(self.timestep_est_label, 2, 0)
		self.timestep_est = QLineEdit(self)
		self.tab1.layout.addWidget(self.timestep_est, 2, 1)
		self.batchsize_est_label = QLabel('Please enter batchsize')
		self.tab1.layout.addWidget(self.batchsize_est_label, 3, 0)
		self.batchsize_est = QLineEdit(self)
		self.tab1.layout.addWidget(self.batchsize_est, 3, 1)

		#ESTIMATION BUTTON
		self.estimate = QPushButton("Estimate wind speed")
		self.estimate.setStyleSheet("background-color : lightblue")
		self.estimate.clicked.connect(self.press_estimate)
		self.tab1.layout.addWidget(self.estimate, 0, 2)
		self.loading_bar_est = QProgressBar()
		self.tab1.layout.addWidget(self.loading_bar_est, 1, 2)

		# DIRECTORY SELECTION
		self.directory_est_button = QPushButton("Select Directory")
		self.directory_est_button.clicked.connect(self.select_est_directory)
		self.tab1.layout.addWidget(self.directory_est_button, 1, 0)
		self.directory_est_label = QLabel('No directory selected')
		self.tab1.layout.addWidget(self.directory_est_label, 1, 1)

		#FIX GEOMETRY
		self.tab1.layout.setColumnStretch(0, 1)
		self.tab1.layout.setColumnStretch(1, 2)
		self.tab1.layout.setColumnStretch(2, 4)
		self.tab1.layout.setRowStretch(0, 1)
		self.tab1.layout.setRowStretch(1, 2)
		self.tab1.layout.setRowStretch(2, 3)
		self.tab1.layout.setRowStretch(3, 4)
		self.tab1.layout.setRowStretch(4, 6)

		self.tab1.setLayout(self.tab1.layout)

		####################### TAB 2 #############################

		#CREATE LABELS AND BUTTONS FOR USER INPUT OF PARAMETERS
		self.tab2.layout = QGridLayout()
		self.user_method_label = QLabel('Please enter the method you want to use (quadratic, logarithmic, deep_learning)')
		self.tab2.layout.addWidget(self.user_method_label, 1, 0)
		self.user_method = QLineEdit(self)
		self.tab2.layout.addWidget(self.user_method, 1, 1)
		self.user_parameters_labels = QLabel('Please enter parameters (separated by a comma)')
		self.tab2.layout.addWidget(self.user_parameters_labels, 2, 0)
		self.user_parameters = QLineEdit(self)
		self.tab2.layout.addWidget(self.user_parameters, 2, 1)
		self.user_frequency_label = QLabel('Please enter the frequency at which wind speed is estimated')
		self.tab2.layout.addWidget(self.user_frequency_label, 3, 0)
		self.user_frequency = QLineEdit(self)
		self.tab2.layout.addWidget(self.user_frequency, 3, 1)
		self.user_samplerate_label = QLabel('Please enter samplerate of analysis of sound files')
		self.tab2.layout.addWidget(self.user_samplerate_label, 4, 0)
		self.user_samplerate = QLineEdit(self)
		self.tab2.layout.addWidget(self.user_samplerate, 4, 1)
		self.user_duration_label = QLabel('Please enter duration of signals to analyze (in seconds)')
		self.tab2.layout.addWidget(self.user_duration_label, 5, 0)
		self.user_duration = QLineEdit(self)
		self.tab2.layout.addWidget(self.user_duration, 5, 1)
		self.user_samplelength_label = QLabel('Please enter duration of subsignals on which to compute the FFT (in seconds)')
		self.tab2.layout.addWidget(self.user_samplelength_label, 6, 0)
		self.user_samplelength = QLineEdit(self)
		self.tab2.layout.addWidget(self.user_samplelength, 6, 1)
		self.user_overlap_label = QLabel('Please enter overlap (between 0 and 1) between two consecutive subsignals')
		self.tab2.layout.addWidget(self.user_overlap_label, 7, 0)
		self.user_overlap = QLineEdit(self)
		self.tab2.layout.addWidget(self.user_overlap, 7, 1)
		self.create_user = QPushButton("Save values")
		self.create_user.clicked.connect(self.save_user)
		self.tab2.layout.addWidget(self.create_user, 8, 0)

		self.sensitivity_label = QLabel(' '*30 + 'Please enter the sensitivity of your hydrophone (dB re 1 \u03BCPa)')
		self.tab2.layout.addWidget(self.sensitivity_label, 2, 2)
		self.hyd_sensitivity = QLineEdit(self)
		self.tab2.layout.addWidget(self.hyd_sensitivity, 2, 3)
		self.gain_label = QLabel(' '*30+'Please enter the gain of your hydrophone (dB)')
		self.tab2.layout.addWidget(self.gain_label, 3, 2)
		self.hyd_gain = QLineEdit(self)
		self.tab2.layout.addWidget(self.hyd_gain, 3, 3)
		self.voltage_label = QLabel(' '*30+'Please enter the voltage of your ADC (V)')
		self.tab2.layout.addWidget(self.voltage_label, 4, 2)
		self.hyd_voltage = QLineEdit(self)
		self.tab2.layout.addWidget(self.hyd_voltage, 4, 3)
		self.bits_label = QLabel(' '*30+'Please enter the number of bits your signal is encoded with')
		self.tab2.layout.addWidget(self.bits_label, 5, 2)
		self.hyd_bits = QLineEdit(self)
		self.tab2.layout.addWidget(self.hyd_bits, 5, 3)
		self.create_hyd = QPushButton("Save values")
		self.create_hyd.clicked.connect(self.save_hyd)
		self.tab2.layout.addWidget(self.create_hyd, 6,3)

		#FIX GEOMETRY
		self.tab2.layout.setColumnStretch(0, 3)
		self.tab2.layout.setColumnStretch(3, 4)
		self.tab2.layout.setColumnStretch(4, 7)
		self.tab2.layout.setColumnStretch(7, 8)
		self.tab2.setLayout(self.tab2.layout)

		####################### TAB 3 #############################

		#QLINE BOXES AND CHECKLISTS FOR ERA DOWNLOAD
		self.tab3.layout = QGridLayout()
		self.era_browser = QTextBrowser()
		self.era_browser.setOpenExternalLinks(True)
		self.era_browser.setReadOnly(True)
		self.era_browser.append('<a href="https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview">All information on ERA5\'s model and for CDS account creation can be found here</a>')
		self.tab3.layout.addWidget(self.era_browser, 0, 0)
		self.uid_label = QLabel('Please enter the UID of your CDS account')
		self.tab3.layout.addWidget(self.uid_label, 1, 0)
		self.uid = QLineEdit(self)
		self.tab3.layout.addWidget(self.uid, 1, 1)
		self.key_label = QLabel('Please enter the key of your CDS account')
		self.tab3.layout.addWidget(self.key_label, 2, 0)
		self.key = QLineEdit(self)
		self.tab3.layout.addWidget(self.key, 2, 1)
		self.north_boundary_label = QLabel('Please enter the northern boundary of your study area (in decimal degrees)')
		self.tab3.layout.addWidget(self.north_boundary_label, 3, 0)
		self.north_boundary = QLineEdit(self)
		self.tab3.layout.addWidget(self.north_boundary, 3, 1)
		self.south_boundary_label = QLabel('Please enter the southern boundary of your study area (in decimal degrees)')
		self.tab3.layout.addWidget(self.south_boundary_label, 4, 0)
		self.south_boundary = QLineEdit(self)
		self.tab3.layout.addWidget(self.south_boundary, 4, 1)
		self.east_boundary_label = QLabel('Please enter the eastern boundary of your study area (in decimal degrees)')
		self.tab3.layout.addWidget(self.east_boundary_label, 5, 0)
		self.east_boundary = QLineEdit(self)
		self.tab3.layout.addWidget(self.east_boundary, 5, 1)
		self.west_boundary_label = QLabel('Please enter the western boundary of your study area (in decimal degrees)')
		self.tab3.layout.addWidget(self.west_boundary_label, 6, 0)
		self.west_boundary = QLineEdit(self)
		self.tab3.layout.addWidget(self.west_boundary, 6, 1)
		self.years_label = QLabel('Please enter the years of your audio recordings (separated by a comma)')
		self.tab3.layout.addWidget(self.years_label, 7, 0)
		self.years = QLineEdit(self)
		self.tab3.layout.addWidget(self.years, 7, 1)
		self.loading_bar_era = QProgressBar()
		self.tab3.layout.addWidget(self.loading_bar_era, 8, 1)


		#Create months checklist
		self.months_label = QLabel('Select the months of your audio recordings')
		self.tab3.layout.addWidget(self.months_label, 1, 3)
		months_layout = QHBoxLayout()
		self.months = []
		for month in range(1, 13):
			checkbox = QCheckBox(f"{month}", self)
			months_layout.addWidget(checkbox)
			self.months.append(checkbox)
		self.tab3.layout.addLayout(months_layout, 2, 3)
		#Create days checklist
		self.days_label = QLabel('Select the days of your audio recordings')
		self.tab3.layout.addWidget(self.days_label, 3, 3)
		days_layout1 = QHBoxLayout()
		days_layout2 = QHBoxLayout()
		self.days = []
		for day in range(1, 32):
			checkbox = QCheckBox(f"{day}", self)
			if day < 16:
				days_layout1.addWidget(checkbox)
			else :
				days_layout2.addWidget(checkbox)	
			self.days.append(checkbox)
		self.tab3.layout.addLayout(days_layout1, 4, 3)
		self.tab3.layout.addLayout(days_layout2, 5, 3)
		self.download_era = QPushButton("Download ERA5 values")
		self.download_era.clicked.connect(self.api_download)
		self.tab3.layout.addWidget(self.download_era, 8, 2)



		#FIX GEOMETRY
		for i in range(11):
			self.tab3.layout.setRowStretch(i, i+1)
		self.tab3.layout.setColumnStretch(0, 2)
		self.tab3.layout.setColumnStretch(2, 3)
		self.tab3.layout.setColumnStretch(3, 4)
		self.tab3.layout.setColumnStretch(4, 6)
		self.tab3.setLayout(self.tab3.layout)


		####################### TAB 4 #############################
	

		self.tab4.layout = QGridLayout()
	
		#METHOD SELECTION
		self.dropdown_cal = QComboBox(self)
		for key in literature.keys():
			self.dropdown_cal.addItem(key)
		self.dropdown_cal.addItem('User Defined')
		self.method_cal = 'Default'
		self.dropdown_cal.currentIndexChanged.connect(self.handle_selection_change_cal)
		self.dropdown_cal_label = QLabel('No method selected')
		self.tab4.layout.addWidget(self.dropdown_cal_label, 0, 1)
		self.tab4.layout.addWidget(self.dropdown_cal, 0, 0)

		#TIMEFORMAT SELECTION AND TIMESTAMPS
		self.timeformat_button = QComboBox(self)
		for key in ['epoch', 'Python datetime', 'yyyymmdd-HHMMSS']:
			self.timeformat_button.addItem(key)
		self.timeformat_cal = 'epoch'
		self.timeformat_button.currentIndexChanged.connect(self.handle_timeformat_change_cal)
		self.timeformat_cal_label = QLabel('No time format selected')
		self.tab4.layout.addWidget(self.timeformat_cal_label, 6, 1)
		self.tab4.layout.addWidget(self.timeformat_button, 6, 0)
		self.timestamps_cal_button = QPushButton('Select timestamps file')
		self.timestamps_cal_button.clicked.connect(self.select_timestamps_cal)
		self.tab4.layout.addWidget(self.timestamps_cal_button, 5, 0)
		self.timestamps_cal_label = QLabel('No timestamps file selected')
		self.tab4.layout.addWidget(self.timestamps_cal_label, 5, 1)

		#BATCH SIZE AND TIME STEP
		self.timestep_cal_label = QLabel('Please enter audio timestep (in seconds)')
		self.tab4.layout.addWidget(self.timestep_cal_label, 4, 0)
		self.timestep_cal = QLineEdit(self)
		self.tab4.layout.addWidget(self.timestep_cal, 4, 1)
		self.batchsize_cal_label = QLabel('Please enter batchsize')
		self.tab4.layout.addWidget(self.batchsize_cal_label, 3, 0)
		self.batchsize_cal = QLineEdit(self)
		self.tab4.layout.addWidget(self.batchsize_cal, 3, 1)

		#CALIBRATION BUTTON
		self.calibrate = QPushButton("Calibrate model parameters")
		self.calibrate.setStyleSheet("background-color : lightblue")
		self.calibrate.clicked.connect(self.press_calibrate)
		self.tab4.layout.addWidget(self.calibrate, 0, 2)
		
		# DIRECTORY SELECTION
		self.directory_cal_button = QPushButton("Select Directory of wav files")
		self.directory_cal_button.clicked.connect(self.select_cal_directory)
		self.tab4.layout.addWidget(self.directory_cal_button, 1, 0)
		self.directory_cal_label = QLabel('No directory selected')
		self.tab4.layout.addWidget(self.directory_cal_label, 1, 1)
		self.directory_wind_button = QPushButton("Select Directory of ERA wind files")
		self.directory_wind_button.clicked.connect(self.select_wind_directory)
		self.tab4.layout.addWidget(self.directory_wind_button, 2, 0)
		self.directory_wind_label = QLabel('No directory selected')
		self.tab4.layout.addWidget(self.directory_wind_label, 2, 1)

		#PREPARE PLOT
		self.calibration_plot = PlotWidget()		
		self.tab4.layout.addWidget(self.calibration_plot, 7, 2)

		#FIX GEOMETRY
		self.tab4.layout.setColumnStretch(0, 1)
		self.tab4.layout.setColumnStretch(1, 2)
		self.tab4.layout.setColumnStretch(2, 5)

		self.tab4.setLayout(self.tab4.layout)

		self.layout.addWidget(self.tabs)
		self.setLayout(self.layout)


	def select_wind_directory(self):
		'''
		Get and store directory path where ERA wind files to analyze are stored
		'''
		self.directory_wind = QFileDialog.getExistingDirectory(self, "Select Directory")
		if self.directory_wind:
			self.directory_wind_label.setText(self.directory_wind)

	def select_est_directory(self):
		'''
		Get and store directory path where sound files to analyze are stored
		'''
		self.directory_est = QFileDialog.getExistingDirectory(self, "Select Directory")
		if self.directory_est:
			self.directory_est_label.setText(self.directory_est)

	def select_timestamps_cal(self):
		'''
		Get and store directory path where sound files to analyze are stored
		'''
		self.timestamps_cal_selection = QFileDialog.getOpenFileName(self, "Select file")
		self.timestamps_cal_file = self.timestamps_cal_selection[0]
		if self.timestamps_cal_file:
			self.timestamps_cal_label.setText(self.timestamps_cal_file)
		self.timestamps_cal_df = pd.read_csv(self.timestamps_cal_file)

	def select_cal_directory(self):
		'''
		Get and store directory path where sound files to calibrate parameters with are stored
		'''
		self.directory_cal = QFileDialog.getExistingDirectory(self, "Select Directory")
		if self.directory_cal:
			self.directory_cal_label.setText(self.directory_cal)

	'''	def handle_selection_date_cal(self):
		self.date_cal = self.date_cal_button.dateTime().toPyDateTime()
	'''
	def handle_selection_change_est(self, index):
		'''
		Dropdown menu to chose method for wind speed estimation
		'''
		self.method_est = self.dropdown_est.currentText()
		self.dropdown_est_label.setText(f'{self.method_est} method selected')

	def handle_selection_change_cal(self, index):
		'''
		Dropdown menu to chose method for parameters calibration
		'''
		self.method_cal = self.dropdown_cal.currentText()
		self.dropdown_cal_label.setText(f'{self.method_cal} method selected')

	def handle_timeformat_change_cal(self, index):
		'''
		Dropdown menu to chose timeformat for parameters calibration
		'''
		self.timeformat_cal = self.timeformat_button.currentText()
		self.timeformat_cal_label.setText(f'{self.timeformat_cal} method selected')	

	def calibration_complete(self):
		"Once calibration is over, show fit on scatter plot"
		self.calibration_plot.plot(self.params.dataset.spl, self.params.dataset.wind, pen=None, symbol = 'o')
		self.calibration_plot.plot(np.arange(min(self.params.dataset.spl), max(self.params.dataset.spl), 1), self.params.function(np.arange(min(self.params.dataset.spl), max(self.params.dataset.spl), 1), *self.params.popt))

	def computation_complete(self):
		'''
		Once computation is complete, show results in table and as a plot. Data are stored in worker class.
		'''
		self.wind_plot.plot(format_plot_axis(self.worker.ts), self.worker.wind_speed)
		#Reshape time, spl and wind estimation data for table widget
		table_data = np.hstack((np.array(self.worker.ts).reshape(-1,1), np.array(self.worker.wind_speed).reshape(-1,1), np.array(self.worker.sound_pressure_level).reshape(-1,1)))
		self.spl_table.setRowCount(len(table_data)+1)
		for i, elem in enumerate(table_data):
			self.spl_table.setItem(i+1, 0, QTableWidgetItem(str(elem[0])))
			self.spl_table.setItem(i+1, 1, QTableWidgetItem(str(np.round(elem[1], 2))))
			self.spl_table.setItem(i+1, 2, QTableWidgetItem(str(np.round(elem[2], 2))))

	def press_estimate(self):
		'''
		Function that gets wav files and launches computation of spl and wind estimation
		'''
		iteration = 0
		fns = get_files(self.directory_est)
		timestamps = get_timestamps(fns, float(self.timestep_est.text())) #Timestamps are created in utils
		self.worker = Worker(timestamps, float(self.timestep_est.text()), loading_bar = self.loading_bar_est, method = self.method_est, batch_size = int(self.batchsize_est.text())) # SPL and wind estimation are launched
		self.worker.signals.finished.connect(self.computation_complete)   # Check if computation is finished to plot data and show table
		self.threadpool.start(self.worker)
		#self.worker = Worker(timestamps, method = self.method_est, batch_size = int(self.batchsize_est.text())) # SPL and wind estimation are launched

	def press_calibrate(self):
		"""
		Function that loads wav files and calibrates using stratified K Folds the parameters for wind estimation
		"""
		iteration = 0
		fns = get_files(self.directory_cal)
		timestamps = get_timestamps(fns, float(self.timestep_cal.text())) #Timestamps are created in utils
		self.params = Calibrate(timestamps, float(self.timestep_cal.text()), self.timeformat_cal, self.timestamps_cal_df, self.directory_wind, method = self.method_cal, batch_size = int(self.batchsize_cal.text())) # SPL and wind estimation are launched
		self.params.signals.finished.connect(self.calibration_complete)
		self.threadpool.start(self.params)

	def save_user(self):
		'''
		Saves user parameters from tab2 to config/literature.toml
		Fills possible blank values with default values defined here.
		'''
		file_path = "config/literature.toml"
		if self.user_samplerate.text() == "":
			self.user_samplerate.setText('32000')
		if self.user_frequency.text() == "":
			self.user_frequency.setText('8000')
		if self.user_duration.text() == "":
			self.user_duration.setText('10')
		if self.user_samplelength.text() == "":
			self.user_samplelength.setText('1')
		if self.user_overlap.text() == "":
			self.user_overlap.setText('0.2')
		#Create dictionary to add to toml file
		new_data = b = {'User_Defined': {'model': self.user_method.text(), 
			'metadata': {'samplerate': self.user_samplerate.text(), 'frequency': self.user_frequency.text()}, 
			'preprocessing': {'window': 'hanning','duration': self.user_duration.text(), 'sample_length': self.user_samplelength.text(), 'sample_number': '', 'overlap': self.user_overlap.text()},
			'parameters': dict(zip(string.ascii_lowercase, self.user_parameters.text().replace(" ", "").split(',')))}}
		append_to_toml(file_path, new_data)  #process is detailed in utils

	def save_hyd(self):
		'''
		Saves user parameters for hydrophone from tab2 to config/hydrophone.toml
		Fills possible blank values with default values defined here.
		'''
		file_path = "config/device.toml"
		if self.hyd_sensitivity.text() == "":
			self.hyd_sensitivity.setText('0')
		if self.hyd_gain.text() == "":
			self.hyd_gain.setText('0')
		if self.hyd_voltage.text() == "":
			self.hyd_voltage.setText('0')
		if self.hyd_bits.text() == "":
			self.hyd_bits.setText('0')
		#Create dictionary to add to toml file
		new_data = b = {'hydrophone': {'sensitivity': float(self.hyd_sensitivity.text()), 
			'gain': float(self.hyd_gain.text()), 
			'Vadc': float(self.hyd_voltage.text()),
			'Nbits': float(self.hyd_bits.text())}}
		write_toml_file(file_path, new_data)  #process is detailed in utils


	def api_download(self):
		self.loading_bar_era.setMaximum(3)
		months = list((np.where([month.isChecked() for month in self.months])[0]+1).astype(str))
		days = list((np.where([day.isChecked() for day in self.days])[0]+1).astype(str))
		if not os.path.isdir('data') :
			os.mkdir('data')
		self.loading_bar_era.setValue(1)
		if os.name == 'nt' :
			make_windows_cds_file(self.key.text(), self.uid.text())
		elif os.name == 'posix':
			make_cds_file(self.key.text(), self.uid.text())
		else :
			raise NotImplementedError('Acoustic wind toolbox not implemented for systems other than Windows or Linux')
		self.loading_bar_era.setValue(2)
		df = download_era_data(os.path.join(os.getcwd(), "data"), "era", self.key, ['10m_u_component_of_wind', '10m_v_component_of_wind'],
			self.years.text().replace(" ", "").split(','), months, days, ['00:00','01:00','02:00','03:00','04:00','05:00','06:00', '07:00','08:00','09:00','10:00','11:00','12:00', '13:00','14:00','15:00','16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00'], 
			[self.north_boundary.text(), self.west_boundary.text(), self.south_boundary.text(), self.east_boundary.text()]) 

		self.loading_bar_era.setValue(3)



app = QApplication.instance() 
if not app:
    app = QApplication(sys.argv)

fen = Fenetre()
fen.resize(2000, 1000)
fen.move(300,50)

fen.show()

app.exec_()
