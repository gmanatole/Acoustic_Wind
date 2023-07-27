import sys
import string
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtCore import Qt as QtDisp
import matplotlib
matplotlib.use('Qt5Agg')
from pyqtgraph import *
import pyqtgraph as pg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from config import literature, device
from models import empirical
from utils import get_files, get_timestamps, append_to_toml
from processing.run import Worker
from processing.preprocessing import Load_Data, compute_spl
import numpy as np
from api.access_cds import make_cds_file, download_era_data

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


class Fenetre(QWidget):
	
	'''
	Class that create GUI. Multiple tabs are created.
	Tab 1 : Main interface to load acoustic data, chose method and get wind speed estimation.
	Tab 2 : Used to enter the user's own method and parameters for wind speed estimation.
	'''
	
	def __init__(self):

		super(QWidget, self).__init__()
		self.layout = QVBoxLayout(self)
		self.setWindowTitle("Estimate wind speed with underwater acoustic recordings")
		#instantiate threadpool for worker class
		self.threadpool = QThreadPool()

		# CREATE TABS
		self.tabs = QTabWidget()
		self.tab1 = QWidget()
		self.tab2 = QWidget()
		self.tab3 = QWidget()
		self.tabs.addTab(self.tab1,"Estimate Weather")
		self.tabs.addTab(self.tab3,"Download ERA wind speed")
		self.tabs.addTab(self.tab2,"Define user parameters")
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
		self.dropdown = QComboBox(self)
		for key in literature.keys():
			self.dropdown.addItem(key)
		self.dropdown.addItem('User Defined')
		self.method = 'Default'
		self.dropdown.currentIndexChanged.connect(self.handle_selection_change)
		self.dropdown_label = QLabel('No method selected')
		self.tab1.layout.addWidget(self.dropdown_label, 0, 1)
		self.tab1.layout.addWidget(self.dropdown, 0, 0)

		#BATCH SIZE AND TIME STEP
		self.timestep_label = QLabel('Please enter audio timestep (in seconds)')
		self.tab1.layout.addWidget(self.timestep_label, 2, 0)
		self.timestep = QLineEdit(self)
		self.tab1.layout.addWidget(self.timestep, 2, 1)
		self.batchsize_label = QLabel('Please enter batchsize')
		self.tab1.layout.addWidget(self.batchsize_label, 3, 0)
		self.batchsize = QLineEdit(self)
		self.tab1.layout.addWidget(self.batchsize, 3, 1)

		#ESTIMATION BUTTON
		self.estimate = QPushButton("Estimate wind speed")
		self.estimate.clicked.connect(self.press_estimate)
		self.tab1.layout.addWidget(self.estimate, 0, 2)
		
		# DIRECTORY SELECTION
		self.directory_button = QPushButton("Select Directory")
		self.directory_button.clicked.connect(self.select_directory)
		self.tab1.layout.addWidget(self.directory_button, 1, 0)
		self.directory_label = QLabel('No directory selected')
		self.tab1.layout.addWidget(self.directory_label, 1, 1)

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

		#FIX GEOMETRY
		self.tab2.layout.setColumnStretch(0, 3)
		self.tab2.layout.setColumnStretch(3, 4)
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

		self.layout.addWidget(self.tabs)
		self.setLayout(self.layout)

	def select_directory(self):
		'''
		Get and store directory path where sound files to analyze are stored
		'''
		self.directory = QFileDialog.getExistingDirectory(self, "Select Directory")
		if self.directory:
			self.directory_label.setText(self.directory)

	def handle_selection_change(self, index):
		'''
		Dropdown menu to chose method for wind speed estimation
		'''
		self.method = self.dropdown.currentText()
		self.dropdown_label.setText(f'{self.method} method selected')

		
	def computation_complete(self):
		'''
		Once computation is complete, show results in table and as a plot. Data are stored in worker class.
		'''
		self.wind_plot.plot(self.worker.ts, self.worker.wind_speed)
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
		fns = get_files(self.directory)
		timestamps = get_timestamps(fns, float(self.timestep.text())) #Timestamps are created in utils
		self.worker = Worker(timestamps, method = self.method, batch_size = int(self.batchsize.text())) # SPL and wind estimation are launched
		self.worker.signals.finished.connect(self.computation_complete)   # Check if computation is finished to plot data and show table
		self.threadpool.start(self.worker)


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

	def api_download(self):
		months = list((np.where([month.isChecked() for month in self.months])[0]+1).astype(str))
		days = list((np.where([day.isChecked() for day in self.days])[0]+1).astype(str))
		if not os.path.isdir('data') :
			os.mkdir('data')
		make_cds_file(self.key.text(), self.uid.text(), os.path.join(os.getcwd(), 'data'))
		df = download_era_data(os.path.join(os.getcwd(), "data"), "era", self.key, ['10m_u_component_of_wind', '10m_v_component_of_wind'],
			self.years.text().replace(" ", "").split(','), months, days, ['00:00','01:00','02:00','03:00','04:00','05:00','06:00', '07:00','08:00','09:00','10:00','11:00','12:00', '13:00','14:00','15:00','16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00'], 
			[self.north_boundary.text(), self.west_boundary.text(), self.south_boundary.text(), self.east_boundary.text()])


app = QApplication.instance() 
if not app:
    app = QApplication(sys.argv)
    
fen = Fenetre()
fen.resize(2000, 1000)
fen.move(300,50)

fen.show()

app.exec_()
