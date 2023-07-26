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
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from config import literature, device
from models import empirical
from utils import get_files, get_timestamps, append_to_toml
from processing.run import Worker
from processing.preprocessing import Load_Data, compute_spl
import numpy as np

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
		self.tabs.resize(300,200)
		self.tabs.addTab(self.tab1,"Estimate Weather")
		self.tabs.addTab(self.tab2,"Define user parameters")



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
		fns = [get_files(self.directory)[1]]
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



app = QApplication.instance() 
if not app:
    app = QApplication(sys.argv)
    
fen = Fenetre()
fen.resize(2000, 1000)
fen.move(300,50)

fen.show()

app.exec_()
