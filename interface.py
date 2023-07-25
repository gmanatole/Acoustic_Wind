import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtCore import Qt as QtDisp
import matplotlib
matplotlib.use('Qt5Agg')
from pyqtgraph import *
import pyqtgraph as pg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from main import Worker
from matplotlib.figure import Figure
from config import literature, device
from models import empirical
from utils import get_files, get_timestamps
from processing.preprocessing import Load_Data, compute_spl
import numpy as np

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

'''class TableModel(QtCore.QAbstractTableModel):
    def __init__(self, data):
        super(TableModel, self).__init__()
        self._data = data

    def data(self, index, role):
        if role == Qt.DisplayRole:
            return self._data[index.row()][index.column()]

    def rowCount(self, index):
        return len(self._data)

    def columnCount(self, index):
        return len(self._data[0])'''


class Fenetre(QWidget):
	
	def __init__(self):

		QWidget.__init__(self)
		self.setWindowTitle("Estimate wind speed with underwater acoustic recordings")
		self.threadpool = QThreadPool()

		layout = QGridLayout()

		#PREPARE PLOT
		#self.sc = MplCanvas(self, width=5, height=4, dpi=100)
		self.wind_plot = PlotWidget()		
		layout.addWidget(self.wind_plot, 4, 2)

		#PREPARE TABLE
		self.spl_table = QTableWidget()
		self.spl_table.setColumnCount(3)
		self.spl_table.setRowCount(1)
		self.spl_table.setItem(0, 0, QTableWidgetItem('Timestamp'))
		self.spl_table.setItem(0, 1, QTableWidgetItem('Wind speed (m/s)'))
		self.spl_table.setItem(0, 2, QTableWidgetItem('SPL (db re 1 uPa)'))
		layout.addWidget(self.spl_table, 4, 1)

		#METHOD SELECTION
		self.dropdown = QComboBox(self)
		for key in literature.keys():
			self.dropdown.addItem(key)
		self.dropdown.addItem('User Defined')
		self.method = 'Default'
		self.dropdown.currentIndexChanged.connect(self.handle_selection_change)
		self.dropdown_label = QLabel('No method selected')
		layout.addWidget(self.dropdown_label, 0, 1)
		layout.addWidget(self.dropdown, 0, 0)

		#BATCH SIZE AND TIME STEP
		self.timestep_label = QLabel('Please enter audio timestep (in seconds)')
		layout.addWidget(self.timestep_label, 2, 0)
		self.timestep = QLineEdit(self)
		layout.addWidget(self.timestep, 2, 1)
		self.batchsize_label = QLabel('Please enter batchsize')
		layout.addWidget(self.batchsize_label, 3, 0)
		self.batchsize = QLineEdit(self)
		layout.addWidget(self.batchsize, 3, 1)


		#ESTIMATION BUTTON
		self.estimate = QPushButton("Estimate wind speed")
		self.estimate.clicked.connect(self.press_estimate)
		#self.estimate.setGeometry(250, 250, 20, 30)  
		#self.estimate_label.setGeometry(250, 260, 20, 30) 
		layout.addWidget(self.estimate, 0, 2)
		layout.setColumnStretch(0, 1)
		layout.setColumnStretch(1, 2)
		layout.setColumnStretch(2, 4)
		layout.setRowStretch(0, 1)
		layout.setRowStretch(1, 2)
		layout.setRowStretch(2, 3)
		layout.setRowStretch(3, 4)
		layout.setRowStretch(4, 6)
		self.setLayout(layout)

		# DIRECTORY SELECTION
		self.directory_button = QPushButton("Select Directory")
		self.directory_button.clicked.connect(self.select_directory)
		layout.addWidget(self.directory_button, 1, 0)
		self.directory_label = QLabel('No directory selected')
		layout.addWidget(self.directory_label, 1, 1)

	def select_directory(self):
		self.directory = QFileDialog.getExistingDirectory(self, "Select Directory")
		if self.directory:
			self.directory_label.setText(self.directory)

	def handle_selection_change(self, index):
		# Retrieve the selected item
		self.method = self.dropdown.currentText()
		self.dropdown_label.setText(f'{self.method} method selected')

		
	def computation_complete(self):
		#self.sc.axes.plot(self.worker.ts, self.worker.wind_speed)
		self.wind_plot.plot(self.worker.ts, self.worker.wind_speed)
		table_data = np.hstack((np.array(self.worker.ts).reshape(-1,1), np.array(self.worker.wind_speed).reshape(-1,1), np.array(self.worker.sound_pressure_level).reshape(-1,1)))
		self.spl_table.setRowCount(len(table_data)+1)
		for i, elem in enumerate(table_data):
			self.spl_table.setItem(i+1, 0, QTableWidgetItem(str(elem[0])))
			self.spl_table.setItem(i+1, 1, QTableWidgetItem(str(np.round(elem[1], 2))))
			self.spl_table.setItem(i+1, 2, QTableWidgetItem(str(np.round(elem[2], 2))))

	def press_estimate(self):

		iteration = 0
		fns = [get_files(self.directory)[1]]
		timestamps = get_timestamps(fns, float(self.timestep.text())) 
		self.worker = Worker(timestamps, method = self.method, batch_size = int(self.batchsize.text()))
		self.worker.signals.finished.connect(self.computation_complete)
		self.threadpool.start(self.worker)

		'''for batch in inst :
			self.estimate_label.setText(f'Estimating wind speed... {int((iteration*100)/len(inst))} %')
			iteration+=1
			inst.run(batch)'''

'''
class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)'''


app = QApplication.instance() 
if not app:
    app = QApplication(sys.argv)
    
fen = Fenetre()
fen.resize(2000, 1000)
fen.move(300,50)

fen.show()

app.exec_()
