#!/usr/bin/env python

from PySide import QtCore, QtGui
from QCore import QslideLimitValue
import numpy as np

class QActorDofsWidget(QtGui.QWidget):
	def __init__(self, name, cb, parent=None):
		self.name = name
		self.cb = cb
		self.parent = parent
		QtGui.QWidget.__init__(self)
		#QtGui.QGroupBox.__init__(self, name, parent)
		self.groupLayout = QtGui.QVBoxLayout(self)
		self.setLayout(self.groupLayout)
		self.reset()

	def reset(self):
		self.valueIsAdjusting = False
		self.sliders = []
		for child in self.groupLayout.children(): child.deleteLater()

	def syncSliders(self, dofValues):
		for di,x in enumerate(dofValues):
			self.sliders[di].sync(x)
		
	def setDofs(self, dofNames, dofValues = None, dofLimits = None):
		self.reset()
		dofNames = [str(name) for name in dofNames]
		if dofValues is None: dofValues = [0.0 for x in dofNames]
		else:                 dofValues = [float(x) for x in dofValues]
		if dofLimits is None: dofLimits = [[-1.5,1.5] for x in dofNames] #[[-float('inf'),float('inf')] for x in dofNames]
		else:                 dofLimits = [[lo,hi] for lo,hi in dofLimits]
		for di,(name,value,(loval,hival)) in enumerate(zip(dofNames, dofValues, dofLimits)):
			slider = QslideLimitValue(name, value, -1.5, 1.5, self.cb, self.name, parent=self)
			self.sliders.append(slider)
			self.groupLayout.addLayout(slider)

if __name__ == '__main__':
	import sys
	global app, win
	app = QtGui.QApplication(sys.argv)

	def test(actor, dof, value):
		print 'cb',actor,dof,value
	
	win = QActorDofsWidget('colin',test)
	win.setDofs(['dof_rx','dof_ry'],[0.0,10.0],[[-3.0,3.0],(-float('inf'),float('inf'))])
	win.show()
	
	app.connect(app, QtCore.SIGNAL('lastWindowClosed()') , app.quit)
	sys.exit(app.exec_())
