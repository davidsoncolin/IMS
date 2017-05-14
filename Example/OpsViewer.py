import sys

from PySide import QtCore, QtGui
from UI import QApp, QGLViewer
from Ops import Runtime, RenderCallback

Runtime = Runtime.getInstance()

def setFrame(frame):
	Runtime.cookOps(win, frame)
	updateApp()

def updateApp():
	QApp.app.updateLayers()
	QApp.app.refreshImageData()
	QApp.app.updateGL()

def dirtyCB(dirty):
	Runtime.cookOps(win, Runtime.getFrame())

def main():
	appIn = QtGui.QApplication(sys.argv)
	appIn.setStyle('plastique')

	global win
	win = QApp.QApp()
	win.setWindowTitle('Imaginarium MoCap Software')
	win.updateMenus()
	Runtime.addOpMenu(win)
	win.dirtyCB = dirtyCB
	Runtime.setCallbacks(win)
	RenderCallback.Factory(win, Runtime)

	appIn.connect(appIn, QtCore.SIGNAL('lastWindowClosed()'), appIn.quit)
	QGLViewer.makeViewer(timeRange=(0, 1400), callback=setFrame, appIn=appIn, win=win, pickCallback=Runtime.picked)

if __name__ == '__main__':
	main()
