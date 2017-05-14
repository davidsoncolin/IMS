#!/usr/bin/env python

from PySide import QtCore, QtGui

class QActorWidget(QtGui.QWidget):
	def __init__(self, cb, parent=None):
		self.cb = cb
		self.parent = parent
		QtGui.QWidget.__init__(self, parent)
		self.groupLayout = QtGui.QVBoxLayout(self)
		self.groupTabs = QtGui.QTabWidget()
		self.groupLayout.addWidget(self.groupTabs)
		self.groupLayout.addStretch(1)
		self.setLayout(self.groupLayout)
		self.actors = {}
		self.actorNames = []
		self.notRecordingPixmap = QtGui.QPixmap("img/NotRecording.png").scaledToHeight(16)
		self.recordingPixmap = QtGui.QPixmap("img/Recording.png").scaledToHeight(16)
	
	def addActor(self, name):
		if self.actors.has_key(name):
			return self.actors[name]
		self.actors[name] = actor = {}
		self.actorNames.append(name)
		actorGroup = QtGui.QWidget()
		actorGroupLayout = QtGui.QVBoxLayout(actorGroup)
		actorVisible = QtGui.QCheckBox('Visible', actorGroup)
		actorVisible.setCheckState(QtCore.Qt.Checked)
		actorGroup.setLayout(actorGroupLayout)
		actorLabel = QtGui.QLabel()
		actor['group'] = actorGroup
		actor['layout'] = actorGroupLayout
		actor['data'] = actorLabel
		actor['visible'] = actorVisible
		actorVisible.cb = lambda x : self.cb(name, x)
		actorVisible.stateChanged.connect(actorVisible.cb)
		self.groupTabs.addTab(actorGroup,name)
		actorGroupLayout.addWidget(actorVisible)
		actorGroupLayout.addWidget(actorLabel)
		actorLabel.setPixmap(self.recordingPixmap)
		return actor

	def setActorDofs(self, name, dofNames, sharedDofs, cb):
		actor = self.actors[name]
		layout = actor['layout']
		import QActorDofsWidget
		actor['dofsWidget'] = dofsWidget = QActorDofsWidget.QActorDofsWidget(name, cb, self)
		layout.addWidget(dofsWidget)
		dofsWidget.setDofs(dofNames,sharedDofs)

	def syncActorDofs(self, name, dofValues):
		self.actors[name]['dofsWidget'].syncSliders(dofValues)
	
	def setActorData(self, name, value):
		self.actors[name]['data'].setPixmap(self.notRecordingPixmap if value else self.recordingPixmap)

	def removeActor(self, name):
		if self.actors.has_key(name):
			self.actorNames.remove(name)
			self.actors.pop(name)['group'].deleteLater() # mark for deletion!

if __name__ == '__main__':
	import sys
	global app, win
	app = QtGui.QApplication(sys.argv)

	def test(actor, value):
		print 'cb',actor,value
	
	win = QActorWidget(test)
	
	win.addActor('charles')
	win.addActor('colin')
	win.addActor('fred')
	win.setActorData('fred', True)
	win.removeActor('colin')
	win.show()
	
	app.connect(app, QtCore.SIGNAL('lastWindowClosed()') , app.quit)
	sys.exit(app.exec_())
