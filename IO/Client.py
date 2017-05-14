#!/usr/bin/env python

import sys

import ReframeClient

from PySide import QtCore, QtGui

from UI import QApp, QActorWidget, GLGrid, GLPoints3D, GLSkeleton

def setActorVisible(name, visible):
	global view, skels, points
	try: view.primitives.remove(skels[name])
	except: pass
	try: view.primitives.remove(points[name])
	except: pass
	if bool(visible):
		view.primitives.append(skels[name])
		view.primitives.append(points[name])



def nextFrame():
	"""
	converts motion capture data into a number of items of global data
	which will be used to perform the OpenGL drawing
	"""
	global view,actors,skels,client
	frame = client.subscribe()

	for name in skels: actors.setActorData(name, False)
	if type(frame) == dict:
		frameHash = frame['hash']
		if frameHash == client.stateHash:
			for subject,subject_state in zip(frame['subjects'], client.state['subjects']):
				name = subject['name']
				if not skels.has_key(name):
					actor = actors.addActor(name)
					print 'adding',name
					boneNames = subject_state['bone_names']
					boneParents = [[boneNames.index([parent,boneNames[0]][parent=='']),-1][parent==''] for parent in subject_state['bone_parents']]
					boneTs = subject['bone_Ts']
					skel = GLSkeleton(boneNames, boneParents, boneTs)
					skel.setName(name)
					view.primitives.append(skel)
					skels[name] = skel
					point = GLPoints3D(subject['marker_Ts'], subject_state['marker_names'])
					points[name] = point
				actors.setActorData(name, True)
				skels[name].vertices[:] = subject['bone_Ts']
				points[name].vertices[:] = subject['marker_Ts']
		if frame.has_key('unlabelled_markers'):
			global unlabelledMarkers
			unlabelledMarkers.vertices = list(frame['unlabelled_markers'])
		else:
			unlabelledMarkers.vertices = []
	view.updateGL()

"""
A reframe client app which reads motion capture data from a
reframe server and displays this using OpenGL, in a QuickTime GUI.
"""

def main():
	# Program options
	hostName = "localhost:18667"
	if (len(sys.argv) > 1):
		hostName = sys.argv[1]
		print 'Setting hostName to ',hostName

	global app,win,view,state,stateHash, actors
	app = QtGui.QApplication(sys.argv)
	app.setStyle('plastique')
	win = QApp.QApp()
	win.setWindowTitle('Imaginarium Client')
	view = win.view()
	win.resize(640, 480)
	actorsDock = QtGui.QDockWidget('Actors')
	actors = QActorWidget.QActorWidget(setActorVisible)
	area = QtGui.QScrollArea()
	area.setMinimumWidth(200)
	area.setWidgetResizable(True)
	area.setWidget(actors)
	actorsDock.setWidget(area)
	actorsDock.setFeatures(QtGui.QDockWidget.DockWidgetMovable|QtGui.QDockWidget.DockWidgetFloatable)
	win.addDockWidget(QtCore.Qt.RightDockWidgetArea, actorsDock)
	win.show()
	global unlabelledMarkers
	unlabelledMarkers = GLPoints3D([])
	view.primitives.append(unlabelledMarkers)
	global skels, points
	skels = {}
	points = {}
	view.primitives.append(GLGrid())
	timer = QtCore.QTimer(app)
	app.connect(timer, QtCore.SIGNAL('timeout()'), nextFrame)
	timer.start(20)

	global client
	client = ReframeClient('localhost',18667)
	
	app.connect(app, QtCore.SIGNAL('lastWindowClosed()') , app.quit)
	sys.exit(app.exec_())

if __name__ == '__main__' :
	main()
