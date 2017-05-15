#!/usr/bin/env python

# (C) Vicon
import os,sys
import PyViconDataStream # thirdparty
import time
from UI import GLSkeleton, GLPoints3D

AdaptBool = { True:'True',False:'False' }
AdaptDirection = { PyViconDataStream.Direction.Forward:"Forward", PyViconDataStream.Direction.Backward:"Backward", PyViconDataStream.Direction.Left:"Left", PyViconDataStream.Direction.Right:"Right", PyViconDataStream.Direction.Up:"Up", PyViconDataStream.Direction.Down:"Down" }

def viconConnect(HostName='localhost:801', TransmitMulticast=False):
	# Make a new client
	MyClient = PyViconDataStream.Client()

	# Connect to a server
	print "Connecting to ", HostName, " ..."
	while (not MyClient.IsConnected().Connected ):
		# Direct connection
		MyClient.Connect( HostName )

		# Multicast connection
		# MyClient.ConnectToMulticast( HostName, "224.0.0.0" )

		print "."
		time.sleep( 1 )

	# Enable some different data types
	MyClient.EnableSegmentData()
	MyClient.EnableMarkerData()
	MyClient.EnableUnlabeledMarkerData()
	MyClient.EnableDeviceData()

	print "Segment Data Enabled: ", AdaptBool[MyClient.IsSegmentDataEnabled().Enabled]
	print "Marker Data Enabled: ", AdaptBool[MyClient.IsMarkerDataEnabled().Enabled]
	print "Unlabeled Marker Data Enabled: ", AdaptBool[MyClient.IsUnlabeledMarkerDataEnabled().Enabled]
	print "Device Data Enabled: ", AdaptBool[MyClient.IsDeviceDataEnabled().Enabled]

	# Set the streaming mode
	MyClient.SetStreamMode( PyViconDataStream.StreamMode.ClientPull )
	# MyClient.SetStreamMode( PyViconDataStream.StreamMode.ClientPullPreFetch )
	# MyClient.SetStreamMode( PyViconDataStream.StreamMode.ServerPush )

	# Set the global up axis
	MyClient.SetAxisMapping( PyViconDataStream.Direction.Forward, PyViconDataStream.Direction.Left, PyViconDataStream.Direction.Up )

	_Output_GetAxisMapping = MyClient.GetAxisMapping()
	print "Axis Mapping: X-", AdaptDirection[_Output_GetAxisMapping.XAxis], \
			" Y-", AdaptDirection[_Output_GetAxisMapping.YAxis], \
			" Z-", AdaptDirection[_Output_GetAxisMapping.ZAxis]

	# Discover the version number
	_Output_GetVersion = MyClient.GetVersion()
	print "Version: ", _Output_GetVersion.Major, ".", _Output_GetVersion.Minor, "." , _Output_GetVersion.Point

	if( TransmitMulticast ):
		MyClient.StartTransmittingMulticast( "localhost", "224.0.0.0" )
	return MyClient

def viconReadFrame(MyClient):
	# Get a frame
	print "Waiting for new frame..."
	while( MyClient.GetFrame().Result != PyViconDataStream.Result.Success ):
		# non-blocking
		return None
		# Sleep a little so that we don't lumber the CPU with a busy poll
		time.sleep( 1 )
		print "."

	# Get the frame number
	_Output_GetFrameNumber = MyClient.GetFrameNumber()
	print "Frame Number: ", _Output_GetFrameNumber.FrameNumber

	# Get the timecode
	_Output_GetTimecode = MyClient.GetTimecode()

	print "Timecode: ", _Output_GetTimecode.Hours, "h " \
			, _Output_GetTimecode.Minutes, "m " \
			, _Output_GetTimecode.Seconds, "s " \
			, _Output_GetTimecode.Frames, "f " \
			, _Output_GetTimecode.SubFrame, "sf "\
			, AdaptBool[_Output_GetTimecode.FieldFlag ], " "\
			, _Output_GetTimecode.Standard, " "\
			, _Output_GetTimecode.SubFramesPerFrame, " "\
			, _Output_GetTimecode.UserBits

	# Get the latency
	print "Latency: ", MyClient.GetLatencyTotal().Total, "s"

	for LatencySampleIndex in range(MyClient.GetLatencySampleCount().Count):

		SampleName = MyClient.GetLatencySampleName( LatencySampleIndex ).Name
		SampleValue = MyClient.GetLatencySampleValue( SampleName ).Value

		print "  ", SampleName, " ", SampleValue, "s"

	# Count the number of subjects
	SubjectCount = MyClient.GetSubjectCount().SubjectCount
	print "Subjects (", SubjectCount, "):"

	for SubjectIndex in range(SubjectCount):

		print "  Subject #", SubjectIndex

		# Get the subject name
		SubjectName = MyClient.GetSubjectName( SubjectIndex ).SubjectName
		print "            Name: ", SubjectName

		# Get the root segment
		RootSegment = MyClient.GetSubjectRootSegmentName( SubjectName ).SegmentName
		print "    Root Segment: ", RootSegment

		# Count the number of segments
		SegmentCount = MyClient.GetSegmentCount( SubjectName ).SegmentCount
		print "    Segments (", SegmentCount, ")"
		for SegmentIndex in range(SegmentCount):

			print "      Segment #", SegmentIndex

			# Get the segment name
			SegmentName = MyClient.GetSegmentName( SubjectName, SegmentIndex ).SegmentName
			print "          Name: ", SegmentName

			# Get the segment parent
			SegmentParentName = MyClient.GetSegmentParentName( SubjectName, SegmentName ).SegmentName
			print "        Parent: ", SegmentParentName

			# Get the segment's children
			ChildCount = MyClient.GetSegmentChildCount( SubjectName, SegmentName ).SegmentCount
			print "     Children (", ChildCount, "):"
			for ChildIndex in range(ChildCount):

				ChildName = MyClient.GetSegmentChildName( SubjectName, SegmentName, ChildIndex ).SegmentName
				print "       ", ChildName

			# Get the static segment translation
			_Output_GetSegmentStaticTranslation = MyClient.GetSegmentStaticTranslation( SubjectName, SegmentName )
			print "        Static Translation: (", _Output_GetSegmentStaticTranslation.Translation[ 0 ] , ", " \
														, _Output_GetSegmentStaticTranslation.Translation[ 1 ] , ", " \
														, _Output_GetSegmentStaticTranslation.Translation[ 2 ] , ") "

			# Get the static segment rotation in helical co-ordinates
			_Output_GetSegmentStaticRotationHelical = MyClient.GetSegmentStaticRotationHelical( SubjectName, SegmentName )
			print "        Static Rotation Helical: (", _Output_GetSegmentStaticRotationHelical.Rotation[ 0 ], ", " \
															, _Output_GetSegmentStaticRotationHelical.Rotation[ 1 ], ", " \
															, _Output_GetSegmentStaticRotationHelical.Rotation[ 2 ], ") "

			# Get the static segment rotation as a matrix
			_Output_GetSegmentStaticRotationMatrix = MyClient.GetSegmentStaticRotationMatrix( SubjectName, SegmentName )
			print "        Static Rotation Matrix: (", _Output_GetSegmentStaticRotationMatrix.Rotation[ 0 ], ", " \
													, _Output_GetSegmentStaticRotationMatrix.Rotation[ 1 ], ", " \
													, _Output_GetSegmentStaticRotationMatrix.Rotation[ 2 ], ", " \
													, _Output_GetSegmentStaticRotationMatrix.Rotation[ 3 ], ", " \
													, _Output_GetSegmentStaticRotationMatrix.Rotation[ 4 ], ", " \
													, _Output_GetSegmentStaticRotationMatrix.Rotation[ 5 ], ", " \
													, _Output_GetSegmentStaticRotationMatrix.Rotation[ 6 ], ", " \
													, _Output_GetSegmentStaticRotationMatrix.Rotation[ 7 ], ", " \
													, _Output_GetSegmentStaticRotationMatrix.Rotation[ 8 ], ") "

			# Get the static segment rotation in quaternion co-ordinates
			_Output_GetSegmentStaticRotationQuaternion = MyClient.GetSegmentStaticRotationQuaternion( SubjectName, SegmentName )
			print "        Static Rotation Quaternion: (", _Output_GetSegmentStaticRotationQuaternion.Rotation[ 0 ], ", " \
																, _Output_GetSegmentStaticRotationQuaternion.Rotation[ 1 ], ", " \
																, _Output_GetSegmentStaticRotationQuaternion.Rotation[ 2 ], ", " \
																, _Output_GetSegmentStaticRotationQuaternion.Rotation[ 3 ], ") "

			# Get the static segment rotation in EulerXYZ co-ordinates
			_Output_GetSegmentStaticRotationEulerXYZ = MyClient.GetSegmentStaticRotationEulerXYZ( SubjectName, SegmentName )
			print "        Static Rotation EulerXYZ: (", _Output_GetSegmentStaticRotationEulerXYZ.Rotation[ 0 ], ", " \
																, _Output_GetSegmentStaticRotationEulerXYZ.Rotation[ 1 ], ", " \
																, _Output_GetSegmentStaticRotationEulerXYZ.Rotation[ 2 ], ") "

			# Get the global segment translation
			_Output_GetSegmentGlobalTranslation = MyClient.GetSegmentGlobalTranslation( SubjectName, SegmentName )
			print "        Global Translation: (", _Output_GetSegmentGlobalTranslation.Translation[ 0 ] , ", " \
												, _Output_GetSegmentGlobalTranslation.Translation[ 1 ] , ", " \
												, _Output_GetSegmentGlobalTranslation.Translation[ 2 ] , ") " \
												, AdaptBool[_Output_GetSegmentGlobalTranslation.Occluded ]

			# Get the global segment rotation in helical co-ordinates
			_Output_GetSegmentGlobalRotationHelical = MyClient.GetSegmentGlobalRotationHelical( SubjectName, SegmentName )
			print "        Global Rotation Helical: (", _Output_GetSegmentGlobalRotationHelical.Rotation[ 0 ], ", " \
													, _Output_GetSegmentGlobalRotationHelical.Rotation[ 1 ], ", " \
													, _Output_GetSegmentGlobalRotationHelical.Rotation[ 2 ], ") " \
													, AdaptBool[_Output_GetSegmentGlobalRotationHelical.Occluded ]

			# Get the global segment rotation as a matrix
			_Output_GetSegmentGlobalRotationMatrix = MyClient.GetSegmentGlobalRotationMatrix( SubjectName, SegmentName )
			print "        Global Rotation Matrix: (", _Output_GetSegmentGlobalRotationMatrix.Rotation[ 0 ], ", " \
													, _Output_GetSegmentGlobalRotationMatrix.Rotation[ 1 ], ", " \
													, _Output_GetSegmentGlobalRotationMatrix.Rotation[ 2 ], ", " \
													, _Output_GetSegmentGlobalRotationMatrix.Rotation[ 3 ], ", " \
													, _Output_GetSegmentGlobalRotationMatrix.Rotation[ 4 ], ", " \
													, _Output_GetSegmentGlobalRotationMatrix.Rotation[ 5 ], ", " \
													, _Output_GetSegmentGlobalRotationMatrix.Rotation[ 6 ], ", " \
													, _Output_GetSegmentGlobalRotationMatrix.Rotation[ 7 ], ", " \
													, _Output_GetSegmentGlobalRotationMatrix.Rotation[ 8 ], ") " \
													, AdaptBool[_Output_GetSegmentGlobalRotationMatrix.Occluded ]


			# Get the global segment rotation in quaternion co-ordinates
			_Output_GetSegmentGlobalRotationQuaternion = MyClient.GetSegmentGlobalRotationQuaternion( SubjectName, SegmentName )
			print "        Global Rotation Quaternion: (", _Output_GetSegmentGlobalRotationQuaternion.Rotation[ 0 ], ", " \
														 , _Output_GetSegmentGlobalRotationQuaternion.Rotation[ 1 ], ", " \
														 , _Output_GetSegmentGlobalRotationQuaternion.Rotation[ 2 ], ", " \
														 , _Output_GetSegmentGlobalRotationQuaternion.Rotation[ 3 ], ") " \
														 , AdaptBool[_Output_GetSegmentGlobalRotationQuaternion.Occluded ]


			# Get the global segment rotation in EulerXYZ co-ordinates
			_Output_GetSegmentGlobalRotationEulerXYZ = MyClient.GetSegmentGlobalRotationEulerXYZ( SubjectName, SegmentName )
			print "        Global Rotation EulerXYZ: (" , _Output_GetSegmentGlobalRotationEulerXYZ.Rotation[ 0 ], ", " \
														, _Output_GetSegmentGlobalRotationEulerXYZ.Rotation[ 1 ], ", " \
														, _Output_GetSegmentGlobalRotationEulerXYZ.Rotation[ 2 ], ")" \
														, AdaptBool[_Output_GetSegmentGlobalRotationEulerXYZ.Occluded ]

			# Get the local segment translation
			_Output_GetSegmentLocalTranslation = MyClient.GetSegmentLocalTranslation( SubjectName, SegmentName )
			print "        Local Translation: (", _Output_GetSegmentLocalTranslation.Translation[ 0 ], ", " \
												, _Output_GetSegmentLocalTranslation.Translation[ 1 ], ", " \
												, _Output_GetSegmentLocalTranslation.Translation[ 2 ], ")" \
												, AdaptBool[_Output_GetSegmentLocalTranslation.Occluded]

			# Get the local segment rotation in helical co-ordinates
			_Output_GetSegmentLocalRotationHelical = MyClient.GetSegmentLocalRotationHelical( SubjectName, SegmentName )
			print "        Local Rotation Helical: (", _Output_GetSegmentLocalRotationHelical.Rotation[ 0 ], ", " \
													 , _Output_GetSegmentLocalRotationHelical.Rotation[ 1 ], ", " \
													 , _Output_GetSegmentLocalRotationHelical.Rotation[ 2 ], ")" \
													 , AdaptBool[_Output_GetSegmentLocalRotationHelical.Occluded]

			# Get the local segment rotation as a matrix
			_Output_GetSegmentLocalRotationMatrix = MyClient.GetSegmentLocalRotationMatrix( SubjectName, SegmentName )
			print "        Local Rotation Matrix: (", _Output_GetSegmentLocalRotationMatrix.Rotation[ 0 ], ", " \
													, _Output_GetSegmentLocalRotationMatrix.Rotation[ 1 ], ", " \
													, _Output_GetSegmentLocalRotationMatrix.Rotation[ 2 ], ", " \
													, _Output_GetSegmentLocalRotationMatrix.Rotation[ 3 ], ", " \
													, _Output_GetSegmentLocalRotationMatrix.Rotation[ 4 ], ", " \
													, _Output_GetSegmentLocalRotationMatrix.Rotation[ 5 ], ", " \
													, _Output_GetSegmentLocalRotationMatrix.Rotation[ 6 ], ", " \
													, _Output_GetSegmentLocalRotationMatrix.Rotation[ 7 ], ", " \
													, _Output_GetSegmentLocalRotationMatrix.Rotation[ 8 ], ")" \
													, AdaptBool[_Output_GetSegmentLocalRotationMatrix.Occluded ]

			# Get the local segment rotation in quaternion co-ordinates
			_Output_GetSegmentLocalRotationQuaternion = MyClient.GetSegmentLocalRotationQuaternion( SubjectName, SegmentName )
			print "        Local Rotation Quaternion: (", _Output_GetSegmentLocalRotationQuaternion.Rotation[ 0 ], ", " \
														, _Output_GetSegmentLocalRotationQuaternion.Rotation[ 1 ], ", " \
														, _Output_GetSegmentLocalRotationQuaternion.Rotation[ 2 ], ", " \
														, _Output_GetSegmentLocalRotationQuaternion.Rotation[ 3 ], ") " \
														, AdaptBool[_Output_GetSegmentLocalRotationQuaternion.Occluded ]

			# Get the local segment rotation in EulerXYZ co-ordinates
			_Output_GetSegmentLocalRotationEulerXYZ = MyClient.GetSegmentLocalRotationEulerXYZ( SubjectName, SegmentName )
			print "        Local Rotation EulerXYZ: (", _Output_GetSegmentLocalRotationEulerXYZ.Rotation[ 0 ], ", " \
													, _Output_GetSegmentLocalRotationEulerXYZ.Rotation[ 1 ], ", " \
													, _Output_GetSegmentLocalRotationEulerXYZ.Rotation[ 2 ], ") " \
													, AdaptBool[_Output_GetSegmentLocalRotationEulerXYZ.Occluded ]

		# Count the number of markers
		MarkerCount = MyClient.GetMarkerCount( SubjectName ).MarkerCount
		print "    Markers (", MarkerCount, ")"
		for MarkerIndex in range(MarkerCount):

			# Get the marker name
			MarkerName = MyClient.GetMarkerName( SubjectName, MarkerIndex ).MarkerName

			# Get the marker parent
			MarkerParentName = MyClient.GetMarkerParentName( SubjectName, MarkerName ).SegmentName

			# Get the global marker translation
			_Output_GetMarkerGlobalTranslation = MyClient.GetMarkerGlobalTranslation( SubjectName, MarkerName )

			print "      Marker #", MarkerIndex, ": "\
					, MarkerName, " ("\
					, _Output_GetMarkerGlobalTranslation.Translation[ 0 ], ", "\
					, _Output_GetMarkerGlobalTranslation.Translation[ 1 ], ", "\
					, _Output_GetMarkerGlobalTranslation.Translation[ 2 ], ") "\
					, AdaptBool[_Output_GetMarkerGlobalTranslation.Occluded ]
	# Get the unlabeled markers
	UnlabeledMarkerCount = MyClient.GetUnlabeledMarkerCount().MarkerCount
	print "  Unlabeled Markers (", UnlabeledMarkerCount, "):"
	for UnlabeledMarkerIndex in range(UnlabeledMarkerCount):
		# Get the global marker translation
		_Output_GetUnlabeledMarkerGlobalTranslation = MyClient.GetUnlabeledMarkerGlobalTranslation( UnlabeledMarkerIndex )

		print "      Marker #", UnlabeledMarkerIndex, ": ("\
				, _Output_GetUnlabeledMarkerGlobalTranslation.Translation[ 0 ], ", " \
				, _Output_GetUnlabeledMarkerGlobalTranslation.Translation[ 1 ], ", " \
				, _Output_GetUnlabeledMarkerGlobalTranslation.Translation[ 2 ], ") "


def viconParseFrame(MyClient):
	ret = {}
	# Get a frame
	#print "Waiting for new frame..."
	while( MyClient.GetFrame().Result != PyViconDataStream.Result.Success ):
		# non-blocking
		return None

	ret['frame_number'] = MyClient.GetFrameNumber().FrameNumber
	# Get the timecode
	tc = MyClient.GetTimecode()
	ret['tc'] = '%02d:%02d:%02d:%02d:%02d %s %s %s %s' % \
				(tc.Hours,tc.Minutes,tc.Seconds,tc.Frames,tc.SubFrame,AdaptBool[tc.FieldFlag ],tc.Standard,tc.SubFramesPerFrame,tc.UserBits)

	# Get the latency (seconds)
	ret['latency'] = MyClient.GetLatencyTotal().Total

	# Count the number of subjects
	SubjectCount = MyClient.GetSubjectCount().SubjectCount
	ret['num_subjects'] = SubjectCount
	ret['subjects'] = []
	for SubjectIndex in xrange(SubjectCount):
		ret['subjects'].append({})
		subject = ret['subjects'][-1]
		subject['index'] = SubjectIndex

		subject['name'] = SubjectName = MyClient.GetSubjectName( SubjectIndex ).SubjectName
		SegmentCount = MyClient.GetSegmentCount( SubjectName ).SegmentCount
		subject['bone_names'] = BoneNames = [MyClient.GetSegmentName( SubjectName, bi ).SegmentName for bi in xrange(SegmentCount)]
		subject['bone_parents'] = [MyClient.GetSegmentParentName( SubjectName, bn ).SegmentName for bn in BoneNames]
		subject['bone_Ts'] = [MyClient.GetSegmentGlobalTranslation( SubjectName, bn ).Translation for bn in BoneNames]
		subject['bone_Rs'] = [MyClient.GetSegmentGlobalRotationMatrix( SubjectName, bn ).Rotation for bn in BoneNames]

		# Count the number of markers
		MarkerCount = MyClient.GetMarkerCount( SubjectName ).MarkerCount
		subject['marker_names'] = MarkerNames = [MyClient.GetMarkerName( SubjectName, mi ).MarkerName for mi in xrange(MarkerCount)]
		subject['marker_bones'] = [MyClient.GetMarkerParentName( SubjectName, mn ).SegmentName for mn in MarkerNames]
		subject['marker_Ts'] = [MyClient.GetMarkerGlobalTranslation( SubjectName, mn ).Translation for mn in MarkerNames]
		subject['marker_occludeds'] = [MyClient.GetMarkerGlobalTranslation( SubjectName, mn ).Occluded for mn in MarkerNames]

	# Get the unlabeled markers
	UnlabeledMarkerCount = MyClient.GetUnlabeledMarkerCount().MarkerCount
	ret['unlabelled_markers'] = [MyClient.GetUnlabeledMarkerGlobalTranslation( mi ).Translation for mi in xrange(UnlabeledMarkerCount)]

	return ret

def setActorVisible(name, visible):
	try: view.primitives.remove(skels[name])
	except: pass
	try: view.primitives.remove(points[name])
	except: pass
	if bool(visible):
		view.primitives.append(skels[name])
		view.primitives.append(points[name])
	
def nextFrame():
	global view,actors,MyClient, skels
	frame = viconParseFrame(MyClient)
	for name in skels: actors.setActorData(name, False)
	if type(frame) == dict:
		for subject in frame['subjects']:
			name = subject['name']
			if not skels.has_key(name):
				actor = actors.addActor(name)
				print 'adding',name
				boneNames = subject['bone_names']
				boneParents = [[boneNames.index([parent,boneNames[0]][parent=='']),-1][parent==''] for parent in subject['bone_parents']]
				boneTs = subject['bone_Ts']
				skel = GLSkeleton(boneNames,boneParents,boneTs)
				skel.setName(name)
				view.primitives.append(skel)
				skels[name] = skel
				point = GLPoints3D(subject['marker_Ts'], subject['marker_names'])
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


def main():
	# Program options
	global actors
	TransmitMulticast = False
	HostName = "localhost:801"
	import sys
	if (len(sys.argv) > 1):
		HostName = sys.argv[1]
		print 'Setting hostname to ',HostName

	from UI import QApp, QActorWidget, GLGrid, GLPoints3D, GLSkeleton
	from PySide import QtCore, QtGui, QtOpenGL
	import numpy as np

	global app,win,view
	app = QtGui.QApplication(sys.argv)
	app.setStyle('plastique')
	win = QApp.QApp()
	win.setWindowTitle('Imaginarium ViconConnect')
	view = win.view()
	view.setMinimumWidth(640)
	view.setMinimumHeight(480)
	win.setCentralWidget(view)
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
	global MyClient, skels, points
	skels = {}
	points = {}
	view.primitives.append(GLGrid())
	timer = QtCore.QTimer(app)
	app.connect(timer, QtCore.SIGNAL('timeout()'), nextFrame)
	timer.start(20)

	MyClient = viconConnect(HostName, TransmitMulticast)
	
	app.connect(app, QtCore.SIGNAL('lastWindowClosed()') , app.quit)
	sys.exit(app.exec_())

	#for i in range(3):
	## repeat to check disconnecting doesn't wreck next connect
		#MyClient = viconConnect(HostName, TransmitMulticast)

		#while(True):
			#frame = viconParseFrame(MyClient)

		#if( TransmitMulticast ):
			#MyClient.StopTransmittingMulticast()

		#MyClient.DisableSegmentData()
		#MyClient.DisableMarkerData()
		#MyClient.DisableUnlabeledMarkerData()
		#MyClient.DisableDeviceData()

		## Disconnect and dispose
		#print " Disconnecting..."
		#MyClient.Disconnect()
if __name__ == '__main__' :
	main()
