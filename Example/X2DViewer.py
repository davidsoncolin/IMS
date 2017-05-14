#!/usr/bin/env python

import numpy as np
from IO import ViconReader, C3D
from UI import QGLViewer, QApp  # gui
from GCore import Label

def intersectRaysCB(fi):
	global x2d_frames,mats,Ps,c3d_frames,view,primitives,primitives2D,track3d,prev_frame
	skipping = prev_frame is None or np.abs(fi - prev_frame) > 10
	prev_frame = fi
	view = QApp.view()
	points,altpoints = primitives
	g2d = primitives2D[0]
	frame = x2d_frames[fi]
	x2ds_data,x2ds_splits = ViconReader.frameCentroidsToDets(frame,mats)
	#print x2ds_data
	g2d.setData(x2ds_data,x2ds_splits)
	if skipping: 
		x3ds,x3ds_labels = track3d.boot(x2ds_data, x2ds_splits)
	else:
		x3ds,x3ds_labels = track3d.push(x2ds_data, x2ds_splits)
	points.setData(x3ds)
	if c3d_frames is not None:
		c3ds = c3d_frames[(fi-832)/2]
		true_labels = np.where(c3ds[:,3]==0)[0].astype(np.int32)
		x3ds_true = c3ds[true_labels,:3]
		altpoints.setData(x3ds_true)
	view.updateGL()

def X2DViewer(x2d_filename, xcp_filename, c3d_filename = None):
	'''Generate a 3D view of an x2d file, using the calibration.'''
	global x2d_frames,mats,Ps,c3d_frames,primitives,primitives2D,track3d,prev_frame
	prev_frame = None
	c3d_frames = None
	if c3d_filename is not None:
		c3d_dict = C3D.read(c3d_filename)
		c3d_frames, c3d_fps, c3d_labels = c3d_dict['frames'],c3d_dict['fps'],c3d_dict['labels']
	mats,xcp_data = ViconReader.loadXCP(xcp_filename)
	camera_ids = [int(x['DEVICEID']) for x in xcp_data]
	print ('loading 2d')
	x2d_dict = ViconReader.loadX2D(x2d_filename)
	x2d_frames = x2d_dict['frames']
	print ('num frames', len(x2d_frames))
	mat = mats[0]
	print ("Mat: {}".format(len(mat)))
	print ("mat = [K,RT,np.dot(K,RT),np.array([k1,k2],dtype=np.float32),d['POSITION'],d['SENSOR_SIZE']")
	print ("K:\n\t{}".format(mat[0]))
	print ("RT:\n\t{}".format(mat[1]))
	print ("K(RT):\n\t{}".format(mat[2]))
	print ("Distortion:\n\t{}".format(mat[3]))
	print ("Position:\n\t{}".format(mat[4]))
	print ("Sensor Size:\n\t{}".format(mat[5]))
	Ps = [m[2]/(m[0][0,0]) for m in mats]
	print (len(Ps))
	print (Ps[0])
	track3d = Label.Track3D(mats)
	
	primitives = QGLViewer.makePrimitives(vertices = [], altVertices=[])
	primitives2D = QGLViewer.makePrimitives2D(([],[0]))

	QGLViewer.makeViewer(primitives=primitives, primitives2D=primitives2D, timeRange=(0, len(x2d_frames) - 1, 1, 100.0), callback=intersectRaysCB, mats=mats,camera_ids=camera_ids)


if __name__=='__main__':
	import sys, os
	print (sys.argv)
	if len(sys.argv) == 1:
		directory = os.path.join(os.environ['GRIP_DATA'],'151110')
		x2d_filename = os.path.join(directory,'50_Grip_RoomCont_AA_02.x2d')
		xcp_filename = os.path.join(directory,'50_Grip_RoomCont_AA_02.xcp')
		X2DViewer(x2d_filename, xcp_filename)
	elif len(sys.argv) == 3:
		x2d_filename, xcp_filename = sys.argv[1:]
		X2DViewer(x2d_filename, xcp_filename)
	elif len(sys.argv) == 4:
		x2d_filename, xcp_filename, c3d_filename = sys.argv[1:]
		X2DViewer(x2d_filename, xcp_filename, c3d_filename)

