#!/usr/bin/env python

from IO import  GiantReader
from UI import QGLViewer, QApp  # gui
from GCore import Calibrate
from GCore import Label
from numpy import abs
import functools

prev_frame = None

def intersectRaysCB(fi, raw_frames, mats, primitives, primitives2D, track3d):
	global prev_frame
	skipping = prev_frame is None or abs(fi - prev_frame) > 10
	prev_frame = fi
	view = QApp.view()
	points,altpoints = primitives

	g2d = primitives2D[0]
	frame = raw_frames[fi]

	# frame = x2d_frames[fi]
	x2ds_data,x2ds_splits= GiantReader.frameCentroidsToDets(frame, None)

	g2d.setData(x2ds_data,x2ds_splits)
	if skipping:
		x3ds,x3ds_labels = track3d.boot(x2ds_data, x2ds_splits)
	else:
		x3ds,x3ds_labels = track3d.push(x2ds_data, x2ds_splits)
	points.setData(x3ds)

	view.updateGL()

def GiantViewer(raw_filename, cal_filename):
	'''Generate a 3D view of an x2d file, using the calibration.'''
	raw_frames = {100: None}
	print 'loading calibration'
	camera_info = GiantReader.readCal(cal_filename)

	camera_ids = None
	print "Camera IDs:\n{}".format(camera_ids)
	print 'loading 2d'
	raw_dict = GiantReader.readAsciiRaw(raw_filename)
	raw_frames = raw_dict['frames']

	print 'num frames', raw_dict['numFrames']

	mats = [Calibrate.makeMat(camera['MAT'], camera['DISTORTION'], (512, 440)) for camera in camera_info['Cameras']]
	track3d = Label.Track3D(mats)

	primitives = QGLViewer.makePrimitives(vertices = [], altVertices=[])
	primitives2D = QGLViewer.makePrimitives2D(([],[0]))
	cb = functools.partial(intersectRaysCB, raw_frames=raw_frames, mats=mats, primitives=primitives, primitives2D=primitives2D, track3d=track3d)
	QGLViewer.makeViewer(primitives=primitives, primitives2D=primitives2D, timeRange=(1, max(raw_frames.keys()), 1, 100.0), callback=cb, mats=mats,camera_ids=camera_ids) # , callback=intersectRaysCB


if __name__=='__main__':
	import sys, os
	print sys.argv
	if len(sys.argv) == 3:
		raw_filename, cal_filename = sys.argv[1:]
		GiantViewer(raw_filename, cal_filename)
	elif len(sys.argv) == 4:
		raw_filename, cal_filename, c3d_filename = sys.argv[1:]
		GiantViewer(raw_filename, cal_filename, c3d_filename)

