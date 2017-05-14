import Op, Interface
import numpy as np
from GCore import Calibrate


class Cameras(Op.Op):
	def __init__(self, name='/Calibrate Cameras', locations='', detections='', x3ds='', solve_focal_length=True, solve_distortion=True,
	             error_threshold=0.05, min_samples=100, jumpFrames=5, showDetections=False, frameRange=''):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('locations', 'Camera locations', 'Camera locations', 'string', locations, {}),
			('detections', '2D Detections', '2D Detections', 'string', detections, {}),
			('x3ds', '3D Points', '3D Points', 'string', x3ds, {}),
			('solve_focal_length', 'Solve Focal Length', 'Solve Focal Length', 'bool', solve_focal_length, {}),
			('solve_distortion', 'Solve Distortion', 'Solve Distortion', 'bool', solve_distortion, {}),
			('error_threshold', 'Error Threshold', 'Error Threshold', 'float', error_threshold, {}),
			('min_samples', 'Min. samples', 'Min. samples to solve distortion', 'int', min_samples, {}),
			('jumpFrames', 'Jump Frames', 'Handle every Nth frame', 'int', jumpFrames, {}),
			('showDetections', 'Show detections', 'Show all collected detections', 'bool', showDetections, {}),
			('frameRange', 'Frame range', 'Frame range', 'string', frameRange, {})
		]

		super(self.__class__, self).__init__(name, fields)

	def cook(self, location, interface, attrs):
		if not self.useFrame(interface.frame(), attrs['frameRange']):
			interface.setAttr('updateMats', False)
			return

		# We need 2D data e.g. wand detections from a wand op
		# We need 3D wand data from e.g. c3d or a 3D wand detector
		dets_location = attrs['detections']
		x3ds_location = attrs['x3ds']
		if not dets_location or not x3ds_location: return

		# Get the 2D and 3D data
		x2ds = interface.attr('rx2ds', atLocation=dets_location)
		x2d_splits = interface.attr('x2ds_splits', atLocation=dets_location)
		x3ds = interface.attr('x3ds', atLocation=x3ds_location)

		if x2ds is None or x2d_splits is None or x3ds is None: return

		numCameras = len(x2d_splits) - 1
		error_threshold = attrs['error_threshold']

		# Get the data we've collected already so we can add to it
		frame = interface.frame()
		dets_colours = interface.attr('x2ds_colours', atLocation=dets_location)
		collectedDets = interface.attr('collect_rx2ds')
		collectedX3ds = interface.attr('collect_x3ds')
		lastFrame = interface.attr('lastFrame', [frame] * numCameras)
		emptyFrame3d = np.array([[]], dtype=np.float32).reshape(-1, 3)

		# This is potentially used by other ops so we only set it when we have some confidence
		# (and we might reset or tweak the values to indicate confidence levels at some point)
		cameraErrors = interface.attr('cameraErrors', [-1] * numCameras)

		# This is never modified to allow checking the camera rms values regardless of what we make of them
		rmsValues = interface.attr('rms', [-1] * numCameras)

		# Get the width and height for the videos
		vwidth = interface.attr('vwidth', [1920] * numCameras)
		vheight = interface.attr('vheight', [1080] * numCameras)

		# Get the frame mapping for x3ds
		x3ds_frames = interface.attr('x3ds_frames', {})
		x2ds_frames = interface.attr('x2ds_frames', [[] for i in xrange(numCameras)])

		# Get the camera matrices. We initialise them with default settings if we don't find any
		mats = interface.attr('mats', atLocation=location)
		if mats is None:
			mats = []
			for ci in range(numCameras):
				mats.append(Calibrate.makeUninitialisedMat(ci, (vheight[ci], vwidth[ci])))

		# Allow overriding the error threshold using an attribute (on the cooked location)
		error_threshold_attr = interface.attr('error_threshold')
		if error_threshold_attr is not None:
			error_threshold = error_threshold_attr

		Ps = interface.attr('Ps')
		if Ps is None: Ps = [np.array([], dtype=np.float32) for n in range(numCameras)]

		# Get the minimum number of samples we need to start solving distortion etc. as specified by the user
		minSamples = attrs['min_samples']

		# Prepare the collected data for further processing (or initialise if nothing has been collected)
		if collectedDets is not None:
			c_x2ds, c_splits = collectedDets
			cams_collected = [c_x2ds[c0:c1] for ci, (c0, c1) in enumerate(zip(c_splits[:-1], c_splits[1:]))]
		else:
			cams_collected = [[] for ci, (c0, c1) in enumerate(zip(x2d_splits[:-1], x2d_splits[1:]))]
			collectedX3ds = []
			for ci, (c0, c1) in enumerate(zip(x2d_splits[:-1], x2d_splits[1:])):
				collectedX3ds.append(emptyFrame3d)

		# Process each camera by looking for a wand and attempt a calibration. If we're happy with the results we'll
		# add it to our collection
		for ci, (c0, c1) in enumerate(zip(x2d_splits[:-1], x2d_splits[1:])):
			elapsed = frame - lastFrame[ci]
			if 0 < elapsed < attrs['jumpFrames']: continue

			# Get the 2Ds and 3Ds for the wand in this camera (if any)
			cameraDetections = x2ds[c0:c1]
			cameraX3ds = x3ds
			if not cameraDetections.any() or not cameraX3ds.any(): continue

			# Add the new detection to the existing collection as a candidate for a new collection
			if cams_collected[ci] is None or len(cams_collected[ci]) == 0:
				proposalDets, proposalX3ds = cameraDetections, cameraX3ds
			else:
				proposalDets = np.concatenate((cams_collected[ci], cameraDetections))
				proposalX3ds = np.concatenate((collectedX3ds[ci], cameraX3ds))

			# Check if we want to solve for distortion and focal length by looking at the number of samples
			# we've got already compared to our minimum number of samples required
			numSamples = len(proposalDets) / 5
			# if numSamples == minSamples: self.logger.info('Camera %d reached min samples of %d' % (ci, minSamples))
			solveTrigger = True if numSamples > minSamples else False
			solve_focal_length = attrs['solve_focal_length'] if solveTrigger else False
			solve_distortion = attrs['solve_distortion'] if solveTrigger else False

			# The wand is assumed to have 5 points so we make sure we've got at least one wand before attempting
			# to calibrate
			if len(proposalDets) >= 5 and len(proposalX3ds) >= 5:
				P, ks, rms = Calibrate.cv2_solve_camera_from_3d(proposalX3ds, proposalDets,
				                                                solve_focal_length=solve_focal_length,
				                                                solve_distortion=solve_distortion)

				if ks[0] < -3. or ks[0] > 3.: ks[0] = 0.
				if ks[1] < -3. or ks[1] > 3.: ks[1] = 0.

				# This shouldn't' happen but if we lose confidence in the camera we can visualise it
				# by resetting the camera error (this will change the colour in the UI)
				if rms > error_threshold:
					cameraErrors[ci] = -1
					continue

				# See how the rms for the calibration compares to the last recorded value for this camera
				prevRms = rms if rmsValues[ci] == -1 else rmsValues[ci]
				rmsDelta = rms - prevRms

				# If the rms is lower than the last recorded error for this camera then
				# we want to keep this data
				if rmsDelta <= 0 or not solveTrigger:
					cams_collected[ci] = proposalDets
					collectedX3ds[ci] = proposalX3ds
					if frame not in x3ds_frames:
						x3ds_frames[frame] = proposalX3ds[-5:]
					x2ds_frames[ci] += ([frame] * 5)
				else:
					continue

				# Record the rms value for the camera
				rmsValues[ci] = rms

				# Once we've solved for distortion etc. we are more confident with the accuracy of our
				# error so we start reporting it, where the value can be used for visualiation etc.
				if solveTrigger: cameraErrors[ci] = rms
				lastFrame[ci] = frame

				# Everything has gone well so far so we create and add the new camera matrix
				mat = Calibrate.makeMat(P, ks, (vheight[ci], vwidth[ci]))
				mats[ci] = mat
				Ps[ci] = P

		# Concatenate the results from all the cameras
		cams = [np.concatenate((cc)) for cc in cams_collected if len(cc)]
		if not cams:
			# We haven't found a wand in any camera so we just keep calm and return
			return

		# Build our collections and write to the interface
		collectedDets = np.array(np.concatenate(cams), dtype=np.float32).reshape(-1, 2), \
		                Interface.makeSplitBoundaries(map(len, cams_collected))
		interface.setAttr('collect_rx2ds', collectedDets)
		interface.setAttr('collect_x3ds', collectedX3ds)
		interface.setAttr('x2ds_frames', x2ds_frames)
		interface.setAttr('x3ds_frames', x3ds_frames)
		interface.setAttr('lastFrame', lastFrame)

		# Write the calibration data to the interface and request an update at render time
		interface.setAttr('mats', mats)
		interface.setAttr('Ps', Ps)
		interface.setAttr('rms', rmsValues)
		interface.setAttr('cameraErrors', cameraErrors)
		interface.setAttr('updateMats', True)

		# Optionally display all the collected wand detections
		if 'showDetections' in attrs and attrs['showDetections']:
			colours = np.tile(dets_colours, (len(collectedDets[0]) / 5, 1))
			allAttrs = {'x2ds': collectedDets[0], 'x2ds_splits': collectedDets[1],
			            'x2ds_colours': colours}
			interface.createChild('collected', 'detections', attrs=allAttrs)


class WandCorrespondences(Op.Op):
	def __init__(self, name='/Wand Correspondences', detections='', matsLocation=''):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('detections', 'Detections', 'Detections', 'string', detections, {}),
			('matsLocation', 'Mats Location', 'Mats locations', 'string', matsLocation, {}),
		]

		super(self.__class__, self).__init__(name, fields)

	def cook(self, location, interface, attrs):
		detections = attrs['detections']
		matsLocation = attrs['matsLocation']
		if not detections or not matsLocation: return

		wand_frames = interface.attr('x2ds', atLocation=detections)
		print wand_frames[1:2]
		vicon_mats = interface.attr('mats', atLocation=matsLocation)

		vicon_solved = [not (m[1][1,3] == 0.0 and m[1][2,3] == 0.0 and m[1][0,3] != 0.0) for m in vicon_mats]
		x2s_cameras, x3s_cameras, frames_cameras, num_kept_frames = Calibrate.generate_wand_correspondences(wand_frames, vicon_mats, vicon_solved)

		# TODO: Finish this bit of code

import numpy as np
import scipy.optimize as sci_opt
from scipy.sparse import lil_matrix
from GCore import Calibrate
import cv2
import ISCV

'''
Implements Bundle Adjustment with Scipy

Cameras are represented as 1 floats, 3 rotation angles, 3 translations, focal length, 2 for optical centre and 2 distortions
Points3d are just a 3 vector in world space
Points2d are a 2 vector of the pixel location
camera_indices is the camera_ids for each observation
points_indices is the index of the 2d point for each observation
'''

def matToVec(P, distortion):
	outVec = np.zeros(11, dtype=np.float32)
	K, RT = Calibrate.decomposeKRT(P)
	outVec[:3] = cv2.Rodrigues(RT[:3, :3])[0].ravel()
	outVec[3:6] = RT[:3, 3]
	outVec[6] = K[0, 0] # Focal Length
	outVec[7:9] = distortion
	outVec[9:] = K[:2, 2] # Optical Centre
	return outVec

def vecToMat(vec):
	f, k1, k2, ox, oy = vec[6:]
	rot = vec[:3]
	trans = vec[3:6]
	K = np.eye(3)
	K[[0,1],[0,1]] = f
	K[:2, 2] = [ox, oy]
	R = cv2.Rodrigues(rot)[0]
	RT = np.zeros((3, 4), dtype=np.float32)
	RT[:3, :3] = R
	RT[:3, 3] = trans
	P = Calibrate.composeKRT(K, RT)[:3,:]
	return np.float32(P), (k1, k2)

def bundle_adjustment_sparsity(n_cameras, n_points, x2ds_splits, point_indices):
	camera_indices = np.zeros(x2ds_splits[-1], dtype=int)
	for i, (c0, c1) in enumerate(zip(x2ds_splits[:-1], x2ds_splits[1:])):
		camera_indices[c0:c1] = i

	m = camera_indices.size * 2
	n = n_cameras * 11 + n_points * 3
	A = lil_matrix((m, n), dtype=int)

	i = np.arange(camera_indices.size)
	for s in range(11):
		A[2 * i, camera_indices * 11 + s] = 1
		A[2 * i + 1, camera_indices * 11 + s] = 1

	for s in range(3):
		A[2 * i, n_cameras * 11 + point_indices * 3 + s] = 1
		A[2 * i + 1, n_cameras * 11 + point_indices * 3 + s] = 1

	return A

def errorFunction(X, n_cameras, n_points, x2d_splits, x2ds_labels, x2ds):
	camera_params = X[:n_cameras * 11].reshape((n_cameras, 11))
	x3ds = X[n_cameras * 11:].reshape((n_points, 3))

	projected_x2ds = np.zeros_like(x2ds)
	for camVec, c0, c1 in zip(camera_params, x2d_splits[:-1], x2d_splits[1:]):
		P, distortion = vecToMat(camVec)
		x3d_labels = np.int32([x2ds_labels[i] for i in xrange(c0, c1)])
		proj_x2ds, proj_splits, proj_labels = ISCV.project(np.float32(x3ds[x3d_labels]), x3d_labels, np.float32([P]))
		assert np.all(x3d_labels == proj_labels)
		ISCV.distort_points(proj_x2ds, float(camVec[9]), float(camVec[10]), float(distortion[0]), float(distortion[1]), proj_x2ds)
		projected_x2ds[c0:c1, :] = proj_x2ds

	return (projected_x2ds - x2ds).ravel()

def printProblemDetails(n_cameras, n_points, x2ds):
	n = 9 * n_cameras + 3 * n_points
	m = 2 * x2ds.shape[0]
	print("n_cameras: {}".format(n_cameras))
	print("n_points: {}".format(n_points))
	print("Total number of parameters: {}".format(n))
	print("Total number of residuals: {}".format(m))

def adjustBundles(x3ds, x2ds, x2ds_splits, x2ds_labels, Ps, distortions):
	n_cameras = x2ds_splits.shape[0] - 1
	n_points = x3ds.shape[0]
	printProblemDetails(n_cameras, n_points, x2ds)
	camera_params = np.float32([matToVec(P, distortion) for P, distortion in zip(Ps, distortions)])
	x0 = np.hstack((camera_params.ravel(), x3ds.ravel()))
	sparsity = bundle_adjustment_sparsity(n_cameras, n_points, x2ds_splits, x2ds_labels)
	res = sci_opt.least_squares(errorFunction, x0, jac_sparsity=sparsity, verbose=2, x_scale='jac', ftol=1e-10,
								method='trf', args=(n_cameras, n_points, x2ds_splits, x2ds_labels, x2ds))
	X = res.x
	error = res.fun
	camera_params = X[:n_cameras * 11].reshape((n_cameras, 11))
	Ps, distortions = [], []
	for vec in camera_params:
		P, distortion = vecToMat(vec)
		Ps.append(P)
		distortions.append(distortion)
	return Ps, distortions, X, error

class BundleAdjust_Wand(Op.Op):
	def __init__(self, name='/BundleAdjust_Wand', locations='', frameRange=''):
		self.fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('frameRange', 'Frame range', 'Frame range', 'string', frameRange, {})
		]

		super(self.__class__, self).__init__(name, self.fields)

	def cook(self, location, interface, attrs):
		if not self.useFrame(interface.frame(), attrs['frameRange']): return

		mats = interface.attr('mats')
		collect_rx2ds = interface.attr('collect_rx2ds')
		x2ds_frames = interface.attr('x2ds_frames')
		x3ds_frames = interface.attr('x3ds_frames')

		frames = sorted(x3ds_frames.keys())
		frameIndex = {frame: fi for fi, frame in enumerate(frames)}
		x3ds = np.array([x3ds_frames[frame] for frame in frames], dtype=np.float32).reshape(-1, 3)
		x2ds = collect_rx2ds[0].copy()
		x2ds_splits = collect_rx2ds[1].copy()
		x2ds_labels = []
		for ci in xrange(len(x2ds_frames)):
			for fi, frame in enumerate(x2ds_frames[ci]):
				x2ds_labels.append(5 * frameIndex[frame] + (fi % 5))
		assert np.max(x2ds_labels) < x3ds.shape[0]
		x2ds_labels = np.array(x2ds_labels, dtype=int)
		Ps = [mat[2] for mat in mats]
		distortions = [mat[3] for mat in mats]

		new_Ps, new_distortions, _, _ = adjustBundles(x3ds, x2ds, x2ds_splits, x2ds_labels, Ps, distortions)

		for i in xrange(len(mats)):
			new_mat = list(mats[i])
			print "\n----\n{}\n{}\n----\n".format(Ps[i], new_Ps[i])
			new_mat[2] = new_Ps[i]
			new_mat[3] = new_distortions[i]
			print "\n----\n{}\n{}\n----\n".format(distortions[i], new_distortions[i])
			mats[i] = tuple(new_mat)

		interface.setAttr('mats', mats)
		interface.setAttr('Ps', new_Ps)
		interface.setAttr('updateMats', True)



# Register Ops
import Registry
Registry.registerOp('Calibrate Cameras', Cameras)
Registry.registerOp('Generate Wand Correspondences', WandCorrespondences)
