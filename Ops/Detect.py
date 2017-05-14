import numpy as np

import Op, Interface
from GCore import Detect as GDetect
#from Detections import FeatureDetector, Utils
from IO import ViconReader

import cv2, os
import numpy
import random
from scipy.ndimage import label as nd_label
from multiprocessing.dummy import Pool as ThreadPool
import itertools


class Blocker(Op.Op):
	def __init__(self, name='/Blocker', locations='', camera=0, topLeft=(-1, 1), bottomRight=(1, -1), portal=False):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('camera', 'Camera', 'Camera', 'int', camera, {}),
			('topLeft', 'Top left', 'Top left', 'string', str(topLeft), {}),
			('bottomRight', 'Bottom right', 'Bottom right', 'string', str(bottomRight), {}),
			('portal', 'Portal', 'Portal', 'bool', portal, {})
		]

		super(self.__class__, self).__init__(name, fields)

	def cook(self, location, interface, attrs):
		blockAttrs = {
			'camera': attrs['camera'],
			'bbox': {'topLeft': eval(attrs['topLeft']), 'bottomRight': eval(attrs['bottomRight'])},
			'portal': attrs['portal']
		}
		interface.createChild(interface.name(), 'blocker', atLocation=interface.parentPath(), attrs=blockAttrs)


class SkeletonBlocker(Op.Op):
	def __init__(self, name='/Skeleton Blocker', locations='', calibration='', padding=0., frameRange=''):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('locations', 'Skeleton locations', 'Skeleton locations', 'string', locations, {}),
			('calibration', 'Calibration', 'Calibration', 'string', calibration, {}),
			('padding', 'Padding', 'Padding %', 'float', padding, {}),
			('frameRange', 'Frame range', 'Frame range', 'string', frameRange, {})
		]

		super(self.__class__, self).__init__(name, fields)

		self.effectorData, self.effectorLabels, self.effectorTargets = None, None, None

	def cook(self, location, interface, attrs):
		if not self.useFrame(interface.frame(), attrs['frameRange']): return

		# Find bounding box for skeleton (in all views)
		skelDict = interface.attr('skelDict')
		if skelDict is None:
			self.logger.error('No skeleton found at: %s' % location)
			return

		Ps = interface.attr('Ps', atLocation=attrs['calibration'])
		if Ps is None:
			self.logger.error('Camera positions (Ps) not found at: %s' % attrs['calibration'])
			return

		from GCore import SolveIK
		if self.effectorData is None:
			self.effectorLabels = np.array([int(mn) for mn in skelDict['markerNames']], dtype=np.int32)
			self.effectorData = SolveIK.make_effectorData(skelDict)
			self.effectorTargets = np.zeros_like(self.effectorData[1])

		markers, labels = SolveIK.skeleton_marker_positions(skelDict, skelDict['rootMat'], skelDict['chanValues'],
		                                                    self.effectorLabels, self.effectorData, skelDict['markerWeights'])

		# markers, labels = Interface.getWorldSpaceMarkerPos(skelDict)

		import ISCV
		x2ds_data, x2ds_splits, x2d_labels = ISCV.project(markers, np.array(labels, dtype=np.int32), Ps)

		camPts = zip(x2ds_splits[:-1], x2ds_splits[1:])
		for ci, (s, e) in enumerate(camPts):
			pts = x2ds_data[s:e]
			if not pts.any(): continue
			minX, minY = np.min(pts, axis=0)
			maxX, maxY = np.max(pts, axis=0)
			padding = attrs['padding']
			# topLeft=(-1, 1), bottomRight=(1, -1)
			paddingX = np.abs(maxX - minX) * padding
			paddingY = np.abs(maxY - minY) * padding

			topLeft, bottomRight = (minX - paddingX, maxY + paddingY), (maxX + paddingX, minY - paddingY)
			blockAttrs = {
				'camera': ci,
				'bbox': {'topLeft': topLeft, 'bottomRight': bottomRight},
				'portal': True
			}
			interface.createChild('blockers/camera%s' % str(ci), 'blocker', attrs=blockAttrs)


class LodBlocker(Op.Op):
	def __init__(self, name='/LOD Blocker', locations='', calibration='', padding=0., distort=False):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('locations', 'LOD locations', 'LOD locations', 'string', locations, {}),
			('calibration', 'Calibration', 'Calibration', 'string', calibration, {}),
			('padding', 'Padding', 'Padding', 'float', padding, {}),
			('distort', 'Distort', 'Distort', 'bool', distort, {})
		]

		super(self.__class__, self).__init__(name, fields)
		self.flush()

	def flush(self):
		self.tris = None
		self.lastFrame = -1

	def cook(self, location, interface, attrs):
		if interface.location(location) is None:
			self.logger.error('No LODs found at: %s' % location)
			return

		# if self.tris is None or self.lastFrame != interface.frame():
		if True:
			lodNames = interface.attr('names')
			lodTris = interface.attr('tris')
			lodVerts = interface.attr('verts')

			# Prune unwanted lods (user parameter?)
			lodTris = np.array(lodTris).reshape(-1, 12, 3)
			whichLods = np.ones(len(lodNames), dtype=np.int32)
			whichLods[lodNames.index('VSS_HeadEnd')] = 0
			whichLods[lodNames.index('VSS_Head')] = 0
			whichLods[lodNames.index('VSS_Neck')] = 0
			lodTris = lodTris[np.where(whichLods == 1)]
			lodTris = lodTris.reshape(-1, 3)

			tris = lodVerts[lodTris]
			self.tris = tris.reshape(-1, 3)

		Ps = interface.attr('Ps', atLocation=attrs['calibration'])
		if Ps is None:
			self.logger.error('Camera positions (Ps) not found at: %s' % attrs['calibration'])
			return

		import ISCV
		x2ds_data, x2ds_splits, x2d_labels = ISCV.project(self.tris, np.arange(len(self.tris), dtype=np.int32), Ps)

		mats = interface.attr('mats', atLocation=attrs['calibration'])

		camPts = zip(x2ds_splits[:-1], x2ds_splits[1:])
		for ci, (s, e) in enumerate(camPts):
			pts = x2ds_data[s:e]
			# print "> Camera:", ci, "| # pts =", len(pts)
			if not pts.any(): continue

			numPts = len(pts)
			m = numPts % 3
			pts = pts[0:numPts - m]

			if attrs['distort']:
				K, RT, P, ks, T, wh = mats[ci]
				ISCV.distort_points(pts, float(-K[0, 2]), float(-K[1, 2]), float(ks[0]), float(ks[1]), pts)

			pts = pts.reshape(-1, 3, 2)

			# lbs = x2d_labels[s:e]
			# lbs = lbs.reshape(-1, 3, 1)
			# for pi, (pt, lb) in enumerate(zip(pts, lbs)):
			# 	randomColour = np.concatenate((np.random.rand(3), np.array([0.8], dtype=np.float32)))
			# 	lengths = np.zeros(len(camPts), dtype=np.int32)
			# 	lengths[ci] = len(pt)
			# 	splits = Interface.makeSplitBoundaries(lengths)
			# 	dAttrs = {
			# 		'x2ds': pt, 'x2ds_splits': splits, 'labels': lb, 'x2ds_colour': randomColour, 'x2ds_pointSize': 8
			# 	}
			# 	interface.createChild('bboxes_%d_%d' % (ci, pi), 'detections', attrs=dAttrs)

			blockAttrs = {
				'camera': ci,
				'tris_x2ds': pts,
				'padding': attrs['padding']
			}
			interface.createChild('lodBlockers/camera%s' % str(ci), 'lodBlocker', attrs=blockAttrs)


def pointsInTriangle(triangle, points, eps=0.05):
	coeffs = np.linalg.lstsq(triangle.T, points.T)[0]
	# print 'coeffs =', coeffs
	whichHits = np.where((np.sum(coeffs >= 0 - eps, axis=0) == 3) & (np.sum(coeffs <= 1 + eps, axis=0) == 3))[0]
	# print 'whichHits =', whichHits
	return len(whichHits) > 0, whichHits


class Dot(Op.Op):
	def __init__(self, name='/Dot Detector', locations='', min_dot_size=1., max_dot_size=12., circularity_threshold=4.,
				 threshold_bright=(160, 160, 160), threshold_dark_inv=20, calibration='', undistort=True, pointSize_bright=10,
				 pointSize_dark=10, colour_bright=(0.3, 0.3, 0.7, 0.7), colour_dark=(1., 0.5, 0, 0.7),
				 blockers='', lodBlockers='', useBlockers=True, useLodBlockers=True, blockerExclusivity=True,
	             frameRange='', assignLabels=False):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('locations', 'Camera locations', 'Camera locations', 'string', locations, {}),
			('min_dot_size', 'Min Det Radius', 'Min Radius of candidate detections.', 'float', min_dot_size, {"min": 0.0, "max": 40.0}),
			('max_dot_size', 'Max Det Radius', 'Max Radius of candidate detections.', 'float', max_dot_size, {"min": 0.0, "max": 40.0}),
			('circularity_threshold', 'Circularity Threshold', 'Required Circularity of candidate detections.', 'float', circularity_threshold, {"min": 0.0, "max": 40.0}),
			# Note: Make this an RGB array with a colour widget
			('threshold_bright_r', 'Bright Dot Threshold (R)', 'Brightness threshold of light detections (Red).', 'int', threshold_bright[0], {"min": 0, "max": 255}),
			('threshold_bright_g', 'Bright Dot Threshold (G)', 'Brightness threshold of light detections (Green).', 'int', threshold_bright[1], {"min": 0, "max": 255}),
			('threshold_bright_b', 'Bright Dot Threshold (B)', 'Brightness threshold of light detections (Blue).', 'int', threshold_bright[2], {"min": 0, "max": 255}),
			('threshold_dark_inv', 'Dark Dot Threshold', 'Darkness threshold of dark detections.', 'int', threshold_dark_inv, {"min": 0, "max": 255}),
			('matsLocation', 'Calibration Location', 'Calibration locations', 'string', calibration, {}),
			('pointSize_bright', 'Point size (bright)', 'Point size (bright)', 'float', pointSize_bright, {}),
			('pointSize_dark', 'Point size (dark)', 'Point size (dark)', 'float', pointSize_dark, {}),
			('colour_bright', 'Bright dot colour', 'Bright dot colour', 'string', str(colour_bright), {}),
			('colour_dark', 'Dark dot colour', 'Dark dot colour', 'string', str(colour_dark), {}),
			('blockers', 'Blockers', 'Blockers', 'string', blockers, {}),
			('lodBlockers', 'LOD blockers', 'LOD blockers', 'string', lodBlockers, {}),
			('useBlockers', 'Use blockers', 'Use blockers', 'bool', useBlockers, {}),
			('useLodBlockers', 'Use LOD blockers', 'Use LOD blockers', 'bool', useLodBlockers, {}),
			('blockerExclusivity', 'Blocker exclusivity', 'Blocker exclusivity', 'bool', blockerExclusivity, {}),
			('frameRange', 'Frame range', 'Frame range', 'string', frameRange, {}),
			('assignLabels', 'Assign labels', 'Assign arbitrary unique labels', 'bool', assignLabels, {})
		]

		super(self.__class__, self).__init__(name, fields)
		self.lastCookedFrame = -1
		self.data0, self.data0_raw = None, None
		self.data1, self.data1_raw = None, None

	def flush(self):
		self.lastCookedFrame = -1

	def get_dark_and_light_points(self, data, attrs, threshold_bright, threshold_dark_inv):
		min_dot_size = attrs['min_dot_size']
		max_dot_size = attrs['max_dot_size']
		circularity_threshold = attrs['circularity_threshold']

		(y1, x1), (y2, x2) = attrs['bbox']

		if threshold_dark_inv > 0:
			#good_darks, pts0 = GDetect.detect_dots(255 - data, threshold_dark_inv, attrs)
			good_darks, pts0 = GDetect.detect_dots_with_box(255 - data, threshold_dark_inv, attrs, x1, y1, x2, y2)
		else:
			good_darks, pts0 = np.zeros((0,2),dtype=np.float32), np.zeros((0,2),dtype=np.float32)

		if threshold_bright[0] > 0 and threshold_bright[1] > 0 and threshold_bright[2] > 0:
			#good_lights, pts1 = GDetect.detect_dots(data, threshold_bright, attrs)
			good_lights, pts1 = GDetect.detect_dots_with_box(data, threshold_bright, attrs, x1, y1, x2, y2)
		else:
			good_lights, pts1 = np.zeros((0,2),dtype=np.float32), np.zeros((0,2),dtype=np.float32)

		return good_darks, pts0, good_lights, pts1

	def cook(self, location, interface, attrs):
		if not self.useFrame(interface.frame(), attrs['frameRange']): return
		training = interface.attr('train', default={'train': False, 'reset': False, 'send_plate': False})
		if training['send_plate']:
			imgs = interface.attr('imgs_blurred')
		else:
			imgs = interface.attr('imgs')

		if imgs is None:
			self.logger.warning('No image data found at: %s' % location)
			return

		# if interface.frame() != self.lastCookedFrame:
		if True:
			# Find matrices
			matsLocation = attrs['matsLocation']
			if not matsLocation: matsLocation = location
			mats = interface.attr('mats', atLocation=matsLocation)
			if not mats:
				self.logger.error('No mats found at: %s' % matsLocation)
				return

			# Create a blocker dict for convenience and to avoid looking up the
			# same things for every camera
			blockersDict = {}
			if attrs['blockers'] and ('useBlockers' in attrs and attrs['useBlockers']):
				blockerLocations = interface.splitLocations(attrs['blockers'])
				for blockerLoc in blockerLocations:
					blockCam = interface.attr('camera', atLocation=blockerLoc)
					blockBbox = interface.attr('bbox', atLocation=blockerLoc)
					blockPortal = interface.attr('portal', atLocation=blockerLoc)
					d = {'bbox': blockBbox, 'portal': blockPortal}
					if blockCam not in blockersDict: blockersDict[blockCam] = []
					blockersDict[blockCam].append(d)

			# Check if we are using visibility LODs
			lodBlockersDict = {}
			if 'lodBlockers' in attrs and attrs['lodBlockers'] and attrs['useLodBlockers']:
				lodBlockerLocs = interface.splitLocations(attrs['lodBlockers'])
				for blockerLoc in lodBlockerLocs:
					blockCam = interface.attr('camera', atLocation=blockerLoc)
					blockTrisX2ds = interface.attr('tris_x2ds', atLocation=blockerLoc)
					blockPadding = interface.attr('padding', atLocation=blockerLoc)
					d = {'tris': blockTrisX2ds, 'padding': blockPadding}
					lodBlockersDict[blockCam] = d

			# TO DO: Make work for arbitrary image size - needs to default to full image to be useful
			bboxes = interface.attr('bbox', default=None)
			if bboxes is not None and len(bboxes) == 0: bboxes = None
			self.p0, self.p1 = [], []
			for ci, img in enumerate(imgs):
				threshold_bright = [attrs['threshold_bright_r'], attrs['threshold_bright_g'], attrs['threshold_bright_b']]
				if ci == 0: threshold_bright[2] = 230 # What?!
				threshold_dark_inv = attrs['threshold_dark_inv']

				# TO DO: Fix multi-region detections
				#attrs['bboxes'] = bboxes if bboxes is not None else ((0,0), (int(img.shape[1]), int(img.shape[0])))
				attrs['bbox'] = ((0,0), (int(img.shape[1]), int(img.shape[0])))
				good_darks, pts0, good_lights, pts1 = self.get_dark_and_light_points(img, attrs, tuple(threshold_bright), threshold_dark_inv)

				if ci in blockersDict:
					pts0_filtered, pts1_filtered = [], []
					blockers = blockersDict[ci]

					for pi, p in enumerate(pts0):
						include = True
						px, py = p

						for blocker in blockers:
							bbox = blocker['bbox']
							insideBbox = bbox['topLeft'][0] < px < bbox['bottomRight'][0] and bbox['bottomRight'][1] < py < bbox['topLeft'][1]

							# Act based on whether we are a blocker or a portal
							isPortal = blocker['portal']
							if (isPortal and not insideBbox) or (not isPortal and insideBbox):
								include = False

						if include:
							pts0_filtered.append(pi)

					for pi,p in enumerate(pts1):
						include = True
						px, py = p

						for blocker in blockers:
							bbox = blocker['bbox']
							insideBbox = bbox['topLeft'][0] < px < bbox['bottomRight'][0] and bbox['bottomRight'][1] < py < bbox['topLeft'][1]

							# Act based on whether we are a blocker or a portal
							isPortal = blocker['portal']
							if (isPortal and not insideBbox) or (not isPortal and insideBbox):
								include = False

						if include:
							pts1_filtered.append(pi)

					pts0 = pts0[pts0_filtered]
					pts1 = pts1[pts1_filtered]

				elif len(blockersDict) > 0 and attrs['blockerExclusivity']:
					# If we are using blockers but none are visible in this camera we don't want any points
					pts0 = np.array([], dtype=np.float32)
					pts1 = np.array([], dtype=np.float32)

				# Now that we've filtered the points based on the skeleton bounding box blockers, we want to filter some more
				# using LODs if we have any
				# Note: If we use Python, we have to look at a more efficient way to do it, i.e. not loop through each triangle,
				#       instead solve for all triangles at once?
				# Note: Only process triangles facing me?
				if lodBlockersDict and ci in lodBlockersDict:
					d = lodBlockersDict[ci]
					tris = d['tris']

					# Dark detections
					intersectsDark, intersectsBright = [], []
					ptsDark, ptsBright = np.array([], dtype=np.float32), np.array([], dtype=np.float32)
					if pts0.any(): ptsDark = np.hstack((pts0, np.ones((pts0.shape[0], 1))))
					if pts1.any(): ptsBright = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
					for tri in tris:
						tri = np.hstack((tri, np.ones((tri.shape[0], 1), dtype=np.float32)))

						if ptsDark.any():
							_, whichDark = pointsInTriangle(tri, ptsDark, d['padding'])
							intersectsDark.extend(whichDark)

						if ptsBright.any():
							_, whichBright = pointsInTriangle(tri, ptsBright, d['padding'])
							intersectsBright.extend(whichBright)

					if intersectsDark:
						intersectsDark = np.unique(intersectsDark)
						pts0 = pts0[intersectsDark]
					else:
						pts0 = np.array([], dtype=np.float32)

					if intersectsBright:
						intersectsBright = np.unique(intersectsBright)
						pts1 = pts1[intersectsBright]
					else:
						pts1 = np.array([], dtype=np.float32)

				self.p0.append(pts0)
				self.p1.append(pts1)

			self.data0, self.data0_raw = processDetections(self.p0, mats)
			self.data1, self.data1_raw = processDetections(self.p1, mats)

		# Create child locations for dark and light detections
		if self.data0:
			darkAttrs = {
				'x2ds': self.data0[0], 'rx2ds': self.data0_raw[0], 'x2ds_splits': self.data0[1], 'detections': self.p0, 'x2ds_colour': eval(attrs['colour_dark']),
				'x2ds_pointSize': attrs['pointSize_dark'], 'x2ds_colours': np.array([])
			}
			if attrs['assignLabels']: darkAttrs['labels'] = np.arange(len(self.data0[0]))
			interface.createChild('dark', 'points2d', attrs=darkAttrs)

		if self.data1:
			brightAttrs = {
				'x2ds': self.data1[0], 'rx2ds': self.data1_raw[0], 'x2ds_splits': self.data1[1], 'detections': self.p1, 'x2ds_colour': eval(attrs['colour_bright']),
				'x2ds_pointSize': attrs['pointSize_bright'], 'x2ds_colours': np.array([])
			}
			interface.createChild('bright', 'points2d', attrs=brightAttrs)


class Wand(Op.Op):
	def __init__(self, name='/Detect_Wand', locations='', ratio=2., x2d_threshold=.5, straightness_threshold=0.01, match_threshold=0.07,
	             pointSize=8, frameRange=''):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('locations', '2D Detection locations', '2D Detections locations', 'string', locations, {}),
			('ratio', 'Ratio', 'Ratio', 'float', ratio, {}),
			('x2d_threshold', 'Detection Threshold', 'Detection Threshold', 'float', x2d_threshold, {}),
			('straightness_threshold', 'Straightness Threshold', 'Straightness Threshold', 'float', straightness_threshold, {}),
			('match_threshold', 'Match Threshold', 'Match Threshold', 'float', match_threshold, {}),
			('pointSize', 'Point size', 'Point size', 'float', pointSize, {}),
			('frameRange', 'Frame range', 'Frame range', 'string', frameRange, {})
		]

		super(self.__class__, self).__init__(name, fields)
		self.colours = np.array([
			[1., 0., 0., .7],
			[0., 1., 0., .7],
			[0., 0., 1., .7],
			[1., 0., 1., .7],
			[1., 1., 0., .7]
		])

	def cook(self, location, interface, attrs):
		if not self.useFrame(interface.frame(), attrs['frameRange']): return

		from GCore import Label
		ratio = attrs['ratio']
		x2d_threshold = attrs['x2d_threshold']
		straightness_threshold = attrs['straightness_threshold']
		match_threshold = attrs['match_threshold']

		wandDetections = []
		emptyFrame = np.array([[]], dtype=np.float32).reshape(-1, 2)

		cameraPts = interface.attr('detections')
		if cameraPts is None:
			self.logger.error('No attribute points found at: %s' % location)
			return

		for ci, pts in enumerate(cameraPts):
			if pts.any() and len(pts) >= 5:
				# print("#### Camera %d ####" % ci)
				labels, hypothesis = Label.find_T_wand_2d(pts, ratio, x2d_threshold, straightness_threshold, match_threshold)
				# print labels, hypothesis
				if len(labels) > 0:
					# Pick the first one (need to look into scoring etc.)
					# print labels
					labels = labels[0]
					wandDetections.append(pts[labels])
				else:
					wandDetections.append(emptyFrame)
			else:
				wandDetections.append(emptyFrame)

		(wand_x2ds, wand_splits), _ = processDetections(wandDetections, None)

		wandAttrs = {
			'rx2ds': wand_x2ds,
			'x2ds_splits': wand_splits,
			'x2ds_pointSize': attrs['pointSize'],
			'x2ds_colours': self.colours
		}
		interface.createChild('wand', 'points2d', attrs=wandAttrs)


class Mser(Op.Op):
	def __init__(self, name='/MSER Detector', locations='', calibration='', numFeatures=2, gammaCorrection=False, monochromeRandom=False,
	             sort=True, plotHistogram=False, plotImage=False, plotMarkers=False, undistort=True, pointSize_bright=10, pointSize_dark=10,
	             colour_bright=(0.3, 0.3, 0.7, 0.7), colour_dark=(1., 0.5, 0, 0.7),
	             blockers='', lodBlockers='', useBlockers=True, useLodBlockers=True):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('locations', 'locations', 'Skeleton locations', 'string', locations, {}),
			('calibration', 'Calibration', 'Calibration', 'string', calibration, {}),
			('n_features', '# Features', 'Number of Features to Detect.', 'int', numFeatures),
			('apply_gamma_correction', 'Gamma Correction', 'Gamma Correction', 'bool', gammaCorrection),
			('do_monochrome_random', 'Monochrome Random', 'Monochrome Random', 'bool', monochromeRandom),
			('do_sorting', 'Sort', 'Sort', 'bool', sort),
			('do_plot_histogram', 'Plot Histogram', 'Plot Histogram', 'bool', plotHistogram),
			('do_plot_image', 'Plot Image', 'Plot Image', 'bool', plotImage),
			('do_plot_markers', 'Plot Markers', 'Plot Markers', 'bool', plotMarkers),
			('undistort', 'Undistort', 'Undistort using Mats', 'bool', undistort, {}),
			('pointSize_bright', 'Point size (bright)', 'Point size (bright)', 'float', pointSize_bright, {}),
			('pointSize_dark', 'Point size (dark)', 'Point size (dark)', 'float', pointSize_dark, {}),
			('colour_bright', 'Bright colour', 'Bright colour', 'string', str(colour_bright), {}),
			('colour_dark', 'Dark colour', 'Dark colour', 'string', str(colour_dark), {}),
			('blockers', 'Blockers', 'Blockers', 'string', blockers, {}),
			('lodBlockers', 'LOD blockers', 'LOD blockers', 'string', lodBlockers, {}),
			('useBlockers', 'Use blockers', 'Use blockers', 'bool', useBlockers, {}),
			('useLodBlockers', 'Use LOD blockers', 'Use LOD blockers', 'bool', useLodBlockers, {})
		]

		super(self.__class__, self).__init__(name, fields)

	def get_dark_and_light_points(self, img, attrs):
		do_plot_histogram = attrs['do_plot_histogram']
		do_plot_image = attrs['do_plot_image']
		do_plot_markers = attrs['do_plot_markers']
		apply_gamma_correction = attrs['apply_gamma_correction']
		n_features = attrs['n_features']  # use (x,y) rather than (x,y,area)
		do_monochrome_random = attrs['do_monochrome_random']  # random colour look-up table
		f = FeatureDetector.run_detector
		image = FeatureDetector.rgb2gray(img)
		result_image, standard = f("davidMSER", img, image, do_plot_histogram, do_plot_image, do_plot_markers,
								   do_monochrome_random, n_features, apply_gamma_correction)

		if attrs['do_sorting']:
			black_standard, blue_standard, white_standard = Utils.sorting_hat(image, standard)
			pts1 = Utils.kpts_to_numpy(white_standard, img)
			pts0 = Utils.kpts_to_numpy(black_standard, img)
			good_lights = []
			good_darks = []
		else:
			pts1 = Utils.kpts_to_numpy(standard, img)
			good_lights = []
			pts0 = np.empty((0, 2), dtype=np.float32)
			good_darks = []

		return good_darks, pts0, good_lights, pts1

	def cook(self, location, interface, attrs):
		imgs = interface.attr('imgs')
		if imgs is None: return

		# Find matrices
		matsLocation = attrs['calibration']
		mats = interface.attr('mats', atLocation=matsLocation)
		if not mats:
			self.logger.error('No mats found at: %s' % matsLocation)
			return

		# Create a blocker dict for convenience and to avoid looking up the
		# same things for every camera
		blockersDict = {}
		if attrs['blockers'] and ('useBlockers' in attrs and attrs['useBlockers']):
			blockerLocations = interface.splitLocations(attrs['blockers'])
			for blockerLoc in blockerLocations:
				blockCam = interface.attr('camera', atLocation=blockerLoc)
				blockBbox = interface.attr('bbox', atLocation=blockerLoc)
				blockPortal = interface.attr('portal', atLocation=blockerLoc)
				d = {'bbox': blockBbox, 'portal': blockPortal}
				if blockCam not in blockersDict: blockersDict[blockCam] = []
				blockersDict[blockCam].append(d)

		# Check if we are using visibility LODs
		lodBlockersDict = {}
		if 'lodBlockers' in attrs and attrs['lodBlockers'] and attrs['useLodBlockers']:
			lodBlockerLocs = interface.splitLocations(attrs['lodBlockers'])
			for blockerLoc in lodBlockerLocs:
				blockCam = interface.attr('camera', atLocation=blockerLoc)
				blockTrisX2ds = interface.attr('tris_x2ds', atLocation=blockerLoc)
				blockPadding = interface.attr('padding', atLocation=blockerLoc)
				d = {'tris': blockTrisX2ds, 'padding': blockPadding}
				lodBlockersDict[blockCam] = d

		p0, p1 = [], []
		for ci, img in enumerate(imgs):
			good_darks, pts0, good_lights, pts1 = self.get_dark_and_light_points(img, attrs)

			if ci in blockersDict:
				pts0_filtered, pts1_filtered = [], []
				blockers = blockersDict[ci]

				for pi, p in enumerate(pts0):
					include = True
					px, py = p

					for blocker in blockers:
						bbox = blocker['bbox']
						insideBbox = bbox['topLeft'][0] < px < bbox['bottomRight'][0] and bbox['bottomRight'][1] < py < bbox['topLeft'][1]

						# Act based on whether we are a blocker or a portal
						isPortal = blocker['portal']
						if (isPortal and not insideBbox) or (not isPortal and insideBbox):
							include = False

					if include:
						pts0_filtered.append(pi)

				for pi,p in enumerate(pts1):
					include = True
					px, py = p

					for blocker in blockers:
						bbox = blocker['bbox']
						insideBbox = bbox['topLeft'][0] < px < bbox['bottomRight'][0] and bbox['bottomRight'][1] < py < bbox['topLeft'][1]

						# Act based on whether we are a blocker or a portal
						isPortal = blocker['portal']
						if (isPortal and not insideBbox) or (not isPortal and insideBbox):
							include = False

					if include:
						pts1_filtered.append(pi)

				pts0 = pts0[pts0_filtered]
				pts1 = pts1[pts1_filtered]

			# Now that we've filtered the points based on the skeleton bounding box blockers, we want to filter some more
			# using LODs if we have any
			# Note: If we use Python, we have to look at a more efficient way to do it, i.e. not loop through each triangle,
			#       instead solve for all triangles at once?
			# Note: Only process triangles facing me?
			if lodBlockersDict and ci in lodBlockersDict:
				d = lodBlockersDict[ci]
				tris = d['tris']

				# Dark detections
				intersectsDark, intersectsBright = [], []
				ptsDark = np.hstack((pts0, np.ones((pts0.shape[0], 1))))
				ptsBright = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
				for tri in tris:
					tri = np.hstack((tri, np.ones((tri.shape[0], 1), dtype=np.float32)))
					_, whichDark = pointsInTriangle(tri, ptsDark, d['padding'])
					_, whichBright = pointsInTriangle(tri, ptsBright, d['padding'])
					intersectsDark.extend(whichDark)
					intersectsBright.extend(whichBright)

				if intersectsDark:
					intersectsDark = np.unique(intersectsDark)
					pts0 = pts0[intersectsDark]
				else:
					pts0 = np.array([], dtype=np.float32)

				if intersectsBright:
					intersectsBright = np.unique(intersectsBright)
					pts1 = pts1[intersectsBright]
				else:
					pts1 = np.array([], dtype=np.float32)

			p0.append(pts0)
			p1.append(pts1)

		data0, data0_raw = processDetections(p0, mats)
		data1, data1_raw = processDetections(p1, mats)

		# Create child locations for dark and light detections
		darkAttrs = {
			'x2ds': data0[0], 'rx2ds': data0_raw[0], 'x2ds_splits': data0[1], 'detections': p0, 'x2ds_colour': eval(attrs['colour_dark']),
			'x2ds_pointSize': attrs['pointSize_dark']
		}
		brightAttrs = {
			'x2ds': data1[0], 'rx2ds': data1_raw[0], 'x2ds_splits': data1[1], 'detections': p1, 'x2ds_colour': eval(attrs['colour_bright']),
			'x2ds_pointSize': attrs['pointSize_bright']
		}
		interface.createChild('dark', 'points2d', attrs=darkAttrs)
		interface.createChild('bright', 'points2d', attrs=brightAttrs)


# Utility functions for both ops
def processDetections(dets, mats=None):
	splits = Interface.makeSplitBoundaries(map(len, dets))
	x2ds_data = np.zeros((splits[-1], 2), dtype=np.float32)
	for det, i0, i1 in zip(dets, splits[:-1], splits[1:]): x2ds_data[i0:i1] = det.reshape(-1, 2)
	data = x2ds_data, splits

	# Undistort using mats
	ux2ds_data, ux2ds_splits = ViconReader.frameCentroidsToDets(data, mats)

	return (ux2ds_data, splits), (x2ds_data, splits)


class Squares(Op.Op):
	def __init__(self, name='/Detect Squares', locations='', maskSize=2, dst=3, threshold=10, threshold2=0, undistort=False):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('locations', 'locations', 'Skeleton locations', 'string', locations, {}),
			('maskSize', 'maskSize', 'Mask size', 'int', maskSize),
			('dst', 'dst', 'Dst', 'int', dst),
			('threshold', 'threshold', 'Threshold', 'int', threshold),
			('threshold2', 'threshold2', 'Threshold 2', 'int', threshold2),
		]

		super(self.__class__, self).__init__(name, fields)

	def segment_on_dt(self, img, attrs):
		dt = cv2.distanceTransform(img, attrs['maskSize'], attrs['dst'])  # L2 norm, 3x3 mask
		dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(numpy.uint8)
		dt = cv2.threshold(dt, attrs['threshold'], 255, cv2.THRESH_BINARY)[1]
		lbl, ncc = nd_label(dt)

		lbl[img == 0] = lbl.max() + 1
		lbl = lbl.astype(numpy.int32)
		cv2.watershed(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), lbl)
		lbl[lbl == -1] = 0
		return lbl


	def detectPts(self, src, attrs):
		pts = []

		# img = cv2.cvtColor(cv2.imread(src), cv2.COLOR_BGR2GRAY)
		img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
		img = cv2.threshold(img, attrs['threshold2'], 255, cv2.THRESH_OTSU)[1]
		img = 255 - img  # White: objects; Black: background

		ws_result = self.segment_on_dt(img, attrs)

		# Colorise
		height, width = ws_result.shape
		ws_color = numpy.zeros((height, width, 3), dtype=numpy.uint8)
		lbl, ncc = nd_label(ws_result)
		for l in xrange(1, ncc + 1):
			a, b = numpy.nonzero(lbl == l)
			if img[a[0], b[0]] == 0:  # Do not color background.
				continue
			rgb = [random.randint(0, 255) for _ in xrange(3)]
			ws_color[lbl == l] = tuple(rgb)

		# cv2.imwrite(out, ws_color)

		# Fit ellipse to determine the rectangles.
		wsbin = numpy.zeros((height, width), dtype=numpy.uint8)
		wsbin[cv2.cvtColor(ws_color, cv2.COLOR_BGR2GRAY) != 0] = 255

		ws_bincolor = cv2.cvtColor(255 - wsbin, cv2.COLOR_GRAY2BGR)
		lbl, ncc = nd_label(wsbin)
		for l in xrange(1, ncc + 1):
			yx = numpy.dstack(numpy.nonzero(lbl == l)).astype(numpy.int64)
			xy = numpy.roll(numpy.swapaxes(yx, 0, 1), 1, 2)
			if len(xy) < 6:  # Too small.
				continue

			ellipse = cv2.fitEllipse(xy)
			center, axes, angle = ellipse
			pts.append(center)
			rect_area = axes[0] * axes[1]
			if 0.9 < rect_area / float(len(xy)) < 1.1:
				rect = numpy.round(numpy.float64(
						cv2.cv.BoxPoints(ellipse))).astype(numpy.int64)
				color = [random.randint(60, 255) for _ in xrange(3)]
				cv2.drawContours(ws_bincolor, [rect], 0, color, 2)

		# cv2.imwrite(r'C:\Users\orng.IMAGINARIUMUK\Documents\Suit_outage.jpg', ws_bincolor)
		return pts, ws_bincolor

	def cook(self, location, interface, attrs):
		imgs = interface.attr('imgs')
		if imgs is None: return

		# Find matrices
		matsLocation = attrs['matsLocation']
		if not matsLocation: matsLocation = location
		mats = interface.attr('mats', atLocation=matsLocation)
		if not mats:
			self.logger.error('No mats found at: %s' % matsLocation)
			return

		p0, p1 = [], []

		for ci, img in enumerate(imgs):
			pts0, data = self.detectPts(img, attrs)

			p0.append(pts0)
			# p1.append(pts1)

			from UI import QApp
			img[:] = data
			QApp.view().cameras[ci + 1].invalidateImageData()

		data0, data0_raw = processDetections(p0, mats)
		data1, data1_raw = processDetections(p1, mats)

		# Create child locations for dark and light detections
		darkAttrs = {
			'x2ds': data0[0], 'x2ds_splits': data0[1], 'detections': p0, 'x2ds_colour': (1., 0.5, 0, 0.7)
		}
		# brightAttrs = {
		#     'x2ds': data1[0], 'x2ds_splits': data1[1], 'detections': p1, 'x2ds_colour': (0., 0.7, 0., 0.7)
		# }
		interface.createChild('dark', 'points2d', attrs=darkAttrs)
		# interface.createChild('bright', 'detections', attrs=brightAttrs)


def makeCoords(height, width):
	''' Generate a uniform grid on the pixels, scaled so that the x-axis runs from -1 to 1. '''
	pix_coord = np.zeros((height, width, 2), dtype=np.float32)
	pix_coord[:, :, 1] = np.arange(height).reshape(-1, 1)
	pix_coord[:, :, 0] = np.arange(width)
	coord = (pix_coord - np.array([0.5 * (width - 1), 0.5 * (height - 1)], dtype=np.float32)) * (2.0 / width)
	return coord, pix_coord


class Template(Op.Op):
	def __init__(self, name='/Detect_Template', locations='', matchThresholds='', undistort=True, templatesPath='', distThreshold=8, blockers=''):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('locations', 'locations', 'Skeleton locations', 'string', locations, {}),
			('matchThresholds', 'Matching threshold', 'Matching threshold', 'string', matchThresholds),
			('templatesPath', 'Templates path', 'Templates path', 'string', templatesPath, {}),
			('distThreshold', 'Distance threshold', 'Distance threshold', 'int', distThreshold, {}),
			('blockers', 'Blockers', 'Blockers', 'string', blockers, {})
		]

		super(self.__class__, self).__init__(name, fields)

		self.templates, self.templateShapes = None, None
		self.matchThresholds = []
		self.coords = None
		# self.lower, self.upper = np.array([70, 70, 140], dtype=np.uint8), np.array([170, 250, 255], dtype=np.uint8)
		self.lower, self.upper = np.array([140, 70, 70], dtype=np.uint8), np.array([255, 250, 170], dtype=np.uint8)

		self.pool = ThreadPool(7)

	def flush(self):
		self.coords = None

	def setup(self, interface, attrs):
		if self.templates is None:
			directory = self.resolvePath(attrs['templatesPath'])
			templateFilenames, templatePaths = [], []
			try:
				for file in os.listdir(directory):
					if file.lower().endswith('.png') or file.lower().endswith('.jpg'):
						templateFilenames.append(file)
						templatePaths.append(os.path.join(directory, file))
			except WindowsError as e:
				self.logger.error('Could not find templates: % s' % str(e))

			self.logger.info('Loading templates: {}'.format(templateFilenames))
			self.templates = [cv2.imread(path, 0) for path in templatePaths]
			self.templateShapes = [template.shape[::-1] for template in self.templates]

	def processImage(self, (img, distThreshold)):
		pts0 = []
		img = cv2.blur(img, (2, 2))
		img_grey = np.array(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), dtype=np.uint8)

		matches = []
		cv2.imwrite(os.path.join(r'D:\IMS\Surrey Calibration', 'Grip.png'), cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
		for template, (w, h), threshold in zip(self.templates, self.templateShapes, self.matchThresholds):
			res = cv2.matchTemplate(img_grey, template, cv2.TM_CCOEFF_NORMED)
			loc = np.where(res >= threshold)

			for pt in zip(*loc[::-1]):
				ptCorner = (pt[0] + w, pt[1] + h)
				centrePointAbs = np.int32(pt) + ((np.int32(ptCorner) - np.int32(pt)) / 2)
				midpointOffset = w / 2, h / 2

				# Check if we've got a match there already
				# intersects = np.sum(np.sum(np.abs(centrePointAbs - matches) < max(midpointOffset), axis=1) > 1) if matches else 0
				# intersects = np.sum(np.sum(np.abs(centrePointAbs - matches) < distThreshold, axis=1) > 1) if matches else 0
				intersects = np.sum(np.linalg.norm(centrePointAbs - matches, axis=1) < float(distThreshold)) if matches else 0
				if intersects: continue

				# Extract region
				matchRegion = img[pt[1]:ptCorner[1], pt[0]:ptCorner[0]]
				# centreColour = matchRegion[h / 2, w / 2]
				# meanColour = np.mean(np.mean(matchRegion, axis=0), axis=0)

				# Check if the colours in the region are within our accepted range
				if True: # This seems to be better
					rw, rh = range(midpointOffset[0] - 2, midpointOffset[0] + 2 + 1), range(midpointOffset[1] - 2, midpointOffset[1] + 2 + 1)
					centreRegion = matchRegion[rh[0]:rh[-1] + 1, rw[0]:rw[-1] + 1, :]
					mask = cv2.inRange(centreRegion, self.lower, self.upper)
					reprColours = centreRegion[mask == 255]
					if len(reprColours) >= 6:
						validColour = True
					else:
						validColour = False
				else:
					mask = cv2.inRange(matchRegion, self.lower, self.upper)
					reprColours = matchRegion[mask == 255]
					if len(reprColours) > int(np.ceil((w * h) * 0.15)):
						validColour = True
					else:
						validColour = False

				# Add the detection if we're happy at this point
				if validColour:
					matches.append(centrePointAbs)
					pts0.append(self.coords[centrePointAbs[1], centrePointAbs[0]])

		return pts0

	def cook(self, location, interface, attrs):
		if interface.frame() == self.lastCookedFrame: return
		imgs = interface.attr('imgs')
		mats = interface.attr('mats')
		if imgs is None or mats is None:
			self.logger.error('Image data not found')
			return

		if not self.templates: return
		if attrs['matchThresholds']:
			self.matchThresholds = eval(attrs['matchThresholds'])

		if not self.matchThresholds: self.matchThresholds = [0.85] * len(self.templates)
		blockersDict = {}
		if attrs['blockers']:
			blockerLocations = interface.splitLocations(attrs['blockers'])
			for blockerLoc in blockerLocations:
				blockCam = interface.attr('camera', atLocation=blockerLoc)
				blockBbox = interface.attr('bbox', atLocation=blockerLoc)
				blockPortal = interface.attr('portal', atLocation=blockerLoc)
				d = {'bbox': blockBbox, 'portal': blockPortal}
				if blockCam not in blockersDict: blockersDict[blockCam] = []
				blockersDict[blockCam].append(d)

		p0, p1 = [], []
		if self.coords is None:
			self.coords, pix_coord = makeCoords(interface.attr('vheight')[0], interface.attr('vwidth')[0])
			self.coords[:, :, 1] *= -1

		camDets = self.pool.map(self.processImage, itertools.izip(imgs, itertools.repeat(attrs['distThreshold'])))
		for ci, dets in enumerate(camDets):
			if ci in blockersDict:
				pts0_filtered = []
				blockers = blockersDict[ci]

				for pi, p in enumerate(dets):
					include = True
					px, py = p

					for blocker in blockers:
						bbox = blocker['bbox']
						insideBbox = bbox['topLeft'][0] < px < bbox['bottomRight'][0] and bbox['bottomRight'][1] < py < bbox['topLeft'][1]

						# Act based on whether we are a blocker or a portal
						isPortal = blocker['portal']
						if (isPortal and not insideBbox) or (not isPortal and insideBbox):
							include = False

					if include:
						pts0_filtered.append(pi)

				dets = np.float32(dets)[pts0_filtered]

			p0.append(np.array(dets, dtype=np.float32))

		# for ci, img in enumerate(imgs):
		# 	if ci == 3:
		# 		pts0 = self.processImage((img, attrs['distThreshold']))
		# 		p0.append(np.array(pts0, dtype=np.float32))
		# 	else:
		# 		p0.append(np.array([], dtype=np.float32))

		if p0:
			data0, data0_raw = processDetections(p0, mats)
			dAttrs = {
				'x2ds': data0[0], 'x2ds_splits': data0[1],
				'detections': p0
			}
			interface.createChild('detections', 'detections', attrs=dAttrs)


class Corners(Op.Op):
	def __init__(self, name='/Detect Corners', locations='', threshold=0.02, blockSize=2, ksize=3, k=0.04, undistort=True,
	             filterThreshold=100000, filterNeighbourhoodSize=10, blockers='', lodBlockers=''):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('locations', 'locations', 'Skeleton locations', 'string', locations, {}),
			('threshold', 'threshold', 'Threshold for an optimal value', 'float', threshold),
			('blockSize', 'Block size', 'Size of neighbourhood considered', 'int', blockSize),
			('ksize', 'k size', 'Aperture parameter of Sobel derivative used', 'int', ksize),
			('k', 'k', 'Harris detector free parameter in the equation', 'float', k),
			('undistort', 'Undistort', 'Undistort using Mats', 'bool', undistort, {}),
			('filterThreshold', 'Filter threshold', 'Filter threshold', 'float', filterThreshold),
			('filterNeighbourhoodSize', 'Filter neighbourhood size', 'Filter neighbourhood size', 'int', filterNeighbourhoodSize),
			('blockers', 'Blockers', 'Blockers', 'string', blockers, {}),
			('lodBlockers', 'LOD blockers', 'LOD blockers', 'string', lodBlockers, {})
		]

		super(self.__class__, self).__init__(name, fields)
		self.coords = None

	def flush(self):
		self.coords = None

	def cook(self, location, interface, attrs):
		import scipy.ndimage as ndimage
		import scipy.ndimage.filters as filters

		imgs = interface.attr('imgs')
		if imgs is None: return

		# Find matrices
		mats = interface.attr('mats')

		# Maxima filter parameters
		neighborhood_size = attrs['filterNeighbourhoodSize']
		threshold = attrs['filterThreshold']

		if True:
			blockersDict = {}
			if attrs['blockers']:
				blockerLocations = interface.splitLocations(attrs['blockers'])
				for blockerLoc in blockerLocations:
					blockCam = interface.attr('camera', atLocation=blockerLoc)
					blockBbox = interface.attr('bbox', atLocation=blockerLoc)
					blockPortal = interface.attr('portal', atLocation=blockerLoc)
					d = {'bbox': blockBbox, 'portal': blockPortal}
					if blockCam not in blockersDict: blockersDict[blockCam] = []
					blockersDict[blockCam].append(d)

			# Check if we are using visibility LODs
			lodBlockersDict = {}
			if 'lodBlockers' in attrs and attrs['lodBlockers']:
				lodBlockerLocs = interface.splitLocations(attrs['lodBlockers'])
				for blockerLoc in lodBlockerLocs:
					blockCam = interface.attr('camera', atLocation=blockerLoc)
					blockTrisX2ds = interface.attr('tris_x2ds', atLocation=blockerLoc)
					blockPadding = interface.attr('padding', atLocation=blockerLoc)
					d = {'tris': blockTrisX2ds, 'padding': blockPadding}
					lodBlockersDict[blockCam] = d

		p0, p1 = [], []
		if self.coords is None:
			h, w = interface.attr('vheight')[0], interface.attr('vwidth')[0]
			self.coords, pix_coord = makeCoords(h, w)
			self.coords[:, :, 1] *= -1

		for ci, img in enumerate(imgs):
			pts0 = []
			try:
				# img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
				gray = np.float32(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
				dst = cv2.cornerHarris(gray, attrs['blockSize'], attrs['ksize'], attrs['k'])
				dst = cv2.dilate(dst, None)
			except:
				continue

			if True:   # Requires OpenCV 3
				ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
				dst = np.uint8(dst)

				# Find centroids
				ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

				# Define the criteria to stop and refine the corners
				criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
				corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

				for ci, (w, h) in enumerate(corners):
					pts0.append(self.coords[h, w])

			if False:
				data_max = filters.maximum_filter(dst, neighborhood_size)
				maxima = (dst == data_max)
				data_min = filters.minimum_filter(dst, neighborhood_size)
				diff = ((data_max - data_min) > threshold)
				maxima[diff == 0] = 0

				labeled, num_objects = ndimage.label(maxima)
				slices = ndimage.find_objects(labeled)
				for dy, dx in slices:
					x_center = (dx.start + dx.stop - 1) / 2
					y_center = (dy.start + dy.stop - 1) / 2
					pts0.append(self.coords[y_center, x_center])

			pts0 = np.array(pts0, dtype=np.float32)
			if ci in blockersDict:
				pts0_filtered, pts1_filtered = [], []
				blockers = blockersDict[ci]

				for pi, p in enumerate(pts0):
					include = True
					px, py = p

					for blocker in blockers:
						bbox = blocker['bbox']
						insideBbox = bbox['topLeft'][0] < px < bbox['bottomRight'][0] and bbox['bottomRight'][1] < py < bbox['topLeft'][1]

						# Act based on whether we are a blocker or a portal
						isPortal = blocker['portal']
						if (isPortal and not insideBbox) or (not isPortal and insideBbox):
							include = False

					if include:
						pts0_filtered.append(pi)

				pts0 = pts0[pts0_filtered]

			if lodBlockersDict and ci in lodBlockersDict:
				d = lodBlockersDict[ci]
				tris = d['tris']

				intersectsDark = []
				ptsDark, ptsBright = np.array([], dtype=np.float32), np.array([], dtype=np.float32)
				if pts0.any(): ptsDark = np.hstack((pts0, np.ones((pts0.shape[0], 1))))
				for tri in tris:
					tri = np.hstack((tri, np.ones((tri.shape[0], 1), dtype=np.float32)))

					if ptsDark.any():
						_, whichDark = pointsInTriangle(tri, ptsDark, d['padding'])
						intersectsDark.extend(whichDark)

				if intersectsDark:
					intersectsDark = np.unique(intersectsDark)
					pts0 = pts0[intersectsDark]
				else:
					pts0 = np.array([], dtype=np.float32)

			p0.append(np.array(pts0, dtype=np.float32))

		if p0:
			data0, data0_raw = processDetections(p0, mats)
			interface.createChild('corners', 'detections', atLocation=self.name, attrs={'x2ds': data0[0], 'x2ds_splits': data0[1]})


class RandomNoise(Op.Op):
	def __init__(self, name='/Random Noise', locations='', noise=0.01, frameRange=''):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('locations', '2D Detection locations', '2D Detections locations', 'string', locations, {}),
			('noiseRange', 'Noise range', 'Noise range', 'float', noise, {}),
			('frameRange', 'Frame range', 'Frame range', 'string', frameRange, {})
		]

		super(self.__class__, self).__init__(name, fields)

	def cook(self, location, interface, attrs):
		if not self.useFrame(interface.frame(), attrs['frameRange']): return

		x2ds = interface.attr('x2ds')
		if x2ds is None:
			self.logger.error('No detections found at: %s' % location)
			return

		noiseRange = attrs['noiseRange']
		offsets = np.random.uniform(-noiseRange, noiseRange, x2ds.shape)
		offsets = offsets.astype(numpy.float32, copy=False)
		x2ds = x2ds + offsets
		interface.setAttr('x2ds', x2ds)


class BackgroundSubtraction(Op.Op):
	def __init__(self, name='/Background_Subtraction', locations='', history=200, varThreshold=16, detectShadows=True, update=False,
	             learningRate=-1, removeNoise=False, kernelType='Box', kernelSize=2, dilate=True, dilateKernelSize=5, dilateIterations=1,
	             disableOpenCL=True, frameRange=''):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('history', 'History', 'History', 'int', history, {}),
			('varThreshold', 'Variance threshold', 'Variance threshold', 'float', varThreshold, {}),
			('detectShadows', 'Detect shadows', 'Detect shadows', 'bool', detectShadows, {}),
			('learningRate', 'Learning rate', 'Learning rate (0.0-1.0) default -1', 'float', learningRate, {}),
			('removeNoise', 'Remove noise', 'Remove noise (using kernel)', 'bool', removeNoise, {}),
			('kernelType', 'Kernel type', 'Kernel type', 'select', kernelType, {'enum': ('Box', 'Ellipse')}),
			('kernelSize', 'Kernel size', 'Kernel size (pixels)', 'int', kernelSize, {}),
			('dilate', 'Dilate', 'Dilate', 'bool', dilate, {}),
			('dilateKernelSize', 'Dilate kernel size', 'Dilate kernel size (pixels)', 'int', dilateKernelSize, {}),
			('dilateIterations', 'Dilate iterations', 'Dilate iterations', 'int', dilateIterations, {}),
			('frameRange', 'Frame range', 'Frame range', 'string', frameRange, {}),
			('update', 'Update setup', 'Update setup', 'bool', update, {}),
			('disableOpenCL', 'Disable OpenCL', 'Disable OpenCL', 'bool', disableOpenCL, {})
		]

		super(self.__class__, self).__init__(name, fields)
		self.subtractor = None
		self.coords = None
		self.kernel = None
		self.kernelDilate = None

	def setup(self, interface, attrs):
		# Duped the code for now
		if self.subtractor is None:
			if attrs['disableOpenCL']: cv2.ocl.setUseOpenCL(False)

			# Create background subtractor MOG2
			self.subtractor = cv2.createBackgroundSubtractorMOG2(history=attrs['history'],
																 varThreshold=attrs['varThreshold'],
																 detectShadows=attrs['detectShadows'])
			# Create kernels
			kernelSize = attrs['kernelSize']
			kernelType = attrs['kernelType']
			if kernelType == 'Box':
				self.kernel = np.ones((kernelSize, kernelSize), np.uint8)
			elif kernelType == 'Ellipse':
				self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernelSize, kernelSize))

			dilateKernelSize = attrs['dilateKernelSize']
			self.kernelDilate = np.ones((dilateKernelSize, dilateKernelSize), np.uint8)

	def cook(self, location, interface, attrs):
		if not self.useFrame(interface.frame(), attrs['frameRange']): return

		imgs = interface.attr('imgs')
		if imgs is None: return

		# mats = interface.attr('mats')

		if attrs['update']:
			if attrs['disableOpenCL']: cv2.ocl.setUseOpenCL(False)

			# Create background subtractor MOG2
			self.subtractor = cv2.createBackgroundSubtractorMOG2(history=attrs['history'], varThreshold=attrs['varThreshold'], detectShadows=attrs['detectShadows'])

			# Create kernels
			kernelSize = attrs['kernelSize']
			kernelType = attrs['kernelType']
			if kernelType == 0: #'Box':
				self.kernel = np.ones((kernelSize, kernelSize), np.uint8)
			elif kernelType == 1: #'Ellipse':
				self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernelSize, kernelSize))

			dilateKernelSize = attrs['dilateKernelSize']
			self.kernelDilate = np.ones((dilateKernelSize, dilateKernelSize), np.uint8)

		if self.coords is None:
			h, w = interface.attr('vheight')[0], interface.attr('vwidth')[0]
			self.coords, pix_coord = makeCoords(h, w)
			self.coords[:, :, 1] *= -1

		for img in imgs:
			foregroundMask = self.subtractor.apply(img, learningRate=attrs['learningRate'])

			if attrs['removeNoise']:
				foregroundMask = cv2.morphologyEx(foregroundMask, cv2.MORPH_OPEN, self.kernel)

			if attrs['dilate']:
				foregroundMask = cv2.dilate(foregroundMask, self.kernelDilate, iterations=attrs['dilateIterations'])

			# Fill gaps in the person
			im2, contours, hierarchy = cv2.findContours(foregroundMask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
			for cnt in contours:
				cv2.drawContours(foregroundMask, [cnt], 0, 255, -1)

			# Apply foreground mask
			# bitwise_or is faster than an bitwise_and in this instance.
			img[:] = cv2.bitwise_or(img, img, mask=foregroundMask)
			#img[:] = cv2.bitwise_and(img, img, mask=foregroundMask)

			#if False:
			#	# gsImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
			#	gsImg = cv2.adaptiveThreshold(img[:, :, 2], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, -2)
			#	gsImg = cv2.blur(gsImg, (2, 2))
			#	img[:, :, 0] = gsImg
			#	img[:, :, 1] = gsImg
			#	img[:, :, 2] = gsImg


class Sift(Op.Op):
	def __init__(self, name='/Sift', locations='', calibration=''):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('calibration', 'calibration', 'calibration', 'string', calibration, {})
		]

		super(self.__class__, self).__init__(name, fields)

	def cook(self, location, interface, attrs):
		import sift
		imgs = interface.attr('imgs')
		if imgs is None:
			self.logger.warning('No image data found at: %s' % location)
			return

		# Find matrices
		calibration = attrs['calibration']
		if not calibration: calibration = location
		mats = interface.attr('mats', atLocation=calibration)
		if not mats:
			self.logger.error('No mats found at: %s' % calibration)
			return

		for ci, img in enumerate(imgs):
			sift.process_image()


# Register Ops
import Registry
Registry.registerOp('Blocker', Blocker)
Registry.registerOp('Skeleton Blocker', SkeletonBlocker)
Registry.registerOp('Detect Dots', Dot)
Registry.registerOp('Detect MSER', Mser)
Registry.registerOp('Detect Wand', Wand)
Registry.registerOp('Detect Squares', Squares)
Registry.registerOp('Detect Corners', Corners)
Registry.registerOp('Random Detection Noise', RandomNoise)
