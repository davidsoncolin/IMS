import numpy as np
import time, math
import matplotlib.pyplot as plt
import pylab
from scipy.interpolate import splprep, splev

import Op, Interface
from GCore import Label
import ISCV

plt.switch_backend('Qt4Agg')


class Track2D(Op.Op):
	def __init__(self, name='/Track 2D', locations='', x2dThreshold=0.012, frameRange=''):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('locations', 'Locations', 'locations', 'string', locations, {}),
			('x2d_threshold', 'X2D Threshold', 'X2D Threshold', 'float', x2dThreshold, {}),
			('frameRange', 'Frame range', 'Frame range', 'string', frameRange, {})
		]

		super(self.__class__, self).__init__(name, fields)
		self.tracker = None

	def flush(self):
		self.tracker = None

	def cook(self, location, interface, attrs):
		if not self.useFrame(interface.frame(), attrs['frameRange']): return

		x2ds = interface.attr('x2ds')
		x2ds_splits = interface.attr('x2ds_splits')
		if x2ds is None or x2ds_splits is None: return

		# TODO: Cache this
		if self.tracker is None:
			self.tracker = Label.Track2D(len(x2ds_splits) - 1, x2d_threshold=attrs['x2d_threshold'])

		self.tracker.push(x2ds, x2ds_splits)
		interface.setAttr('labels', self.tracker.labels)


class Track3D(Op.Op):
	def __init__(self, name='/Track 3D', locations='', detections='', calibration='',
	             x2d_threshold=6./2000, pred_2d_threshold=100./2000, x3d_threshold=30,
	             tilt_threshold=0.0002, min_rays=3, numPolishIts=3, forceRayAgreement=True, boot=True, bootInterval=10,
	             skeleton='', pointSize=12.0, colour=(0.8, 0.0, 0.8, 0.7), intersect_threshold=100., generateNormals=False,
	             showContributions=False, frameRange='', enable=False):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('locations', 'Locations', 'Locations', 'string', locations, {}),
			('enable', 'enable', 'enable', 'bool', enable, {}),
			('detections', 'Detections location', 'Detections location', 'string', detections, {}),
			('calibration', 'Calibration', 'Calibration location', 'string', calibration, {}),
			('x2d_threshold', '2D threshold', '2D threshold', 'float', x2d_threshold, {}),
			('pred_2d_threshold', '2D threshold prediction', '2D threshold prediction', 'float', pred_2d_threshold, {}),
			('x3d_threshold', '3D threshold', '3D threshold', 'float', x3d_threshold, {}),
			('tilt_threshold', 'Tilt treshold', 'Tilt threshold', 'float', tilt_threshold, {}),
			('min_rays', 'Min. rays', 'Minimum number of intersecting rays', 'int', min_rays, {}),
			('numPolishIts', '# Polish its.', 'Number of polish iterations', 'int', numPolishIts, {'min': 1}),
			('forceRayAgreement', 'Ray agreement', 'Force ray agreement', 'bool', forceRayAgreement, {}),
			('boot', 'Enable booting', 'Enable booting', 'bool', boot, {}),
			('boot_interval', 'Boot interval', 'Boot interval', 'int', bootInterval, {}),
			('skeleton', 'Skeleton', 'Skeleton with visibility LODs', 'string', skeleton, {}),
			('pointSize', '3D Point size', '3D Point size', 'float', pointSize, {}),
			('colour', '3D Point colour', '3D Point colour', 'string', str(colour), {}),
			('intersect_threshold', 'Intersect threshold', 'Intersect threshold', 'float', intersect_threshold, {}),
			('generateNormals', 'Generate normals', 'Generate normals for visibility checks', 'bool', generateNormals, {}),
			('show_contributions', 'Show contributions', 'Show camera contributions', 'bool', showContributions, {}),
			('frameRange', 'Frame range', 'Frame range', 'string', frameRange, {})
		]

		super(self.__class__, self).__init__(name, fields)
		self.flush()

	def flush(self):
		self.tracker = None
		self.boot = False
		self.visibility = None
		self.cameraPositions = None
		self.frames = []
		self.x3ds = None
		self.x3ds_labels = None
		self.lastFrame = -1

	def cook(self, location, interface, attrs):
		if not attrs['enable']: return
		if not self.useFrame(interface.frame(), attrs['frameRange']):
			self.lastFrame = interface.frame()
			return

		if not attrs['calibration'] or not attrs['detections']: return
		if interface.frame() == self.lastFrame and not interface.isDirty(): return

		# Get 2D data and push to the tracker
		detections = attrs['detections']
		x2ds_data = interface.attr('x2ds', atLocation=detections)
		x2ds_splits = interface.attr('x2ds_splits', atLocation=detections)

		if x2ds_data is None or x2ds_splits is None:
			self.logger.error('No 2D data found at: %s' % detections)
			return

		settings = Label.PushSettings()

		calibrationLocation = attrs['calibration']
		if not calibrationLocation: calibrationLocation = interface.root()
		self.mats = interface.attr('mats', atLocation=calibrationLocation)
		if not self.mats: return

		# Make sure we've got the as many calibration matrices as the number of cameras with detections
		numCamsWithDets = len(x2ds_splits) - 1
		if numCamsWithDets != len(self.mats):
			# TODO: Don't allow going further, return
			# NOTE: Temp for Reframe
			self.mats = self.mats[:numCamsWithDets]

		if 'skeleton' in attrs and attrs['skeleton']:
			skeletonLoc = attrs['skeleton']
			skelDict = interface.attr('skelDict', atLocation=skeletonLoc)
			visibilityLod = interface.getChild('visibilityLod', parent=skeletonLoc)
			if visibilityLod is None:
				self.logger.warning('No visibility LODs found at skeleton: %s' % attrs['skeleton'])
				return

			lodTris = visibilityLod['tris']
			lodVerts = visibilityLod['verts']
			lodNormals = visibilityLod['faceNormals']

			settings.useVisibility = True
			settings.generateNormals = True
			settings.triangles = lodVerts[lodTris]
			settings.triangleNormals = np.concatenate((lodNormals))
			settings.cameraPositions = np.array([m[4] for m in self.mats], dtype=np.float32)
			settings.intersect_threshold = 100.

			tris = lodVerts[lodTris]
			cameraPositions = np.array([m[4] for m in self.mats], dtype=np.float32)
			if self.visibility is None: self.visibility = ISCV.ProjectVisibility.create()
			self.visibility.setLods(tris, cameraPositions, np.concatenate((lodNormals)),
			                        attrs['intersect_threshold'], attrs['generateNormals'])

		settings.visibility = self.visibility
		settings.numPolishIts = attrs['numPolishIts']
		settings.forceRayAgreement = attrs['forceRayAgreement']

		if self.tracker is None:
			self.tracker = Label.Track3D(self.mats, attrs['x2d_threshold'], attrs['pred_2d_threshold'], attrs['x3d_threshold'],
			                             attrs['tilt_threshold'], attrs['min_rays'], boot_interval=attrs['boot_interval'])

		# booting = interface.attr('booting', atLocation='/root')
		if not self.boot and attrs['boot']: #booting == 0 or not self.boot:# and attrs['boot']: #self.tracker.next_id == 0:
			self.x3ds, x2ds_labels = self.tracker.boot(x2ds_data, x2ds_splits, settings=settings)
			self.boot = True
		else:
			self.x3ds, x2ds_labels = self.tracker.push(x2ds_data, x2ds_splits, settings=settings)

		trackAttrs = {
			'x3ds': self.x3ds,
			'x3ds_labels': self.tracker.x3ds_labels,
			'x3ds_colour': eval(attrs['colour']),
			'x3ds_pointSize': attrs['pointSize']
		}

		if attrs['show_contributions']:
			# Find which cameras contribute to the 3D reconstructions (optional?)
			trackAttrs['camerasLocation'] = calibrationLocation
			trackAttrs['showCameraContributions'] = attrs['show_contributions']
			# trackAttrs['cameraPositions'] = self.cameraPositions
			trackAttrs['labels'] = x2ds_labels
			trackAttrs['x2ds_splits'] = x2ds_splits
			# interface.setAttr('labels', self.tracker.x2ds_labels, atLocation=attrs['detections'])

		interface.createChild(interface.name(), 'points3d', atLocation=interface.parentPath(), attrs=trackAttrs)
		self.frames.append(interface.frame())

		interface.setAttr('labels', x2ds_labels, atLocation=detections)

		# Show labelled detections as green for clarity
		labelColour = interface.attr('x2ds_colour', atLocation=detections)
		labelColours = interface.getLabelColours(x2ds_labels, labelColour)
		if labelColours.any():
			numLabelled = len(np.unique(x2ds_labels)) - 1
			# self.logger.info('# Labelled: %d' % numLabelled)
			interface.setAttr('x2ds_colours', labelColours, atLocation=detections)

		self.lastFrame = interface.frame()

		# Test
		interface.setAttr('model', self.tracker, atLocation='/root')


class Model(Op.Op):
	def __init__(self, name='/Tracking Model', locations='', detections='', calibration='', tracking='', its=1, normals=False,
	             x2d_threshold=20./2000, pred_2d_threshold=100./2000, x3d_threshold=30, boot=False, unlabelledPenalty=100.0, maxHypotheses=500,
				 bootIts=5, mesh='', useWeights=False, useVisibility=False, visibilityLod='', intersection_threshold=100., generateNormals=False,
	             showContributions=True, pointSize=8., colour=(0.8, 0.8, 0., 0.7), showLabelAssignment=True, visualiseLabels=False,
	             frameRange='', showLabellingGraph=False, bootResetTo=10, bootReset=False, forceBoot=False, enable=False,
	             use3dTracks=False):
		fields = [
			('name', 'Name', 'name', 'string', name, {}),
			('locations', 'Skeleton locations', 'Skeleton locations', 'string', locations, {}),
			('enable', 'enable', 'enable', 'bool', enable, {}),
			('detections', 'Detections location', 'Detections location', 'string', detections, {}),
			('calibration', 'Calibration location', 'Calibration location', 'string', calibration, {}),
			('tracking', 'Tracking location', '3D Tracking location', 'string', tracking, {}),
			('its', '# Iterations', 'Number of iterations', 'int', its, {}),
			('normals', 'Use normals', 'Use normals if available', 'bool', normals, {}),
			('x2d_threshold', '2D Threshold', '2D Threshold', 'float', x2d_threshold, {}),
			('pred_2d_threshold', '2D Threshold Prediction', '2D Threshold Prediction', 'float', pred_2d_threshold, {}),
			('x3d_threshold', '3D Threshold', '3D Threshold', 'float', x3d_threshold, {}),
			('boot', 'Boot Labels', 'Boot Labels', 'bool', boot, {}),
			('maxHypotheses', '# Max. Hypotheses', 'Number of hypotheses to maintain', 'int', maxHypotheses, {}),
			('unlabelledPenalty', 'Unlabelled Penalty', 'Penalty for unlabelled points', 'float', unlabelledPenalty, {}),
			('bootIts', 'Boot iterations', 'Boot iterations', 'int', bootIts, {}),
			('use3dTracks', 'Use 3D tracks', 'Use 3D tracks', 'bool', use3dTracks, {}),
			('mesh', 'Mesh', 'Mesh location', 'string', mesh, {}),
			('useWeights', 'Use weights', 'Use weights', 'bool', useWeights, {}),
			('useVisibility', 'Visibility check', 'Do a visibility check if possible', 'bool', useVisibility, {}),
			('visibilityLod', 'Visibility LOD location', 'Visibility LOD location', 'string', visibilityLod, {}),
			('intersection_threshold', 'Intersection threshold', 'Intersection threshold', 'float', intersection_threshold, {}),
			('generateNormals', 'Generate normals', 'Generate normals for visibility checks', 'bool', generateNormals, {}),
			('show_contributions', 'Show contributions', 'Show camera contributions', 'bool', showContributions, {}),
			('pointSize', '3D Point size', '3D Point size', 'float', pointSize, {}),
			('colour', '3D Point colour', '3D Point colour', 'string', str(colour), {}),
			('showLabelAssignment', 'Show label assignment', 'Show label assignment | unlabelled (R), labelled (G), 1-ray (G)', 'bool', showLabelAssignment, {}),
			('visualiseLabels', 'Visualise labels', 'Visualise labels', 'bool', visualiseLabels, {}),
			('frameRange', 'Frame range', 'Frame range', 'string', frameRange, {}),
			('showLabellingGraph', 'Show labelling graph', 'Show labelling graph', 'bool', showLabellingGraph, {}),
			('bootResetTo', 'Boot reset to', 'Boot reset to (skipping or manual)', 'int', bootResetTo, {}),
			('bootReset', 'Boot reset', 'Boot reset', 'bool', bootReset, {}),
			('forceBoot', 'Force boot', 'Force boot', 'bool', forceBoot, {}) # Temp
		]

		super(self.__class__, self).__init__('Tracking Model', fields)
		self.flush()

		self.trackerDirty = False
		self.lastFrame = -1

	def flush(self):
		self.cameraPositions = None
		self.model = None
		self.visibility = None
		self.booting = None
		self.Ps = None

	def update(self):
		self.trackerDirty = True

	def getEffectorLabels(self, skelDict):
		if isinstance(skelDict['markerNames'][0], str):
			try:
				effectorLabels = np.array([int(mn) for mn in skelDict['markerNames']], dtype=np.int32)
			except:
				skelDict['labelNames'] = list(np.unique(skelDict['markerNames']))
				effectorLabels = np.array(
						[skelDict['labelNames'].index(ln) if ln in skelDict['labelNames'] else -1 for ln in skelDict['markerNames']],
						dtype=np.int32)
		else:
			effectorLabels = np.array(skelDict['markerNames'], dtype=np.int32)

		return effectorLabels

	def cook(self, location, interface, attrs):
		if not attrs['enable']: return
		if not self.useFrame(interface.frame(), attrs['frameRange']):
			self.lastFrame = interface.frame()
			return

		if interface.frame() == self.lastFrame and not interface.isDirty(): return
		if self.booting is None: self.booting = attrs['bootResetTo']

		its = attrs['its']
		normals = attrs['normals']
		x2d_threshold = attrs['x2d_threshold']
		pred_2d_threshold = attrs['pred_2d_threshold']
		x3d_threshold = attrs['x3d_threshold']
		detections = attrs['detections']
		if not location or not detections: return

		# Define push settings for track model
		settings = Label.PushSettings()
		settings.useWeights = attrs['useWeights']

		# Get skeleton
		skelDict = interface.attr('skelDict')
		if skelDict is None:
			self.logger.error('No skeleton dictionary found!')
			return

		# Get calibration
		calibrationLocation = attrs['calibration']
		if not calibrationLocation: calibrationLocation = interface.root()
		mats = interface.attr('mats', atLocation=calibrationLocation)
		if not mats:
			self.logger.error('No mats found at: %s' % calibrationLocation)
			return

		if self.cameraPositions is None: self.cameraPositions = np.array([m[4] for m in mats], dtype=np.float32)

		# Note: This should be split into x2ds and x2ds_splits (temporarily support both cases)
		# data = interface.attr('data', atLocation=detections)
		data = None
		if data is not None:
			x2ds_data, x2ds_splits = data
		else:
			x2ds_data = interface.attr('x2ds', atLocation=detections)
			x2ds_splits = interface.attr('x2ds_splits', atLocation=detections)

		if x2ds_data is None or x2ds_splits is None:
			# self.logger.info('Could not find detection data at: %s' % detections)
			self.logger.error('Could not find 2D data (x2ds, x2ds_splits) at: %s' % detections)
			return

		Ps = interface.attr('Ps', atLocation=calibrationLocation)
		if Ps is None:
			if self.Ps is None: self.Ps = np.array([m[2] / (np.sum(m[2][0, :3] ** 2) ** 0.5) for m in mats], dtype=np.float32)
			Ps = self.Ps

		# Make sure we've got the as many calibration matrices as the number of cameras with detections
		numCamsWithDets = len(x2ds_splits) - 1
		if numCamsWithDets != len(mats):
			# TODO: Don't allow going further, return
			# NOTE: Temp for Reframe
			mats = mats[:numCamsWithDets]
			Ps = Ps[:numCamsWithDets]

		if self.model is None or interface.isDirty():
			if 'markerNames' not in skelDict:
				self.logger.error('No markerNames found in skeleton!')
				return

			if len(skelDict['markerNames']) == 0:
				self.logger.error('No markers in skeleton markerNames!')
				return

			effectorLabels = self.getEffectorLabels(skelDict)
			self.model = Label.TrackModel(skelDict, effectorLabels, mats, x2d_threshold, pred_2d_threshold, x3d_threshold)

			# Check if we want to use a 3D tracker to provide 3D data for the tracking model
			if attrs['use3dTracks']:
				self.logger.info('Using 3D Tracks')
				self.model.track3d = interface.attr('model', atLocation='/root')
				# self.model.track3d = Label.Track3D(mats, 6./2000., 100./2000., 30., boot_interval=1)
				settings.numPolishIts = 3
				settings.forceRayAgreement = True

			# Attempt to pick up skeleton root mat if present
			try:
				rootMat = interface.attr('rootMat')
				if rootMat.any():
					self.model.rootMat = rootMat
			except:
				pass  # Probably no skeleton in the scene

		if normals:
			if attrs['mesh'] and interface.hasAttr('normals', atLocation=attrs['mesh']):
				settings.x3ds_normals = interface.attr('normals', atLocation=attrs['mesh'])

			if 'markerNormals' in skelDict:
				settings.x3ds_normals = skelDict['markerNormals']

			if self.visibility is None: self.visibility = ISCV.ProjectVisibility.create()
			self.visibility.setNormals(settings.x3ds_normals)

		if interface.frame() == 0:
			self.model.bootLabels(x2ds_data, x2ds_splits)

		# Check if we should boot (and have all the attributes we need)
		# Check boot countdown
		graph = interface.attr('label_graph')
		if attrs['boot']: self.booting -= 1
		if (attrs['boot'] and self.booting == 0) or (attrs['forceBoot']):
			if not graph:
				self.logger.error('Could not boot because the label graph was not found!')
				return

			trackingLocation = attrs['tracking']
			if not trackingLocation: trackingLocation = location
			_x3ds = interface.attr('x3ds', atLocation=trackingLocation)
			if _x3ds is None:
				self.logger.error('Could not boot because the x3ds were not found at: %s' % trackingLocation)
				return

			self.logger.info('Boot Pose...')
			maxHyps = attrs['maxHypotheses']
			penalty = attrs['unlabelledPenalty']
			# if attrs['forceBoot']: self.booting = 1 #attrs['bootResetTo']

			numGraphLabels = len(graph[0])
			x3dIndicesForLabels = -np.ones(numGraphLabels, dtype=np.int32)
			label_score = ISCV.label_from_graph(_x3ds, graph[0], graph[1], graph[2], graph[3], maxHyps, penalty, x3dIndicesForLabels)

			clouds = ISCV.HashCloud2DList(x2ds_data, x2ds_splits, x2d_threshold)

			whichLabels = np.array(np.where(x3dIndicesForLabels != -1)[0], dtype=np.int32)
			x3ds = _x3ds[x3dIndicesForLabels[whichLabels]]
			self.logger.info('Labelled %d out of %d markers' % (len(whichLabels), numGraphLabels))
			interface.setAttr('numLabelled', len(whichLabels))

			# if 'labelNames' in skelDict: labelNames = np.int32(skelDict['labelNames'])
			# else: labelNames = np.arange(len(skelDict['markerNames']))
			if 'labelNames' in skelDict: labelNames = np.int32(skelDict['markerNames']) # This will break Spader, DotsTool should change
			else: labelNames = np.arange(len(skelDict['markerNames']))

			x3ds_labels = np.array(skelDict['markerNames'], dtype=np.str)[whichLabels]

			pras_score, x2d_labels, vels = Label.project_assign(clouds, x3ds, whichLabels, Ps, x2d_threshold=x2d_threshold)
			self.logger.info('Frame: %d | Label score: %.2f | Pras score: %.2f' % (interface.frame(), label_score, pras_score))

			# Initialise the pose using the assigned labels
			bootScore = self.model.bootPose(x2ds_data, x2ds_splits, x2d_labels, its=attrs['bootIts'])
			self.logger.info('Boot score: %.2f' % bootScore)

			if False:
				# Check distance after booting
				from GCore import SolveIK
				m_x3ds, m_x3ds_labels = SolveIK.skeleton_marker_positions(skelDict, skelDict['rootMat'], skelDict['chanValues'],
																		  self.model.effectorLabels, self.model.effectorData,
																		  skelDict['markerWeights'])
				diffs = m_x3ds[whichLabels] - x3ds
				meanDiff = np.mean(diffs, axis=0)
				diffSum = np.linalg.norm(diffs)
				self.logger.info('Mean 3D distance = {}'.format(meanDiff))
				self.logger.info('Total 3D distance = %.2f' % diffSum)

			# Character.pose_skeleton(skelDict['Gs'], skelDict)

		else:
			if False and not self.booting >= 0:
				self.model.track3d = interface.attr('model', atLocation='/root')
				# self.model.track3d = Label.Track3D(mats, 6./2000., 100./2000., 30., boot_interval=1)
				settings.numPolishIts = 3
				settings.forceRayAgreement = True

			# Check if we've got visibility lods
			if 'useVisibility' in attrs and attrs['useVisibility']:
				settings.useVisibility = attrs['useVisibility']
				settings.generateNormals = attrs['generateNormals']
				if 'visibilityLod' in attrs and attrs['visibilityLod']:
					visibilityLod = interface.location(attrs['visibilityLod'])
				else:
					visibilityLod = interface.getChild('visibilityLod')

				if visibilityLod is None:
					self.logger.error('No visibility LODs found at skeleton: %s' % location)
					return

				lodTris = visibilityLod['tris']
				lodVerts = visibilityLod['verts']
				lodNormals = visibilityLod['faceNormals']
				settings.triangleNormals = np.concatenate((lodNormals))

				if 'generateCb' in visibilityLod: settings.generateVisibilityLodsCb = visibilityLod['generateCb']

				tris = lodVerts[lodTris]
				if self.visibility is None: self.visibility = ISCV.ProjectVisibility.create()
				self.visibility.setLods(tris, self.cameraPositions, np.concatenate((lodNormals)),
				                        attrs['intersection_threshold'], attrs['generateNormals'])

			if self.trackerDirty:
				self.model.rebuildEffectorData(skelDict, self.getEffectorLabels(skelDict))
				self.trackerDirty = False

			# Allow overriding the 2D threshold using an attribute
			settings.x2d_thresholdOverride = interface.attr('x2d_thresholdOverride')

			settings.visibility = self.visibility
			self.model.push(x2ds_data, x2ds_splits, its=its, settings=settings)

			x3ds = self.model.x3ds
			# x3ds = self.model.trackX3ds
			x3ds_labels = self.model.x3d_labels

		#if attrs['bootReset']: self.booting = attrs['bootResetTo']
		if self.lastFrame != -1 and np.abs(interface.frame() - self.lastFrame) >= attrs['bootResetTo']:
			self.booting = attrs['bootResetTo']

		self.lastFrame = interface.frame()

		# -- Grab all the information and update --
		skelDict = self.model.skelDict

		# Colour marker points based on labels if we have been given any (from a detection location)
		#   Not labelled: Red
		#   Labelled (more than one ray): Green
		#   Labelled (one ray): Blue
		# start = time.time()
		# TODO: Make efficient
		x3ds_colours = np.array([], dtype=np.float32)
		if attrs['visualiseLabels']:
			x3ds_colours = np.tile((1, 0, 0, 0.7), (x3ds_labels.shape[0], 1))
			labelHits = np.array([len(np.where(self.model.labels == x3d_label)[0]) for x3d_label in x3ds_labels], dtype=np.int32)
			x3ds_colours[np.where(labelHits == 1)[0]] = (0, 0, 1, 0.7)
			x3ds_colours[np.where(labelHits > 1)[0]] = (0, 1, 0, 0.7)
		# print '  > label hits:', (time.time() - start)

		# Create reconstructed 3D points from the model
		modelAttrs = {
			'x3ds': x3ds,
			'x3ds_labels': x3ds_labels,
			'normals': settings.x3ds_normals,
			'x3ds_colour': eval(attrs['colour']),
			'x3ds_pointSize': attrs['pointSize'],
			'x3ds_colours': x3ds_colours
		}

		modelAttrs['boot'] = attrs['boot'] and self.booting == 0

		if attrs['showLabellingGraph'] and graph is not None:
			edges = Label.find_graph_edges_for_labels(graph, self.model.x3d_labels)
			modelAttrs['edges'] = edges

		# Find which cameras contribute to the 3D reconstructions
		# start = time.time()
		cameraContributions = {}
		if attrs['show_contributions']:
			modelAttrs['showCameraContributions'] = attrs['show_contributions']
			modelAttrs['camerasLocation'] = calibrationLocation
			modelAttrs['x2ds_splits'] = x2ds_splits
			modelAttrs['labels'] = self.model.labels

		interface.createChild('reconstruction', 'points3d', attrs=modelAttrs)

		if interface.attr('originalNormals') is not None:
			n = []
			normals = interface.attr('originalNormals').copy()
			for ni, (parent, normal) in enumerate(zip(skelDict['markerParents'], normals)):
				Gs = skelDict['Gs'][parent].copy()
				n.append(np.dot(Gs[:3, :3], normal))
			skelDict['markerNormals'] = np.float32(n)

		# Update Skeleton data
		interface.setAttr('skelDict', self.model.skelDict)
		interface.setAttr('Gs', skelDict['Gs'].copy())

		# NOTE: Shouldn't this be done in the update mesh op?
		#       (maybe good to keep it as an option if we make it efficient)
		# Update mesh data if any
		# if attrs['mesh']:
		# 	vs, vs_labels = getWorldSpaceMarkerPos(skelDict)
		# 	interface.setAttr('vs', vs, atLocation=attrs['mesh'])

		# Add detection labels
		interface.setAttr('labels', self.model.labels, atLocation=detections)
		interface.setAttr('labels', self.model.labels)

		# Show labelled detections as green for clarity
		labelColour = interface.attr('x2ds_colour', atLocation=detections)
		labelColours = interface.getLabelColours(self.model.labels, labelColour)
		if labelColours.any():
			# numLabelled = len(np.unique(self.model.labels)) - 1
			# self.logger.info('# Labelled: %d' % len(numLabelled))
			interface.setAttr('x2ds_colours', labelColours, atLocation=detections)

		# Temporary hack to help improve labelled data
		interface.setAttr('model', self.model)


class Error(Op.Op):
	def __init__(self, name='/Track Error', locations='', source='', x3ds='', printRule=''):
		self.fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'locations', 'Skeleton locations', 'string', locations, {}),
			('source', 'source', 'source skeleton location', 'string', source, {}),
			('x3ds', '3D points', '3D points (optional)', 'string', x3ds, {}),
			('printRule', 'Print on frames', 'Print on frames', 'string', printRule, {})
		]

		super(self.__class__, self).__init__(name, self.fields)

		self.numFrames = 0
		self.effectorsDist = 0
		self.minEffectorsDist = 0
		self.maxEffectorsDist = 0
		self.jointsDiffs = 0
		self.minJointDiff = 0
		self.maxJointDiff = 0
		self.labelHits = 0
		self.minLabelHits = 1
		self.maxLabelHits = 0

		self.stats = {
			'frames': [],
			'labels': [],
			'effectors': [],
			'joints': []
		}

	def cook(self, location, interface, attrs):
		# Make sure we have some source to compare with
		if not attrs['source']: return

		# Get cooked skeleton
		skelDict = interface.attr('skelDict')
		if not skelDict: return

		labels = interface.attr('labels')
		if labels is None: return

		# Get the reconstructions if we have any
		# x3ds = None
		# if 'x3ds' in attrs: x3ds = interface.attr('x3ds', atLocation=attrs['x3ds'])

		# Get the source we want to compare against (assume for now that the source is a skeleton)
		sourceSkelDict = interface.attr('skelDict', atLocation=attrs['source'])

		if not sourceSkelDict: return

		# Get effectors for ground truth skeleton
		from GCore import SolveIK
		effectorLabels_gt = np.array([int(mn) for mn in sourceSkelDict['markerNames']], dtype=np.int32)
		effectorData_gt = SolveIK.make_effectorData(skelDict)
		x3ds_gt, x3ds_labels_gt = SolveIK.skeleton_marker_positions(sourceSkelDict, sourceSkelDict['rootMat'], sourceSkelDict['chanValues'],
		                                                            effectorLabels_gt, effectorData_gt, sourceSkelDict['markerWeights'])

		# Get effectors for target skeleton
		effectorLabels = np.array([int(mn) for mn in skelDict['markerNames']], dtype=np.int32)
		effectorData = SolveIK.make_effectorData(skelDict)
		x3ds, x3ds_labels = SolveIK.skeleton_marker_positions(skelDict, skelDict['rootMat'], skelDict['chanValues'],
		                                                      effectorLabels, effectorData, skelDict['markerWeights'])

		d = (x3ds - x3ds_gt) ** 2
		ed = np.sqrt(np.sum(d, axis=1))
		totalEd = np.sum(ed)
		minEd, maxEd = np.min(ed), np.max(ed)
		self.minEffectorsDist = max(minEd, self.minEffectorsDist)
		self.maxEffectorsDist = max(maxEd, self.maxEffectorsDist)
		self.effectorsDist += totalEd
		self.stats['effectors'].append(totalEd)

		frame = interface.frame()
		self.stats['frames'].append(frame)
		self.numFrames += 1

		# for vi, (v, d) in enumerate(zip(x3ds, ed)):
		# 	pAttrs = {'x3ds': np.array([v], dtype=np.float32), 'x3ds_pointSize': np.sqrt(d) + 0.1, 'x3ds_colour': (0, 0, 0, 0.5)}
		# 	interface.createChild('p_%d' % vi, 'points3d', attrs=pAttrs)

		# Now that we have two skeletons, calculate distances between joints
		dists = []
		jointDiffs = 0
		for jointName in skelDict['jointNames']:
			d = []
			for ci, (cv, cn) in enumerate(zip(sourceSkelDict['chanValues'], sourceSkelDict['chanNames'])):
				if jointName in cn and cn[-2:] in ['rx', 'ry', 'rz']:
					idx = skelDict['chanNames'].index(cn)
					jointDiff = abs(skelDict['chanValues'][idx] - cv)
					jointDiffs += jointDiff
					self.jointsDiffs += jointDiff
					d.append(jointDiff)

			if d:
				dists.append(np.array(d, dtype=np.float32))

		allDists = np.concatenate((dists))
		minJointDiff = abs(np.min(allDists))
		maxJointDiff = abs(np.max(allDists))
		# self.stats['joints'].append(np.sum(allDists))
		self.stats['joints'].append(maxJointDiff)
		self.minJointDiff = max(minJointDiff, self.minJointDiff)
		self.maxJointDiff = max(maxJointDiff, self.maxJointDiff)

		# Check how many labels we've found
		numMarkers = skelDict['numMarkers']
		hits = np.where(labels != -1)[0]
		numHits = float(len(hits))
		perc = numHits / numMarkers
		self.stats['labels'].append(perc)
		self.labelHits += perc
		self.minLabelHits = min(perc, self.minLabelHits)
		self.maxLabelHits = max(perc, self.maxLabelHits)

		# TODO: Measure label accuracy by checking which ones are correct (not just assigned)

		# Print stats for frame
		# print "> Frame:", frame
		# print "  - Effectors dists (min | max | total):", minEd, "|", maxEd, "|", totalEd
		# print "  - Joint diffs (min | max | total):", minJointDiff, "|", maxJointDiff, "|", jointDiffs
		# print "  - Label hits:", perc, "% |", int(numHits)

		# Print average stats
		if self.useFrame(interface.frame(), attrs['printRule']):
			avgEffDist = self.effectorsDist / self.numFrames
			avgJointDiff = self.jointsDiffs / self.numFrames
			avgLabelHits = self.labelHits / self.numFrames
			print "> AVERAGE:"
			print "  - Effs (min | max | avg | total):", self.minEffectorsDist, "|", self.maxEffectorsDist, "|", avgEffDist, "|", self.effectorsDist
			print "  - Joints (min | max | avg | total):", self.minJointDiff, "|", self.maxJointDiff, "|", avgJointDiff, "|", self.jointsDiffs
			print "  - Labels (min | max | avg):", self.minLabelHits, "|", self.maxLabelHits, "|", avgLabelHits

			if True:
				import datetime, os
				from os.path import expanduser
				home_directory = expanduser('~')
				dumpDir = os.path.join(home_directory, 'Documents\IMS')

				import matplotlib.pyplot as plt
				fig, (ax1, ax2, ax3) = plt.subplots(3)

				ax1.set_title('Effectors')
				ax1.plot(self.stats['frames'], self.stats['effectors'])

				ax2.set_title('Joints')
				ax2.plot(self.stats['frames'], self.stats['joints'])

				ax3.set_title('Labels')
				ax3.plot(self.stats['frames'], self.stats['labels'])

				dumpName = 'Stats ' + str(datetime.datetime.now().strftime('%d-%m-%Y %H-%M-%S'))
				fname = os.path.join(dumpDir, dumpName)
				plt.savefig(fname + '.png')
				plt.show()

		# ed = np.array([np.sqrt(np.sum(np.power(d, 2))) for d in dists], dtype=np.float32)

		# minDist, maxDist = np.min(ed), np.max(ed)
		# self.logger.info('Min Dist = %f | Max Dist = %f' % (minDist, maxDist))

		# for vi, v in enumerate(vs):
		# 	idx = skelDict['markerParents'][vi]
		# 	d = ed[idx]
		# 	pAttrs = {'x3ds': np.array([v], dtype=np.float32), 'x3ds_pointSize': d*10 + 0.1, 'x3ds_colour': (0, 0, 0, 0.5)}
		# 	interface.createChild('p_%d' % vi, 'points3d', attrs=pAttrs)


class Count3Ds(Op.Op):
	def __init__(self, name='/Count_3D_Tracks', locations='', collectRule='', printRule='', exportRule='', exportPath='',
	             numMaxElms=3, minNumPoints=100, reverse=False, allowOverrides=False, displayTracks=False):
		self.fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'locations', 'X3ds locations', 'string', locations, {}),
			('collectRule', 'Collect on frames', 'Collect on frames', 'string', collectRule, {}),
			('printRule', 'Print on frames', 'Print on frames', 'string', printRule, {}),
			('exportRule', 'Export on frames', 'Export on frames', 'string', exportRule, {}),
			('exportPath', 'Export path', 'Export path', 'string', exportPath, {}),
			('numMaxElms', 'numMaxElms', 'numMaxElms', 'int', numMaxElms, {}),
			('minNumPoints', 'minNumPoints', 'minNumPoints', 'int', minNumPoints, {}),
			('reverse', 'Reverse', 'Reverse', 'bool', reverse, {}),
			('allowOverrides', 'Overrides', 'Allow overrides', 'bool', allowOverrides, {}),
			('displayTracks', 'Display tracks', 'Display tracks', 'bool', displayTracks, {})
		]

		super(self.__class__, self).__init__(name, self.fields)

		self.stats = {
			'frames': [],
			'num_x3ds': [],
			'num_tracks': [],
			'track_lifetime': {},
			'lastFrame': -1
		}

		self.frames = 0
		self.x3ds_frames = {}
		self.cacheManualOverride = False
		self.trackColours = {}

	def setup(self, interface, attrs):
		self.cacheManualOverride = attrs['allowOverrides']

	def cook(self, location, interface, attrs):
		# if interface.frame() == self.stats['lastFrame']: return
		if not self.useFrame(interface.frame(), attrs['collectRule']): return

		if attrs['reverse']:
			if self.stats['lastFrame'] == -1: self.stats['lastFrame'] = interface.frame()
		else:
			self.stats['lastFrame'] = interface.frame()

		# Temp boot hack
		isBoot = interface.attr('boot')
		if isBoot is None or not isBoot: return

		# Get cooked skeleton
		x3ds = interface.attr('x3ds')
		if x3ds is None: return

		x3ds_labels = interface.attr('x3ds_labels')
		if x3ds_labels is None or len(x3ds_labels) == 0: return

		if len(x3ds) < attrs['minNumPoints']:
			self.logger.warning('Not enough markers (%d)' % len(x3ds))
			return

		# Note: Assumption here for now is that we're using the labels (ints)
		x3ds_labels = np.int32(x3ds_labels)
		maxLabel = np.max(x3ds_labels)
		frame = interface.frame()
		if frame not in self.stats['frames']:
			self.stats['frames'].append(frame)
			self.stats['num_x3ds'].append(int(len(x3ds)))
			self.stats['num_tracks'].append(int(maxLabel))
			self.frames += 1

		# Collect the x3ds if we're exporting them later
		# if attrs['collectRule']:
		# 	for x3d, x3d_label in zip(x3ds, x3ds_labels):
		# 		if x3d_label not in self.x3ds_frames: self.x3ds_frames[x3d_label] = []
		# 		self.x3ds_frames[x3d_label].append(x3d)

		frameLabels = []
		labelSwitch = np.zeros((maxLabel + 1, 1), dtype=np.int32)
		for x3d, label in zip(x3ds, x3ds_labels):
			label = int(label)
			if label not in self.x3ds_frames:
				self.x3ds_frames[label] = []

				colour = np.float32(np.random.rand(4))
				colour[3] = 1.0
				self.trackColours[label] = colour

			if label not in self.stats['track_lifetime']: self.stats['track_lifetime'][label] = []
				# self.stats['track_lifetime'][label] = [frame]
				# self.x3ds_frames[label].append(x3d)
			# else:
			if frame not in self.stats['track_lifetime'][label]:
				self.stats['track_lifetime'][label].append(frame)
				self.x3ds_frames[label].append(x3d)

				frameLabels.append(label)
				labelSwitch[label, 0] = 1

			elif attrs['allowOverrides']:
				frameIdx = self.stats['track_lifetime'][label].index(frame)
				self.x3ds_frames[label][frameIdx] = x3d

		refFrame = float(self.frames)
		if refFrame > 0:
			colours = np.zeros_like(x3ds)
			for li, l in enumerate(x3ds_labels):
				if l in self.stats['track_lifetime']:
					frames = self.stats['track_lifetime'][l]
					c = float(len(frames)) / refFrame
					colours[li][0] = 1. - c
					colours[li][2] = c
				else:
					colours[li][0] = 1.

			interface.setAttr('x3ds_colours', np.array(colours, dtype=np.float32))

		if attrs['exportRule'] and self.useFrame(interface.frame(), attrs['exportRule']):
			import collections
			trackLifetimes = self.stats['track_lifetime']
			if trackLifetimes:
				# Check which track length is the most common to use as a base track, where we look for other tracks of
				# the same length
				trackIds = collections.Counter([len(frames) for tid, frames in trackLifetimes.iteritems()]).most_common(attrs['numMaxElms'])
				# print 'Most common (#frames, #points):', trackIds
				minNumPoints = attrs['minNumPoints']
				numFrames, numPoints = -1, -1
				if trackIds:
					numFrames, numPoints = trackIds[0]

				if numPoints < minNumPoints:
					self.logger.warning('Not enough points found in tracks: #points [%d] < [%d]' % (numPoints, minNumPoints))
				else:
					# We should be verifying that the selected tracks line up with the base track
					# trackFirstFrame, trackLastFrame = track[0], track[-1]

					# Go through each track and pick out the tracks that have survived as long as the base track
					c3ds, c3ds_labels = [], []
					for label, trackFrames in trackLifetimes.iteritems():
						# For now exclude tracks with a longer lifetime. We should pick out the block of data
						# by identifying which frames are solid within the timeline.
						if len(trackFrames) != numFrames: continue
						c3ds.append(self.x3ds_frames[label])
						c3ds_labels.append(label)

					c3ds = np.array(c3ds, dtype=np.float32)
					c3ds_labels = np.array(c3ds_labels, dtype=np.int32)

					# Either dump the c3ds to file (if a path is given) or alternatively write the c3ds to the interface
					if attrs['exportPath']:
						from IO import IO
						exportPath = self.resolvePath(attrs['exportPath'] + '_' + str(interface.frame()) + '.c3dio')
						import os
						if not os.path.isfile(exportPath):
							IO.save(exportPath, {'/root/tracks': {'x3ds': c3ds, 'x3ds_labels': c3ds_labels}})
							self.logger.info('Exported C3Ds to: %s' % exportPath)
					else:
						c3dsAttrs = {
							'x3ds': c3ds,
							'x3ds_labels': c3ds_labels
						}
						interface.createChild('c3ds', 'group', attrs=c3dsAttrs)

		# Print stats
		# if self.useFrame(interface.frame(), attrs['printRule']):
		if False:
			import datetime, os
			from os.path import expanduser
			home_directory = expanduser('~')
			dumpDir = os.path.join(home_directory, 'Documents\IMS')

			self.logger.info('# tracks = %d' % self.stats['num_tracks'][-1])
			self.logger.info('# x3ds = %d' % self.stats['num_x3ds'][-1])
			# print 'labels:', frameLabels

			import matplotlib.pyplot as plt
			fig, (ax1, ax2) = plt.subplots(2)

			# ax1.set_title('# X3Ds')
			# ax1.plot(self.stats['frames'], self.stats['num_x3ds'])
			#
			# ax2.set_title('# Tracks')
			# ax2.plot(self.stats['frames'], self.stats['num_tracks'])

			trackLifetimes = np.array([(l, len(f), np.min(f), np.max(f)) for (l, f) in self.stats['track_lifetime'].iteritems()], dtype=np.int32)
			trackLifetimes.view('i32,i32,i32,i32').sort(order=['f1'], axis=0)

			ax1.set_title('Track lifetimes')
			ax1.barh(range(len(trackLifetimes)), trackLifetimes[:, 1][::-1], color='blue')

			ax2.set_title('Active labels (frame %s)' % str(interface.frame()))
			ax2.bar(range(maxLabel + 1), labelSwitch[:, 0])

			# from IO import IO
			# IO.save(r'D:\IMS\TracksStats.io', {'/root/data': {'tracks': trackLifetimes}})

			dumpName = 'Stats ' + str(datetime.datetime.now().strftime('%d-%m-%Y %H-%M-%S'))
			fname = os.path.join(dumpDir, dumpName)
			plt.savefig(fname + '.png')
			# plt.show()

		logAttrs = {
			'stats': self.stats,
			'x3ds_frames': self.x3ds_frames,
			'track_colours': self.trackColours
		}
		interface.createChild('log', 'group', attrs=logAttrs)

		if attrs['displayTracks']:
			for trackId, trackX3ds in self.x3ds_frames.iteritems():
				colour = self.trackColours[trackId]
				tAttrs = {
					'x3ds': trackX3ds,
					'x3ds_colour': colour
				}
				interface.createChild('track_%d' % trackId, 'points', attrs=tAttrs)


class Visualise(Op.Op):
	def __init__(self, name='/Visualise_Tracks', locations='', maxFrames=0, singleLocation=False, update=True):
		self.fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('maxFrames', 'Max frames', 'Max frames', 'int', maxFrames, {}),
			('singleLocation', 'Single location', 'Single location', 'bool', singleLocation, {}),
			('update', 'Update', 'Update', 'bool', update, {})
		]

		super(self.__class__, self).__init__(name, self.fields)

	def cook(self, location, interface, attrs):
		if not attrs['update']: return
		x3ds_frames = interface.attr('x3ds_frames')
		maxFrames = attrs['maxFrames']

		if x3ds_frames is not None:
			interface.deleteLocationsByName(location + '/track_')
			trackColours = interface.attr('track_colours')

			if attrs['singleLocation']:
				x3ds, colours = [], []
				for trackId, trackX3ds in x3ds_frames.iteritems():
					if len(trackX3ds) == 0: continue
					if maxFrames and len(trackX3ds) > maxFrames: continue
					colour = trackColours[trackId] if trackColours is not None else (0., 0., 0.7, 0.7)
					x3ds.extend(trackX3ds)
					colours.extend(np.repeat([colour], len(trackX3ds), axis=0))

				tAttrs = {
					'x3ds': np.float32(x3ds),
					'x3ds_colours': np.float32(colours),
					'x3ds_pointSize': 8.
				}

				interface.createChild('tracks', 'points', attrs=tAttrs)

			else:
				for trackId, trackX3ds in x3ds_frames.iteritems():
					if len(trackX3ds) == 0: continue
					if maxFrames and len(trackX3ds) > maxFrames: continue
					colour = trackColours[trackId] if trackColours is not None else (0., 0., 0.7, 0.7)
					tAttrs = {
						'x3ds': trackX3ds,
						'x3ds_colour': colour,
						'x3ds_pointSize': 8.
					}

					interface.createChild('track_%d' % trackId, 'points', attrs=tAttrs)


class ExportX3ds(Op.Op):
	def __init__(self, name='/Export_Track_Log_To_X3Ds', locations='', saveTo='', numMaxElms=3, minNumPoints=30, frameRange=''):
		self.fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('saveTo', 'Save to', 'Save to (.x3d)', 'filename', saveTo, {}),
			('numMaxElms', 'numMaxElms', 'numMaxElms', 'int', numMaxElms, {'min': 1}),
			('minNumPoints', 'minNumPoints', 'minNumPoints', 'int', minNumPoints, {'min': 1}),
			('frameRange', 'Frame range', 'Frame range', 'string', frameRange, {})
		]

		super(self.__class__, self).__init__(name, self.fields)

	def cook(self, location, interface, attrs):
		if not self.useFrame(interface.frame(), attrs['frameRange']): return
		if not location or not attrs['saveTo']: return

		stats = interface.attr('stats')
		x3ds_frames = interface.attr('x3ds_frames')

		import collections
		trackLifetimes = stats['track_lifetime']

		if True:
			# Go through and save out frames. Missing frames are marked as -1
			maxTrackId, minTrackId, numTracks = max(trackLifetimes.keys()), min(trackLifetimes.keys()), len(trackLifetimes)
			trackInds = np.ones(maxTrackId + 1, dtype=np.int32) * -1
			trackInds[trackLifetimes.keys()] = np.arange(numTracks + 1)
			tracksNumFrames = [len(frames) for frames in trackLifetimes.values()]
			tracksMinFrames = [min(frames) for frames in trackLifetimes.values()]
			frameMin, frameMax = min(tracksNumFrames), max(tracksNumFrames)
			frameStart, frameEnd = 0, frameMax
			numFrames = frameEnd - frameStart
			# c3ds = np.zeros((numFrames, numTracks, 4), dtype=np.float32)
			c3ds = np.ones((numTracks, numFrames, 4), dtype=np.float32) * -1
			self.logger.info("Shape: {}".format(c3ds.shape))

			labels = []
			allFrames = np.int32(stats['frames'])
			for tid, trackFrames in trackLifetimes.iteritems():
				trackFrames = np.where(trackFrames == allFrames.reshape(-1, 1))[0]
				if tid not in x3ds_frames: continue
				tid_index = trackInds[tid]
				c3ds[tid_index, trackFrames, :3] = x3ds_frames[tid]
				c3ds[tid_index, trackFrames, 3] = 0.0
				labels.append(tid)

			c3ds_labels = np.int32(trackLifetimes.keys())
		else:
			trackIds = collections.Counter([len(frames) for tid, frames in trackLifetimes.iteritems()]).most_common(attrs['numMaxElms'])

			minNumPoints = minNumPoints = attrs['minNumPoints']
			numFrames, numPoints = -1, -1
			if trackIds:
				numFrames, numPoints = trackIds[0]

			c3ds, c3ds_labels = [], []
			for label, trackFrames in trackLifetimes.iteritems():
				if len(trackFrames) != numFrames: continue
				if len(x3ds_frames[label]) != numFrames:
					print 'Unexpected frame length for label %s: %d instead of %d' % (label, len(x3ds_frames[label]), numFrames)
				c3ds.append(x3ds_frames[label])
				c3ds_labels.append(label)

			c3ds = np.float32(c3ds)
			c3ds_labels = np.int32(c3ds_labels)

		if attrs['saveTo']:
			from IO import IO
			exportPath = self.resolvePath(attrs['saveTo'])
			IO.save(exportPath, {'/root/tracks': {'x3ds': c3ds, 'x3ds_labels': c3ds_labels}})
			self.logger.info('Exported C3Ds to: %s' % exportPath)
		else:
			c3dsAttrs = {
				'x3ds': c3ds,
				'x3ds_labels': c3ds_labels
			}
			interface.createChild('c3ds', 'group', attrs=c3dsAttrs)


def calculateMissingFrames(trackLifetimes, x3ds_frames, trackId, mergeId):
	if trackId not in trackLifetimes or mergeId not in trackLifetimes: return
	mergeStart, mergeEnd = trackLifetimes[trackId][-1], trackLifetimes[mergeId][0]
	numMissingFrames = mergeEnd - mergeStart
	if numMissingFrames <= 0 or len(x3ds_frames[trackId]) <= 1 or len(x3ds_frames[mergeId]) <= 1:
		return None, None

	v0_idx = -2 if len(x3ds_frames[trackId]) > 1 else -1
	v3_idx = 1 if len(x3ds_frames[mergeId]) > 1 else 0
	cpts = np.float32([
		x3ds_frames[trackId][v0_idx],
		x3ds_frames[trackId][-1],
		x3ds_frames[mergeId][0],
		x3ds_frames[mergeId][v3_idx]
	])

	tck, u = splprep(cpts.T, u=None, s=0.0, per=0)
	u_new = np.linspace(0, 1, numMissingFrames + 3)
	x_new, y_new, z_new = splev(u_new, tck, der=0)
	fillPts = np.float32([[x, y, z] for (x, y, z) in zip(x_new, y_new, z_new)])
	fillFrameNumbers = range(mergeStart + 1, mergeEnd)

	fillPts = fillPts[2:-2]
	fillFrameNumbers = fillFrameNumbers
	assert len(fillPts) == len(fillFrameNumbers)
	return fillPts, fillFrameNumbers


class MergeTracks(Op.Op):
	def __init__(self, name='/Merge_Tracks', locations='', trackId=-1, mergeIds='', x3d_threshold=100., frame_threshold=30,
	             suggest=True, executeMerge=False, fillMissingFrames=True, visualiseCandidates=False, visualisePrecedingCandidates=False,
				 pointSize=12.0, colour1=(0, 0, 0, 1), colour2=(0.5, 0.5, 0.5, 1), clearCache=True):
		self.fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('trackId', 'Track ID', 'Track ID', 'int', trackId, {'min': -1}),
			('mergeIds', 'Merge IDs', 'Merge IDs', 'string', mergeIds, {}),
			('x3d_threshold', '3D threshold', '3D threshold', 'float', x3d_threshold, {}),
			('frame_threshold', 'Frame threshold', 'Frame threshold', 'float', frame_threshold, {}),
			('suggest', 'Suggest', 'Suggest merge', 'bool', suggest, {}),
			('executeMerge', 'Execute merge', 'Execute merge', 'bool', executeMerge, {}),
			('fillMissingFrames', 'Fill missing frames', 'Fill missing frames', 'bool', fillMissingFrames, {}),
			('visualiseCandidates', 'Visualise candidates', 'Visualise candidates', 'bool', visualiseCandidates, {}),
			('visualisePrecedingCandidates', 'Visualise prec. candidates', 'Visualise preceding candidates', 'bool', visualisePrecedingCandidates, {}),
			('pointSize', '3D Point size', '3D Point size', 'float', pointSize, {}),
			('colour1', 'Colour (filler)', 'Filler colour to track', 'string', str(colour1), {}),
			('colour2', 'Colour (filler prec.)', 'Filler colour to preceding track', 'string', str(colour2), {}),
			('clearCache', 'Clear cache', 'Clear cache', 'bool', clearCache, {})
		]

		super(self.__class__, self).__init__(name, self.fields)

		self.cands, self.candsPreceding = [], []

	def cook(self, location, interface, attrs):
		if attrs['trackId'] == -1: return
		x3ds_frames = interface.attr('x3ds_frames')
		interface.deleteLocationsByName(location + '/filler_')

		mergeIds = np.int32(attrs['mergeIds'].split()) if attrs['mergeIds'] else None

		if x3ds_frames is not None:
			trackId = attrs['trackId']

			# Find candidate tracks (tracks that don't overlap with the track in question)
			stats = interface.attr('stats')
			if stats is not None:
				trackLifetimes = stats['track_lifetime']
				if trackId not in trackLifetimes:
					self.logger.warning('Could not find track %d in log' % trackId)
					return

				trackFrames = trackLifetimes[trackId]
				s, e = min(trackFrames), max(trackFrames)
				self.logger.info('Track %d duration: %d -> %d' % (trackId, s, e))

				if attrs['suggest']:
					if not self.cands or attrs['clearCache']:
						self.cands, self.candsPreceding = [], []
						for tid, frames in trackLifetimes.iteritems():
							if trackId == tid or tid not in x3ds_frames: continue
							ts, te = min(frames), max(frames)
							# print('Compare with track %d: %d -> %d' % (tid, ts, te))
							# Check overlap (accept frame gaps within threshold)
							if 0 < ts - e < attrs['frame_threshold']:
								# Distance test
								x3d = x3ds_frames[trackId][trackFrames.index(e)]
								x3d_cand = x3ds_frames[tid][frames.index(ts)]
								dist = np.linalg.norm(x3d - x3d_cand)
								print('Track %d is within threshold (after) with dist %f' % (tid, dist))
								if dist < attrs['x3d_threshold']:
									self.cands.append((tid, ts, te, dist))

							elif 0 < s - te < attrs['frame_threshold']:
								x3d = x3ds_frames[trackId][trackFrames.index(s)]
								x3d_cand = x3ds_frames[tid][frames.index(te)]
								dist = np.linalg.norm(x3d - x3d_cand)
								if dist < attrs['x3d_threshold']:
									self.candsPreceding.append((tid, ts, te, dist))

					if self.cands:
						self.logger.info("Candidate tracks: {}".format(self.cands))

						if attrs['visualiseCandidates']:
							interface.setAttr('visible', True, atLocation='%s/track_%d' % (location, trackId))
							for tid, ts, te, dist in self.cands:
								if mergeIds is not None and tid not in mergeIds: continue
								interface.setAttr('visible', True, atLocation='%s/track_%d' % (location, tid))
								fillPts, fillFrames = calculateMissingFrames(trackLifetimes, x3ds_frames, trackId, tid)
								if fillPts is not None:
									self.logger.info('Estimated %d points to connect tracks %d and %d (%d frames)' % (len(fillPts), trackId, tid, (te - ts)))
									# print 'Track join (front):', trackLifetimes[trackId][-2:], '>', fillFrames[:2]
									# print 'Track join (back):', fillFrames[-2:], '>', trackLifetimes[tid][:2]
									pAttrs = {
										'x3ds': np.float32(fillPts),
										'x3ds_colour': eval(attrs['colour1']),
										'x3ds_pointSize': attrs['pointSize'],
									}
									interface.createChild('filler_prec_%d_%d' % (trackId, tid), 'points3d', atLocation='%s' % location, attrs=pAttrs)

					if self.candsPreceding:
						self.logger.info("Candidate tracks (preceding): {}".format(self.candsPreceding))

						if attrs['visualisePrecedingCandidates']:
							interface.setAttr('visible', True, atLocation='%s/track_%d' % (location, trackId))
							for tid, ts, te, dist in self.candsPreceding:
								if mergeIds is not None and tid not in mergeIds: continue
								interface.setAttr('visible', True, atLocation='%s/track_%d' % (location, tid))
								fillPts, fillFrames = calculateMissingFrames(trackLifetimes, x3ds_frames, trackId, tid)
								if fillPts is not None:
									self.logger.info('Estimated %d points to connect tracks %d and %d' % (len(fillPts), trackId, tid))
									pAttrs = {
										'x3ds': np.float32(fillPts),
										'x3ds_colour': eval(attrs['colour2']),
										'x3ds_pointSize': attrs['pointSize'],
									}
									interface.createChild('filler_%d_%d' % (trackId, tid), 'points3d', atLocation='%s' % location, attrs=pAttrs)

			# Merge tracks and make sure we remove any overlap
			if mergeIds is None or not attrs['executeMerge']: return
			trackLifetimes = stats['track_lifetime']
			self.cands, self.candsPreceding = [], []

			# Go through each merge id requested by the user and merge
			for mergeId in mergeIds:
				if mergeId == -1: continue
				if mergeId not in trackLifetimes or mergeId not in x3ds_frames:
					self.logger.warning('Could not find track id %d to merge into %d' % (mergeId, trackId))
					continue

				if mergeId < trackId:
					self.logger.warning('At the moment we can only merge to an earlier track: %d > %d' % (trackId, mergeId))

				# Fill missing frames between the tracks if requested
				if attrs['fillMissingFrames']:
					fillPts, fillFrameNumbers = calculateMissingFrames(trackLifetimes, x3ds_frames, trackId, mergeId)
					# Extend the track data if there are any frames to fill with
					if fillPts is not None and fillFrameNumbers is not None:
						x3ds_frames[trackId].extend(fillPts)
						trackLifetimes[trackId].extend(fillFrameNumbers)

				trackFrames = trackLifetimes[trackId]
				mergeFrames = trackLifetimes[mergeId]

				# Check if there's overlap and if so resolve it by excluding the overlapping points from the merge track
				trackId_lastFrame, mergeId_firstFrame = trackFrames[-1], mergeFrames[0]
				mergeFrom = mergeId_firstFrame
				if trackId_lastFrame >= mergeId_firstFrame:
					mergeFrom = mergeFrames.index(trackId_lastFrame + 1)
					self.logger.info('Merge track starts before the target track ends (%d >= %d): Merge from %d' % (trackId_lastFrame, mergeId_firstFrame, mergeFrom))

				# Update the x3ds for track frames
				x3ds_frames[trackId].extend(x3ds_frames[mergeId][mergeFrom:])
				del x3ds_frames[mergeId]

				# Update the track stats to reflect the merged frames
				if mergeFrames:
					trackFrames.extend(mergeFrames[mergeFrom:])
				del trackLifetimes[mergeId]

			stats['track_lifetime'] = trackLifetimes

		interface.setAttr('x3ds_frames', x3ds_frames)
		interface.setAttr('stats', stats)


class AutoMergeTracks(Op.Op):
	def __init__(self, name='/Auto_Merge_Tracks', locations='', x3d_threshold=100., frame_threshold=30,
	             suggest=False, executeMerge=False, strictMerge=False, fillMissingFrames=True, minNumFrames=4):
		self.fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('x3d_threshold', '3D threshold', '3D threshold', 'float', x3d_threshold, {}),
			('frame_threshold', 'Missing frames threshold', 'Missing frames threshold', 'int', frame_threshold, {}),
			('suggest', 'Suggest', 'Suggest merge', 'bool', suggest, {}),
			('executeMerge', 'Execute merge', 'Execute merge', 'bool', executeMerge, {}),
			('strictMerge', 'Strict merge', 'Only merge if one track option is available', 'bool', strictMerge, {}),
			('fillMissingFrames', 'Fill missing frames', 'Fill missing frames', 'bool', fillMissingFrames, {}),
			('minNumFrames', 'Min. # frames', 'Min. # frames', 'int', minNumFrames, {'min': 1})
		]

		super(self.__class__, self).__init__(name, self.fields)

	def cook(self, location, interface, attrs):
		if not attrs['suggest']: return
		mergeCount = 0
		x3ds_frames = interface.attr('x3ds_frames')
		if x3ds_frames is not None:
			# Find candidates (tracks that don't overlap)
			stats = interface.attr('stats')
			if stats is not None:
				trackLifetimes = stats['track_lifetime']
				keysToRemove = []
				for trackId, trackFrames in trackLifetimes.iteritems():
					if trackId not in x3ds_frames: continue
					s, e = min(trackFrames), max(trackFrames)
					si, ei = trackFrames.index(s), trackFrames.index(e)
					cands, candsBackward = [], []
					x3ds_trackFrames = x3ds_frames[trackId]
					numTrackFrames = len(x3ds_trackFrames)
					dists = []

					for tid, frames in trackLifetimes.iteritems():
						if trackId == tid or tid not in x3ds_frames: continue
						ts, te = min(frames), max(frames)
						# Check overlap (accept frame gaps within threshold)
						if 0 < ts - e < attrs['frame_threshold']:
							# Distance test
							if ei >= numTrackFrames:
								self.logger.warning('Track %d (%d -> %d) exceeds frame length: %d' % (trackId, s, e, numTrackFrames))
								continue

							x3d = x3ds_trackFrames[ei]
							x3d_cand = x3ds_frames[tid][frames.index(ts)]
							dist = np.linalg.norm(x3d - x3d_cand)
							if dist < attrs['x3d_threshold']:
								cands.append((tid, ts, te, dist))
								dists.append(dist)

					if cands:
						self.logger.info('Track %d duration: %d -> %d' % (trackId, s, e))
						self.logger.info(" -> Candidate tracks: {}".format(cands))

						# Merge tracks
						if attrs['executeMerge']:
							# Find the track we want to merge (merge Id)
							if attrs['strictMerge'] and len(cands) != 1: continue
							if len(cands) == 1:
								mergeId = cands[0][0]
							else:
								# Find lowest distance (seems the most sensible given our simple heuristics)
								mergeId = cands[np.argmin(dists)][0]

							if trackId not in trackLifetimes: continue
							if mergeId not in trackLifetimes: continue

							# Fill missing frames between tracks if necessary
							if attrs['fillMissingFrames']:
								fillPts, fillFrameNumbers = calculateMissingFrames(trackLifetimes, x3ds_frames, trackId, mergeId)
								if fillPts is not None and fillFrameNumbers is not None:
									self.logger.info('Using %d estimated points to connect tracks %d and %d' % (len(fillPts), trackId, tid))
									x3ds_frames[trackId].extend(fillPts)
									trackLifetimes[trackId].extend(fillFrameNumbers)

							# Update x3ds to reflect the merged frames
							x3ds_frames[trackId].extend(x3ds_frames[mergeId])
							#x3ds_frames[mergeId] = []
							del x3ds_frames[mergeId]
							mergeCount += 1
							self.logger.info(' -> Merged track %d into %d' % (mergeId, trackId))

							# Update track stats to reflect the merged frames
							mergeFrames = trackLifetimes[mergeId]
							if mergeFrames:
								trackLifetimes[trackId].extend(mergeFrames)
							# del trackLifetimes[mergeId]
							keysToRemove.append(mergeId)

				for key in keysToRemove: del trackLifetimes[key]
				stats['track_lifetime'] = trackLifetimes

		# Log the number of tracks after merging
		if mergeCount:
			self.logger.info('Number of tracks after %d merge operations: %d' % (mergeCount, len(x3ds_frames)))
		elif not mergeCount and attrs['executeMerge']:
			self.logger.info('No merging required')

		# Eliminate tracks shorter than a certain length (in frames)?
		for tid, frames in x3ds_frames.iteritems():
			numFrames = len(frames)
			frameThreshold = attrs['minNumFrames']
			if numFrames < frameThreshold:
				self.logger.info('Track %d has fewer than %d frames (%d)' % (tid, frameThreshold, numFrames))

		interface.setAttr('x3ds_frames', x3ds_frames)
		interface.setAttr('stats', stats)


class Interpolate(Op.Op):
	def __init__(self, name='/Interpolate_Tracks', locations='', track1=-1, track2=-1, type=1):
		self.fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('track1', 'Track 1 ID', 'Track 1 ID', 'int', track1, {'min': 0}),
			('track2', 'Track 2 ID', 'Track 2 ID', 'int', track2, {'min': 0}),
			('type', 'Type', 'Type', 'int', type, {}) # TODO: Make drop-down
		]

		super(self.__class__, self).__init__(name, self.fields)

	def cook(self, location, interface, attrs):
		x3ds_frames = interface.attr('x3ds_frames')
		if x3ds_frames is None: return

		stats = interface.attr('stats')
		if stats is None: return
		trackLifetime = stats['track_lifetime']

		track1_id = attrs['track1']
		track2_id = attrs['track2']
		if track1_id == -1 or track2_id == -1:
			return

		track1 = x3ds_frames[track1_id]
		track2 = x3ds_frames[track2_id]
		frameGap = trackLifetime[track2_id][0] - trackLifetime[track1_id][-1] - 1
		self.logger.info('Gap frames: %d' % frameGap)
		pts = np.float32([])

		if attrs['type'] == 1:
			cpts = np.float32([
				track1[-2], track1[-1], track2[0], track2[1]
			])
			targetGap = np.linalg.norm(track1[-1] - track1[-2])
			gapDist = np.linalg.norm(track1[-1] - track2[0])
			ratio = math.ceil(gapDist / targetGap)
			self.logger.info('Gap distance: %.2f' % gapDist)

			tck, u = splprep(cpts.T, u=None, s=0.0, per=0)
			u_new = np.linspace(0, 1, frameGap + 3)
			x_new, y_new, z_new = splev(u_new, tck, der=0)
			pts = np.float32([[x, y, z] for (x, y, z) in zip(x_new, y_new, z_new)])

		pAttrs = {
			'x3ds': np.float32(pts[1:-2]),
			'x3ds_colour': (0, 0, 0, 1),
			'x3ds_pointSize': 12.
		}
		interface.createChild('interpolatedPts', 'points3d', attrs=pAttrs)


class Info(Op.Op):
	def __init__(self, name='/Tracks_Info', locations='', basicInfo=True, detailedInfo=False, printInfo=False,
	            plotTimeline=False, useFilters=True, filterMaxFrames=0):
		self.fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('basicInfo', 'Basic info', 'Basic info', 'bool', basicInfo, {}),
			('detailedInfo', 'Detailed info', 'Detailed info', 'bool', detailedInfo, {}),
			('printInfo', 'Print info', 'Print info', 'bool', printInfo, {}),
			('plotTimeline', 'Plot timeline', 'Plot timeline', 'bool', plotTimeline, {}),
			('useFilters', 'Use filters', 'Use filters', 'bool', useFilters, {}),
			('filterMaxFrames', 'Filter max frames', 'Only show if frames less than', 'int', filterMaxFrames, {'min': 0})
		]

		super(self.__class__, self).__init__(name, self.fields)

	def cook(self, location, interface, attrs):
		x3ds_frames = interface.attr('x3ds_frames')
		stats = interface.attr('stats')
		if x3ds_frames is None or stats is None: return
		tracksLifetime = stats['track_lifetime']

		if attrs['basicInfo']:
			numTracks = len(stats['track_lifetime'])
			numX3dsTracks = len(x3ds_frames)
			interface.setAttr('numTracks', numTracks)
			interface.setAttr('numX3dsTracks', numX3dsTracks)
			if attrs['printInfo']: self.logger.info('Number of tracks (x3ds): %d (%d)' % (numTracks, numX3dsTracks))

		if attrs['detailedInfo']:
			for tid, frames in tracksLifetime.iteritems():
				ts, te = min(frames), max(frames)
				if attrs['useFilters'] and te - ts >= attrs['filterMaxFrames']: continue
				if attrs['printInfo']: print('Track %d: %d -> %d' % (tid, ts, te))

		if attrs['plotTimeline']:
			labels, trackFrames = [], []
			for label, frames in tracksLifetime.iteritems():
				ts, te = min(frames), max(frames)
				if attrs['useFilters'] and te - ts >= attrs['filterMaxFrames']: continue
				labels.append(label)
				trackFrames.append(frames)

			trackColours = interface.attr('track_colours')
			if trackColours is None: trackColours = ['blue'] * len(labels)

			fig = plt.figure()
			ax = fig.add_subplot(111)

			for i, (label, frames) in enumerate(zip(labels, trackFrames)):
				ax.barh((i * 0.5) + 0.5, len(frames), left=frames[0], height=0.3, align='center', color=trackColours[label], alpha=0.75)

			y_max = float(len(labels)) * 0.5 + 0.25
			pos = np.arange(0.5, y_max, 0.5)
			locs_y, labels_y = pylab.yticks(pos, labels)
			plt.setp(labels_y, fontsize=6)

			ax.axis('tight')
			ax.set_ylim(ymin=0.25, ymax=y_max)
			ax.grid(color='g', linestyle=':')
			ax.invert_yaxis()
			plt.show()


class VisualiseTrackHealth(Op.Op):
	def __init__(self, name='/Visualise_Track_Health', locations='', frame=0, enable=True):
		self.fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('frame', 'Frame', 'Frame', 'int', frame, {'min': 0}),
			('enable', 'Enable', 'Enable', 'bool', enable, {})
		]

		super(self.__class__, self).__init__(name, self.fields)
		self.x3ds_frames = None
		self.trackLifetime = None

	def cook(self, location, interface, attrs):
		if not attrs['enable']: return
		if not location or location == self.getName(): return
		if self.x3ds_frames is None:
			self.x3ds_frames = interface.attr('x3ds_frames')

		if self.trackLifetime is None:
			stats = interface.attr('stats')
			if stats is not None:
				self.trackLifetime = stats['track_lifetime']

		if self.x3ds_frames is None:
			self.logger.error('3D frames not found at location: %s' % location)
			return

		if self.trackLifetime is None:
			self.logger.error('Stats not found at location: %s' % location)
			return

		if not attrs['frame']: return
		refFrame = attrs['frame']
		frame = interface.frame()
		pts, labels, colours = [], [], []
		for tid, frames in self.trackLifetime.iteritems():
			if frame in frames:
				if tid not in labels: labels.append(tid)
				trackFrames = self.x3ds_frames[tid]
				pts.append(trackFrames[frames.index(frame)])
				c = min(1., float(len(frames)) / float(refFrame))
				colours.append([1. - c, 0., c, 1.])

		pAttrs = {
			'x3ds': np.float32(pts),
			'x3ds_labels': np.int32(labels),
			'x3ds_colours': np.float32(colours)
		}
		interface.createChild('snapshot', 'points3d', attrs=pAttrs)


class VisualiseAnimatedX3ds(Op.Op):
	def __init__(self, name='/Visualise_Animated_X3Ds', locations='', pointSize=12., useColours=False):
		self.fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('pointSize', '3D Point size', '3D Point size', 'float', pointSize, {'min': 1.}),
			('useColours', 'Use colours', 'Use colours', 'bool', useColours, {})
		]

		super(self.__class__, self).__init__(name, self.fields)
		self.flush()

	def flush(self):
		self.x3ds_frames, self.x3ds_labels, self.x3ds_colours = None, None, None

	def cook(self, location, interface, attrs):
		if self.x3ds_frames is None:
			self.x3ds_frames = interface.attr('x3ds_frames')
			self.x3ds_labels = interface.attr('x3ds_labels')
			self.x3ds_colours = interface.attr('x3ds_colours')

		if self.x3ds_frames is not None and interface.frame() in self.x3ds_frames:
			frameAttrs = {
				'x3ds': self.x3ds_frames[interface.frame()],
				'x3ds_pointSize': attrs['pointSize']
			}
			if self.x3ds_labels is not None:
				frameAttrs['x3ds_labels'] = self.x3ds_labels[interface.frame()]
			if attrs['useColours'] and self.x3ds_colours is not None:
				frameAttrs['x3ds_colours'] = self.x3ds_colours[interface.frame()]
			interface.createChild('points', 'points3d', attrs=frameAttrs)


class AddMarkersToSkeleton(Op.Op):
	def __init__(self, name='/Add_Markers', locations='', x3ds='', collectRule='', frameRange='', useMeanMarkers=True ):
		self.fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'locations', 'Skeleton locations', 'string', locations, {}),
			('x3dsLocation', 'x3ds', 'X3ds locations', 'string', x3ds, {}),
			('collectRule', 'Collect on frames', 'Collect on frames', 'string', collectRule, {}),
			('frameRange', 'Frame range', 'Frame range', 'string', frameRange, {}),
			('useMeanMarkers', 'Use mean markers', 'Use mean markers', 'bool', useMeanMarkers, {})
		]

		super(self.__class__, self).__init__(name, self.fields)
		self.RTs = []

	def cook(self, location, interface, attrs):
		skelDict = interface.attr('skelDict')
		if skelDict is None:
			self.logger.error('No skeleton found at: %s' % location)
			return

		from IO import ASFReader
		if self.useFrame(interface.frame(), attrs['collectRule']):
			self.RTs.append(ASFReader.invert_matrix_array(skelDict['Gs']))

		if not self.useFrame(interface.frame(), attrs['frameRange']): return

		x3dsLocation = attrs['x3dsLocation']
		x3ds = interface.attr('x3ds', atLocation=x3dsLocation)
		x3ds_labels = interface.attr('x3ds_labels', atLocation=x3dsLocation)
		if x3ds is None or x3ds_labels is None:
			self.logger.error('No x3ds data found at: %s' % x3dsLocation)
			return

		# Now we've got a skeleton and x3ds which indicate candidate marker data
		# We have to find out which bones and joints the markers map to
		frames = np.transpose(x3ds, axes=(1, 0, 2))#[:50, :, :3]
		data = frames.copy()

		_RTs = np.transpose(self.RTs, axes=(1, 0, 2, 3))

		pointToGroup, pointResiduals, stabilisedFrames = ASFReader.assignAndStabilize(data, _RTs, thresholdDistance=200.)
		print pointToGroup

		# jointIndices = [int(jn) for jn in skelDict['jointNames']]

		Gs = skelDict['Gs']
		markerParents = [gi for gi in pointToGroup if gi != -1]
		markerNames = [('%d' % pi) for pi, gi in enumerate(pointToGroup) if gi != -1]

		if attrs['useMeanMarkers']:
			markerOffsets = np.mean(
					[[np.dot(Gs[gi][:3, :3].T, data[fi][pi] - Gs[gi][:3, 3]) for pi, gi in enumerate(pointToGroup) if gi != -1] for fi in
					 range(data.shape[0])], axis=0)
		else:
			markerOffsets = [np.dot(Gs[gi][:3, :3].T, data[-1][pi] - Gs[gi][:3, 3]) for pi, gi in enumerate(pointToGroup) if gi != -1]

		skelDict['markerParents'] = np.int32(markerParents)
		skelDict['markerNames'] = markerNames
		skelDict['markerOffsets'] = np.float32(markerOffsets)
		skelDict['markerWeights'] = np.ones(len(markerNames), dtype=np.float32)
		interface.setAttr('skelDict', skelDict)
		interface.setAttr('override', True)


class Graph(Op.Op):
	def __init__(self, name='/Track_Graph', locations='', frameRange='', x3d_threshold=300, nearestN=4, updateRange='',
	             trackedX3ds=''):
		self.fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'locations', 'X3ds locations', 'string', locations, {}),
			('frameRange', 'Frame range', 'Frame range', 'string', frameRange, {}),
			('updateRange', 'Update range', 'Update range', 'string', updateRange, {}),
			('x3d_threshold', '3D threshold', '3D threshold', 'float', x3d_threshold, {}),
			('nearestN', 'Nearest N', 'Nearest N', 'int', nearestN, {}),
			('trackedX3ds', 'Tracked X3Ds', 'Tracked X3Ds', 'string', trackedX3ds, {})
		]

		super(self.__class__, self).__init__(name, self.fields)
		self.graph = None
		self.edges = None

	def setup(self, interface, attrs):
		if self.graph is None:
			self.graph = Label.TrackGraph(attrs['x3d_threshold'], attrs['nearestN'])

	def cook(self, location, interface, attrs):
		if not self.useFrame(interface.frame(), attrs['frameRange']): return
		updateGraph = True if self.useFrame(interface.frame(), attrs['updateRange']) else False

		x3ds = interface.attr('x3ds')
		x3ds_labels = interface.attr('x3ds_labels')
		if x3ds is None or x3ds_labels is None: return

		x3ds_joints = None
		if attrs['trackedX3ds']:
			x3ds_joints = np.ones((len(x3ds_labels)), dtype=np.int32) * -1
			trackedX3ds = interface.attr('x3ds', atLocation=attrs['trackedX3ds'])
			trackedLabels = interface.attr('x3ds_labels', atLocation=attrs['trackedX3ds'])
			trackedJoints = interface.attr('joints', atLocation=attrs['trackedX3ds'])
			if trackedJoints is None:
				self.logger.warning('No tracked data found at: %s' % attrs['trackedX3ds'])
			else:
				_, _labels, _vels = Label.label_3d_from_3d(trackedX3ds, trackedLabels, None, x3ds, attrs['x3d_threshold'])
				matchingLabels = np.where(_labels != -1)[0]
				whichJoints = np.where(_labels[matchingLabels] == trackedLabels.reshape(-1, 1))[1]
				if len(matchingLabels) != 0:
					x3ds_joints[matchingLabels] = trackedJoints[whichJoints]
					# print x3ds_joints

		self.graph.push(x3ds, x3ds_labels, updateGraph, x3ds_joints)

		interface.setAttr('trackGraph', self.graph.graph)
		# if self.edges is None:
		self.edges = self.graph.drawing_graph()
		# interface.setAttr('edges', self.graph.drawing_graph())

		pAttrs = {
			'x3ds': self.graph.x3ds,
			'x3ds_labels': self.graph.x3ds_labels,
			'x3ds_pointSize': 14.,
			'x3ds_colour': (1., 0.5, 0., 0.7),
			'edges': self.edges
		}
		interface.createChild('points', 'points3d', attrs=pAttrs)


class FrameDiff(Op.Op):
	def __init__(self, name='/Track_Frame_Diff', locations='', frameRange=''):
		self.fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'locations', 'X3ds locations', 'string', locations, {}),
			('frameRange', 'Frame range', 'Frame range', 'string', frameRange, {})
		]

		super(self.__class__, self).__init__(name, self.fields)
		self.flush()

	def flush(self):
		self.x3ds_labels = None

	def cook(self, location, interface, attrs):
		if not self.useFrame(interface.frame(), attrs['frameRange']): return
		x3ds_labels = interface.attr('x3ds_labels')

		if self.x3ds_labels is not None:
			x3ds_colours = np.zeros((len(x3ds_labels), 4), dtype=np.float32)
			x3ds_colours[:, :] = [1, 0, 0, 0.7]
			shared = np.where(x3ds_labels == self.x3ds_labels.reshape(-1, 1))[1]
			x3ds_colours[shared] = [0, 0, 1, 0.7]
			interface.setAttr('x3ds_colours', x3ds_colours)

		self.x3ds_labels = x3ds_labels


def getWorldSpaceMarkerPos(skelDict):
	vs, lbls = [], []
	for mi in range(skelDict['numMarkers']):
		parentJointGs = np.append(skelDict['Gs'][skelDict['markerParents'][mi]], [[0, 0, 0, 1]], axis=0)
		mOffset = skelDict['markerOffsets'][mi]
		mOffset = np.array([[mOffset[0], mOffset[1], mOffset[2], 1]], dtype=np.float32)

		v = np.dot(parentJointGs, mOffset.T)
		vs.append(np.concatenate(v[:3]))
		lbls.append(skelDict['markerNames'][mi])

	vs = np.array(vs, dtype=np.float32)
	return vs, lbls

def det2imgXY(detection, (h, w)):
	"""
	Convert detection space (-1..1) to image space.  Compensate for non-square images
	w: 1920    h:1080
	--
	det       [0.48002064,  0.29927447]
	measured  [1420,               253]
	compute   [1420.8198165893555, 701.60821616649628]
	--
	det       [ 0.78030837  0.49955559]
	measured  [1709,                60]
	computed  [1709.0960311889648, 809.76001739501953]

	"""

	width, height = np.float32(w), np.float32(h)
	x = (width / 2.) + (width * detection[0] / 2.)
	y = (height / 2.) - (width * detection[1] / 2.)

	return [x, y]


# Register Ops
import Registry
Registry.registerOp('Track 2D', Track2D)
Registry.registerOp('Track 3D', Track3D)
Registry.registerOp('Track Model', Model)
Registry.registerOp('Track Error', Error)
Registry.registerOp('Track Graph', Graph)
Registry.registerOp('Count 3D Tracks', Count3Ds)
Registry.registerOp('Visualise Tracks', Visualise)
Registry.registerOp('Interpolate Tracks', Interpolate)
Registry.registerOp('Tracks Info', Info)
Registry.registerOp('Visualise X3Ds Animation', VisualiseAnimatedX3ds)
Registry.registerOp('Visualise Track Health', VisualiseTrackHealth)
Registry.registerOp('Export Track Log to X3Ds', ExportX3ds)

