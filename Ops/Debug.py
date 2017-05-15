import numpy as np
import Op, Interface
from GCore import State, SolveIK, Recon, Label, Character
import ISCV
import cv2

class PrintState(Op.Op):
	def __init__(self, name='/Print State', location='', unique=False, dirtyOnly=False):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('location', 'Location', 'Location', 'string', location, {}),
			('unique', 'Unique', 'Unique', 'bool', unique, {}),
			# ('dirtyOnly', 'Dirty Only', 'Dirty Only', 'bool', dirtyOnly, {}),
		]

		super(self.__class__, self).__init__(name, fields)

	def cook(self, location, interface, attrs):
		location = attrs['location']

		if location == '':
			unique = attrs['unique']
			if unique:
				keys = State.uniqueKeys()
				for key in keys:
					print key, State.getKey(key)
			else:
				keys = State.allKeys()
				for k in keys[1:]:
					print k, State.getKey(k)

		else:
			if not State.hasKey(location):
				self.logger.info('State does not have key: % s' % location)
				return

			print location, State.getKey(location)


class PrintInterface(Op.Op):
	def __init__(self, name='/Print Interface', filter='', enable=False, dirtyOnly=False, namesOnly=True, frameRange=''):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('filter', 'Filter', 'Filter', 'string', filter, {}),
			('enable', 'Enable', 'Enable', 'bool', enable, {}),
			('dirtyOnly', 'Dirty only', 'Dirty only', 'bool', dirtyOnly, {}),
			('namesOnly', 'Names only', 'Names only', 'bool', namesOnly, {}),
			('frameRange', 'Frame range', 'Frame range', 'string', frameRange, {})
		]

		super(self.__class__, self).__init__(name, fields)

	def cook(self, location, interface, attrs):
		attrs = self.getAttrs()
		filter = attrs['filter']
		dirtyOnly = attrs['dirtyOnly']
		namesOnly = attrs['namesOnly']
		enable = attrs['enable']

		if not enable: return
		if not self.useFrame(interface.frame(), attrs['frameRange']): return

		locations = interface.locations()
		for location in locations:
			if filter and not location.startswith(filter):
				continue

			if dirtyOnly and interface.isDirty(location):
				continue

			if namesOnly:
				print location
			else:
				print location, interface.attrs(location)


class CalculatePoseEffectorPositions(Op.Op):
	def __init__(self, name='/Calculate Effector Positions', locations='', effectorsName='effectors',
				pointSize=10., colour=(0.6, 0.1, 0.7, 0.7), frameRange='', useWeights=False, labels='',
				calibration='', showContributions=True, visualiseLabels=False, enableCache=True,
				colourGroups=False, seed=0):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('locations', 'Skeleton locations', 'Skeleton locations', 'string', locations, {}),
			('effectorsName', 'Effectors name', 'Effectors name', 'string', effectorsName, {}),
			('pointSize', '3D Point size', '3D Point size', 'float', pointSize, {}),
			('colour', '3D Point colour', '3D Point colour', 'string', str(colour), {}),
			('frameRange', 'Frame range', 'Frame range', 'string', frameRange, {}),
			('useWeights', 'Use weights', 'Use weights', 'bool', useWeights, {}),
			('labels', 'Labels location', 'Labels location', 'string', labels, {}),
			('calibration', 'Calibration', 'Calibration (camera contributions)', 'string', calibration, {}),
			('show_contributions', 'Show contributions', 'Show camera contributions', 'bool', showContributions, {}),
			('visualiseLabels', 'Visualise labels', 'Visualise labels', 'bool', visualiseLabels, {}),
			('enableCache', 'Enable cache', 'Enable cache', 'bool', enableCache, {}),
			('colourGroups', 'Colour groups', 'Colour groups', 'bool', colourGroups, {}),
			('seed', 'Seed', 'Seed', 'int', seed, {'min': 0})
		]

		super(self.__class__, self).__init__(name, fields)
		self.effectorData = {}
		self.effectorLabels = {}

	def get_pose_effector_positions(self, chanValues, effectorLabels, skelDict, effectorData, effectorTargets, rootMat, effectorWeights=None):
		Character.pose_skeleton(skelDict['Gs'], skelDict, chanValues, rootMat)
		numEffectors = len(effectorTargets)
		effectors = np.zeros((numEffectors, 3, 4), dtype=np.float32)
		residual = np.zeros((numEffectors, 3, 4), dtype=np.float32)
		sc = ISCV.pose_effectors(effectors, residual, skelDict['Gs'], effectorData[0], effectorData[1], effectorData[2], effectorTargets)
		labels = np.unique(effectorLabels)
		x3ds = np.zeros((len(labels), 3), dtype=np.float32)
		if effectorWeights is not None:
			x3ds[:, 0] = np.bincount(effectorLabels, weights=effectors[:, 0, 3] * effectorWeights, minlength=labels[-1] + 1)[labels]
			x3ds[:, 1] = np.bincount(effectorLabels, weights=effectors[:, 1, 3] * effectorWeights, minlength=labels[-1] + 1)[labels]
			x3ds[:, 2] = np.bincount(effectorLabels, weights=effectors[:, 2, 3] * effectorWeights, minlength=labels[-1] + 1)[labels]
		else:
			x3ds[:, 0] = np.bincount(effectorLabels, weights=effectors[:, 0, 3], minlength=labels[-1] + 1)[labels]
			x3ds[:, 1] = np.bincount(effectorLabels, weights=effectors[:, 1, 3], minlength=labels[-1] + 1)[labels]
			x3ds[:, 2] = np.bincount(effectorLabels, weights=effectors[:, 2, 3], minlength=labels[-1] + 1)[labels]
			x3ds_count = np.bincount(effectorLabels, minlength=labels[-1] + 1)[labels]
			x3ds /= x3ds_count.reshape(-1, 1)

		return labels, x3ds

	def cook(self, location, interface, attrs):
		if not self.useFrame(interface.frame(), attrs['frameRange']): return
		skelDict = interface.attr('skelDict')
		if skelDict is None: return

		if 'markerNames' not in skelDict or 'markerWeights' not in skelDict or 'markerOffsets' not in skelDict: return

		if attrs['enableCache']:
			if location not in self.effectorLabels:
				try: # TODO: Fix properly
					self.effectorLabels[location] = np.int32([int(mn) for mn in skelDict['markerNames']])
				except:
					self.effectorLabels[location] = np.arange(len(skelDict['markerNames']))
			if location not in self.effectorData: self.effectorData[location] = SolveIK.make_effectorData(skelDict)
			effectorLabels = self.effectorLabels[location]
			effectorData = self.effectorData[location]
		else:
			effectorLabels = np.array([int(mn) for mn in skelDict['markerNames']], dtype=np.int32)
			effectorData = SolveIK.make_effectorData(skelDict)

		interface.setAttr('effectorLabels', effectorLabels)
		interface.setAttr('effectorData', effectorData)

		markerWeights = skelDict['markerWeights'] # if attrs['useWeights'] else None

		x3ds, x3ds_labels = SolveIK.skeleton_marker_positions(skelDict, skelDict['rootMat'], skelDict['chanValues'],
															effectorLabels, effectorData, markerWeights)
		# effectorTargets = np.zeros_like(effectorData[1])
		# weights = skelDict['markerWeights'] if attrs['useWeights'] else None
		# x3ds_labels, x3ds = self.get_pose_effector_positions(skelDict['chanValues'], effectorLabels, skelDict, effectorData, effectorTargets, skelDict['rootMat'], weights)
		x3ds_colours = np.array([], dtype=np.float32)
		x3ds_joints = []
		if attrs['colourGroups']:
			# Create random colours for each joint/group
			np.random.seed(attrs['seed'])
			colours = np.random.rand(skelDict['numJoints'], 3)
			colours = np.hstack((colours, np.ones((colours.shape[0], 1))))

			x3ds_colours = np.zeros((len(x3ds), 4), dtype=np.float32)
			labels = x3ds_labels.tolist()
			for li, label in enumerate(x3ds_labels):
				if str(label) not in skelDict['markerNames']:
					x3ds_colours[li, :] = [1, 0, 0, 1]
					continue

				# Find the index based on the label (typically in order but just to make sure)
				mi = skelDict['markerNames'].index(str(label))
				parent = skelDict['markerParents'][mi]
				x3ds_joints.append(parent)
				if parent >= len(colours):
					x3ds_colours[li] = np.float32([0, 0, 0, 1])
				else:
					x3ds_colours[li] = colours[parent]

		# Colour marker points based on labels if we have been given any (from a detection location)
		# Not labelled: Red
		# Labelled (more than one ray): Green
		# Labelled (one ray): Blue
		#x3ds_colours = np.array([], dtype=np.float32)
		cameraContributions = {}
		cameraPositions = None
		# TODO: The following needs to be optimised as it's massively slowing things down (35fps -> 5fps)
		if attrs['visualiseLabels'] and 'labels' in attrs and attrs['labels']:
			labels = interface.attr('labels', atLocation=attrs['labels'])
			if labels is not None:
				x3ds_colours = np.tile((1, 0, 0, 0.7), (x3ds_labels.shape[0], 1))
				labelHits = np.array([len(np.where(labels == x3d_label)[0]) for x3d_label in x3ds_labels], dtype=np.int32)
				x3ds_colours[np.where(labelHits == 1)[0]] = (0, 0, 1, 0.7)
				x3ds_colours[np.where(labelHits > 1)[0]] = (0, 1, 0, 0.7)

			# If we've been given the calibration data we can build the info for contributing cameras per 3D point
			if 'calibration' in attrs and attrs['calibration']:
				mats = interface.attr('mats', atLocation=attrs['calibration'])
				if mats is not None:
					cameraPositions = np.array([m[4] for m in mats], dtype=np.float32)
					splits = interface.attr('x2ds_splits', atLocation=attrs['labels'])
					if splits is not None:
						for label3d in x3ds_labels:
							camIds = [interface.findCameraIdFromRayId(rayId, splits) for rayId in np.where(labels == label3d)[0]]
							cameraContributions[label3d] = camIds

		markerFilter = interface.attr('markerFilter')
		if markerFilter:
			whichInds = np.where(x3ds_labels.reshape(-1, 1) == markerFilter)[0]
			x3ds, x3ds_labels = x3ds[whichInds], x3ds_labels[whichInds]
			if x3ds_colours:
				x3ds_colours = x3ds_colours[whichInds]

		pAttrs = {
			'x3ds': x3ds,
			'x3ds_labels': x3ds_labels,
			'x3ds_colour': eval(attrs['colour']),
			'x3ds_pointSize': attrs['pointSize'],
			'x3ds_colours': np.float32(x3ds_colours),
			'showCameraContributions': attrs['show_contributions'],
			'cameraContributions': cameraContributions,
			'cameraPositions': cameraPositions,
			'joints': np.int32(x3ds_joints)
		}

		interface.createChild(attrs['effectorsName'], 'points3d', attrs=pAttrs)


class Label3DsFrom3Ds(Op.Op):
	def __init__(self, name='/Label 3Ds From 3Ds', locations='', seedX3ds='', x3d_threshold=30):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('locations', 'Skeleton locations', 'Skeleton locations', 'string', locations, {}),
			('seedX3ds', 'Seed 3Ds', 'Seed 3Ds', 'string', seedX3ds, {}),
			('x3d_threshold', '3D threshold', '3D threshold', 'float', x3d_threshold, {'min': 0})
		]

		super(self.__class__, self).__init__(name, fields)

	def cook(self, location, interface, attrs):
		x3ds = interface.attr('x3ds')
		if x3ds is None or not x3ds.any(): return

		if not attrs['seedX3ds']: return
		seedX3ds = interface.attr('x3ds', atLocation=attrs['seedX3ds'])
		if seedX3ds is None or not seedX3ds.any(): return

		_, prev_labels, prev_vels = Label.label_3d_from_3d(x3ds, np.array(range(len(x3ds)), dtype=np.int32), None, seedX3ds, attrs['x3d_threshold'])
		keepers = np.where(prev_labels != -1)[0]
		if len(keepers) != 0:
			x3ds[prev_labels[keepers]] = seedX3ds[keepers]

		interface.setAttr('x3ds', x3ds)


class LabelX3DsFromX3Ds(Op.Op):
	def __init__(self, name='/Label_X3Ds', locations='', seedX3ds='', x3d_threshold=30, updateX3ds=False):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('locations', 'X3Ds locations', 'X3Ds locations', 'string', locations, {}),
			('seedX3ds', 'Seed X3Ds', 'Seed X3Ds', 'string', seedX3ds, {}),
			('x3d_threshold', '3D threshold', '3D threshold', 'float', x3d_threshold, {'min': 0}),
			('updateX3ds', 'Update x3ds', 'Update x3ds', 'bool', updateX3ds, {})
		]

		super(self.__class__, self).__init__(name, fields)

	def cook(self, location, interface, attrs):
		if not location: return
		x3ds_src = interface.attr('x3ds')
		labels_src = interface.attr('x3ds_labels')

		if x3ds_src is None or not x3ds_src.any(): return

		x3ds_target = interface.attr('x3ds', atLocation=attrs['seedX3ds'])
		labels_target = interface.attr('x3ds_labels', atLocation=attrs['seedX3ds'])

		if x3ds_target is None or not x3ds_target.any(): return

		# print labels_target
		# print labels_src

		from GCore import Label
		sc, labels, vels = Label.label_3d_from_3d(x3ds_target, labels_target, None, x3ds_src, attrs['x3d_threshold'])

		# print "labels:", labels
		# print "vels:", vels

		keepers = np.where(labels != -1)[0]

		# If we don't find a point nearby but we want to retain the target, we copy the target (expected) for the
		# missing labels
		if len(keepers) != 0:
			if attrs['updateX3ds']:
				# Copy nearby points
				#x3ds_src[keepers] = x3ds_target[keepers]
				idxs = np.where(labels_target == labels[keepers].reshape(-1, 1))[1]
				# idxs = np.unique(np.where(labels_target == labels[keepers].reshape(-1, 1))[0])
				x3ds_src[keepers] = x3ds_target[idxs]
				# print 'Frame:', interface.frame(), '| Label:', labels[keepers]
			else:
				# If we find a point nearby we override the label with the target
				# labels_src[labels[keepers]] = labels[keepers]
				labels_src[keepers] = labels[keepers]

			# print "labels_src:", labels_src
			interface.setAttr('x3ds_labels', labels_src)
			interface.setAttr('x3ds', x3ds_src)


class AssignLabelsWithVelocities(Op.Op):
	def __init__(self, name='/Assign Labels with Vels', locations='', calibration='', x2d_threshold=0.03, pred_2d_threshold=0.015, frameRange=''):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('calibration', 'calibration', 'calibration', 'string', calibration, {}),
			('x2d_threshold', 'X2D Threshold', 'X2D Threshold', 'float', x2d_threshold, {}),
			('pred_2d_threshold', 'Pred X2D Threshold', 'Pred X2D Threshold', 'float', pred_2d_threshold, {}),
			('frameRange', 'Frame range', 'Frame range', 'string', frameRange, {})
		]

		self.x2ds = np.zeros((0, 2), dtype=np.float32)
		self.vels = np.zeros((0, 2), dtype=np.float32)
		self.labels = np.zeros(0, dtype=np.int32)
		self.cookedLabels = np.zeros(0, dtype=np.int32)
		self.splits = None
		self.score = -1

		super(self.__class__, self).__init__(name, fields)

	def cook(self, location, interface, attrs):
		import ISCV
		if not self.useFrame(interface.frame(), attrs['frameRange']): return

		new_x2ds = interface.attr('x2ds')
		splits = interface.attr('x2ds_splits')
		if new_x2ds is None or splits is None: return

		if not attrs['calibration']: return
		if self.splits is None:
			mats = interface.attr('mats', atLocation=attrs['calibration'])
			self.splits = np.zeros(len(mats) + 1, dtype=np.int32)

		assignmentLocation = location + '/assignment'
		dirty = interface.attr('dirty', atLocation=assignmentLocation)

		x2d_threshold = attrs['x2d_threshold']
		pred_2d_threshold = attrs['pred_2d_threshold']
		clouds = ISCV.HashCloud2DList(new_x2ds, splits, max(pred_2d_threshold, x2d_threshold))

		if interface.location(assignmentLocation) is None:
			self.score, self.labels, self.vels = clouds.assign_with_vel(self.x2ds, self.vels, self.splits, self.labels, pred_2d_threshold)
			self.x2ds, self.splits = new_x2ds, splits

			a = {
				'score': self.score,
				'labels': self.labels,
				'vels': self.vels,
				'x2ds': self.x2ds,
				'x2ds_splits': self.splits,
				'x2ds_colour': (0, 0, 1, 1),
				'dirty': False
			}
			interface.createChild('assignment', 'group', attrs=a)
			self.cookedLabels = self.labels

		if dirty:
			labels = interface.attr('labels', atLocation=assignmentLocation)
			vels = interface.attr('vels', atLocation=assignmentLocation)

			if labels is not None: self.labels = labels
			if vels is not None: self.vels = vels

			self.score, labels, vels = clouds.assign_with_vel(self.x2ds, self.vels, self.splits, self.labels, pred_2d_threshold)
			self.x2ds, self.splits = new_x2ds, splits

			interface.setAttr('score', self.score, atLocation=assignmentLocation)
			interface.setAttr('labels', labels, atLocation=assignmentLocation)
			interface.setAttr('vels', vels, atLocation=assignmentLocation)
			# interface.setAttr('x2ds', new_x2ds, atLocation=assignmentLocation)
			# interface.setAttr('x2ds_splits', self.splits, atLocation=assignmentLocation)
			# interface.setAttr('x2ds_colour', (0, 0, 1, 1), atLocation=assignmentLocation)
			interface.setAttr('dirty', False, atLocation=assignmentLocation)

			self.cookedLabels = labels

		interface.setAttr('labels', self.cookedLabels)
		interface.setAttr('score', self.score)

		# Show labelled detections as green for clarity
		labelColours = interface.getLabelColours(self.cookedLabels, interface.attr('x2ds_colour'))
		if labelColours.any():
			interface.setAttr('x2ds_colours', labelColours)


class UpdateVelocities(Op.Op):
	def __init__(self, name='/Update vels', locations='', source=''):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('source', 'source', 'source', 'string', source, {})
		]

		super(self.__class__, self).__init__(name, fields)

		self.x2ds = np.zeros((0, 2), dtype=np.float32)
		self.vels = np.zeros((0, 2), dtype=np.float32)
		self.splits = None
		self.labels = np.zeros(0, dtype=np.int32)

	def cook(self, location, interface, attrs):
		import ISCV

		source = attrs['source']
		if not source: source = location

		x2ds = interface.attr('x2ds')
		splits = interface.attr('x2ds_splits')
		if x2ds is None or splits is None:
			self.logger.warning('No detections found at: %s' % location)
			return

		if self.splits is None: self.splits = np.zeros_like(splits)

		vels = interface.attr('vels', atLocation=location + '/assignment')
		labels = interface.attr('labels', atLocation=source)
		if vels is None or labels is None: return

		ISCV.update_vels(x2ds, splits, labels, self.x2ds, self.splits, self.labels, vels)
		self.x2ds, self.splits, self.labels, self.vels = x2ds, splits, labels, vels

		interface.setAttr('labels', labels, atLocation=location + '/assignment')
		interface.setAttr('vels', vels, atLocation=location + '/assignment')
		interface.setAttr('dirty', True, atLocation=location + '/assignment')


class SolveX3ds(Op.Op):
	def __init__(self, name='/Solve 3D from labels', locations='', calibration='', pointSize=12., colour=(0.3, 0.9, 0.7, 0.7),
				showContributions=True):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('locations', 'Detections locations', 'Detections locations', 'string', locations, {}),
			('calibration', 'Calibration location', 'Calibration location', 'string', calibration, {}),
			('pointSize', '3D Point size', '3D Point size', 'float', pointSize, {}),
			('colour', '3D Point colour', '3D Point colour', 'string', str(colour), {}),
			('show_contributions', 'Show contributions', 'Show camera contributions', 'bool', showContributions, {})
		]

		super(self.__class__, self).__init__(name, fields)

	def cook(self, location, interface, attrs):
		x2ds = interface.attr('x2ds', np.array([]))
		splits = interface.attr('x2ds_splits')
		labels = interface.attr('labels')
		if not x2ds.any() or splits is None or labels is None: return

		calibration = attrs['calibration']
		Ps = interface.attr('Ps', atLocation=calibration)
		mats = interface.attr('mats', atLocation=calibration)

		x3ds, x3ds_labels, E, x2d_labels = Recon.solve_x3ds(x2ds, splits, np.array(labels, dtype=np.int32), Ps)

		# cameraPositions = np.array([m[4] for m in mats], dtype=np.float32)
		# cameraContributions = {}
		# for label3d in x3ds_labels:
		# 	camIds = [interface.findCameraIdFromRayId(rayId, splits) for rayId in np.where(labels == label3d)[0]]
		# 	cameraContributions[label3d] = camIds

		pAttrs = {
			'x3ds': x3ds,
			'x3ds_labels': x3ds_labels,
			'x3ds_colour': eval(attrs['colour']),
			'x3ds_pointSize': attrs['pointSize']
		}

		if attrs['show_contributions']:
			pAttrs['showCameraContributions'] = attrs['show_contributions']
			pAttrs['camerasLocation'] = calibration
			pAttrs['x2ds_splits'] = splits
			pAttrs['labels'] = labels

		interface.createChild('solved', 'points3d', attrs=pAttrs)


class SolveSkeletonFrom3D(Op.Op):
	def __init__(self, name='/Solve skeleton from 3D', locations='', x3ds=''):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('locations', 'Skeleton locations', 'Skeleton locations', 'string', locations, {}),
			('x3ds', '3D points', '3D points', 'string', x3ds, {})
		]

		super(self.__class__, self).__init__(name, fields)

	def cook(self, location, interface, attrs):
		skelDict = interface.attr('skelDict')
		if skelDict is None: return

		effectorLabels = interface.attr('effectorLabels')
		effectorData = interface.attr('effectorData')
		if effectorLabels is None or effectorData is None: return

		effectorLabels = list(effectorLabels)

		if not attrs['x3ds']: return
		x3ds = interface.attr('x3ds', atLocation=attrs['x3ds'])
		labels = interface.attr('x3ds_labels', atLocation=attrs['x3ds'])

		score = SolveIK.solve_skeleton_from_3d(x3ds, labels, effectorLabels, skelDict, effectorData, skelDict['rootMat'])
		interface.setAttr('skelDict', skelDict)
		interface.setAttr('score', score)


class SolveSkeletonFrom2D(Op.Op):
	def __init__(self, name='/Solve skeleton from 2D', locations='', detections='', calibration='', labels='', outerIterations=5,
				pointSize=14., colour=(0.4, 0.3, 0.4, 0.7), showContributions=True):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('locations', 'Skeleton locations', 'Skeleton locations', 'string', locations, {}),
			('detections', 'Labelled detections', 'Labelled detections', 'string', detections, {}),
			('calibration', 'Calibration', 'Calibration', 'string', calibration, {}),
			('outerIts', '# Outer iterations', 'Number of outer iterations', 'int', outerIterations, {}),
			('pointSize', '3D Point size', '3D Point size', 'float', pointSize, {}),
			('colour', '3D Point colour', '3D Point colour', 'string', str(colour), {}),
			('show_contributions', 'Show contributions', 'Show camera contributions', 'bool', showContributions, {})
		]

		super(self.__class__, self).__init__(name, fields)

	def cook(self, location, interface, attrs):
		skelDict = interface.attr('skelDict')
		if skelDict is None: return

		effectorLabels = interface.attr('effectorLabels')
		effectorData = interface.attr('effectorData')
		if effectorLabels is None or effectorData is None: return

		effectorLabels = list(effectorLabels)

		if not attrs['detections']: return
		detections = attrs['detections']
		x2ds = interface.attr('x2ds', atLocation=detections)
		splits = interface.attr('x2ds_splits', atLocation=detections)
		labels = interface.attr('labels', atLocation=detections)

		if not attrs['calibration']: return
		calibration = attrs['calibration']
		Ps = interface.attr('Ps', atLocation=calibration)
		mats = interface.attr('mats', atLocation=calibration)

		s_x3ds, s_x3d_labels, E, s_x2d_labels = SolveIK.solve_skeleton_from_2d(x2ds, splits, labels, effectorLabels, Ps, skelDict, effectorData,
																			skelDict['rootMat'], outerIts=attrs['outerIts'])
		interface.setAttr('skelDict', skelDict)

		# Find which cameras contribute to the 3D reconstructions (optional?)
		cameraPositions = np.array([m[4] for m in mats], dtype=np.float32)
		cameraContributions = {}
		for label3d in s_x3d_labels:
			camIds = [interface.findCameraIdFromRayId(rayId, splits) for rayId in np.where(labels == label3d)[0]]
			cameraContributions[label3d] = camIds

		pAttrs = {
			'x3ds': s_x3ds,
			'x3ds_labels': s_x3d_labels,
			'x3ds_colour': eval(attrs['colour']),
			'x3ds_pointSize': attrs['pointSize'],
			'cameraContributions': cameraContributions,
			'showCameraContributions': attrs['show_contributions'],
			'cameraPositions': cameraPositions
		}
		interface.createChild('ikSolve', 'points3d', attrs=pAttrs)

	# Add detection labels
	# if s_x2d_labels.any():
	# 	interface.setAttr('labels', s_x2d_labels, atLocation=detections)
	#
	# 	# Show labelled detections as green for clarity
	# 	labelColour = interface.attr('x2ds_colour', atLocation=detections)
	# 	labelColours = interface.getLabelColours(s_x2d_labels, labelColour)
	# 	if labelColours.any():
	# 		interface.setAttr('x2ds_colours', labelColours, atLocation=detections)


class MarkerCopy(Op.Op):
	def __init__(self, name='/Marker Copy', locations='', sourceSkeleton=''):
		self.fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'locations', 'Skeleton locations', 'string', locations, {}),
			('source', 'source', 'Source skeleton', 'string', sourceSkeleton, {})
		]

		super(self.__class__, self).__init__(name, self.fields)

	def cook(self, location, interface, attrs):
		if not location or not attrs['source']: return

		# Get the source skeleton dict
		sourceLocation = attrs['source']
		skelDict_source = interface.attr('skelDict', atLocation=sourceLocation)
		if not skelDict_source: return

		# Get the target skeleton dict (the location we are cooking)
		skelDict_target = interface.attr('skelDict')
		if not skelDict_target: return

		if 'numMarkers' in skelDict_source: skelDict_target['numMarkers'] = skelDict_source['numMarkers']
		if 'markerNames' in skelDict_source: skelDict_target['markerNames'] = skelDict_source['markerNames']
		if 'markerParents' in skelDict_source: skelDict_target['markerParents'] = skelDict_source['markerParents']
		if 'markerOffsets' in skelDict_source: skelDict_target['markerOffsets'] = skelDict_source['markerOffsets']
		if 'markerWeights' in skelDict_source: skelDict_target['markerWeights'] = skelDict_source['markerWeights']
		if 'markerColour' in skelDict_source: skelDict_target['markerColour'] = skelDict_source['markerColour']

		interface.setAttr('skelDict', skelDict_target)


class UpdateSkeletonMarkers(Op.Op):
	def __init__(self, name='/Update Skeleton Markers', locations='', x3ds='', frameRange=''):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('locations', 'Skeleton locations', 'Skeleton locations', 'string', locations, {}),
			('x3ds', '3D points', '3D points', 'string', x3ds, {}),
			('frameRange', 'Frame range', 'Frame range', 'string', frameRange, {})
		]

		super(self.__class__, self).__init__(name, fields)

	def cook(self, location, interface, attrs):
		if not self.useFrame(interface.frame(), attrs['frameRange']): return
		skelDict = interface.attr('skelDict')
		if skelDict is None: return

		if not attrs['x3ds']: return
		x3ds = interface.attr('x3ds', atLocation=attrs['x3ds'])
		x3ds_labels = interface.attr('x3ds_labels', atLocation=attrs['x3ds'])

		if x3ds is None or x3ds_labels is None:
			self.logger.error('No 3D data found at: %s' % attrs['x3ds'])
			return

		if 'numMarkers' not in skelDict:
			skelDict['numMarkers'] = len(x3ds)

		if 'markerOffsets' not in skelDict:
			skelDict['markerOffsets'] = np.zeros((skelDict['numMarkers'], 3), dtype=np.float32)

		# Calculate the offsets for the new x3ds using the skeleton joint global matrices
		for x3d, label in zip(x3ds, x3ds_labels):
			idx = skelDict['markerNames'].index(str(label))
			jointIdx = skelDict['markerParents'][idx]
			jointGs = skelDict['Gs'][jointIdx].copy()
			jointGs = np.append(jointGs, [[0, 0, 0, 1]], axis=0)
			x3d = x3d.transpose()
			x3d = np.append(x3d, [1])
			offset = np.dot(np.linalg.inv(jointGs), x3d)
			skelDict['markerOffsets'][idx] = offset[:3]

		interface.setAttr('skelDict', skelDict)


class RemoveMarkerData(Op.Op):
	def __init__(self, name='/Remove Marker Data', locations='', frameRange=''):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('locations', 'Skeleton locations', 'Skeleton locations', 'string', locations, {}),
			('frameRange', 'Frame range', 'Frame range', 'string', frameRange, {})
		]

		super(self.__class__, self).__init__(name, fields)

	def cook(self, location, interface, attrs):
		if not self.useFrame(interface.frame(), attrs['frameRange']): return
		skelDict = interface.attr('skelDict')
		if skelDict is None: return

		# if 'numMarkers' in skelDict: del skelDict['numMarkers']
		if 'markerNames' in skelDict: del skelDict['markerNames']
		# if 'markerNamesUnq' in skelDict: del skelDict['markerNamesUnq']
		if 'markerParents' in skelDict: del skelDict['markerParents']
		if 'markerOffsets' in skelDict: del skelDict['markerOffsets']
		if 'markerWeights' in skelDict: del skelDict['markerWeights']
		if 'markerColour' in skelDict: del skelDict['markerColour']
		if 'markerNormals' in skelDict: del skelDict['markerNormals']

		interface.setAttr('skelDict', skelDict)


class AddMarkersFrom3dLabels(Op.Op):
	def __init__(self, name='/Add Markers From 3D Labels', locations='', x3ds='', x3d_threshold=100.0, frameRange=''):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('locations', 'Skeleton locations', 'Skeleton locations', 'string', locations, {}),
			('x3ds', '3D Points', '3D Points', 'string', x3ds, {}),
			('x3d_threshold', '3D threshold', '3D threshold', 'float', x3d_threshold, {}),
			('frameRange', 'Frame range', 'Frame range', 'string', frameRange, {})
		]

		super(self.__class__, self).__init__(name, fields)

	def getWorldSpaceMarkerPos(self, skelDict, offsets, parents):
		vs = []
		lbls = []
		for mi in range(len(offsets)):
			parentJointGs = np.append(skelDict['Gs'][parents[mi]], [[0, 0, 0, 1]], axis=0)
			mOffset = offsets[mi]
			mOffset = np.array([[mOffset[0], mOffset[1], mOffset[2], 1]], dtype=np.float32)

			v = np.dot(parentJointGs, mOffset.T)
			vs.append(np.concatenate(v[:3]))
			lbls.append(mi)

		vs = np.array(vs, dtype=np.float32)
		return vs, lbls

	def cook(self, location, interface, attrs):
		if not self.useFrame(interface.frame(), attrs['frameRange']): return
		if not attrs['x3ds']: return
		x3ds = interface.attr('x3ds', atLocation=attrs['x3ds'])
		x3d_labels = interface.attr('x3d_labels', atLocation=attrs['x3ds'])

		skelDict = interface.attr('skelDict')
		if skelDict is None:
			self.logger.error('No skeleton found at: %s' % location)
			return

		if 'geom_Vs' not in skelDict:
			self.logger.error('No geometry (geom_Vs) found in skeleton at: %s' % location)
			return

		# Go through the markers and find the nearest vertices on the geometry
		cloud = ISCV.HashCloud3D(skelDict['geom_Vs'], attrs['x3d_threshold'])
		scores, matches, matches_splits = cloud.score(x3ds)

		marker_inds = []
		vert_inds = []
		for i, (s, e) in enumerate(zip(matches_splits[:-1], matches_splits[1:])):
			if s - e == 0: continue
			lowestScoreIdx = matches[s:e][np.argmin(scores[s:e])]
			marker_inds.append(i)
			vert_inds.append(lowestScoreIdx)

		# Find joints to which the vertices map to
		weights = skelDict['shape_weights'][0]
		markerJointsMap = {}
		for m_idx, v_idx in zip(marker_inds, vert_inds):
			markerJointsMap[m_idx] = []
			for ji, verts in weights[0].iteritems():
				if v_idx in verts[0]:
					# Find joint name
					for jname, jnum in weights[1].iteritems():
						if ji == jnum and jname in skelDict['jointNames']:
							markerJointsMap[m_idx].append((jname, verts[1][ji]))

		markerOffsets = skelDict['m_offsets'] if 'm_offsets' in skelDict else []
		markerParents = skelDict['m_parents'] if 'm_parents' in skelDict else []
		markerWeights = skelDict['m_weights'] if 'm_weights' in skelDict else []
		markerNames = skelDict['m_names'] if 'm_names' in skelDict else []

		for mi, joints in markerJointsMap.iteritems():
			for jointName, jointWeight in joints:
				markerNames.append(skelDict['markerNames'][mi])
				jointIndex = skelDict['jointIndex'][jointName]
				markerParents.append(jointIndex)
				jointGs = skelDict['Gs'][jointIndex].copy()
				jointGs = np.append(jointGs, [[0, 0, 0, 1]], axis=0)
				markerWorldPos = x3ds[mi].transpose()
				markerWorldPos = np.append(markerWorldPos, [1])
				offset = np.dot(np.linalg.inv(jointGs), markerWorldPos)
				markerOffsets.append(offset[:3])
				markerWeights.append(jointWeight[3])

		skelDict['m_offsets'] = markerOffsets
		skelDict['m_parents'] = markerParents
		skelDict['m_weights'] = markerWeights
		skelDict['m_names'] = markerNames

		interface.setAttr('skelDict', skelDict)

		vs, vs_labels = self.getWorldSpaceMarkerPos(skelDict, markerOffsets, markerParents)
		pAttrs = {
			'x3ds': vs,
			'x3ds_labels': vs_labels,
			'x3ds_colour': (0, 0, 1, 0.7),
			'x3ds_pointSize': 14
		}
		interface.createChild('x3ds', 'points3d', attrs=pAttrs)


class CreateMarkersFromLabels(Op.Op):
	def __init__(self, name='/Create Markers From Labels', locations='', x3ds='', x2d_labels='', x3d_threshold=100.0, type='default', useAllWeights=True, frameRange=''):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('locations', 'Skeleton locations', 'Skeleton locations', 'string', locations, {}),
			('x3ds', '3D Points', '3D Points', 'string', x3ds, {}),
			('x2d_labels', '2D Labels', '2D Labels', 'string', x2d_labels, {}),
			('x3d_threshold', '3D threshold', '3D threshold', 'float', x3d_threshold, {}),
			('type', 'Marker type', 'Marker type', 'string', type, {}),
			('useAllWeights', 'Use all weights', 'Use all weights', 'bool', useAllWeights, {}),
			('frameRange', 'Frame range', 'Frame range', 'string', frameRange, {})
		]

		super(self.__class__, self).__init__(name, fields)
		self.jointMarkersMappingData = None

	def cook(self, location, interface, attrs):
		if not self.useFrame(interface.frame(), attrs['frameRange']): return
		if not attrs['x3ds']: return
		# if not attrs['x2d_labels']: return

		x3ds = interface.attr('x3ds', atLocation=attrs['x3ds'])
		x3ds_labels = interface.attr('x3ds_labels', atLocation=attrs['x3ds'])
		if x3ds is None or len(x3ds) == 0 or x3ds_labels is None:
			# self.logger.warning('No 3D data found at: %s' % attrs['x3ds'])
			if self.jointMarkersMappingData is not None:
				interface.setAttr('jointMarkersMap', self.jointMarkersMappingData)
			return

		# labels = interface.attr('labels', atLocation=attrs['x2d_labels'])
		# if labels is None:
		# 	self.logger.error('No labels found at: %s' % attrs['x2d_labels'])

		skelDict = interface.attr('skelDict')
		if skelDict is None:
			self.logger.error('No skeleton found at: %s' % location)
			return

		if 'geom_Vs' not in skelDict:
			self.logger.error('No geometry (geom_Vs) found in skeleton at: %s' % location)
			return

		# ulabels = np.unique(labels[np.where(labels != -1)[0]])
		# usedX3ds = np.concatenate(([np.where(x3ds_labels == l)[0] for l in ulabels]))

		# chosen_x3ds = x3ds[usedX3ds]
		# chosen_labels = x3ds_labels[usedX3ds]
		chosen_x3ds = np.array(x3ds, dtype=np.float32)
		chosen_labels = np.array(x3ds_labels, dtype=np.int32)

		a = {
			'x3ds': chosen_x3ds,
			'x3ds_colour': (0.8, 0, 0.0, 0.7),
			'x3ds_pointSize': 18
		}
		# interface.createChild('chosen', 'points3d', attrs=a)

		# Go through the markers and find the nearest vertices on the geometry
		cloud = ISCV.HashCloud3D(skelDict['geom_Vs'], attrs['x3d_threshold'])
		scores, matches, matches_splits = cloud.score(chosen_x3ds)
		if len(matches) == 0:
			self.logger.warning('No matches were found.')
			return

		marker_inds = []
		vert_inds = []
		for i, (s, e) in enumerate(zip(matches_splits[:-1], matches_splits[1:])):
			if s - e == 0: continue
			lowestScoreIdx = matches[s:e][np.argmin(scores[s:e])]
			marker_inds.append(i)
			vert_inds.append(lowestScoreIdx)

		linesAttrs = {
			'colour': (0.4, 0.1, 0.4, 0.7),
			'edgeColour': (0.6, 0.1, 0.6, 0.7),
			'pointSize': 2
		}

		marker_inds = np.array(marker_inds, dtype=np.int32)
		linesAttrs['x0'] = chosen_x3ds[marker_inds]

		vert_inds = np.array(vert_inds, dtype=np.int32)
		linesAttrs['x1'] = skelDict['geom_Vs'][vert_inds]
		# interface.createChild('markerToVertex', 'edges', attrs=linesAttrs)

		# Find joints to which the vertices map to
		weights = skelDict['shape_weights'][0]
		markerJointsMap = {}
		for m_idx, v_idx in zip(marker_inds, vert_inds):
			markerJointsMap[m_idx] = []
			for ji, verts in weights[0].iteritems():
				if v_idx in verts[0]: # and ji in jointIndsMap:
					# Find joint name
					# markerJointsMap[m_idx].append(jointIndsMap[ji]['name'])
					for jname, jnum in weights[1].iteritems():
						if ji == jnum and jname in skelDict['jointNames']:
							markerJointsMap[m_idx].append((jname, verts[1][ji]))

		# markerOffsets = skelDict['m_offsets'] if 'm_offsets' in skelDict else []
		# markerParents = skelDict['m_parents'] if 'm_parents' in skelDict else []
		# markerWeights = skelDict['m_weights'] if 'm_weights' in skelDict else []
		# markerNames = skelDict['m_names'] if 'm_names' in skelDict else []
		# markerTypes = skelDict['m_types'] if 'm_types' in skelDict else []

		markerOffsets = skelDict['markerOffsets'].tolist() if 'markerOffsets' in skelDict else []
		markerParents = skelDict['markerParents'].tolist() if 'markerParents' in skelDict else []
		markerWeights = skelDict['markerWeights'].tolist() if 'markerWeights' in skelDict else []
		markerNames = skelDict['markerNames'] if 'markerNames' in skelDict else []
		markerTypes = skelDict['markerTypes'] if 'markerTypes' in skelDict else []

		# for mi, joints in markerJointsMap.iteritems():
		# 	maxWeight, maxIndex = 0, -1
		# 	for jointName, jointWeight in joints:
		# 		if jointWeight[3] > maxWeight:
		# 			maxWeight = jointWeight[3]
		# 			maxIndex = skelDict['jointIndex'][jointName]
		#
		# 	if maxIndex != -1:
		# 		lbl = chosen_labels[mi]
		# 		if 'm_names' not in skelDict or lbl not in skelDict['m_names']:
		# 			markerNames.append(str(lbl))
		# 			markerParents.append(maxIndex)
		# 			jointGs = skelDict['Gs'][maxIndex].copy()
		# 			jointGs = np.append(jointGs, [[0, 0, 0, 1]], axis=0)
		# 			markerWorldPos = chosen_x3ds[mi].transpose()
		# 			markerWorldPos = np.append(markerWorldPos, [1])
		# 			offset = np.dot(np.linalg.inv(jointGs), markerWorldPos)
		# 			markerOffsets.append(offset[:3])
		# 			markerWeights.append(1.0)
		# 			markerTypes.append(attrs['type'])
		print 'Calculate weights:'
		if attrs['useAllWeights']:
			# Go through existing known markers and build new marker data
			# for mi, markerName in enumerate(skelDict['markerNames']):
			for mi, joints in markerJointsMap.iteritems():
				lbl = chosen_labels[mi]
				for jointName, jointWeight in joints:
					markerNames.append(str(lbl))
					jointIndex = skelDict['jointIndex'][jointName]
					markerParents.append(jointIndex)
					jointGs = skelDict['Gs'][jointIndex].copy()
					jointGs = np.append(jointGs, [[0, 0, 0, 1]], axis=0)
					markerWorldPos = chosen_x3ds[mi].transpose()
					markerWorldPos = np.append(markerWorldPos, [1])
					offset = np.dot(np.linalg.inv(jointGs), markerWorldPos)
					markerOffsets.append(offset[:3])

					print ' Weight:', jointWeight[3], '->', jointName
					initialWeight = 1. / pow(np.linalg.norm(offset[:3]), 2)

					if 'Spine' in jointName or 'Chest' in jointName:
						initialWeight = 0.
					else:
						coords = skelDict['Gs'][:, :3, 3]
						pointMarker = markerWorldPos[:3]
						ap = pointMarker - coords[jointIndex]
						ab = np.dot(jointGs[:3,:3], skelDict['Bs'][jointIndex])
						projectedPoint = coords[jointIndex] + (np.dot(ap, ab) / np.dot(ab, ab)) * ab

						projectionVector = projectedPoint - pointMarker
						dist = np.linalg.norm(projectionVector)
						initialWeight = 1. / pow(dist, 2)

					markerWeights.append(initialWeight)
					# markerWeights.append(jointWeight[3])
					markerTypes.append(attrs['type'])
		else:
			for mi, joints in markerJointsMap.iteritems():
				maxWeight, maxIndex = 0, -1
				for jointName, jointWeight in joints:
					if jointWeight[3] > maxWeight:
						maxWeight = jointWeight[3]
						maxIndex = skelDict['jointIndex'][jointName]

				if maxIndex != -1:
					lbl = chosen_labels[mi]
					markerNames.append(str(lbl))
					markerParents.append(maxIndex)
					jointGs = skelDict['Gs'][maxIndex].copy()
					jointGs = np.append(jointGs, [[0, 0, 0, 1]], axis=0)
					markerWorldPos = chosen_x3ds[mi].transpose()
					markerWorldPos = np.append(markerWorldPos, [1])
					offset = np.dot(np.linalg.inv(jointGs), markerWorldPos)
					markerOffsets.append(offset[:3])
					markerWeights.append(1.0)
					markerTypes.append(attrs['type'])

		# skelDict['m_offsets'] = markerOffsets
		# skelDict['m_parents'] = markerParents
		# skelDict['m_weights'] = markerWeights
		# skelDict['m_names'] = markerNames
		# skelDict['m_types'] = markerTypes

		skelDict['numMarkers'] = len(markerNames)
		skelDict['markerNames'] = markerNames
		skelDict['markerParents'] = np.array(markerParents, dtype=np.int32)
		skelDict['markerOffsets'] = np.array(markerOffsets, dtype=np.float32)
		skelDict['markerWeights'] = np.array(markerWeights, dtype=np.float32)
		skelDict['markerTypes'] = markerTypes
		interface.setAttr('override', True)

		print 'Marker weights:', skelDict['markerWeights']

		# Renormalise
		if attrs['useAllWeights']:
			effectorLabels = np.array([int(mn) for mn in skelDict['markerNames']], dtype=np.int32)
			labels = np.unique(effectorLabels)
			lengths = np.bincount(skelDict['markerNames'], minlength=labels[-1]+1)[labels]
			splits = Interface.makeSplitBoundaries(lengths)
			for s, e in zip(splits[:-1], splits[1:]):
				weights = skelDict['markerWeights'][s:e]
				weights /= sum(weights)
				# weights = np.ones_like(weights) * (1. / (e - s))
				skelDict['markerWeights'][s:e] = weights
		else:
			for ji in range(skelDict['numJoints']):
				whichJoints = np.where(skelDict['markerParents'] == ji)[0]
				if not whichJoints.any(): continue
				weightSum = np.sum(skelDict['markerWeights'][whichJoints])
				skelDict['markerWeights'][whichJoints] /= weightSum

		# skelDict['markerWeights'] = np.array([0.5, 0.5], dtype=np.float32)
		print 'Marker weights (normalised):', skelDict['markerWeights']

		# print "> Update skeleton"
		interface.setAttr('skelDict', skelDict)

		if True:
			vs = x3ds
			jointMarkersMap = {}
			weights = skelDict['shape_weights'][0]
			for jointName, idx in weights[1].iteritems():
				if jointName not in skelDict['jointNames']: continue
				jointMarkersMap[jointName] = []
				jointVerts = weights[0][idx][0]
				for i, vi in enumerate(vert_inds):
					if vi in jointVerts:
						jointMarkersMap[jointName].append(marker_inds[i])

			j_inds = []
			m_inds = []
			for jointName, markerInds in jointMarkersMap.iteritems():
				_j_inds, _m_inds = [], []
				for idx in markerInds:
					_j_inds.append(skelDict['jointIndex'][jointName])
					_m_inds.append(idx)

				j_inds.append(_j_inds)
				m_inds.append(_m_inds)

			np.random.seed(100)
			colours = np.random.rand(len(j_inds), 3)
			colours = np.hstack((colours, np.ones((colours.shape[0], 1))))

			self.jointMarkersMappingData = [j_inds, m_inds, colours]
			interface.setAttr('jointMarkersMap', self.jointMarkersMappingData)

			# np.random.seed(100)
			# colours = np.random.rand(len(j_inds), 3)
			# colours = np.hstack((colours, np.ones((colours.shape[0], 1))))
			#
			# for i, (ji, mi) in enumerate(zip(j_inds, m_inds)):
			# 	if not ji or not mi: continue
			# 	mappingAttrs = {
			# 		'colour': (0.1, 0.4, 0.1, 0.5),
			# 		'edgeColour': colours[i],
			# 		'pointSize': 8
			# 	}
			# 	mappingAttrs['x0'] = skelDict['Gs'][ji][:, :3, 3]
			# 	mappingAttrs['x1'] = vs[mi]
			# 	interface.createChild('jointToMarkers_%s' % skelDict['jointNames'][ji[0]], 'edges', attrs=mappingAttrs)

	# skelDict['m_numMarkers'] = len(markerNames)
	# skelDict['m_names'] = markerNames
	# skelDict['m_parents'] = np.array(markerParents, dtype=np.int32)
	# skelDict['m_offsets'] = np.array(markerOffsets, dtype=np.float32)
	# skelDict['m_weights'] = np.array(markerWeights, dtype=np.float32)


class LabelTest(Op.Op):
	def __init__(self, name='/Label Test', locations='', source=''):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('locations', 'Skeleton locations', 'Skeleton locations', 'string', locations, {}),
			('source', 'Source skeleton', 'Source skeleton', 'string', source, {})
		]

		super(self.__class__, self).__init__(name, fields)

	def cook(self, location, interface, attrs):
		if not attrs['source']: return
		skelDict = interface.attr('skelDict')
		if skelDict is None:
			self.logger.error('No skeleton found at: %s' % location)
			return

		sourceSkelDict = interface.attr('skelDict', atLocation=attrs['source'])
		if sourceSkelDict is None:
			self.logger.error('No source skeleton found at: %s' % attrs['source'])
			return

		skelDict['numMarkers'] = sourceSkelDict['numMarkers']
		skelDict['markerOffsets'] = sourceSkelDict['markerOffsets']
		skelDict['markerParents'] = sourceSkelDict['markerParents']
		skelDict['markerWeights'] = sourceSkelDict['markerWeights']
		skelDict['markerNames'] = sourceSkelDict['markerNames']

		interface.setAttr('skelDict', skelDict)


class ReconstructionError(Op.Op):
	def __init__(self, name='/Reconstruction Error', locations='', baseline='', x3d_threshold=30.0):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('baseline', 'baseline', 'baseline', 'string', baseline, {}),
			('x3d_threshold', '3D threshold', '3D threshold', 'float', x3d_threshold, {})
		]

		super(self.__class__, self).__init__(name, fields)

	def cook(self, location, interface, attrs):
		if not attrs['baseline']: return

		x3ds = interface.attr('x3ds')
		if x3ds is None:
			self.logger.error('Could not find 3D data at location: %s' % location)
			return

		base_x3ds = interface.attr('vs', atLocation=attrs['baseline'])
		if base_x3ds is None:
			base_x3ds = interface.attr('x3ds', atLocation=attrs['baseline'])
			if base_x3ds is None:
				self.logger.error('Could not find baseline 3D data at location: %s' % attrs['baseline'])
				return

		cloud = ISCV.HashCloud3D(x3ds, attrs['x3d_threshold'])
		scores, matches, matches_splits = cloud.score(base_x3ds)
		interface.setAttr('scores', scores)
		interface.setAttr('matches', matches)
		interface.setAttr('matches_splits', matches_splits)

		self.logger.info('Num. points: %d | %d' % (len(x3ds), len(base_x3ds)))
		numMatches = len(matches)
		numUniqueMatches = len(np.unique(matches))
		self.logger.info('Found %d unique matches (%d)' % (numUniqueMatches, numMatches))

		# Highlight the matches using red to make it visually stand out
		colours = np.tile(np.array([0.5, 0.5, 0.5, 0.7], dtype=np.float32), (x3ds.shape[0], 1))
		colours[matches] = np.array([1.0, 0.0, 0.0, 0.8], dtype=np.float32)
		interface.setAttr('x3ds_colours', colours)


class CollectTrackingPoints(Op.Op):
	def __init__(self, name='/Collect_Tracking_Points', locations='', visualise=True, groupFrames=True, frameRange=''):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('locations', '3D locations', '3D locations', 'string', locations, {}),
			('visualise', 'Visualise', 'Visualise', 'bool', visualise, {}),
			('groupFrames', 'Group frames', 'Group frames', 'bool', groupFrames, {}),
			('frameRange', 'Frame range', 'Frame range', 'string', frameRange, {})
		]

		super(self.__class__, self).__init__(name, fields)
		self.labelCollection = {}
		self.labelColours = {}

		self.x3ds_frames, self.x3ds_labels, self.x3ds_colours = {}, {}, {}

	def flush(self):
		self.labelCollection = {}
		self.labelColours = {}

	def cook(self, location, interface, attrs):
		if not self.useFrame(interface.frame(), attrs['frameRange']): return

		# Temp boot hack
		isBoot = interface.attr('boot')
		if isBoot is None or not isBoot: return

		x3ds = interface.attr('x3ds')
		x3ds_labels = interface.attr('x3ds_labels')
		x3ds_colours = interface.attr('x3ds_colours')
		if x3ds is None or x3ds_labels is None:
			self.logger.warning('No 3D data found at: %s' % location)
			return

		if 'groupFrames' in attrs and attrs['groupFrames']:
			if interface.frame() not in self.x3ds_frames:
				if len(x3ds_labels) >= 472:
					self.x3ds_frames[interface.frame()] = x3ds
					self.x3ds_labels[interface.frame()] = x3ds_labels
				# x3ds_colours = np.hstack((x3ds_colours, np.ones((x3ds_colours.shape[0], 1), dtype=np.float32)))
				# self.x3ds_colours[interface.frame()] = x3ds_colours

			cAttrs = {
				'x3ds_frames': self.x3ds_frames,
				'x3ds_labels': self.x3ds_labels,
				'x3ds_colours': self.x3ds_colours
			}
			interface.createChild('collection', 'group', attrs=cAttrs)

		else:
			for xi, (x3d, label) in enumerate(zip(x3ds, x3ds_labels)):
				if label not in self.labelCollection:
					self.labelCollection[label] = [x3d]
					if x3ds_colours is not None: self.labelColours[label] = [x3ds_colours[xi]]
				else:
					self.labelCollection[label].append(x3d)
					if x3ds_colours is not None: self.labelColours[label].append(x3ds_colours[xi])

			# Add collections to interface for debugging purposes
			# interface.setAttr('labelCollections', self.labelCollection)
			# interface.setAttr('labelColours', self.labelColours)

			points = np.concatenate(([c for c in self.labelCollection.values()]))
			colours = np.concatenate(([c for c in self.labelColours.values()]))
			cAttrs = {
				'x3ds': points,
				'x3ds_colours': colours
			}
			collectionType = 'points3d' if attrs['visualise'] else 'group'
			interface.createChild('collection', collectionType, attrs=cAttrs)


class CollectTrackingDetections(Op.Op):
	def __init__(self, name='/Collect Tracking Detections', locations=''):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('locations', 'Detections locations', 'Detections locations', 'string', locations, {})
		]

		super(self.__class__, self).__init__(name, fields)
		self.cameraLabelCollection = {}
		self.cameraLabelColours = {}

	def flush(self):
		self.cameraLabelCollection = {}
		self.cameraLabelColours = {}

	def cook(self, location, interface, attrs):
		x2ds = interface.attr('x2ds')
		splits = interface.attr('x2ds_splits')
		labels = interface.attr('labels')
		x2ds_colours = interface.attr('x2ds_colours')
		if x2ds is None or splits is None or labels is None:
			self.logger.warning('No detections found at: %s' % location)

		numCams = len(splits) - 1
		if not self.cameraLabelCollection:
			for ci in range(numCams):
				self.cameraLabelCollection[ci] = []
				self.cameraLabelColours[ci] = []

		for ci, (s, e) in enumerate(zip(splits[:-1], splits[1:])):
			self.cameraLabelCollection[ci].extend(x2ds[s:e])
			if x2ds_colours is not None: self.cameraLabelColours[ci].extend(x2ds_colours[s:e])

		# Add collections to interface for debugging purposes
		# interface.setAttr('cameraLabelCollection', self.cameraLabelCollection)
		# interface.setAttr('cameraLabelColours', self.cameraLabelColours)

		dets, colours = [], []
		for cameraIndex, cameraLabels in self.cameraLabelCollection.iteritems():
			dets.extend(cameraLabels)

		for cameraIndex, cameraColours in self.cameraLabelColours.iteritems():
			colours.extend(cameraColours)

		lengths = [len(camLabels) for camLabels in self.cameraLabelCollection.values()]
		camSplits = Interface.makeSplitBoundaries(lengths)

		cAttrs = {
			'x2ds': np.array(dets, dtype=np.float32),
			'x2ds_splits': camSplits,
			'x2ds_colours': np.array(colours, dtype=np.float32)
		}
		interface.createChild('collection', 'detections', attrs=cAttrs)


class LabellingTest(Op.Op):
	def __init__(self, name='/Labelling Test', locations='', calibration='', skeleton='', intersect_threshold=100., generateNormals=True,
				tiltThreshold=0.0002, x2dThreshold=0.01, x3dThreshold=30.0, minRays=3):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('calibration', 'Calibration location', 'Calibration location', 'string', calibration, {}),
			('skeleton', 'Skeleton', 'Skeleton with visibility LODs', 'string', skeleton, {}),
			('intersect_threshold', 'Intersect threshold', 'Intersect threshold', 'float', intersect_threshold, {}),
			('generateNormals', 'Generate normals', 'Generate normals for visibility checks', 'bool', generateNormals, {}),
			('tilt_threshold', 'Tilt threshold', 'Slack factor for tilt pairing', 'float', tiltThreshold, {}),
			('x2d_threshold', 'Detection threshold', 'Detections threshold', 'float', x2dThreshold, {}),
			('x3d_threshold', '3D threshold', '3D threshold', 'float', x3dThreshold, {}),
			('min_rays', 'Min. number of rays', 'Minimum number of rays', 'int', minRays, {})
		]

		super(self.__class__, self).__init__(name, fields)

	def dets_to_rays(self, x2ds, splits, mats):
		rays = np.zeros((len(x2ds), 3), dtype=np.float32)
		for c0, c1, m in zip(splits[:-1], splits[1:], mats):
			K, RT, T = m[0], m[1], m[4]
			crays = rays[c0:c1]
			np.dot(x2ds[c0:c1], RT[:2, :3], out=crays) # ray directions (unnormalized)
			crays -= np.dot([-K[0, 2], -K[1, 2], K[0, 0]], RT[:3, :3])
		rays /= (np.sum(rays * rays, axis=1) ** 0.5).reshape(-1, 1) # normalized ray directions
		return rays

	def intersect_rays(self, attrs, x2ds, splits, Ps, mats, seed_x3ds=None, tilt_threshold=0.0002, x2d_threshold=0.01, x3d_threshold=30.0, min_rays=3, lod={}):
		Ks = np.array(zip(*mats)[0], dtype=np.float32)
		RTs = np.array(zip(*mats)[1], dtype=np.float32)
		Ts = np.array(zip(*mats)[4], dtype=np.float32)
		# ret2 = ISCV.intersect_rays(x2ds, splits, Ps, Ks, RTs, Ts, seed_x3ds, tilt_threshold, x2d_threshold, x3d_threshold, min_rays)
		# return ret2

		import itertools
		numCameras = len(splits) - 1
		numDets = splits[-1]
		labels = -np.ones(numDets, dtype=np.int32)
		E = ISCV.compute_E(x2ds, splits, Ps)
		rays = self.dets_to_rays(x2ds, splits, mats)
		Ts = np.array([m[4] for m in mats], dtype=np.float32)

		def norm(a):
			return a / (np.sum(a ** 2) ** 0.5)

		tilt_axes = np.array([norm(np.dot([-m[0][0, 2], -m[0][1, 2], m[0][0, 0]], m[1][:3, :3])) for m in mats], dtype=np.float32)
		corder = np.array(list(itertools.combinations(range(numCameras), 2)), dtype=np.int32) # all combinations ci < cj
		clouds = ISCV.HashCloud2DList(x2ds, splits, x2d_threshold)
		x3ds_ret = []
		if seed_x3ds is not None:
			x3ds_ret = list(seed_x3ds)
			# initialise labels from seed_x3ds
			_, labels, _ = clouds.project_assign(seed_x3ds, np.arange(len(x3ds_ret), dtype=np.int32), Ps, x2d_threshold)

		for ci in xrange(numCameras):
			for cj in xrange(ci + 1, numCameras):
				ui, uj = np.where(labels[splits[ci]:splits[ci + 1]] == -1)[0], np.where(labels[splits[cj]:splits[cj + 1]] == -1)[0]
				if len(ui) == 0 or len(uj) == 0: continue
				ui += splits[ci];
				uj += splits[cj]
				axis = Ts[cj] - Ts[ci]
				tilt_i = np.dot(map(norm, np.cross(rays[ui], axis)), tilt_axes[ci])
				tilt_j = np.dot(map(norm, np.cross(rays[uj], axis)), tilt_axes[ci]) # NB tilt_axes[ci] not a bug
				io = np.argsort(tilt_i)
				jo = np.argsort(tilt_j)
				ii, ji = 0, 0
				data = []
				while ii < len(io) and ji < len(jo):
					d0, d1 = tilt_i[io[ii]], tilt_j[jo[ji]]
					diff = d0 - d1
					if abs(diff) < tilt_threshold:
						# test for colliding pairs
						# if ii + 1 < len(io) and tilt_i[io[ii + 1]] - d0 < tilt_threshold: ii += 2; continue
						# if ji + 1 < len(jo) and tilt_j[jo[ji + 1]] - d1 < tilt_threshold: ji += 2; continue
						# test for colliding triples
						# if ii > 0 and d0 - tilt_i[io[ii - 1]] < tilt_threshold: ii += 1; continue
						# if ji > 0 and d1 - tilt_j[jo[ji - 1]] < tilt_threshold: ji += 1; continue
						d = [ui[io[ii]], uj[jo[ji]]]
						data.append(d)
						ii += 1
						ji += 1
					elif diff < 0:
						ii += 1
					else:
						ji += 1
				if len(data) != 0:
					# intersect rays
					for d in data:
						E0, e0 = E[d, :, :3].reshape(-1, 3), E[d, :, 3].reshape(-1)
						x3d = np.linalg.solve(np.dot(E0.T, E0) + np.eye(3) * 1e-7, -np.dot(E0.T, e0))

						# if lod:
						# 	proj_x2ds, proj_splits, proj_labels = ISCV.project_visibility2(np.array([x3d], dtype=np.float32), np.array([0], dtype=np.int32), Ps,lod['tris'], lod['cameraPositions'], lod['normals'],attrs['intersect_threshold'], attrs['generateNormals'])
						# 	sc, labels_out, _ = clouds.assign(proj_x2ds, proj_splits, proj_labels, x2d_threshold)
						# else:
						sc, labels_out, _ = clouds.project_assign(np.array([x3d], dtype=np.float32), np.array([0], dtype=np.int32), Ps, x2d_threshold)

						tmp = np.where(labels_out == 0)[0]
						if len(tmp) >= min_rays:
							tls_empty = np.where(labels[tmp] == -1)[0]
							if len(tls_empty) >= min_rays:
								labels[tmp[tls_empty]] = len(x3ds_ret)
								x3ds_ret.append(x3d)

		# merge
		x3ds_ret = np.array(x3ds_ret, dtype=np.float32).reshape(-1, 3)
		cloud = ISCV.HashCloud3D(x3ds_ret, x3d_threshold)
		scores, matches, matches_splits = cloud.score(x3ds_ret)
		mergers = np.where(matches_splits[1:] - matches_splits[:-1] > 1)[0]
		for li in mergers:
			i0, i1 = matches_splits[li:li + 2]
			collisions = np.where(scores[i0:i1] < x3d_threshold ** 2)[0]
			if len(collisions) > 1:
				collisions += i0

		# now cull the seed_x3ds, because they could confuse matters
		if seed_x3ds is not None:
			labels[np.where(labels < len(seed_x3ds))] = -1

		# final polish
		x3ds_ret, x3ds_labels, E_x2ds_single, x2ds_single_labels = Recon.solve_x3ds(x2ds, splits, labels, Ps, True)
		# throw away the single rays and their 3d points by renumbering the generated 3d points

		if lod:
			if self.visibility is None: self.visibility = ISCV.ProjectVisibility.create()
			self.visibility.setLods(lod['tris'], lod['cameraPositions'], lod['normals'],
									attrs['intersect_threshold'], attrs['generateNormals'])
			proj_x2ds, proj_splits, proj_labels = ISCV.project_visibility(x3ds_ret, x3ds_labels, Ps, self.visibility)
			sc, labels_out, _ = clouds.assign(proj_x2ds, proj_splits, proj_labels, x2d_threshold)
		else:
			_, labels, _ = clouds.project_assign(x3ds_ret, None, Ps, x2d_threshold)

		return x3ds_ret, labels

	def cook(self, location, interface, attrs):

		x2ds = interface.attr('x2ds')
		splits = interface.attr('x2ds_splits')

		calibration = attrs['calibration']
		if not calibration:
			self.logger.warning('No calibration data found at: %s' % calibration)
			return

		Ps = interface.attr('Ps', atLocation=calibration)
		mats = interface.attr('mats', atLocation=calibration)

		# Check if we've got visibility lods
		lod = {}
		if 'skeleton' in attrs and attrs['skeleton']:
			skeletonLoc = attrs['skeleton']
			skelDict = interface.attr('skelDict', atLocation=skeletonLoc)
			visibilityLod = interface.getChild('visibilityLod', parent=skeletonLoc)
			if visibilityLod is None:
				self.logger.error('No visibility LODs found at skeleton: %s' % attrs['skeleton'])
				return

			lodNames = visibilityLod['names']
			lodTris = visibilityLod['tris']
			lodVerts = visibilityLod['verts']
			lodNormals = visibilityLod['faceNormals']

			mats = interface.attr('mats', atLocation=attrs['calibration'])
			lod['cameraPositions'] = np.array([m[4] for m in mats], dtype=np.float32)
			lod['tris'] = lodVerts[lodTris]
			lod['normals'] = np.concatenate((lodNormals))

		x3ds, x3d_labels = self.intersect_rays(attrs, x2ds, splits, Ps, mats, tilt_threshold=attrs['tilt_threshold'], x2d_threshold=attrs['x2d_threshold'],
											x3d_threshold=attrs['x3d_threshold'], min_rays=attrs['min_rays'], lod=lod)

		pAttrs = {
			'x3ds': x3ds,
			'x3ds_labels': x3d_labels,
			'x3ds_colour': (0, 0, 1, 0.8)
		}
		interface.createChild('x3ds', 'points3d', attrs=pAttrs)


class MakeStuff(Op.Op):
	def __init__(self, name='/MakeStuff', locations='/root/stuff'):
		self.fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {})
		]

		super(self.__class__, self).__init__(name, self.fields)

	def cook(self, location, interface, attrs):
		interface.createChild(interface.name(), 'group', atLocation=interface.parentPath(), attrs={'hello': 'world'})


class ChangeStuff(Op.Op):
	def __init__(self, name='/ChangeStuff', locations=''):
		self.fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {})
		]

		super(self.__class__, self).__init__(name, self.fields)

	def cook(self, location, interface, attrs):
		interface.setAttr('gunk', {'numbers': [0, 1, 2, 3]})


class Prune3Ds(Op.Op):
	def __init__(self, name='/Prune3Ds', locations='', x3d_threshold=600.):
		self.fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('x3d_threshold', '3D threshold', '3D threshold', 'float', x3d_threshold, {})
		]

		super(self.__class__, self).__init__(name, self.fields)

	def cook(self, location, interface, attrs):
		x3ds = interface.attr('x3ds')
		x3ds_labels = interface.attr('x3ds_labels')
		if x3ds is None or x3ds_labels is None: return

		cloud = ISCV.HashCloud3D(np.array(x3ds, dtype=np.float32), attrs['x3d_threshold'])
		scores, matches, matches_splits = cloud.score(np.array([[0, 0, 0]], dtype=np.float32))

		interface.setAttr('x3ds', x3ds[matches])
		interface.setAttr('x3ds_labels', x3ds_labels[matches])

		missingDataFlags = interface.attr('missingDataFlags')
		if missingDataFlags is not None:
			interface.setAttr('missingDataFlags', missingDataFlags[matches])


class Filter3Ds(Op.Op):
	def __init__(self, name='/Filter3Ds', locations='', whiteList='', blackList=''):
		self.fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('whiteList', 'Label white list', 'Label white list', 'string', whiteList, {}),
			('blackList', 'Label black list', 'Label black list', 'string', blackList, {})
		]

		super(self.__class__, self).__init__(name, self.fields)
		self.flush()

	def flush(self):
		self.printInfo = True

	def cook(self, location, interface, attrs):
		x3ds = interface.attr('x3ds')
		x3ds_labels = interface.attr('x3ds_labels')
		if x3ds is None or x3ds_labels is None: return

		x3ds_colours = interface.attr('x3ds_colours')

		whiteList, blackList = None, None
		if attrs['whiteList']:
			whiteList = np.unique(eval(attrs['whiteList']))
			if self.printInfo:
				self.logger.info('White list contains %d elements' % len(whiteList))
				self.printInfo = False

		if attrs['blackList']:
			blackList = np.unique(eval(attrs['blackList']))
			if self.printInfo:
				self.logger.info('Black list contains %d elements' % len(blackList))
				self.printInfo = False

		if whiteList is not None:
			whichInds = np.where(x3ds_labels.reshape(-1, 1) == whiteList)[0]
			x3ds, x3ds_labels = x3ds[whichInds], x3ds_labels[whichInds]
			interface.setAttr('x3ds', x3ds)
			interface.setAttr('x3ds_labels', x3ds_labels)

		if x3ds_colours is not None and x3ds_colours.any():
			x3ds_colours = x3ds_colours[whichInds]
			interface.setAttr('x3ds_colours', x3ds_colours)


class CollectPoints(Op.Op):
	def __init__(self, name='/Collect_Points', locations='', x3d_threshold=30.):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('locations', '3D locations', '3D locations', 'string', locations, {}),
			('x3d_threshold', '3D threshold', '3D threshold', 'float', x3d_threshold, {})
		]

		super(self.__class__, self).__init__(name, fields)
		self.x3ds = None

	def cook(self, location, interface, attrs):
		x3ds = interface.attr('x3ds')
		x3ds_labels = interface.attr('x3ds_labels')
		if x3ds is None or x3ds_labels is None:
			self.logger.warning('No 3D data found at: %s' % location)
			return

		cloud = ISCV.HashCloud3D(skelDict['geom_Vs'], attrs['x3d_threshold'])
		scores, matches, matches_splits = cloud.score(x3ds)

		for xi, (x3d, label) in enumerate(zip(x3ds, x3ds_labels)):
			if label not in self.labelCollection:
				self.labelCollection[label] = [x3d]
			else:
				self.labelCollection[label].append(x3d)

		# Add collections to interface for debugging purposes
		# interface.setAttr('labelCollections', self.labelCollection)
		# interface.setAttr('labelColours', self.labelColours)

		points = np.concatenate(([c for c in self.labelCollection.values()]))
		colours = np.concatenate(([c for c in self.labelColours.values()]))
		cAttrs = {
			'x3ds': points,
			'x3ds_colours': colours
		}
		interface.createChild('collection', 'points3d', attrs=cAttrs)

class CreateROI(Op.Op):
	def __init__(self, name='/CreateROI', locations=''):
		self.fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {})
		]

		super(self.__class__, self).__init__(name, self.fields)

	def make_coords(self, height, width):
		''' Generate a uniform grid on the pixels, scaled so that the x-axis runs from -1 to 1. '''
		pix_coord = np.zeros((height, width, 2), dtype=np.float32)
		pix_coord[:, :, 1] = np.arange(height).reshape(-1, 1)
		pix_coord[:, :, 0] = np.arange(width)
		coord = (pix_coord - np.array([0.5 * (width - 1), 0.5 * (height - 1)], dtype=np.float32)) * (2.0 / width)
		return coord, pix_coord

	def getX2Ds(self, points, height, width):
		from collections import OrderedDict
		new_points = {}
		for key in points:
			cam0, cam1 = points[key].keys()
			new_points[key] = OrderedDict()
			new_points[key][0] = [points[key][cam0][1], points[key][cam0][0]]
			new_points[key][1] = [points[key][cam1][1], points[key][cam1][0] - width]

		coord, pix_coord = self.make_coords(height, width)
		coord[:, :, 1] *= -1

		p2ds = np.zeros((2, len(new_points.keys()), 2), dtype=int)
		for ki, key in enumerate(new_points.keys()):
			for cam in new_points[key]:
				p2ds[cam, ki, :] = new_points[key][cam] # copy(new_points[key][cam])
				new_points[key][cam] = coord[new_points[key][cam][0], new_points[key][cam][1]]

		view0, view1, labels = [], [], []
		for key, pair in new_points.items():
			labels.append(key)
			view0.append(pair[0])
			view1.append(pair[1])

		return np.array(view0), np.array(view1), np.array(labels), p2ds, coord, pix_coord

	def boxes_intersect(self, a, b):
		(a_min_x, a_min_y), (a_max_x, a_max_y) = a
		(b_min_x, b_min_y), (b_max_x, b_max_y) = b

		if a_max_x < b_min_x: return False
		if a_min_x > b_max_x: return False
		if a_max_y < b_min_y: return False
		if a_min_y > b_max_y: return False
		return True

	def box_extended(self, a, b):
		(min_x, min_y), (max_x, max_y) = a
		(b_min_x, b_min_y), (b_max_x, b_max_y) = b

		if min_x > b_min_x: min_x = b_min_x
		if min_y > b_min_y: min_y = b_min_y

		if max_x < b_max_x: max_x = b_max_x
		if max_y < b_max_y: max_y = b_max_y

		return (min_x, min_y), (max_x, max_y)


	def cook(self, location, interface, attrs):
		imgs = interface.attr('imgs')

		for image_index, img in enumerate(imgs):
			grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			_, thresh = cv2.threshold(grey, 1, 255, cv2.THRESH_BINARY)

			im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			boxes = []
			index = 0
			for cnt in contours:
				x, y, w, h = cv2.boundingRect(cnt)

				# Ignore small boxes.
				if w*h < 5000: continue
				# TODO: Only store the largest area

				min_corners = [(x, y), (x + w, y + h)]
				for b_i, box in enumerate(boxes):
					result = self.boxes_intersect(min_corners, box)
					if result:
						boxes[b_i] = self.box_extended(min_corners, box)
						break
				else:
					boxes.append(min_corners)
					index += 1

			cropped_images = []
			cropped_bboxes = []
			for b_i, box in enumerate(boxes):
				(x, y), (max_x, max_y) = box
				crop = img[y:max_y, x:max_x, :]

				cropped_bboxes.append((x, y, max_x-x, max_y-y))
				cropped_images.append(np.ascontiguousarray(crop))


			bbAttrs = {'imgs': cropped_images, 'bboxes': cropped_bboxes, 'image_index': image_index}
			interface.createChild('img_%d' % image_index, 'group', attrs=bbAttrs)

class CleanROI(Op.Op):
	def __init__(self, name='/CleanROI', locations=''):
		self.fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {})
		]

		super(self.__class__, self).__init__(name, self.fields)

	def cook(self, location, interface, attrs):
		interface.deleteChild(location)

class ResizeFrame(Op.Op):
	def __init__(self, name='/ResizeFrame', locations='', camerasRootLocation=''):
		self.fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('camerasRootLocation', 'camerasRootLocation', 'camerasRootLocation', 'string', camerasRootLocation, {})
		]

		super(self.__class__, self).__init__(name, self.fields)

	def cook(self, location, interface, attrs):
		bboxes = interface.attr('bboxes')
		imgs = interface.attr('imgs')
		camera_imgs = interface.attr('imgs', atLocation=attrs['camerasRootLocation'])
		image_index = interface.attr('image_index')


		for camera_index, cam_img in enumerate(camera_imgs):
			# TO DO: REDO based on the current index instead of looping.
			if camera_index != image_index: continue
			# TO DO: Reshape based on the original camera images
			a = np.zeros(shape=camera_imgs[0].shape, dtype=np.uint8)
			for bbox, img in zip(bboxes, imgs):
				x, y, w, h = bbox
				a[y:y + img.shape[0], x:x + img.shape[1], :] = img
			cam_img[:] = a


class VisualiseNormals(Op.Op):
	def __init__(self, name='/Visualise_Normals', locations='', lengthFactor=50.):
		self.fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('lengthFactor', 'Length factor', 'Length factor', 'float', lengthFactor, {})
		]

		super(self.__class__, self).__init__(name, self.fields)

	def cook(self, location, interface, attrs):
		# TODO: Remove special case where we accept either a list of vs's or x3ds (ugly and causes normals which should be handled in the same way)
		locAttrs = interface.location(location)
		if locAttrs and locAttrs['type'] == 'skeleton': # Note: Shouldn't this be a utility function or something?
			skelDict = interface.attr('skelDict')
			if skelDict is None:
				self.logger.error('Skeleton dictionary not found at: %s' % location)#
				return

			effectorLabels = np.array([int(mn) for mn in skelDict['markerNames']], dtype=np.int32)
			effectorData = SolveIK.make_effectorData(skelDict)
			x3ds, x3ds_labels = SolveIK.skeleton_marker_positions(skelDict, skelDict['rootMat'], skelDict['chanValues'],
																effectorLabels, effectorData, skelDict['markerWeights'])

			normals = skelDict['markerNormals']
			# for ni, (parent, normal) in enumerate(zip(skelDict['markerParents'], normals)):
			# 	Gs = skelDict['Gs'][parent]
			# 	normals[ni] = np.dot(Gs[:3, :3], normal)
		else:
			x3ds = interface.attr('vs')
			if x3ds is None:
				x3ds = interface.attr('x3ds')
				if x3ds is None:
					self.logger.error('No x3ds or vs found at: %s' % location)
					return
			else:
				x3ds = x3ds[0]

			normals = interface.attr('normals')
			if normals is None:
				self.logger.error('No normals found at: %s' % location)
				return

		if len(x3ds) != len(normals):
			self.logger.error('The number of x3ds and normals must match')
			return

		origs = x3ds
		normalsTo = origs + (normals * attrs['lengthFactor'])
		lineAttrs = {
			'x0': origs,
			'x1': normalsTo,
			'colour': (1, 1, 0, 0.5),
			'lineColour': (1, 1, 0, 1),
			'pointSize': 1
		}
		interface.createChild('normals', 'edges', attrs=lineAttrs)


class TransformX3d(Op.Op):
	def __init__(self, name='/TransformX3d', locations='', label='', x=0., y=0., z=0.):
		self.fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', '3D locations', '3D locations', 'string', locations, {}),
			('label', 'label', 'label', 'string', label, {}),
			('x', 'x', 'x', 'float', x, {}),
			('y', 'y', 'y', 'float', y, {}),
			('z', 'z', 'z', 'float', z, {})
		]

		super(self.__class__, self).__init__(name, self.fields)

	def cook(self, location, interface, attrs):
		x3ds = interface.attr('x3ds')
		x3ds_labels = interface.attr('x3ds_labels')
		if x3ds is None or x3ds_labels is None:
			self.logger.error('No 3D data found at: %s' % location)

		if not attrs['label']: return
		label = int(attrs['label'])
		findLabelIdx = np.where(x3ds_labels == label)[0]
		if not findLabelIdx.any():
			self.logger.warning('Could not find label: %d' % label)
			return

		idx = findLabelIdx[0]
		x3ds[idx] = np.float32([attrs['x'], attrs['y'], attrs['z']])
		interface.setAttr('x3ds', x3ds)


class CompareX3ds(Op.Op):
	def __init__(self, name='/CompareX3d', locations='', target='', x3d_threshold=10.):
		self.fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', '3D locations', '3D locations', 'string', locations, {}),
			('target', 'Target', 'Target', 'string', target, {}),
			('x3d_threshold', '3D threshold', '3D threshold', 'float', x3d_threshold, {})
		]

		super(self.__class__, self).__init__(name, self.fields)
		self.flush()

	def flush(self):
		self.markerMap = {}
		self.targetLabels = {}
		self.targetLabelIdxs = {}

	def cook(self, location, interface, attrs):
		if not location or not attrs['target']: return

		x3ds_source = interface.attr('x3ds')
		labels_source = interface.attr('x3ds_labels')
		x3ds_target = interface.attr('x3ds', atLocation=attrs['target'])
		labels_target = interface.attr('x3ds_labels', atLocation=attrs['target'])
		missingDataFlags = interface.attr('missingDataFlags', atLocation=attrs['target'])

		if x3ds_source is None or labels_source is None:
			self.logger.error('Could not find 3D data at: %s' % location)
			return

		if x3ds_target is None or labels_target is None:
			self.logger.error('Could not find 3D data at: %s' % attrs['target'])
			return

		from GCore import Label
		if interface.frame() == 0:
			labelInds = np.arange(len(labels_target))
			self.targetLabelIdxs = {lblInd: lblName for lblName, lblInd in zip(labels_target, labelInds)}
			self.targetLabels = {lblName: lblInd for lblName, lblInd in zip(labels_target, labelInds)}

		targetInds = [li for li, lbl in enumerate(labels_target) if lbl in self.targetLabels]
		targetLbls = [self.targetLabels[lbl] for lbl in labels_target if lbl in self.targetLabels]
		sc, labels, vels = Label.label_3d_from_3d(np.ascontiguousarray(x3ds_target[targetInds]), np.ascontiguousarray(targetLbls), None, np.ascontiguousarray(x3ds_source), attrs['x3d_threshold'])

		if interface.frame() == 0:
			self.markerMap = {sourceMarker: targetMarker for sourceMarker, targetMarker in zip(labels_source, labels)}
		else:
			# Compare with our marker map
			for li, lbl in enumerate(labels):
				if lbl == -1:
					pass
					# self.logger.warning('No matching marker found for source label %d' % li)
				else:
					if lbl != self.markerMap[li]:
						# if self.markerMap[li] == -1:
						# 	self.logger.warning('Nothing was assigned to source label %d' % li)
						# else:
						if self.markerMap[li] != -1:
							targetMarkerBefore = self.targetLabelIdxs[self.markerMap[li]]
							targetMarkerCurrent = self.targetLabelIdxs[lbl]
							self.logger.warning('Marker %d maps to %s (was %s) on frame %d' % (li, targetMarkerCurrent, targetMarkerBefore, interface.frame()))


# Register Ops
import Registry

Registry.registerOp('Print State', PrintState)
Registry.registerOp('Print Interface', PrintInterface)
Registry.registerOp('Calculate Pose Effector Positions', CalculatePoseEffectorPositions)
Registry.registerOp('Label 3Ds from 3Ds', Label3DsFrom3Ds)
Registry.registerOp('Label x3ds from x3ds', LabelX3DsFromX3Ds)
Registry.registerOp('Solve 3Ds from Labels', SolveX3ds)
Registry.registerOp('Solve Skeleton From 3D', SolveSkeletonFrom3D)
Registry.registerOp('Solve Skeleton From 2D', SolveSkeletonFrom2D)
Registry.registerOp('Marker Data Update', UpdateSkeletonMarkers)
Registry.registerOp('Marker Data Create', CreateMarkersFromLabels)
Registry.registerOp('Marker Data Remove', RemoveMarkerData)
Registry.registerOp('Labelling Test', LabellingTest)
Registry.registerOp('Prune 3Ds', Prune3Ds)
Registry.registerOp('Filter 3Ds', Filter3Ds)
Registry.registerOp('Visualise Normals', VisualiseNormals)

Registry.registerOp('Make Stuff', MakeStuff)
Registry.registerOp('Change Stuff', ChangeStuff)

Registry.registerOp('Transform X3D', TransformX3d)
