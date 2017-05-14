import numpy as np
import Op, Interface
from GCore import Label as GLabel
from IO import C3D
import ISCV


class GraphC3d(Op.Op):
	def __init__(self, name='/Label_Graph', locations='', c3dFilename='', threshold=35):
		fields = [
			('name', 'name', 'Name', 'string', name, {}),
			('locations', 'Locations', 'Skeleton locations', 'string', locations, {}),
			('c3d', 'C3D', 'C3D filename', 'filename', c3dFilename, {}),
			('threshold', 'Threshold', 'Threshold', 'float', threshold, {})
		]

		super(self.__class__, self).__init__(name, fields)
		self.graph = None
		self.c3d_frames = None
		self.c3d_labels = None

	def flush(self):
		self.graph, self.c3d_frames, self.c3d_labels = None, None, None

	def cook(self, location, interface, attrs):
		if not location or not attrs['c3d']: return
		if not attrs and not interface.isDirty(): return

		# Note: We need a flush features
		if self.c3d_frames is None:
			skelDict = interface.attr('skelDict')
			if not skelDict: return

			c3dFilename = attrs['c3d']
			try:
				c3d_dict = C3D.read(c3dFilename)
				self.c3d_frames, c3d_fps, self.c3d_labels = c3d_dict['frames'], c3d_dict['fps'], c3d_dict['labels']
			except IOError as e:
				if c3dFilename: self.logger.error('Could not load C3D: %s' % str(e))
				return

			threshold = attrs['threshold']
			skelDict['labelNames'] = list(np.unique(skelDict['markerNames']))
			which_labels = [self.c3d_labels.index(l) for l in skelDict['labelNames'] if l in self.c3d_labels]
			self.graph = GLabel.graph_from_c3ds(skelDict, skelDict['labelNames'], self.c3d_frames[:, which_labels, :], threshold=threshold)

		# Temp: Write the graph to the interface (prefer POD)
		interface.setAttr('label_graph', self.graph)


class ProjectAssign(Op.Op):
	def __init__(self, name='/Project Assign', locations='', detections='', calibration='', x2d_threshold=0.03, newLocation=False,
	             useNormals=False, useVisibility=False, pointSize=6., colour=(0, 0, 1, .6), showProjected=False, skeleton='',
	             intersect_threshold=100., generateNormals=False):
		fields = [
			('name', 'name', 'Name', 'string', name, {}),
			('locations', '3D Locations', '3D locations', 'string', locations, {}),
			('x2ds', 'Detection location', 'Detection location', 'string', detections, {}),
			('calibration', 'Calibration location', 'Calibration location', 'string', calibration, {}),
			('x2d_threshold', '2D threshold', '2D threshold', 'float', x2d_threshold, {}),
			('newLocation', 'Create new location', 'Create new location', 'bool', newLocation, {}),
			('useNormals', 'Use normals', 'Use normals if available', 'bool', useNormals, {}),
			('useVisibility', 'Visibility check', 'Do a visibility check if possible', 'bool', useVisibility, {}),
			('intersect_threshold', 'Intersect threshold', 'Intersect threshold', 'float', intersect_threshold, {}),
			('generateNormals', 'Generate normals', 'Generate normals for visibility checks', 'bool', generateNormals, {}),
			('pointSize', '2D Point size', '3D Point size (if new location)', 'float', pointSize, {}),
			('colour', '2D Point colour', '3D Point colour', 'string', str(colour), {}),
			('showProjected', 'Show projected', 'Show projected if available', 'bool', showProjected, {}),
			('skeleton', 'Skeleton', 'Skeleton with visibility LODs', 'string', skeleton, {})
		]

		super(self.__class__, self).__init__(name, fields)
		self.visibility = None

	def cook(self, location, interface, attrs):
		# Get x3ds and 3D labels from the cooked location
		x3ds = interface.attr('x3ds')
		if x3ds is None:
			self.logger.error('Could not find attribute: x3ds')
			return

		x3ds_labels = interface.attr('x3ds_labels')
		if x3ds_labels is None:
			self.logger.error('Could not find attribute: x3ds_labels')
			return

		normals = interface.attr('normals')

		x2d_threshold = attrs['x2d_threshold']

		# Set the detections and calibration locations as the cook location if not defined
		x2ds_location = attrs['x2ds']
		if not x2ds_location: x2ds_location = location

		calibrationLocation = attrs['calibration']
		if not calibrationLocation: calibrationLocation = interface.root()

		# Fetch 2D and calibration data
		x2ds = interface.attr('x2ds', atLocation=x2ds_location)
		x2ds_splits = interface.attr('x2ds_splits', atLocation=x2ds_location)
		Ps = interface.attr('Ps', atLocation=interface.root() + '/cameras')

		if x2ds is None or x2ds_splits is None:
			self.logger.error('2D detection data at %s is not valid' % x2ds_location)
			return

		if Ps is None:
			mats = interface.attr('mats', atLocation=interface.root())
			if mats:
				Ps = np.array([m[2] / (np.sum(m[2][0, :3] ** 2) ** 0.5) for m in mats], dtype=np.float32)
			else:
				self.logger.error('Attribute mats not found at %s' % calibrationLocation)
				self.logger.error('Attribute Ps not found at %s' % calibrationLocation)
				return

		# Check if we've got visibility lods
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
			tris = lodVerts[lodTris]

			mats = interface.attr('mats', atLocation=attrs['calibration'])
			cameraPositions = np.array([m[4] for m in mats], dtype=np.float32)

		clouds = ISCV.HashCloud2DList(x2ds, x2ds_splits, x2d_threshold)
		if self.visibility is None: self.visibility = ISCV.ProjectVisibility.create()
		proj_x2ds = None

		if attrs['useVisibility'] and normals is not None:
			self.visibility.setNormalsAndLods(normals, tris, cameraPositions, np.concatenate((lodNormals)), attrs['intersect_threshold'], attrs['generateNormals'])
			# proj_x2ds, proj_splits, proj_labels = ISCV.project_visibility(x3ds, x3ds_labels, Ps, self.visibility)
			# score, x2d_labels, residuals = clouds.assign(proj_x2ds, proj_splits, proj_labels, x2d_threshold)
			score, x2d_labels, residuals = clouds.project_assign_visibility(x3ds, x3ds_labels, Ps, x2d_threshold, self.visibility)
		elif attrs['useNormals'] and normals is not None:
			self.visibility.setNormals(normals)
			proj_x2ds, proj_splits, proj_labels = ISCV.project_visibility(x3ds, x3ds_labels, Ps, self.visibility)
			score, x2d_labels, residuals = clouds.assign(proj_x2ds, proj_splits, proj_labels, x2d_threshold)
		else:
			proj_x2ds, proj_splits, proj_labels = ISCV.project(x3ds, x3ds_labels, Ps)
			score, x2d_labels, vels = clouds.assign(proj_x2ds, proj_splits, proj_labels, x2d_threshold)

		if proj_x2ds is not None:
			projectedLocsAttrs = {
				'x2ds': proj_x2ds, 'x2ds_splits': proj_splits, 'labels': proj_labels,
				'x2ds_colour': (1.0, 0.0, 0.0, 0.7), 'x2ds_pointSize': 10,
				'score': score
			}
			if 'showProjected' in attrs and attrs['showProjected']:
				interface.createChild('projected', 'points2d', attrs=projectedLocsAttrs)
			else:
				interface.createChild('projected', 'group', attrs=projectedLocsAttrs)

		if attrs['newLocation']:
			locAttrs = {
				'x2ds': x2ds, 'x2ds_splits': x2ds_splits, 'labels': x2d_labels,
				'x2ds_colour': eval(attrs['colour']), 'x2ds_pointSize': attrs['pointSize'],
				'score': score
			}

			labelColours = interface.getLabelColours(x2d_labels, eval(attrs['colour']))
			if labelColours.any():
				locAttrs['x2ds_colours'] = labelColours

			interface.createChild('assigned', 'points2d', attrs=locAttrs)
		else:
			interface.setAttr('labels', x2d_labels, atLocation=x2ds_location)
			interface.setAttr('score', score)

			labelColours = interface.getLabelColours(x2d_labels, interface.attr('x2ds_colour', atLocation=x2ds_location))
			if labelColours.any():
				interface.setAttr('x2ds_colours', labelColours, atLocation=x2ds_location)


class Wand(Op.Op):
	def __init__(self, name='/Label_Wand_3D', locations='', calibration='', pointSize=8, frameRange=''):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('locations', '2D Detection locations', '2D Detections locations', 'string', locations, {}),
			('calibration', 'Calibration', 'Calibration', 'string', calibration, {}),
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

		from GCore import Label, Calibrate

		x2ds = interface.attr('x2ds')
		x2ds_splits = interface.attr('x2ds_splits')
		if x2ds is None or x2ds_splits is None:
			self.logger.error('No detections found at: %s' % location)
			return

		mats = interface.attr('mats', atLocation=attrs['calibration'])
		if mats is None:
			self.logger.error('No calibration found at: %s' % attrs['calibration'])
			return

		x3ds, x3ds_labels, x2ds_labels = Calibrate.detect_wand(x2ds, x2ds_splits, mats)
		if x3ds is None or x2ds_labels is None: return

		wandAttrs = {
			'x3ds': x3ds,
			'x3ds_labels': x3ds_labels,
			'x3ds_pointSize': attrs['pointSize'],
			'x3ds_colours': self.colours
		}
		interface.createChild('wand3d', 'points3d', attrs=wandAttrs)


class Override(Op.Op):
	def __init__(self, name='/Label_Override', locations='', labels='', newLabel=-1):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('locations', '2D Detection locations', '2D Detections locations', 'string', locations, {}),
			('labels', 'Labels', 'Labels', 'string', labels, {}),
			('newLabel', 'New label', 'New label', 'int', newLabel, {})
		]

		super(self.__class__, self).__init__(name, fields)

	def cook(self, location, interface, attrs):
		labels = interface.attr('labels')
		if labels is None:
			self.logger.warning('No labels found at: %s' % location)
			return

		if not attrs['labels']: return

		labelInds = np.int32(eval(attrs['labels']))
		labels[labelInds] = attrs['newLabel']
		interface.setAttr('labels', labels)


class AssignUnlabelled(Op.Op):
	def __init__(self, name='/Label_Assign_Unlabelled', locations='', startLabel=0):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('locations', '2D Detection locations', '2D Detections locations', 'string', locations, {}),
			('startLabel', 'Start label', 'Start label', 'int', startLabel, {})
		]

		super(self.__class__, self).__init__(name, fields)

	def cook(self, location, interface, attrs):
		labels = interface.attr('labels')
		if labels is None:
			self.logger.warning('No labels found at: %s' % location)
			return

		which = np.where(labels == -1)[0]
		if which.any():
			start = attrs['startLabel']
			labels[which] = np.arange(start, start + len(which))

		interface.setAttr('labels', labels)


# Register Ops
import Registry
Registry.registerOp('Label Graph (C3D)', GraphC3d)
Registry.registerOp('Label with Project-Assign', ProjectAssign)
Registry.registerOp('Label Wand', Wand)