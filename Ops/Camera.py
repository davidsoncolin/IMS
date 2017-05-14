import numpy as np
import Op
import ISCV
from GCore import Calibrate

try:
	import ncam
	class NCamCamera(Op.Op): # Feel free to change the name when we figure out style guidelines
		def __init__(self, name='/NCAM_Camera', locations='/root', ipAddress='127.0.0.1', port=38860):
			fields = [
				('name', 'name', 'name', 'string', name, {}),
				('locations', 'locations', 'locations', 'string', locations, {}),
				('ipAddress', 'ipAddress', 'IP Address', 'string', ipAddress, {}),
				('port', 'port', 'Port', 'int', port, {})
			]

			self.client = ncam.Client()

			super(self.__class__, self).__init__(name, fields)

		def cook(self, location, interface, attrs):
			if not self.client.IsConnected():
				print "Not connected"
				if not self.client.Connect(attrs['ipAddress'], int(attrs['port'])):
					print "Connection Failed"
					return False
				print "New Connection!"

			currentFrame = self.client.GetCurrentFrame()

			RT = currentFrame.CameraTracking.Transform

			fovX, fovY = currentFrame.OpticalParameters.FOV

			if np.abs(fovY) < 1e-10:
				return False

			cx, cy = currentFrame.OpticalParameters.ProjectionCenter # TODO: Investigate format
			ox, oy = 2*(cx-0.5), 2*(cy-0.5)


			K = Calibrate.composeK(fovX, ox=ox, oy=oy, square=(fovY / fovX), skew=0)[:3,:3]
			width_height = currentFrame.OpticalParameters.Resolution
			P = np.dot(K,RT)
			mat = Calibrate.makeMat(P, (0.0, 0.0), width_height) # TODO Distortion Param

			self.setupAttrs = {
				'Ps': [P],
				'camera_ids': ["1"],
				'camera_names': ["NCAM_Camera"],
				'mats': [mat],
				'updateMats': True
			}
			interface.createChild('cameras', 'cameras', atLocation=location, attrs=self.setupAttrs)

			return True
except:
	NCamCamera = Op.Op

class Project(Op.Op):
	def __init__(self, name='/Camera Project', locations='/root/projected', x3ds='', calibration='', skeleton='', useNormals=False,
	             useVisibility=False, distort=False, intersect_threshold=100., generateNormals=False, pointSize=6., colour=(1, 1, 0, .6),
	             cameraOffset=0, frameRange='', x3dIndex=-1):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('locations', 'Locations', 'locations', 'string', locations, {}),
			('x3ds', '3D points', '3D points', 'string', x3ds, {}),
			('calibration', 'Calibration', 'Calibration location', 'string', calibration, {}),
			('skeleton', 'Skeleton', 'Skeleton with visibility LODs', 'string', skeleton, {}),
			('useVisibility', 'Visibility check', 'Do a visibility check if possible', 'bool', useVisibility, {}),
			('useNormals', 'Use normals', 'Use normals if available', 'bool', useNormals, {}),
			('distort', 'Distort', 'Distort', 'bool', distort, {}),
			('intersect_threshold', 'Intersect threshold', 'Intersect threshold', 'float', intersect_threshold, {}),
			('generateNormals', 'Generate normals', 'Generate normals for visibility checks', 'bool', generateNormals, {}),
			('pointSize', '2D Point size', '3D Point size', 'float', pointSize, {}),
			('colour', '2D Point colour', '3D Point colour', 'string', str(colour), {}),
			('cameraOffset', 'Camera offset', 'Camera offset', 'int', cameraOffset, {}),
			('frameRange', 'Frame range', 'Frame range', 'string', frameRange, {}),
			('x3dIndex', '3D index', '3D index (-1=off)', 'int', x3dIndex, {}),
		]

		super(self.__class__, self).__init__(name, fields)
		self.tracker = None
		self.visibility = None

	def flush(self):
		self.tracker = None

	def cook(self, location, interface, attrs):
		if not self.useFrame(interface.frame(), attrs['frameRange']): return
		if not attrs['calibration'] or not attrs['x3ds']: return

		calibrationLocation = attrs['calibration']
		Ps = interface.attr('Ps', atLocation=calibrationLocation)
		mats = interface.attr('mats', atLocation=calibrationLocation)
		if Ps is None:
			if mats is None:
				self.logger.warning('Could not find calibration data at: %s' % calibrationLocation)
				return

			Ps = interface.getPsFromMats(mats)
			if Ps is None: return

		# Get the x3ds (and labels if available) from the cooked location
		x3ds = interface.attr('x3ds', atLocation=attrs['x3ds'])
		if x3ds is None:
			self.logger.error('No 3D points found at: %s' % attrs['x3ds'])
			return

		which_labels = interface.attr('which_labels')
		if which_labels is None:
			which_labels = np.arange(len(x3ds))

		x3ds = np.ascontiguousarray(x3ds, dtype=np.float32)
		normals = interface.attr('normals', atLocation=attrs['x3ds'])

		if 'x3dIndex' in attrs and attrs['x3dIndex'] >= 0:
			idx = attrs['x3dIndex']
			x3ds = x3ds[idx].reshape(1, -1)
			which_labels = [idx]

		# Check if we've got visibility lods
		visibilityLod = None
		if 'skeleton' in attrs and attrs['skeleton']:
			skeletonLoc = attrs['skeleton']
			skelDict = interface.attr('skelDict', atLocation=skeletonLoc)
			visibilityLod = interface.getChild('visibilityLod', parent=skeletonLoc)
			if attrs['useVisibility'] and visibilityLod is None:
				self.logger.error('No visibility LODs found at skeleton: %s' % attrs['skeleton'])
				return

			mats = interface.attr('mats', atLocation=calibrationLocation)
			cameraPositions = np.array([m[4] for m in mats], dtype=np.float32)

		if self.visibility is None: self.visibility = ISCV.ProjectVisibility.create()

		# if attrs['useVisibility'] and normals is not None and visibilityLod is not None:
		if attrs['useVisibility'] and visibilityLod is not None:
			lodNames = visibilityLod['names']
			lodTris = visibilityLod['tris']
			lodVerts = visibilityLod['verts']
			lodNormals = visibilityLod['faceNormals']
			tris = lodVerts[lodTris]

			if attrs['useNormals'] and normals is not None:
				self.visibility.setNormalsAndLods(normals, tris, cameraPositions, np.concatenate((lodNormals)), attrs['intersect_threshold'], attrs['generateNormals'])
			else:
				self.visibility.setLods(tris, cameraPositions, np.concatenate((lodNormals)), attrs['intersect_threshold'], attrs['generateNormals'])

			x2ds, x2ds_splits, x2d_labels = ISCV.project_visibility(x3ds, which_labels, Ps, self.visibility)
		elif attrs['useNormals'] and normals is not None:
			self.visibility.setNormals(normals)
			x2ds, x2ds_splits, x2d_labels = ISCV.project_visibility(x3ds, which_labels, Ps, self.visibility)
		else:
			x2ds, x2ds_splits, x2d_labels = ISCV.project(x3ds, which_labels, Ps)

		# Distort if needed
		if 'distort' in attrs and attrs['distort']:
			for ci, (s, e) in enumerate(zip(x2ds_splits[:-1], x2ds_splits[1:])):
				K, RT, P, ks, T, wh = mats[ci]
				dets = x2ds[s:e]
				ISCV.distort_points(dets, float(-K[0, 2]), float(-K[1, 2]), float(ks[0]), float(ks[1]), dets)
				x2ds[s:e] = dets

		detsAttrs = {
			'x2ds': x2ds,
			'x2ds_splits': x2ds_splits,
			'labels': x2d_labels,
			'x2ds_colour': eval(attrs['colour']),
			'x2ds_pointSize': attrs['pointSize']
		}

		if 'cameraOffset' in attrs and attrs['cameraOffset'] > 0:
			x2ds_splits_render = np.insert(x2ds_splits, np.zeros(attrs['cameraOffset'], dtype=np.int32), 0)
			detsAttrs['x2ds_splits_render'] = x2ds_splits_render

		interface.createChild(interface.name(), 'detections', atLocation=interface.parentPath(), attrs=detsAttrs)


class VisualiseErrors(Op.Op):
	def __init__(self, name='/Visualise_Camera_Errors', locations='', thresholdLow=0.01, thresholdMedium=0.05, forceDefault=False):
		self.fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'Camera locations', 'Camera locations', 'string', locations, {}),
			('thresholdLow', 'Threshold low', 'Threshold low', 'float', thresholdLow, {}),
			('thresholdMedium', 'Threshold medium', 'Threshold medium', 'float', thresholdMedium, {}),
			('forceDefault', 'Force default colour', 'Force default colour', 'bool', forceDefault, {})
		]

		super(self.__class__, self).__init__(name, self.fields)
		self.defaultColour = [0.4, 0.4, 0.4, 1]

	def cook(self, location, interface, attrs):
		cameraErrors = interface.attr('cameraErrors')
		if cameraErrors is None:
			interface.setAttr('colours', None)
			if attrs['forceDefault']:
				cameraIds = interface.attr('camera_ids')
				if cameraIds is not None:
					interface.setAttr('colours', np.tile(self.defaultColour, (len(cameraIds), 1)))

			return

		colours = []
		for error in cameraErrors:
			if error == -1: colours.append(self.defaultColour)
			elif error < attrs['thresholdLow']: colours.append([0, 1, 0, 1])
			elif error < attrs['thresholdMedium']: colours.append([1, 1, 0, 1])
			else: colours.append([1, 0, 0, 1])

		colours = np.array(colours, dtype=np.float32)
		interface.setAttr('colours', colours)


# Register Ops
import Registry
Registry.registerOp('Camera Project', Project)
Registry.registerOp('Visualise Camera Errors', VisualiseErrors)
Registry.registerOp('NCAM Camera', NCamCamera)