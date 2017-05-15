import numpy as np

import Op
from IO import ViconReader, C3D, OptitrackReader


class C3d(Op.Op):
	def __init__(self, name='/Read_C3D', locations='', c3dFilename='', markerSrc='', subjectName='', offset=0, step=1, debug=False,
				 timecodeLocation='', pointSize=16., colour=(0., 0., 1., 0.7), includeMissing=False):
		fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('c3d', 'C3D', 'C3D filename', 'filename', c3dFilename, {}),
			('markerSrc', 'Marker source', 'Marker source', 'string', markerSrc, {}),
			('subject', 'Subject name', 'Subject name', 'string', subjectName, {}),
			('offset', 'Frame offset', 'Frame offset', 'int', offset, {}),
			('step', 'Frame step', 'Frame step', 'int', step, {}),
			('debug', 'Debug', 'Debug', 'bool', debug, {}),
			('pointSize', 'Point size', '3D Point size', 'float', pointSize, {}),
			('colour', 'Point colour', '3D Point colour', 'string', str(colour), {}),
			('timecodeLocation', 'Timecode location', 'Timecode location (start)', 'string', timecodeLocation, {}),
			('includeMissing', 'Include missing', 'Include missing points', 'bool', includeMissing, {})
		]

		super(self.__class__, self).__init__(name, fields)
		self.offset = 0
		self.firstFrame, self.lastFrame = 0, 0
		self.c3d_frames = None
		self.c3d_labels = None
		self.which_labels = None
		self.timecode = None

	def flush(self):
		self.c3d_frames, self.c3d_labels = None, None

	def setup(self, interface, attrs):
		# Note: Fix dirty states to avoid this condition as we may want to change the source (currently runs twice)
		if self.c3d_frames is not None: return

		if 'c3d' in attrs:# and self.isDirty('c3d'):
			c3dFilename = self.resolvePath(attrs['c3d'])

			# Read c3d file
			try:
				c3d_dict = C3D.read(c3dFilename)
				self.c3d_frames, c3d_fps, self.c3d_labels = c3d_dict['frames'], c3d_dict['fps'], c3d_dict['labels']
				self.timecode = ':'.join(c3d_dict['tc'].astype(str))
				self.logger.info('Timecode found: %s' % self.timecode)
			except IOError as e:
				if c3dFilename: self.logger.error('Could not load C3D: %s' % str(e))
				return

			self.lastFrame = len(self.c3d_frames)
			self.logger.info('# Frames: %d' % self.lastFrame)
			interface.setAttr('numFrames', self.lastFrame)

			tcSyncTime, self.offset = interface.getTimecodeSync(self.timecode, 'timecodeLocation', attrs, 100, 25., 2., attrs['offset'])
			if tcSyncTime: self.logger.info('Setting timecode to: %s (offset = %d)' % (tcSyncTime, self.offset))

		if self.c3d_labels and (self.isDirty('markers') or self.isDirty('subject')):
			# At the moment we support a skelDict reference as a marker source
			markerSrc = attrs['markerSrc']
			subjectName = attrs['subject']
			skelDict = interface.attr('skelDict', atLocation=markerSrc)
			if skelDict:
				if subjectName:
					skelDict['markerNames'] = [subjectName + ':' + n for n in skelDict['markerNames']]

				skelDict['labelNames'] = list(np.unique(skelDict['markerNames']))
				self.which_labels = [self.c3d_labels.index(l) for l in skelDict['labelNames'] if l in self.c3d_labels]
				if not self.which_labels:
					self.logger.warn('No labels could be matched in C3D labels')

	def cook(self, location, interface, attrs):
		if self.c3d_frames is None: return

		offset = int(self.offset) if self.offset != 0 else attrs['offset']
		stepSize = attrs['step'] if 'step' in attrs else 1
		frameNum = max((interface.frame() + offset) * stepSize, 0)

		if self.which_labels:
			x3ds = self.c3d_frames[frameNum, self.which_labels, :]
			x3ds_labels = np.array(self.c3d_labels)[self.which_labels]
			missingDataFlags = self.c3d_frames[frameNum, self.which_labels, 3]
		else:
			x3ds = self.c3d_frames[frameNum, :, :]
			x3ds_labels = np.array(self.c3d_labels)
			missingDataFlags = self.c3d_frames[frameNum, :, 3]

		if not attrs['includeMissing']:
			trueLabels = np.int32(np.where(x3ds[:, 3] == 0)[0])
			x3ds = x3ds[trueLabels, :3]
			x3ds_labels = x3ds_labels[trueLabels]
		else:
			x3ds = x3ds[:, :3]

		if 'debug' in attrs and attrs['debug']:
			self.logger.info('Using x3d labels (%d): %s' % (len(x3ds_labels), str(x3ds_labels)))

		pAttrs = {
			'x3ds': x3ds,
			'x3ds_labels': x3ds_labels,
			'which_labels': self.which_labels,
			'missingDataFlags': missingDataFlags,
			'x3ds_pointSize': attrs['pointSize'],
			'x3ds_colour': eval(attrs['colour']),
			'timecode': self.timecode,
			'frameRange': [self.firstFrame, self.lastFrame]
		}
		interface.createChild(interface.name(), 'points3d', atLocation=interface.parentPath(), attrs=pAttrs)


class C3dIo(Op.Op):
	def __init__(self, name='/Read_C3D_IO', locations='', c3dFilename='', offset=0, stepSize=1, pointSize=10.,
				 colour=(0., 0., 1., 0.7), adjustTimeline=True, reverse=False):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('c3d', 'C3D', 'C3D filename', 'filename', c3dFilename, {}),
			('offset', 'Frame offset', 'Frame offset', 'int', offset, {}),
			('step', 'Frame step', 'Frame step', 'int', stepSize, {}),
			('pointSize', 'Point size', '3D Point size', 'float', pointSize, {}),
			('colour', 'Point colour', '3D Point colour', 'string', str(colour), {}),
			('adjustTimeline', 'Adjust timeline', 'Adjust timeline', 'bool', adjustTimeline, {}),
			('reverse', 'Reverse', 'Reverse timeline', 'bool', reverse, {})
		]

		super(self.__class__, self).__init__(name, fields)
		self.data = None
		self.x3ds = None

	def flush(self):
		self.data = None
		self.x3ds = None
		self.x3ds_labels = None

	def cook(self, location, interface, attrs):
		from IO import IO
		if 'c3d' not in attrs or not attrs['c3d']: return
		filename = self.resolvePath(attrs['c3d'])

		if self.data is None:
			try:
				_, self.data = IO.load(filename)
				for locationName, locAttrs in self.data.iteritems():
					if 'x3ds' in locAttrs:
						self.x3ds = locAttrs['x3ds']
						[numMarkers, numFrames, numChannels] = self.x3ds.shape

						if 'x3ds_labels' in locAttrs:
							self.x3ds_labels = locAttrs['x3ds_labels']
						else:
							self.x3ds_labels = np.arange(numMarkers)

						self.logger.info('# Markers: %d | # Frames: %d' % (numMarkers, numFrames))
						if attrs['reverse']: self.x3ds = self.x3ds[:, ::-1]

			except Exception as e:
				self.logger.error('Could not import data: %s' % str(e))
				return

		if self.x3ds is None: return
		[numMarkers, numFrames, numChannels] = self.x3ds.shape

		offset = attrs['offset'] if 'offset' in attrs else 0
		stepSize = attrs['step'] if 'step' in attrs else 1
		frameNum = max((interface.frame() + offset) * stepSize, 0)
		if frameNum >= numFrames: return

		x3ds_frame = self.x3ds[:, frameNum]
		labels = self.x3ds_labels

		if numChannels == 4:
			trueLabels = np.int32(np.where(x3ds_frame[:, 3] == 0)[0])
			x3ds_frame = x3ds_frame[trueLabels, :3]
			labels = labels[trueLabels]

		pAttrs = {
			'x3ds': x3ds_frame,
			'x3ds_labels': labels,
			'x3ds_pointSize': attrs['pointSize'],
			'x3ds_colour': eval(attrs['colour']),
			'frameRange': [0, numFrames]
		}
		interface.createChild(interface.name(), 'points3d', atLocation=interface.parentPath(), attrs=pAttrs)

		if attrs['adjustTimeline']:
			interface.setFrameRange(0, numFrames)


class Xcp(Op.Op):
	def __init__(self, name='/Read_XCP', locations='/root', xcpFilename=''):
		fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('xcp', 'XCP', 'XCP filename', 'filename', xcpFilename, {}),
		]

		self.mats, self.Ps, self.xcp_data = None, None, None

		super(self.__class__, self).__init__(name, fields)

	def cook(self, location, interface, attrs):
		if not 'xcp' in attrs or not attrs['xcp']: return

		if self.mats is None:
			xcp_filename = self.resolvePath(attrs['xcp'])

			try:
				self.mats, self.xcp_data = ViconReader.loadXCP(xcp_filename)
				self.Ps = interface.getPsFromMats(self.mats)
			except IOError as e:
				self.logger.error('Could not load XCP file: %s' % str(e))
				return False

		interface.setAttr('mats', self.mats)
		interface.setAttr('xcp_data', self.xcp_data)

		xcp_camera_ids = np.array([int(x['DEVICEID']) for x in self.xcp_data],dtype=np.int32)
		camera_names = ['%s:%s'%(x['LABEL'],x['DEVICEID']) for x in self.xcp_data]

		self.setupAttrs = {
			'Ps': self.Ps,
			'camera_ids': xcp_camera_ids,
			'camera_names': camera_names,
			'mats': self.mats
		}
		interface.createChild('cameras', 'cameras', atLocation=location, attrs=self.setupAttrs)


class X2d(Op.Op):
	def __init__(self, name='/Read X2D', locations='/root', x2dFilename='',
				 pointSize=8.0, colour=(0.0, 1.0, 0.0, 0.5)):
		fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('x2d', 'X2D', 'X2D filename', 'filename', x2dFilename, {}),
			('pointSize', '2D Point size', '2D Point size', 'float', pointSize, {}),
			('colour', '2D Point colour', '2D Point colour', 'string', str(colour), {})
		]

		super(self.__class__, self).__init__(name, fields)
		self.x2d_frames = None
		self.header = None

	def setup(self, interface, attrs):
		if not 'x2d' in attrs or not attrs['x2d']: return
		x2d_filename = self.resolvePath(attrs['x2d'])

		if self.isDirty('x2d'):
			try:
				x2d_dict = ViconReader.loadX2D(x2d_filename)
				self.x2d_frames = x2d_dict['frames']
				self.header = x2d_dict['header']
			except IOError as e:
				self.logger.error('Could not load X2D file: %s' % str(e))
				return False

			self.firstFrame, self.lastFrame = max(0, interface.frame()), len(x2d_dict)

	def cook(self, location, interface, attrs):
		if self.x2d_frames is None: return

		interface.setType('points')

		x2ds, x2ds_splits = self.x2d_frames[interface.frame()][:2]

		pointAttrs = {
			'x2ds': x2ds[:, :2],
			'x2ds_splits': x2ds_splits,
			'x2ds_colour': eval(attrs['colour']),
			'x2ds_pointSize': attrs['pointSize'],
			'frameRange': [self.firstFrame, self.lastFrame]
		}
		interface.createChild('detections', 'points2d', atLocation=location, attrs=pointAttrs)


class XcpAndX2d(Op.Op):
	def __init__(self, name='/Read XCP and X2D', locations='/root', xcpFilename='', x2dFilename='',
				 offset=0, step=1, pointSize=8.0, colour=(0.0, 1.0, 0.0, 0.5), cameraOffset=0, timecodeLocation='',
				 timecodeOverride='', reverse=False, adjustTimeline=True):
		fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('xcp', 'XCP', 'XCP filename', 'filename', xcpFilename, {}),
			('x2d', 'X2D', 'X2D filename', 'filename', x2dFilename, {}),
			('offset', 'Frame offset', 'Frame offset', 'int', offset, {}),
			('step', 'Frame step', 'Frame step', 'int', step, {}),
			('pointSize', '2D Point size', '2D Point size', 'float', pointSize, {}),
			('colour', '2D Point colour', '2D Point colour', 'string', str(colour), {}),
			('cameraOffset', 'Camera offset', 'Camera offset', 'int', cameraOffset, {}),
			('timecodeLocation', 'Timecode location', 'Timecode location (start)', 'string', timecodeLocation, {}),
			('timecodeOverride', 'Timecode override', 'Timecode override (frame)', 'string', timecodeOverride, {}),
			('reverse', 'Reverse', 'Reverse timeline', 'bool', reverse, {}),
			('adjustTimeline', 'Adjust timeline', 'Adjust timeline', 'bool', adjustTimeline, {})
		]

		super(self.__class__, self).__init__(name, fields)
		self.flush()

	def flush(self):
		self.x2d_frames = None
		self.setupAttrs = {}
		self.mats = None
		self.timecode = ''
		self.offset = 0

	def setup(self, interface, attrs):
		if not 'xcp' in attrs or not attrs['xcp']: return
		if not 'x2d' in attrs or not attrs['x2d']: return

		# if self.isDirty('xcp') or self.isDirty('x2d'):
		if self.x2d_frames is None:
			xcp_filename = self.resolvePath(attrs['xcp'])
			x2d_filename = self.resolvePath(attrs['x2d'])

			# Note: Remove x2d_frames condition when dirty states are more reliable
			try:
				Ps, self.mats, camera_ids, camera_names, self.x2d_frames, header = ViconReader.load_xcp_and_x2d(xcp_filename, x2d_filename)
			except IOError as e:
				self.logger.error('Could not load XCP and X2D files: %s' % str(e))
				return False

			offset = attrs['offset']

			# Find timecode in header
			for info in header:
				if info[0] == 1:
					# The timecode should be last
					tcStr = info[-1]
					tcElms = tcStr.split()[:4]
					self.timecode = ':'.join(tcElms)
					self.logger.info('Timecode found: %s' % self.timecode)
					break

			tcSyncTime, self.offset = interface.getTimecodeSync(self.timecode, 'timecodeLocation', attrs, 100, 25., 4., attrs['offset'])
			if tcSyncTime: self.logger.info('Setting timecode to: %s (offset = %d)' % (tcSyncTime, self.offset))

			self.firstFrame, self.lastFrame = max(0, interface.frame()), len(self.x2d_frames)
			if attrs['reverse']: self.x2d_frames = self.x2d_frames[::-1]

			self.setupAttrs = {
				'Ps': Ps,
				'camera_ids': camera_ids,
				'camera_names': camera_names,
				'mats': self.mats
			}

	def cook(self, location, interface, attrs):
		if self.x2d_frames is None: return

		interface.createChild('cameras', 'cameras', atLocation=location, attrs=self.setupAttrs)

		if 'timecodeOverride' in attrs and attrs['timecodeOverride']:
			tcSyncTime, offset = interface.getTimecodeSync(self.timecode, 'timecodeOverride', attrs, 100, 25., 4., 0)
			if offset == -1: return
			if tcSyncTime: self.logger.info('Override timecode to: %s (offset = %d)' % (tcSyncTime, offset))
			frameNum = int(offset)
		else:
			offset = int(self.offset) if self.offset != 0 else attrs['offset']
			stepSize = attrs['step'] if 'step' in attrs else 1
			frameNum = max((interface.frame() + int(offset)) * stepSize, 0)

		if frameNum >= len(self.x2d_frames):
			self.logger.error('Frame number exceeds number of frames in C3D (%d > %d)' % (frameNum, len(self.x2d_frames)))
			return

		x2ds, x2ds_splits = self.x2d_frames[frameNum]

		pointAttrs = {
			'x2ds': x2ds,
			'x2ds_splits': x2ds_splits,
			'x2ds_colour': eval(attrs['colour']),
			'x2ds_pointSize': attrs['pointSize'],
			'frameRange': [self.firstFrame, self.lastFrame]
		}

		if 'cameraOffset' in attrs and attrs['cameraOffset'] > 0:
			x2ds_splits_render = np.insert(x2ds_splits, np.zeros(attrs['cameraOffset'], dtype=np.int32), 0)
			pointAttrs['x2ds_splits_render'] = x2ds_splits_render

		if self.timecode:
			pointAttrs['timecode'] = self.timecode
			interface.setAttr('timecode', self.timecode)

		interface.createChild('detections', 'points2d', atLocation=location, attrs=pointAttrs)

		if attrs['adjustTimeline']:
			interface.setFrameRange(self.firstFrame, self.lastFrame)


class Cal(Op.Op):
	def __init__(self, name='/Read Cal', locations='/root', calFilename=''):
		fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('cal', 'CAL', 'CAL filename', 'filename', calFilename, {})
		]

		super(self.__class__, self).__init__(name, fields)
		self.mats = None

	def cook(self, location, interface, attrs):
		if not self.mats:
			if not 'cal' in attrs or not attrs['cal']: return

			calibrationFilename = self.resolvePath(attrs['cal'])

			try:
				self.mats, rawCalData = OptitrackReader.load_CAL(calibrationFilename)
			except IOError as e:
				self.logger.error('Could not load CAL file: %s' % str(e))
				return False

		interface.setAttr('mats', self.mats)
		# interface.setAttr('cal_data', rawCalData)


class SurreyCal(Op.Op):
	def __init__(self, name='/Surrey_Cal', locations='/root', calFilename='', width=1920, height=1080):
		fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('cal', 'CAL', 'CAL filename', 'filename', calFilename, {}),
			('width', 'Width', 'Width Override', 'int', width, {}),
			('height', 'Height', 'Height Override', 'int', height, {})
		]

		super(self.__class__, self).__init__(name, fields)
		self.mats = None
		self.camAttrs = {}
		self.previous_size = (None, None)

	def cook(self, location, interface, attrs):
		size_override = (None, None)

		if attrs['width'] != '-1' and attrs['height'] != '-1':
			size_override = (int(attrs['height']), int(attrs['width']))

		if not self.mats or size_override != self.previous_size:
			self.previous_size = size_override
			if not 'cal' in attrs or not attrs['cal']: return
			print 'recalculating mats with size - {}'.format(size_override)

			if "{}/{}".format(location, 'cameras') in interface.locations():
				interface.deleteLocationsByName("{}/{}".format(location, 'cameras'))

			calibrationFilename = self.resolvePath(attrs['cal'])

			try:
				from Reframe import SurreyReader
				cameras = SurreyReader.readCal(calibrationFilename, sensor_size=size_override)
				cameras = cameras[:-1]
				self.mats = [camera['MAT'] for camera in cameras]
				self.camera_ids = range(len(self.mats))

			except IOError as e:
				self.logger.error('Could not load CAL file: %s' % str(e))
				return False

			img = np.zeros(shape=(attrs['height'], attrs['width'], 3), dtype=np.uint8)

			self.camAttrs = {
				'camera_ids': self.camera_ids,
				'mats': self.mats,
				'vheight': [attrs['height']] * len(self.camera_ids),
				'vwidth': [attrs['width']] * len(self.camera_ids),
				'imgs': [img] * len(self.camera_ids)
			}

		# TODO : A Method for updating the cameras instead of creating the Child
		interface.createChild('cameras', 'cameras', attrs=self.camAttrs)


def tryInt(s):
	try:
		return int(s)
	except:
		return s

def alphaNumKey(s):
	import re
	# Turn a string into a list of string and number chunks: "z23a" -> ["z", 23, "a"]
	return [tryInt(c) for c in re.split('([0-9]+)', s)]

class SurreyCalBBC(Op.Op):
	def __init__(self, name='/Surrey_Cal', locations='/root'):
		fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {})
		]

		super(self.__class__, self).__init__(name, fields)
		self.mats = None
		self.camAttrs = {}
		self.previous_size = (None, None)

		self.cals = []

	def cook(self, location, interface, attrs):
		if "{}/{}".format(location, 'cameras') in interface.locations():
			interface.deleteLocationsByName("{}/{}".format(location, 'cameras'))

		height, width = 1080, 1920
		if not self.cals:
			import os
			calPath = os.path.join(os.environ['GRIP_DATA'],'moving_calibration')
			try:
				for file in os.listdir(calPath):
					if file.endswith('.cal'):
						from Reframe import SurreyReader
						cameras = SurreyReader.readCal(os.path.join(calPath, file))
						mats = [camera['MAT'] for camera in cameras]
						camera_ids = range(len(mats))
						self.cals.append((mats, camera_ids))

			except WindowsError as e:
				self.logger.error('Could not find calibration: % s' % str(e))

			print '# Cals:', len(self.cals)

			img = np.zeros(shape=(height, width, 3), dtype=np.uint8)
			camera_ids = self.cals[0][1]
			self.camAttrs = {
				'camera_ids': camera_ids,
				# 'mats': mats,
				'vheight': [height] * len(camera_ids),
				'vwidth': [width] * len(camera_ids),
				'imgs': [img] * len(camera_ids),
				'updateMats': True
			}

		cal = self.cals[interface.frame()]
		mats, camera_ids = cal
		self.camAttrs['mats'] = mats

		# TODO : A Method for updating the cameras instead of creating the Child
		interface.createChild('cameras', 'cameras', attrs=self.camAttrs)


class SurreyTimecodeInfo(Op.Op):
	def __init__(self, name='/SurreyTC', locations='', filename=''):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('filename', 'Filename', 'Filename', 'filename', filename, {})
		]

		super(self.__class__, self).__init__(name, fields)
		self.data = None

	def cook(self, location, interface, attrs):
		from IO import IO
		if 'filename' not in attrs or not attrs['filename']: return
		filename = self.resolvePath(attrs['filename'])

		if self.data is None:
			try:
				_, self.data = IO.load(filename)
				self.logger.info('Found %d entries.' % len(self.data))
			except Exception as e:
				self.logger.error('Could not import data: %s' % str(e))
				return

		if self.data is None: return
		tc, frame, jump = self.data[interface.frame()]

		tcAttrs = {
			'timecode': tc,
			'frame': frame,
			'jump': jump
		}
		interface.createChild(interface.name(), 'group', atLocation=interface.parentPath(), attrs=tcAttrs)


# Register Ops
import Registry
Registry.registerOp('Read C3D', C3d)
Registry.registerOp('Read X3DS (C3D IO)', C3dIo)
Registry.registerOp('Read XCP', Xcp)
Registry.registerOp('Read CAL', Cal)
Registry.registerOp('Read X2D', X2d)
Registry.registerOp('Read XCP and X2D', XcpAndX2d)
Registry.registerOp('Read Surrey Calibration', SurreyCal)
