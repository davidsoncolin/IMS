import os, re, math, time
import numpy as np
import cv2
import Op, Timecode
from IO import MovieReader, OptitrackReader
from GCore import Calibrate
from threading import Thread


import PIL
from IO import Reframe


class Sequence(Op.Op):
	def __init__(self, name='/Video Sequence', locations='/root/cameras', directory='', prefix='', offset=0, offsets='', step=1,
	             useCalibration=True, calibrationFilename='', calibrationLocation='', onlyActiveCamera=False,
	             useTimecode=True, timecodeLocation='', colour=(0.2, 0.0, 0.2, 1)):
		fields = [
			('name', 'Name', 'name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('directory', 'Video directory', 'Video dir', 'string', directory, {}),
			('prefix', 'Use prefix', 'Use prefix', 'bool', bool(prefix), {}),
			('prefixFilename', 'Filename prefix', 'Filename prefix', 'string', prefix, {}),
			('offset', 'Frame offset', 'Frame offset', 'int', offset, {}),
			('offsets', 'Frame offsets', 'Frame offsets', 'string', offsets, {}),
			('step', 'Frame step size', 'Frame step size', 'int', step, {}),
			('calibration', 'Use calibration', 'Use calibration', 'bool', useCalibration, {}),
			('calibrationFilename', 'Calibration filename', 'Calibration filename', 'filename', calibrationFilename, {}),
			('calibrationLocation', 'Calibration location', 'Calibration location', 'string', calibrationLocation, {}),
			('onlyActiveCamera', 'Only active camera', 'Only active camera', 'bool', onlyActiveCamera, {}),
			('useTimecode', 'Use timecode sync', 'Use timecode sync (if available)', 'bool', useTimecode, {}),
			('timecodeLocation', 'Timecode location', 'Timecode location (start)', 'string', timecodeLocation, {}),
			('colour', 'Camera colour', 'Camera colour', 'string', str(colour), {})
		]

		self.frame = None
		self.attrs = {}
		self.flush()

		super(self.__class__, self).__init__(name, fields)
		self.cacheManualOverride = True

	def flush(self):
		self.initialised = False
		self.movies = None
		self.timecodeOffsets = []
		self.timecode = None

	def tryInt(self, s):
		try:
			return int(s)
		except:
			return s

	def alphaNumKey(self, s):
		# Turn a string into a list of string and number chunks: "z23a" -> ["z", 23, "a"]
		return [self.tryInt(c) for c in re.split('([0-9]+)', s)]

	def initialise(self, interface, attrs):
		directory = self.resolvePath(attrs['directory'])
		if not directory: return False

		prefix = attrs['prefix']
		prefixFilename = self.resolvePath(attrs['prefixFilename'])
		if prefix and not prefixFilename: return

		calibration = attrs['calibration']
		calibrationFilename = self.resolvePath(attrs['calibrationFilename'])
		calibrationLocation = self.resolvePath(attrs['calibrationLocation'])
		if calibration and (not calibrationFilename and not calibrationLocation): return False

		movieFilenames = []
		try:
			for file in os.listdir(directory):
				if prefixFilename and not file.startswith(prefixFilename): continue
				if file.endswith('.avi') or file.endswith('.mov') or file.endswith('mp4'):
					movieFilenames.append(os.path.join(directory, file))
		except WindowsError as e:
			self.logger.error('Could not find videos: % s' % str(e))

		if not movieFilenames:
			# TODO: Here we'll have to clear the cameras etc.
			return False

		# Windows will produce a wonky order, i.e. 1, 10, 11, .., 2, 3, ..
		# Use natural sorting to rectify
		movieFilenames.sort(key=self.alphaNumKey)

		self.camera_ids = []
		self.camera_names = []
		self.movies = []
		self.mats = []
		vheights = []
		vwidths = []
		timecodes = []
		hasTimecode = False
		useTimecode = attrs['useTimecode'] if 'useTimecode' in attrs else True

		offset = attrs['offset']
		if 'offsets' in attrs and attrs['offsets']:
			offsets = eval(attrs['offsets'])
		else:
			offsets = [offset] * len(movieFilenames)

		for ci, mf in enumerate(movieFilenames):
			self.logger.info('Loading MovieReader: %s' % mf)
			movieData = MovieReader.open_file(mf, audio=False, frame_offset=offsets[ci])

			if movieData['vbuffer'] is not None:
				self.movies.append(movieData)

				self.timecodeOffsets.append(0)
				if 'timecode' in movieData and movieData['timecode']:
					hasTimecode = True
					timecodes.append(movieData['timecode'])

		# Make sure we have all the cameras before continuing
		if len(self.movies) != len(movieFilenames):
			self.logger.error('Could not load all movies in sequence')
			return

		# Make sure we have as many time codes as movies (if we have any)
		if hasTimecode and len(self.movies) != len(timecodes):
			self.logger.error('Not all movie files have a time code')
			return

		# See if we can get the offsets using the time codes
		if hasTimecode and useTimecode:
			print 'Video timecodes:', timecodes
			fps_all = [round(m['fps']) for m in self.movies]
			print 'FPS:', fps_all
			timecodeValues = [Timecode.TCFtoInt(tc, fps) for tc, fps in zip(timecodes, fps_all)]
			tcOrderDesc = [timecodes.index(tc) for tc in sorted(timecodes, reverse=True)]
			
			# Set the first offset to 0
			firstTcIndex = tcOrderDesc[0]
			self.timecodeOffsets[firstTcIndex] = 0
			largestTc = timecodes[firstTcIndex]
			offsetStartIndex = 1

			# We can also get the timecode destination from an incoming location, e.g. 2D detections
			if 'timecodeLocation' in attrs and attrs['timecodeLocation']:
				tcSyncTime = interface.attr('timecode', atLocation=attrs['timecodeLocation'])
				if tcSyncTime is not None:
					tcSyncValue = Timecode.TCFtoInt(tcSyncTime, fps_all[0])
					if tcSyncValue < timecodeValues[firstTcIndex]:
						self.logger.error('Sync timecode %s is smaller than video timecodes (%s).' % (tcSyncTime, largestTc))
						return

					largestTc = tcSyncTime
					offsetStartIndex = 0

			self.timecode = largestTc
			self.logger.info('Setting timecode to: %s' % (largestTc))

			# Calculate the offset for each camera to get it up to speed with the target timecode
			# TODO: Replace hard coded timecode fps and multiplier
			timecodeFps, timecodeMultiplier = 25., 2.
			for tcInd in tcOrderDesc[offsetStartIndex:]:
				diff = Timecode.TCSub(largestTc, timecodes[tcInd], timecodeFps)
				self.timecodeOffsets[tcInd] = Timecode.TCFtoInt(diff, timecodeFps) * timecodeMultiplier

		if self.timecodeOffsets: print 'Video timecode offsets:', self.timecodeOffsets

		self.camera_ids = ['Camera %d' % ci for ci in xrange(len(movieFilenames))]
		self.movies = self.movies

		if not calibrationLocation: calibrationLocation = interface.root()
		if calibrationFilename or interface.hasAttr('mats', atLocation=calibrationLocation):
			if calibrationFilename:
				# TODO: Detect filetype, e.g. .cal and .xcp and handle accordingly
				try:
					self.mats, rawCalData = OptitrackReader.load_CAL(calibrationFilename)
					if not self.mats: return False
				except IOError as e:
					self.logger.error('Could not load calibration file: %s' % str(e))
					return False
			else:
				self.mats = interface.attr('mats', atLocation=calibrationLocation)
				if not self.mats:
					self.logger.error('Could not find calibration mats: %s' % calibrationLocation)
					return False

		else:
			from GCore import Calibrate
			for ci, (cid, md) in enumerate(zip(self.camera_ids, self.movies)):
				if md is not None:
					self.mats.append(Calibrate.makeUninitialisedMat(ci, (md['vheight'], md['vwidth'])))

		for md in self.movies:
			vheights.append(md['vheight'])
			vwidths.append(md['vwidth'])

		Ps = interface.getPsFromMats(self.mats)
		self.attrs = {
			'vheight': vheights, 'vwidth': vwidths, 'camera_ids': self.camera_ids, 'Ps': Ps, 'mats': self.mats,
			'colour': eval(attrs['colour'])
		}

		if self.camera_names:
			self.attrs['camera_names'] = self.camera_names

		self.initialised = True
		return True

	def setup(self, interface, attrs):
		dirtyAttrs = self.getAttrs(onlyDirty=True)

		if not self.initialised or dirtyAttrs:
			self.initialise(interface, attrs)

	def cook(self, location, interface, attrs):
		if not self.initialised: return
		self.frame = interface.frame()
		imgs = []

		offset = attrs['offset'] if 'offset' in attrs else 0
		stepSize = attrs['step'] if 'step' in attrs else 1

		# Check if we are looking through a single active camera or not as that will be more efficient.
		# Here we are not interested in knowing whether or not we found anything
		activeCameraIdx = interface.attr('activeCameraIdx', atLocation=interface.root(), log=False)
		if 'onlyActiveCamera' in attrs and attrs['onlyActiveCamera'] and activeCameraIdx is not None and activeCameraIdx != -1:
			frameNum = max((self.frame + offset + self.timecodeOffsets[activeCameraIdx]) * stepSize, 0)
			md = self.movies[activeCameraIdx]

			try:
				MovieReader.readFrame(md, seekFrame=frameNum, playingAudio=False)
			except:
				self.logger.error('Could not read frame: %d for active camera %d' % (self.frame, activeCameraIdx))
				return

			img = np.frombuffer(md['vbuffer'], dtype=np.uint8).reshape(md['vheight'], md['vwidth'], 3)
			imgs.append(img)

		else:
			# Process all cameras (slower but necessary for processes/Ops that need all the data)
			for ci, md in enumerate(self.movies):
				try:
					frameNum = max((self.frame + offset + self.timecodeOffsets[ci]) * stepSize, 0)
					MovieReader.readFrame(md, seekFrame=frameNum, playingAudio=False)
					img = np.frombuffer(md['vbuffer'], dtype=np.uint8).reshape(md['vheight'], md['vwidth'], 3)
					imgs.append(img)

				except:
					self.logger.error('Could not read frame: %d for camera %d' % (self.frame, ci))
					return

		self.attrs['imgs'] = imgs
		interface.createChild(interface.name(), 'cameras', atLocation=interface.parentPath(), attrs=self.attrs)

		if self.timecode: interface.setAttr('timecode', self.timecode)


class CamVideoStream:
	def __init__(self, src=0):
		# initialise the video camera stream and read the first frame from the stream
		self.stream = cv2.VideoCapture(src)
		self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
		self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
		# self.stream.set(cv2.CV_CAP_PROP_FOURCC, cv2.cv.CV_FOURCC('H','2','6','4'))
		# self.stream.set(cv2.CV_CAP_PROP_FPS, 30)

		(self.grabbed, self.frame) = self.stream.read()

		# initialise the variable used to indicate if the thread should be stopped
		self.stopped = False

	def start(self):
		# start the thread to read frames from the video stream
		Thread(target=self.update, args=()).start()
		return self

	def update(self):
		# keep looping infinitely until the thread is stopped
		while True:
			# if the thread indicator variable is set, stop the thread
			if self.stopped:
				return

			# otherwise, read the next frame from the stream
			(self.grabbed, self.frame) = self.stream.read()

	def read(self):
		# return the frame most recently read
		return self.frame

	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True


class Capture(Op.Op):
	def __init__(self, name='/Video_Capture', locations='/root/cameras/cam0', width=1920, height=1200):
		fields = [
			('name', 'Name', 'name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('vwidth', 'Width', 'width', 'int', width, {}),
			('vheight', 'Height', 'height', 'int', height, {})
		]

		super(self.__class__, self).__init__(name, fields)
		self.cam, self.mats = None, None
		self.camAttrs = {}

	def cook(self, location, interface, attrs):
		vwidth, vheight = attrs['vwidth'], attrs['vheight']
		if self.cam is None:
			self.cam = CamVideoStream(src=0).start()
			self.mats = [Calibrate.makeUninitialisedMat(0, (vheight, vwidth))]
		self.camAttrs = {
			'vheight': [vheight], 'vwidth': [vwidth], 'camera_ids': [0], 'mats': self.mats,
			'updateImage': True
		}

		md = {'frame': self.cam.read()}
		self.camAttrs['imgs'] = [md['frame']]

		# self.attrs['imgs'] = np.array(frame, dtype=np.uint8)
		interface.createChild(interface.name(), 'cameras', atLocation=interface.parentPath(), attrs=self.camAttrs)


class SurreyStream(Op.Op):
	def __init__(self, name='/Video_SurreyStream', locations='/root/cameras', calibration='', width=1920, height=1080):
		fields = [
			('name', 'Name', 'name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('calibration', 'calibration', 'calibration', 'string', calibration, {}),
			('vwidth', 'Width', 'width', 'int', width, {}),
			('vheight', 'Height', 'height', 'int', height, {})
		]

		super(self.__class__, self).__init__(name, fields)
		self.listeners = {}
		self.camAttrs = {}
		self.initialised = False

	def flush(self):
		pass

	def processData(self, data):
		pass

	def setup(self, interface, attrs):
		if not self.listeners:
			# for r in range(155, 163):
			for r in range(1, 9):
				self.listeners[r] = Reframe.Listener(timeout=0)
				# self.listeners[r].addSubscription(Reframe.tcp('131.227.94.%d' % r, 5550), self.processData, filters=[''])
				self.listeners[r].addSubscription('epgm://239.193.1.%d:5550' % r, self.processData, filters=[''])

			time.sleep(1)

	def cook(self, location, interface, attrs):
		if not self.listeners: return

		import StringIO

		imgs, vwidths, vheights, camera_ids, camera_names, mats = [], [], [], [], [], []
		ci = 0
		for r, listener in self.listeners.iteritems():
			data = listener.poll()
			if data is None:
				self.logger.error('No data on %d' % r)
				continue
			print data
			timeCode, imgStr = data
			sio = StringIO.StringIO(imgStr)
			img = PIL.Image.open(sio)

			if not self.initialised:
				vwidth, vheight = attrs['vwidth'], attrs['vheight']
				mat = Calibrate.makeUninitialisedMat(0, (vheight, vwidth))

				vwidths.append(vwidth)
				vheights.append(vheight)
				camera_ids.append('Camera %d' % ci)
				camera_names.append(str(r))
				mats.append(mat)

			imgs.append(img.tobytes())
			ci += 1

		if not self.initialised:
			self.camAttrs['vheight'] = vheights
			self.camAttrs['vwidth'] = vwidths
			# self.camAttrs['camera_ids'] = camera_ids
			# self.camAttrs['camera_names'] = camera_names
			self.camAttrs['camera_ids'] = interface.attr('camera_ids')
			self.camAttrs['camera_names'] = camera_names
			self.camAttrs['mats'] = interface.attr('mats')
			self.initialised = True

		if imgs:
			self.camAttrs['imgs'] = imgs
			# self.camAttrs['updateImage'] = True
			interface.createChild(interface.name(), 'cameras', atLocation=interface.parentPath(), attrs=self.camAttrs)

			tcAttrs = {
				'x3ds': np.array([[0, 0, 0]], dtype=np.float32),
				'x3ds_labels': [timeCode]
			}
			interface.createChild(interface.name() + '/tc', 'points', atLocation=interface.parentPath(), attrs=tcAttrs)


class Stream(Op.Op):
	def __init__(self, name='/Video_Stream', locations='/root/stream', cameraLocations='', width=1920, height=1080):
		fields = [
			('name', 'Name', 'name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('cameraLocations', 'cameraLocations', 'cameraLocations', 'string', cameraLocations, {}),
			('vwidth', 'Width', 'width', 'int', width, {}),
			('vheight', 'Height', 'height', 'int', height, {})
		]

		super(self.__class__, self).__init__(name, fields)
		self.cam, self.mats = None, None
		self.camAttrs = {}

	def cook(self, location, interface, attrs):
		imgs = interface.attr('jpegimgs')
		if imgs is None: return
		img_count = len(imgs)
		vwidth, vheight = attrs['vwidth'], attrs['vheight']
		self.mats = interface.attr('mats', atLocation=attrs['cameraLocations'])
		updateMats = interface.attr('updateMats', atLocation=attrs['cameraLocations'], default=False)
		if self.mats is None or updateMats:
			self.mats = []
			for i in xrange(img_count):
				self.mats.append(Calibrate.makeUninitialisedMat(i, (vheight, vwidth)))

		interface.setAttr('updateMats', False, atLocation=attrs['cameraLocations'])

		self.camAttrs = {
			'vheight': [vheight] * img_count,
			'vwidth': [vwidth] * img_count,
			'camera_ids': range(img_count),
			'mats': self.mats,
			'updateImage': True,
			'jpegimgs': imgs,
			'updateMats': updateMats
		}

		for k, v in self.camAttrs.iteritems():
			interface.setAttr(k,v)

# Register Ops
import Registry
Registry.registerOp('Video Sequence', Sequence)
Registry.registerOp('Video Capture', Capture)
