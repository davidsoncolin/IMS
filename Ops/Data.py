import os, sys
import datetime
import multiprocessing
import Queue
import numpy as np
import Op, Interface
import IO


class Export(Op.Op):
	def __init__(self, name='/Export_Data', locations='', saveTo='', exactMatch=True, enable=True,
				attrsWhiteList='', attrsBlackList='', frameRange='', allowOverride=True):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('dataLocations', 'locations', 'Data locations', 'string', locations, {}),
			('saveTo', 'Save to', 'Save to', 'filename', saveTo, {}),
			('exactMatch', 'Exact match', 'Exact match', 'bool', exactMatch, {}),
			('enable', 'Enable', 'Enable', 'bool', enable, {}),
			('frameRange', 'Frame range', 'Frame range', 'string', frameRange, {}),
			('attrsBlackList', 'Exclude attributes', 'Exclude attributes', 'string', attrsBlackList, {}),
			('attrsWhiteList', 'Include attributes', 'Include attributes', 'string', attrsWhiteList, {}),
			('allowOverride', 'Allow override', 'Allow override', 'bool', allowOverride, {})
		]

		super(self.__class__, self).__init__(name, fields)

	def cook(self, location, interface, attrs):
		if 'dataLocations' not in attrs or not attrs['dataLocations']: return
		if 'saveTo' not in attrs or not attrs['saveTo']: return

		enable = attrs['enable']

		if not enable: return
		if not self.useFrame(interface.frame(), attrs['frameRange']): return

		dataLocations = attrs['dataLocations']
		saveTo = self.resolvePath(attrs['saveTo'])
		exactMatch = attrs['exactMatch']
		if not attrs['allowOverride'] and os.path.isfile(saveTo): return

		attrsBlackList = self.resolveLocations(attrs['attrsBlackList'], prefix='') if attrs['attrsBlackList'] else []
		attrsWhiteList = self.resolveLocations(attrs['attrsWhiteList'], prefix='') if attrs['attrsWhiteList'] else []

		exportData = {}
		if exactMatch:
			for dataLoc in self.resolveLocations(dataLocations):
				if dataLoc not in interface.locations(): continue
				attrs = interface.location(dataLoc)
				if attrsBlackList:
					exportData[dataLoc] = {}
					for name, value in attrs.iteritems():
						if name not in attrsBlackList:
							exportData[dataLoc][name] = value
				elif attrsWhiteList:
					exportData[dataLoc] = {}
					for name, value in attrs.iteritems():
						if name == 'type' or name in attrsWhiteList:
							exportData[dataLoc][name] = value
				else:
					exportData[dataLoc] = interface.location(dataLoc)

			# exportData = {dataLoc: interface.attr(dataLoc) for dataLoc in self.resolveLocations(dataLocations)}
		else:
			# TODO: Fix bug here (list not dict)
			locations = interface.locations()
			# for locName, loc in locations.iteritems():
			for locName in locations:
				for dataLoc in self.resolveLocations(dataLocations):
					if locName.startswith(dataLoc):
						if attrsBlackList:
							for name, value in interface.location(locName):
								if name not in attrsBlackList:
									exportData[dataLoc][name] = value
						elif attrsWhiteList:
							for name, value in interface.location(locName):
								if name in attrsWhiteList:
									exportData[dataLoc][name] = value
						else:
							exportData[locName] = interface.location(locName)

		if not exportData: return

		IO.save(saveTo, exportData)
		self.logger.info('Data exported to: %s' % saveTo)


class Import(Op.Op):
	def __init__(self, name='/Import_Data', filename='', importHere=False, eval=False):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('filename', 'Filename', 'Filename', 'filename', filename, {}),
			# ('importHere', 'Import Here', 'Import Here', 'bool', importHere, {})
			('eval', 'Eval', 'Eval (in case the data is stored as a string)', 'bool', eval, {})
		]

		super(self.__class__, self).__init__(name, fields)
		self.data = None

	def flush(self):
		self.data = None

	def cook(self, location, interface, attrs):
		if 'filename' not in attrs or not attrs['filename']: return
		filename = self.resolvePath(attrs['filename'])

		if self.data is None:
			try:
				_, self.data = IO.load(filename)
			except Exception as e:
				self.logger.error('Could not import data: %s' % str(e))
				return

		if self.data is None: return
		if attrs['eval'] and isinstance(self.data, str):
			self.data = eval(self.data)

		# importHere = attrs['importHere']
		try:
			for locationName, attrs in self.data.iteritems():
				if 'type' not in attrs:
					# self.logger.warning('Type attribute not found in (created as group): %s' % locationName)
					type = 'group'
				else:
					type = attrs['type']

				loc = interface.location(locationName)
				if loc is None:
					interface.createChild(None, type, atLocation=locationName, attrs=attrs)
				else:
					for attrName, attrValue in attrs.iteritems():
						interface.setAttr(attrName, attrValue, atLocation=locationName)
		except:
			# The data has probably been saved without an op structure so we just shove the imported data onto
			# a child location
			interface.createChild('imported', 'group', attrs={'data': self.data})


class Send(Op.Op):
	def __init__(self, name='/SendData', locations='', camerasRootLocation='', pubPort=-1, calibration='', topic='',
				id_='', cameraIndex=0, timecode='99:99:99:99'):
		self.fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('camerasRootLocation', 'camerasRootLocation', 'camerasRootLocation', 'string', camerasRootLocation, {}),
			('pubPort', 'pubPort', 'pubPort', 'int', pubPort, {'min': 1024 , 'max' : 65535}),
			('topic', 'Topic', 'Topic', 'string', topic, {}),
			('id', 'ID', 'ID', 'string', id_, {}),
			('cameraIndex', 'cameraIndex', 'cameraIndex', 'int', cameraIndex, {}),
			('timecode', 'timecode', 'timecode', 'string', timecode, {})
		]

		super(self.__class__, self).__init__(name, self.fields)
		self.publisher = None
		self.last_timecode = '-00:00:00:00'

	# @profile
	def setup(self, interface, attrs):
		if self.publisher is None:
			from IO import Reframe
			self.publisher = Reframe.Publisher(Reframe.tcpBind(attrs['pubPort']))

	# @profile
	def flush(self):
		self.publisher = None

	# @profile
	def cook(self, location, interface, attrs):
		if self.publisher:
			bAttrs = {}
			#for k in interface.attrKeys(location):
			#	bAttrs[k] = interface.attr(k)

			bAttrs['imgs'] = interface.attr('imgs')
			bAttrs['mats'] = interface.attr('mats', atLocation='/root/stream/cameras')

			# Send data to a separate process
			self.publisher.publish({"location": '/root/camera',
									'data': bAttrs,
									'frame': interface.frame(),
									'id': attrs['id'],
									'camera_index': interface.attr('cId'),
									'dark_detections': interface.attrs('/root/camera/dark'),
									'bright_detections': interface.attrs('/root/camera/bright'),
									'timecode': attrs['timecode']},
									attrs['topic'])
		else:
			self.logger.error("No publisher set to send data on to.")


class SendImage(Op.Op):
	def __init__(self, name='/SendData', locations='', streamLocation='', camerasRootLocation='', pubPort=-1, calibration='', topic='',
				id_='', cameraIndex=0, timecode='99:99:99:99'):
		self.fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('streamLocation', 'streamLocation', 'streamLocation', 'string', streamLocation, {}),
			('camerasRootLocation', 'camerasRootLocation', 'camerasRootLocation', 'string', camerasRootLocation, {}),
			('pubPort', 'pubPort', 'pubPort', 'int', pubPort, {'min': 1024 , 'max' : 65535}),
			('topic', 'Topic', 'Topic', 'string', topic, {}),
			('id', 'ID', 'ID', 'string', id_, {}),
			('cameraIndex', 'cameraIndex', 'cameraIndex', 'int', cameraIndex, {}),
			('timecode', 'timecode', 'timecode', 'string', timecode, {})
			#('timecode', 'timecode', 'timecode', 'string', timecode, {}),
		]

		super(self.__class__, self).__init__(name, self.fields)
		self.publisher = None
		self.last_timecode = '-00:00:00:00'

	# @profile
	def setup(self, interface, attrs):
		if self.publisher is None:
			from IO import Reframe
			self.publisher = Reframe.Publisher(Reframe.tcpBind(attrs['pubPort']))

	# @profile
	def flush(self):
		self.publisher = None

	# @profile
	def cook(self, location, interface, attrs):
		if self.publisher:
			imgs = interface.attr('jpegimgs', atLocation=attrs['streamLocation'])
			if imgs is None or attrs['cameraIndex'] >= len(imgs): return
			updateMats = interface.attr('updateMats', default=False, atLocation=attrs['streamLocation'])
			mats = interface.attr('mats')
			bAttrs = {
				'imgs': [imgs[attrs['cameraIndex']]] if imgs is not None else [],
				'mats': [mats[attrs['cameraIndex']]] if mats is not None else [],
				'vheight': [1080],
				'vwidth': [1920],
				'updateMats': updateMats
			}
			interface.setAttr('updateMats', False)
			trainData = interface.attr('train', atLocation=attrs['streamLocation'])
			timecode = interface.attr('timecode', atLocation=attrs['streamLocation'])

			if self.last_timecode == timecode: return
			self.last_timecode = timecode
			#print '{} sending camera {} with tc {}'.format(datetime.datetime.utcnow(), attrs['cameraIndex'], timecode)
			data = {'location': '/root/camera',
					'data': bAttrs,
					'frame': interface.frame(),
					'id': attrs['id'],
					'camera_index': attrs['cameraIndex'],
					'dark_detections': interface.attrs('/root/camera/dark'),
					'bright_detections': interface.attrs('/root/camera/bright'),
					'timecode': timecode,
					'train': trainData}
			# Send data to a separate process
			self.publisher.publish(data, attrs['topic'])
		else:
			self.logger.error("No publisher set to send data on to.")

class SendImages(Op.Op):
	def __init__(self, name='/SendData', locations='', camerasRootLocation='', pubPort=-1, calibration='', topic='',
				id_='', cameraIndex=0, timecode='99:99:99:99'):
		self.fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('camerasRootLocation', 'camerasRootLocation', 'camerasRootLocation', 'string', camerasRootLocation, {}),
			('pubPort', 'pubPort', 'pubPort', 'int', pubPort, {'min': 1024 , 'max' : 65535}),
			('topic', 'Topic', 'Topic', 'string', topic, {}),
			('id', 'ID', 'ID', 'string', id_, {}),
			('cameraIndex', 'cameraIndex', 'cameraIndex', 'int', cameraIndex, {}),
			('timecode', 'timecode', 'timecode', 'string', timecode, {})
		]

		super(self.__class__, self).__init__(name, self.fields)
		self.publisher = None
		self.last_timecode = '-00:00:00:00'

	# @profile
	def setup(self, interface, attrs):
		if self.publisher is None:
			from IO import Reframe
			self.publisher = Reframe.Publisher(Reframe.tcpBind(attrs['pubPort']))

	# @profile
	def flush(self):
		self.publisher = None

	# @profile
	def cook(self, location, interface, attrs):
		if self.publisher:

			bAttrs = {}
			bAttrs['imgs'] = interface.attr('imgs')
			bAttrs['mats'] = interface.attr('mats')

			# Send data to a separate process
			self.publisher.publish({"location": '/root/camera',
									'data': bAttrs,
									'frame': interface.frame(),
									'id': attrs['id'],
									'camera_index': [0,1,2,3,4,5,6,7],
									'dark_detections': interface.attrs('/root/camera/dark'),
									'bright_detections': interface.attrs('/root/camera/bright'),
									'timecode': attrs['timecode']},
									attrs['topic'])
		else:
			self.logger.error("No publisher set to send data on to.")


class SendDetections(Op.Op):
	def __init__(self, name='/SendData', locations='', camerasRootLocation='', pubPort=-1, topic='',
				id_='', cameraIndex=0, timecode='99:99:99:99'):
		self.fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('camerasRootLocation', 'camerasRootLocation', 'camerasRootLocation', 'string', camerasRootLocation, {}),
			('pubPort', 'pubPort', 'pubPort', 'int', pubPort, {'min': 1024 , 'max' : 65535}),
			('topic', 'Topic', 'Topic', 'string', topic, {}),
			('id', 'ID', 'ID', 'string', id_, {}),
			('cameraIndex', 'cameraIndex', 'cameraIndex', 'int', cameraIndex, {}),
			('timecode', 'timecode', 'timecode', 'string', timecode, {})
		]

		super(self.__class__, self).__init__(name, self.fields)
		self.publisher = None
		self.last_timecode = '-00:00:00:00'

	# @profile
	def setup(self, interface, attrs):
		if self.publisher is None:
			from IO import Reframe
			self.publisher = Reframe.Publisher(Reframe.tcpBind(attrs['pubPort']))

	# @profile
	def flush(self):
		self.publisher = None

	# @profile
	def cook(self, location, interface, attrs):
		training = interface.attr('train', default={'train': False, 'reset': False, 'send_plate': False})

		if self.publisher:
			self.publisher.publish({'location': '/root/camera',
									'frame': interface.frame(),
									'id': attrs['id'],
									'camera_index': interface.attr('cId'),
									'bright_detections': interface.attrs('/root/camera/bright'),
									'dark_detections': interface.attrs('/root/camera/dark'),
									'timecode': interface.attr('timecode'),
									'mats': interface.attr('mats'),
									'imgs': interface.attr('imgs')[0] if training['send_plate'] else None,
									'updateMats': interface.attr('updateMats', False)
									},
									attrs['topic'])
		else:
			self.logger.error("No publisher set to send data on to.")



class Merge(Op.Op):
	def __init__(self, name='/MergeData', locations='/root/remotecameras/*', camerasRootLocation='/root/cameras'): # detections=''
		self.fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('camerasRootLocation', 'camerasRootLocation', 'camerasRootLocation', 'string', camerasRootLocation, {}),
			# TODO: Make generic by not assuming detections

		]

		super(self.__class__, self).__init__(name, self.fields)

	def cook(self, location, interface, attrs):

		#camera_imgs = interface.attr('imgs', default=[], atLocation=attrs['camerasRootLocation'])
		#mats = interface.attr('mats', default=[], atLocation=attrs['camerasRootLocation'])

		data = interface.attr('data')
		camera_index = data['camera_index']
		if camera_index is None: return
		dark_detections = data['dark_detections']
		darks = interface.attrs('/root/detections/dark')

		camera_index = int(camera_index)

		# TO DO: Check for mats that already exist
		#mats = interface.attr('mats', default=[], atLocation=attrs['camerasRootLocation'])
		#while len(mats) < camera_index: mats.append([])
		#if 'mats' in data:
		#	mats[camera_index] = data['mats'][0]

		# TO DO : Check the 'blur' of the image.
		#imgs = interface.attr('imgs', default=[], atLocation=attrs['camerasRootLocation'])
		#imgs[camera_index] = data['imgs'][0]
		#interface.setAttr('imgs', imgs, atLocation=attrs['camerasRootLocation'])

		for key, value in dark_detections.iteritems():
			if key in ['type', 'x2ds_pointSize', 'x2ds_colour', 'x2ds_colours']:
				if key not in darks:
					darks[key] = value
			elif key in ['x2ds']:
				if key not in darks:
					darks[key] = [np.float32([])]
				while len(darks[key]) <= camera_index:
					darks[key].append(np.float32([]))
				darks[key][camera_index] = value
			else:
				continue
			interface.setAttr(key, darks[key], '/root/detections/dark')

			# TODO: Add an empty list if we don't have any dets for a given camera


class TestingABC(Op.Op):
	def __init__(self, name='/DetectionsSort', locations='/root/detections/dark', camerasRootLocation='/root/cameras'):
		self.fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('camerasRootLocation', 'camerasRootLocation', 'camerasRootLocation', 'string', camerasRootLocation, {}),
		]

		super(self.__class__, self).__init__(name, self.fields)

	def cook(self, location, interface, attrs):
		darks = interface.attrs()
		if not darks: return
		import Ops.Interface
		splits = Ops.Interface.makeSplitBoundaries(map(len, darks['x2ds']))

		x2ds = np.zeros((splits[-1], 2), dtype=np.float32)
		for det, i0, i1 in zip(darks['x2ds'], splits[:-1], splits[1:]): x2ds[i0:i1] = det.reshape(-1, 2)
		#x2ds = np.concatenate(darks['detections'])
		detsAttrs = {
			'x2ds' : x2ds, # []
			#'rx2ds' : None, # []
			'x2ds_splits' : splits, # []
			#'x2ds_pointSize' : 10., # int
			#'x2ds_colours' : interface.attr('x2ds_colours'), # []
			#'detections' : None, # []
			#'type' : interface.attr('type'), # ""
			#'x2ds_colour' : interface.attr('x2ds_colour') # ()
		}

		if x2ds is None:
			print 'NO X2DS!', x2ds
			return
		interface.createChild('features', 'detections', attrs=detsAttrs)
		interface.deleteChild(location)


class CreateSplits(Op.Op):
	def __init__(self, name='/CreateSplits', locations='', subPort=-1, topic='', type_='', host='localhost'):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {})
		]

		super(self.__class__, self).__init__(name, fields)

	def cook(self, location, interface, attrs):
		dets = interface.attr('detections')
		splits = Interface.makeSplitBoundaries(map(len, dets))
		interface.setAttr('x2ds_splits', splits)
		#interface.deleteChild(location)

class Receive(Op.Op):
	def __init__(self, name='/ReceiveData', locations='', subPort=-1, topic='', type_='', host='localhost'):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('subPort', 'SUB port', 'SUB port', 'int', subPort, {'min': 1024 , 'max' : 65535}),
			('topic', 'Topic', 'Topic', 'string', topic, {}),
			('type', 'Type', 'Type', 'string', type_, {}),
			('host', 'Host', 'Host', 'string', host, {}),
		]

		super(self.__class__, self).__init__(name, fields)
		self.listener = None
		self.interface = None
		self.process_pool = None
		self.q = None
		self.wait = False
		self.isSetup = False

	#@profile
	def setup(self, interface, attrs):
		if self.process_pool is None:
			if not len(attrs['topic']):
				self.logger.error("No topic to listen on specified")
				return

			# Get the interface for the processData item
			self.interface = interface

			self.process_pool = multiprocessing.Pool(processes=1)
			manager = multiprocessing.Manager()
			self.q = manager.Queue()
			self.process_pool.map_async(listen_on_thread, ((attrs, self.q),))
			self.isSetup = True

	#@profile
	def cook(self, location, interface, attrs):
		if self.process_pool is None:
			return

		if not self.wait:
			if self.q.empty(): return

		data = self.q.get()
		if isinstance(data, dict) and data['camera_index'] is not None:
			at_location = data['location']
			data.pop('location', None)
			interface.setAttr('data', data, atLocation='/root/remotecameras/{}'.format(data['camera_index']))
		self.q.task_done()


class ReceiveTest(Op.Op):

	data = None

	def __init__(self, name='/ReceiveData', locations='', subPort=-1, topic='', type_='', host='localhost'):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('subPort', 'SUB port', 'SUB port', 'int', subPort, {'min': 1024 , 'max' : 65535}),
			('topic', 'Topic', 'Topic', 'string', topic, {}),
			('type', 'Type', 'Type', 'string', type_, {}),
			('host', 'Host', 'Host', 'string', host, {}),
		]

		super(self.__class__, self).__init__(name, fields)
		self.listener = None
		self.interface = None
		self.wait = True
		self.isSetup = False
		self.listener = None

	#@profile
	def setup(self, interface, attrs):
		if self.listener is None:
			if not len(attrs['topic']):
				self.logger.error("No topic to listen on specified")
				return

			# Get the interface for the processData item
			self.interface = interface
			self.listener = IO.Reframe.Listener()
			self.listener.addSubscription(IO.Reframe.tcp(attrs['host'], attrs['subPort']), ReceiveTest.DataReceived, filters=[attrs['topic']])
			self.listener.poll(timeout=0)
			print "LISTENING"

	@classmethod
	def DataReceived(cls, data):
		cls.data = data
		print "GOT DATA!"

	#@profile
	def cook(self, location, interface, attrs):
		if self.listener is None:
			return
		self.listener.poll(timeout=0.1)
		data = self.data
		if data:
			print data.keys()
			#at_location = data['location']
			#data.pop('location', None)
			interface.setAttr('data', data, atLocation='/root/remotecameras/{}'.format(data['camera_index']))
			print data['timecode']
			#print "Camera ID:",data['camera_index']
			#detections = interface.attrs(atLocation=r'/root/detections/dark')
			#merged_detections = {}
			#for k,value in detections:
			#	if k not in merged_detections: merged_detections[k] = []
			#	while len(merged_detections[k]) <= data['camera_index']:
			#		merged_detections[k].append(None)
			#	merged_detections[k][data['camera_index']] = v
			#	interface.setAttr(k, merged_detections[k], r'/root/detections/dark')

			# detections[int(data['camera_index'])] = data['dark_detections']
			#for k,v in data['bright_detections'].iteritems():
			#	interface.setAttr(k, v, r'/root/detections/bright')
			self.data = None


class ReceiveX(Op.Op):
	def __init__(self, name='/ReceiveData', locations='', subPort=-1, topic='', type_='', host='localhost'):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('subPort', 'SUB port', 'SUB port', 'int', subPort, {'min': 1024 , 'max' : 65535}),
			('topic', 'Topic', 'Topic', 'string', topic, {}),
			('type', 'Type', 'Type', 'string', type_, {}),
			('host', 'Host', 'Host', 'string', host, {}),
		]

		super(self.__class__, self).__init__(name, fields)
		self.listener = None
		self.interface = None
		self.process_pool = None
		self.q = None
		self.wait = True
		self.isSetup = False

	#@profile
	def setup(self, interface, attrs):
		if self.process_pool is None:
			if not len(attrs['topic']):
				self.logger.error("No topic to listen on specified")
				return

			# Get the interface for the processData item
			self.interface = interface

			self.process_pool = multiprocessing.Pool(processes=1)
			manager = multiprocessing.Manager()
			self.q = manager.Queue()
			self.process_pool.map_async(listen_on_thread, ((attrs, self.q),))

			location, key = attrs['locations'].rsplit('/', 1)
			self.isSetup = True

	#@profile
	def cook(self, location, interface, attrs):
		if self.process_pool is None:
			return

		if not self.wait:
			if self.q.empty(): return

		data = self.q.get()
		if data:
			at_location = data['location']

			data.pop('location', None)
			interface.setAttr('data', data, atLocation='/root/remotecameras/{}'.format(data['camera_index']))

			#print "Camera ID:",data['camera_index']
			#detections = interface.attrs(atLocation=r'/root/detections/dark')
			#merged_detections = {}
			#for k,value in detections:
			#	if k not in merged_detections: merged_detections[k] = []
			#	while len(merged_detections[k]) <= data['camera_index']:
			#		merged_detections[k].append(None)
			#	merged_detections[k][data['camera_index']] = v
			#	interface.setAttr(k, merged_detections[k], r'/root/detections/dark')

			# detections[int(data['camera_index'])] = data['dark_detections']
			#for k,v in data['bright_detections'].iteritems():
			#	interface.setAttr(k, v, r'/root/detections/bright')

		self.q.task_done()
		del data


class ReceiveImages(Op.Op):
	def __init__(self, name='/ReceiveData', locations='', subPort=-1, topic='', type_='', host='localhost'):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('subPort', 'SUB port', 'SUB port', 'int', subPort, {'min': 1024 , 'max' : 65535}),
			('topic', 'Topic', 'Topic', 'string', topic, {}),
			('type', 'Type', 'Type', 'string', type_, {}),
			('host', 'Host', 'Host', 'string', host, {}),
		]

		super(self.__class__, self).__init__(name, fields)
		self.listener = None
		self.interface = None
		self.process_pool = None
		self.q = None
		self.wait = True

	def setup(self, interface, attrs):
		if self.listener is None:
			# Commented out due to adding min and max as an aurgument parar
			#if attrs['subPort'] <= 1024:
			#	self.logger.error('Port \'{}\' requires elevation'.format(attrs['subPort']))
			#	return

			if not len(attrs['topic']):
				self.logger.error("No topic to listen on specified")
				return

			# Get the interface for the processData item
			self.interface = interface

			self.listener = IO.Reframe.Listener()
			self.listener.addSubscription(IO.Reframe.tcp(attrs['host'], attrs['subPort']), self.processData, filters=[attrs['topic']])

			#self.process_pool = multiprocessing.Pool(processes=1)
			#manager = multiprocessing.Manager()
			#self.q = manager.Queue()
			#self.process_pool.map_async(listen_on_thread, ((attrs, self.q),))

			location, key = attrs['locations'].rsplit('/', 1)

	def processData(self, data):
		assert isinstance(data, dict)
		self.logger.debug("RECEIVED FRAME: {} | Current Frame: {}".format(data['frame'], self.interface.frame()))

	#@profile
	def cook(self, location, interface, attrs):
		self.listener.poll(timeout=1)
		return
		if self.process_pool is None:
			return

		if not self.wait:
			if self.q.empty(): return

		data = self.q.get()
		at_location = data['location']

		for k,v in data['data'].iteritems():
			interface.setAttr(k,v,at_location)

		interface.setAttr('cId', data['camera_index'], at_location)
		self.q.task_done()
		del data


last_frame = -1

def placeData(input_data, queue):
	while not queue.empty():
		queue.get_nowait()
		queue.task_done()

	queue.put(input_data)


def listen_on_thread((attrs, queue)):
	# Using partial to send in queue into the place data
	from functools import partial
	func = partial(placeData, queue=queue)

	listener = IO.Reframe.Listener()
	listener.addSubscription(IO.Reframe.tcp(attrs['host'], attrs['subPort']), func, filters=[attrs['topic']])
	listener.start()



# Register Ops
import Registry
Registry.registerOp('Export Data', Export)
Registry.registerOp('Import Data', Import)
Registry.registerOp('Recieve Data', Receive)
