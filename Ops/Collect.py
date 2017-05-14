import numpy as np
import Op, Interface
from IO import Reframe

class WandDetectionsAnd3Ds(Op.Op):
	def __init__(self, name='/Collect 2Ds And 3Ds', detections='', x3ds='', jumpFrames=20, camsWandSeen=2, showAll=True):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('detections', 'Detections', 'Detections', 'string', detections, {}),
			('x3ds', '3D Data', '3D Data', 'string', x3ds, {}),
			('jumpFrames', 'Jump Frames', 'Handle every Nth frame', 'int', jumpFrames, {}),
			('camsWandSeen', 'Wands in Cameras', 'Include if wand is seen in at least N number of cameras', 'int', camsWandSeen, {}),
			('showAll', 'Show all', 'Show all collected wand detections', 'bool', showAll, {})
		]

		super(self.__class__, self).__init__(name, fields)

	def cook(self, location, interface, attrs):
		# Grab 2D detections and 3D points
		detections_location = attrs['detections']
		x3ds_location = attrs['x3ds']
		if not detections_location or not x3ds_location: return
		frame = interface.frame()

		dets_x2ds = interface.attr('rx2ds', atLocation=detections_location)
		dets_splits = interface.attr('x2ds_splits', atLocation=detections_location)
		x3ds = interface.attr('x3ds', default=np.array([]), atLocation=x3ds_location)
		if dets_x2ds is None or len(dets_x2ds) < 2 or sum(dets_x2ds[1]) == 0 or not x3ds.any(): return
		dets_colours = interface.attr('x2ds_colours', atLocation=detections_location)

		# Check how many cameras we have and how many cameras see the wand in this frame
		numCameras = len(dets_splits) - 1
		numWandsInCameras = len(np.unique(dets_splits)) - 1

		# Gather already collected data
		collectedDets = interface.attr('rx2ds')
		collectedX3ds = interface.attr('x3ds_cams')
		lastFrame = interface.attr('lastFrame', frame)
		currentWands = interface.attr('numCollectedWandsInCameras', 0)
		goodCams = interface.attr('goodCams', [])

		jumpFrames = attrs['jumpFrames']
		camsWandSeen = attrs['camsWandSeen']
		emptyFrame3d = np.array([[]], dtype=np.float32).reshape(-1, 3)

		# Add to collection if it already exists
		if collectedDets is not None and collectedX3ds is not None:
			cams_detected = [dets_x2ds[c0:c1] for ci, (c0, c1) in enumerate(zip(dets_splits[:-1], dets_splits[1:]))]

			# Split the collections back into a list of cameras
			c_x2ds, c_splits = collectedDets
			cams_collected = [c_x2ds[c0:c1] for ci, (c0, c1) in enumerate(zip(c_splits[:-1], c_splits[1:]))]

			# Add 2D detections to cameras
			if len(cams_collected) != len(cams_detected): return
			cams = [np.concatenate((cc, cd), axis=0) for cc, cd in zip(cams_collected, cams_detected)]

			# Create a split version of the newly concatenated detections
			collectedDets = np.array(np.concatenate(cams), dtype=np.float32).reshape(-1, 2), Interface.makeSplitBoundaries(map(len, cams))

			# Check our new collection to see how many cameras are seeing the wand if we were to include
			# the 2D detections for this frame.
			c_x2ds2, c_splits2 = collectedDets
			numCollectedWandsInCameras = len(np.unique(c_splits2)) - 1

			# Decide whether or not to include the wand detections:
			#  - The wand detection is added if at least 'camsWandSeen' cameras see the wand.
			#  - Wand detections are added every 'jumpFrames' frames, regardless of how many cameras see the wand.
			if frame - lastFrame < jumpFrames: return # and numWandsInCameras < camsWandSeen: return

			# Log how many cameras are now seeing the wand (not used at the moment but leaving here if it becomes useful)
			interface.setAttr('numCollectedWandsInCameras', numCollectedWandsInCameras)

			# Add the corresponding 3D points for the wand
			for ci, (c0, c1) in enumerate(zip(dets_splits[:-1], dets_splits[1:])):
				if c1 - c0 > 0:
					collectedX3ds[ci] = np.append(collectedX3ds[ci], x3ds, axis=0)

			interface.setAttr('lastFrame', frame)

			if 'showAll' in attrs and attrs['showAll']:
				colours = np.tile(dets_colours, (len(collectedDets[0]) / 5, 1))
				allAttrs = {'x2ds': collectedDets[0], 'x2ds_splits': collectedDets[1],
				            'x2ds_colours': colours}
				interface.createChild('collected', 'detections', attrs=allAttrs)

		else:
			collectedDets = (dets_x2ds, dets_splits)
			collectedX3ds = []
			for ci, (c0, c1) in enumerate(zip(dets_splits[:-1], dets_splits[1:])):
				if c1 - c0 > 0:
					collectedX3ds.append(x3ds)
				else:
					collectedX3ds.append(emptyFrame3d)

			interface.setAttr('lastFrame', frame)
			interface.setAttr('numCollectedWandsInCameras', numWandsInCameras)

		# Set Collection
		interface.setAttr('rx2ds', collectedDets)
		interface.setAttr('x3ds_cams', collectedX3ds)
		interface.setAttr('x3ds', np.array(np.concatenate(collectedX3ds), dtype=np.float32).reshape(-1, 3))
		interface.setAttr('goodCams', goodCams)


class SendWandDetectionsAnd3Ds(Op.Op):
	def __init__(self, name='/Send 2Ds And 3Ds', detections='', x3ds='', pubPort=19002):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('detections', 'Detections', 'Detections', 'string', detections, {}),
			('x3ds', '3D Data', '3D Data', 'string', x3ds, {}),
			('pubPort', 'PUB port', 'PUB port', 'int', pubPort, {}),
		]

		super(self.__class__, self).__init__(name, fields)
		self.publisher = None

	def flush(self):
		self.publisher = None

	def cook(self, location, interface, attrs):
		if self.publisher is None:
			self.publisher = Reframe.Publisher(Reframe.tcpBind(attrs['pubPort']))

		# Grab 2D detections and 3D points
		detections_location = attrs['detections']
		x3ds_location = attrs['x3ds']
		if not detections_location or not x3ds_location: return

		dets_x2ds = interface.attr('rx2ds', atLocation=detections_location)
		dets_splits = interface.attr('x2ds_splits', atLocation=detections_location)
		x3ds = interface.attr('x3ds', default=np.array([]), atLocation=x3ds_location)
		if dets_x2ds is None or len(dets_x2ds) < 2 or sum(dets_x2ds[1]) == 0 or not x3ds.any(): return

		# Send data to a separate process
		data = {
			'rx2ds': dets_x2ds,
			'splits': dets_splits,
			'x3ds': x3ds,
			'frame': interface.frame()
		}
		self.publisher.publish(data, 'wand')


# class ReceiveCalibration(Op.Op):
# 	def __init__(self, name='/Receive calibration', locations='', subPort=19003):
# 		fields = [
# 			('name', 'Name', 'Name', 'string', name, {}),
# 			('locations', 'locations', 'locations', 'string', locations, {}),
# 			('subPort', 'SUB port', 'SUB port', 'int', subPort, {})
# 		]
#
# 		super(self.__class__, self).__init__(name, fields)
# 		self.listener = None
# 		self.interface = None # Cheeky hack for now
#
# 	def processCalibration(self, calibration):
# 		mats = calibration['mats']
# 		self.interface.setAttr('mats', mats)
#
# 	def cook(self, location, interface, attrs):
# 		if self.listener is None:
# 			self.listener = Reframe.Listener()
# 			self.listener.addSubscription(Reframe.tcp(attrs['subPort']), self.processCalibration, filters=['calibration'])
#
# 		self.interface = interface


class SkeletonJoints(Op.Op):
	def __init__(self, name='/Collect_Skeleton_Joints', locations='', frameRange='', exportRule='', exportPath=''):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('locations', 'Skeleton locations', 'locations', 'string', locations, {}),
			('frameRange', 'Frame range', 'Frame range', 'string', frameRange, {}),
			('exportRule', 'Export on frames', 'Export on frames', 'string', exportRule, {}),
			('exportPath', 'Export path', 'Export path', 'string', exportPath, {})
		]

		super(self.__class__, self).__init__(name, fields)
		self.jointChannelFrames = []

	def cook(self, location, interface, attrs):
		if not self.useFrame(interface.frame(), attrs['frameRange']): return
		skelDict = interface.attr('skelDict')
		if not skelDict: return

		self.jointChannelFrames.append(skelDict['chanValues'].copy())

		if attrs['exportRule'] and self.useFrame(interface.frame(), attrs['exportRule']):
			from IO import IO
			exportPath = self.resolvePath(attrs['exportPath'])
			IO.save(exportPath, self.jointChannelFrames)


# Register Ops
import Registry
Registry.registerOp('Collect 2D and 3D', WandDetectionsAnd3Ds)
