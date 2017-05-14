import math, logging
import numpy as np
from GCore import State
import Timecode

logging.basicConfig()


class Interface:
	def __init__(self):
		# Note: Probably best to remove this at some point and refactor
		self._win = None

		self.logger = logging.getLogger('Op Interface')
		self.logger.setLevel(logging.WARNING)

		self.cache_scenegraph = {}
		self.timeRange = None
		self.reset()

	def reset(self):
		self.path = ''
		self.opName = ''
		self.scenegraph = {}
		self._frame = 0
		self._lastFrame = 0 #1e10
		self._skipping = False

		self._cookedLocations, self._lastCookedLocations = [], []
		self._cookRegistry = {}
		self.clearCookedLocations()

	def resetFrame(self, frame):
		self._frame = frame
		self._lastFrame = frame

	def addFrameToCache(self):
		if self._frame in self.cache_scenegraph: return
		self.cache_scenegraph[self._frame] = {}
		for k, v in self.scenegraph.iteritems():
			self.cache_scenegraph[self._frame][k] = v.copy()

	def hasFrameCache(self, frame):
		if frame in self.cache_scenegraph and self.cache_scenegraph[frame]: return True
		return False

	def fetchFrameCache(self, frame):
		if not self.hasFrameCache(frame): return
		self._frame = frame
		self.scenegraph = self.cache_scenegraph[frame]
		self._cookedLocations = self.locations()

	def clearFrameCache(self, frame=None):
		if frame is None: frame = self.frame()
		self.cache_scenegraph[frame] = None

	def getCache(self):
		return self.cache_scenegraph

	def setCache(self, cache):
		self.cache_scenegraph = cache

	def clearCookedLocations(self):
		# Do we have anything other than the root?
		if len(self._cookedLocations) > 1: self._lastCookedLocations = self._cookedLocations
		self._cookedLocations = []

	def cookedLocations(self):
		return self._cookedLocations

	def lastCookedLocations(self):
		return self._lastCookedLocations

	def cookRegistry(self):
		return self._cookRegistry

	# This adds an op-cooked location relationship
	# Can also be used to avoid having to create and delete an attribute to define that
	# we want an op to affect a particular cooked location
	def registerCook(self, location):
		if location not in self._cookRegistry: self._cookRegistry[location] = []
		if self.opName not in self._cookRegistry[location]: self._cookRegistry[location].append(self.opName)
		# if self.opName not in self._cookRegistry: self._cookRegistry[self.opName] = []
		# if location not in self._cookRegistry[self.opName]: self._cookRegistry[self.opName].append(location)

	def setAttr(self, name, value, atLocation=None, forceCreate=True):
		if atLocation is None: atLocation = self.path
		if atLocation not in self.scenegraph and not forceCreate: return

		if atLocation not in self.scenegraph:
			self.createChild(None, 'group', atLocation, attrs={name: value})
		else:
			self.scenegraph[atLocation][name] = value

		if atLocation not in self._cookedLocations: self._cookedLocations.append(atLocation)
		self.registerCook(atLocation)

	def hasAttr(self, name, atLocation=None):
		if atLocation is None: atLocation = self.path
		if atLocation not in self.scenegraph: return False
		location = self.scenegraph[atLocation]
		if not location: return False
		if name in location: return True
		return False

	def attr(self, name, default=None, atLocation=None, onlyDirty=False, log=False, deep=False):
		if atLocation is None: atLocation = self.path
		if atLocation not in self.scenegraph: return default

		if onlyDirty and not self._isLocationDirty(atLocation, name):
			return default

		location = self.scenegraph[atLocation]
		if not location:
			if log: self.logger.warn('Location %s not found' % atLocation)
			return default

		if name not in location:
			if log: self.logger.warn('No attribute named %s found at %s' % (name, atLocation))
			return default

		# The queried (incoming) attributes should be const/read only. Modifying it must be done using setAttr
		# Note: How can we get rid of this nasty deepcopy?
		from copy import deepcopy
		if deep:
			return deepcopy(location[name])
		else:
			return location[name]

	# Note: If we use attrs={}, i.e. with a default value, Python will use it as a global variable
	def createChild(self, name, type, atLocation=None, attrs=None):
		if atLocation is None: atLocation = self.path
		if atLocation not in self.scenegraph:
			self.scenegraph[atLocation] = {
				'type': 'group'
			}

		if not name:
			location = atLocation
		else:
			location = '/'.join([atLocation, name])

		if location not in self._cookedLocations: self._cookedLocations.append(location)
		self.registerCook(atLocation)

		if attrs is None: self.scenegraph[location] = {}
		else: self.scenegraph[location] = attrs
		self.scenegraph[location]['type'] = type
		State.setKey('/interface' + location, {'dirty': True})
		return self.scenegraph[location]

	def getChild(self, name, parent=None):
		if parent is None: parent = self.path
		childName = '/'.join([parent, name])
		if childName in self.scenegraph: return self.scenegraph[childName]
		return None

	def deleteChild(self, location):
		if location in self.scenegraph:
			del self.scenegraph[location]
			self.registerCook(location)

	def deleteLocationsByName(self, locationPrefix):
		for loc in self.scenegraph.keys():
			if loc.startswith(locationPrefix):
				del self.scenegraph[loc]

	def setType(self, type):
		if self.path not in self.scenegraph:
			self.createChild(None, type, self.path)
		else:
			location = self.scenegraph[self.path]
			location['type'] = type

	def scenegraphLocation(self, locationFullName):
		return self.location(locationFullName)

	def scenegraphLocations(self):
		return self.locations()

	def locations(self):
		return self.scenegraph.keys()

	def location(self, locationFullName):
		if locationFullName not in self.scenegraph: return None
		return self.scenegraph[locationFullName]

	def copyLocation(self, sourceLoc, targetLoc, deep=True, asType=None):
		if sourceLoc not in self.scenegraph: return
		from copy import deepcopy
		if deep:
			self.scenegraph[targetLoc] = deepcopy(self.scenegraph[sourceLoc])
		else:
			self.scenegraph[targetLoc] = self.scenegraph[sourceLoc]

		if asType:
			self.scenegraph[targetLoc]['type'] = asType

	def attrs(self, atLocation=None):
		if atLocation is None: atLocation = self.path
		sgLoc = self.scenegraphLocation(atLocation)
		if not sgLoc: return {}
		return sgLoc

	def attrNames(self, atLocation=None):
		return self.attrKeys(atLocation)

	def attrKeys(self, atLocation=None):
		attrs = self.attrs(atLocation)
		if not attrs: return []
		return attrs.keys()

	def fullName(self):
		return self.path

	def name(self):
		return self.path.split('/')[-1]

	def parentPath(self):
		return '/'.join(self.path.split('/')[:-1])

	def setLocation(self, location):
		self.path = location

	def appendLocation(self, location):
		self.path = '/'.join([self.path, location])

	def setOpName(self, opName):
		self.opName = opName

	def setWin(self, win):
		self._win = win

	def win(self):
		return self._win

	def setFrame(self, frame):
		self._frame = frame

	def frame(self):
		return self._frame

	def lastFrame(self):
		return self._lastFrame

	def _setLastFrame(self, lastFrame):
		self._lastFrame = lastFrame

	def skipping(self):
		return self._skipping

	def _setSkipping(self, skipping):
		self._skipping = skipping

	def isDirty(self, location=None, dirtyKeys=None):
		if location is None: location = self.path
		return self._isLocationDirty(location, dirtyKeys)

	def areLocationsDirty(self, locations):
		dirtyKeys = State.g_dirty
		for location in locations:
			if self._isLocationDirty(location, dirtyKeys):
				return True

		return False

	def _isLocationDirty(self, location, dirtyKeys=None):
		if dirtyKeys is None: dirtyKeys = State.g_dirty
		if location in dirtyKeys: return True
		for key in dirtyKeys:
			if key.startswith(location):
				return True

		return False

	def opParamsDirty(self):
		dirtyKeys = State.g_dirty
		opParamsPrefix = self.opName + '/attrs'
		if opParamsPrefix in dirtyKeys: return True
		for key in dirtyKeys:
			if key.startswith(opParamsPrefix):
				return True

		return False

	def setFrameRange(self, start, end):
		self.timeRange = (start, end)

	def fps(self):
		return math.ceil(self._win.qtimeline.fps)

	def updateTimeline(self):
		if self.timeRange is not None and self._win:
			self._win.qtimeline.setRange(*self.timeRange)
			self.timeRange = None

	def printInfo(self, atLocation=None):
		print '#### Cook:', self.path
		print '## Locations:', self.scenegraphLocations()
		print '## Attrs:', self.attrs(atLocation=atLocation).keys()

	def root(self):
		return '/root'

	def getPsFromMats(self, mats):
		return np.array([m[2] / (np.sum(m[2][0, :3] ** 2) ** 0.5) for m in mats], dtype=np.float32)

	def expandLocation(self, location):
		# This could interpret an expression language to process /*{attr('tx') == 0. etc.
		# but at the moment it just makes sure the location starts with '/'
		if not location.startswith('/'): return '/' + location
		return location

	def splitLocations(self, locations):
		splitLocations = locations.split(' /')
		for li, loc in enumerate(splitLocations):
			splitLocations[li] = self.expandLocation(loc)

		return splitLocations

	def findCameraIdFromRayId(self, rayId, camRaySplits):
		dists = rayId - camRaySplits[:-1]
		dists[np.where(dists < 0)[0]] = np.sum(camRaySplits)

		# There could be more than one value if some cameras have no detections.
		# Get all entries with the minimum value and return the last one.
		return np.where(dists == np.min(dists))[0][-1]

	def getLabelColours(self, labels, defaultColour=(1, 0, 0, 0.7)):
		labelColours = np.tile(np.array(defaultColour, dtype=np.float32), (labels.shape[0], 1))
		labelled = np.where(labels != -1)[0]
		if labelled.any():
			labelColours[labelled, :] = np.array([0.0, 0.7, 0.0, 0.7], dtype=np.float32)
			return labelColours

		return np.array([])

	def getTimecodeSync(self, timecode, tcAttrName, attrs, fps, timecodeFps, timecodeMultiplier, offset=0):
		tcSyncTime = None
		if timecode is not None and tcAttrName in attrs and attrs[tcAttrName]:
			tcSyncTime = self.attr('timecode', atLocation=attrs[tcAttrName])
			if tcSyncTime is not None:
				tcSyncValue = Timecode.TCFtoInt(tcSyncTime, fps)
				try:
					diff = Timecode.TCSub(tcSyncTime, timecode, timecodeFps)
					offset += Timecode.TCFtoInt(diff, timecodeFps) * timecodeMultiplier
				except Exception as e:
					self.logger.error('Error calculating timecode difference: %s' % str(e))
					return tcSyncTime, -1
					# import traceback
					# traceback.print_exc()

		return tcSyncTime, offset


def getLabelNames(labels):
	return [str(l) for l in labels]

def translate(RT, v): RT[:,3] += np.dot(RT[:,:3], v)
def rotateX(RT, v): cv, sv=math.cos(v), math.sin(v); RT[:,1], RT[:,2] = RT[:,1]*cv+RT[:,2]*sv, RT[:,2]*cv-RT[:,1]*sv
def rotateY(RT, v): cv, sv=math.cos(v), math.sin(v); RT[:,2], RT[:,0] = RT[:,2]*cv+RT[:,0]*sv, RT[:,0]*cv-RT[:,2]*sv
def rotateZ(RT, v): cv, sv=math.cos(v), math.sin(v); RT[:,0], RT[:,1] = RT[:,0]*cv+RT[:,1]*sv, RT[:,1]*cv-RT[:,0]*sv

def makeSplitBoundaries(lengths):
	return np.array([sum(lengths[:x]) for x in xrange(len(lengths) + 1)], dtype=np.int32)

def getWorldSpaceMarkerPos(skelDict):
	vs = []
	lbls = []
	for mi in range(skelDict['numMarkers']):
		parentJointGs = np.append(skelDict['Gs'][skelDict['markerParents'][mi]], [[0, 0, 0, 1]], axis=0)
		mOffset = skelDict['markerOffsets'][mi]
		mOffset = np.array([[mOffset[0], mOffset[1], mOffset[2], 1]], dtype=np.float32)

		v = np.dot(parentJointGs, mOffset.T)
		vs.append(np.concatenate(v[:3]))
		lbls.append(skelDict['markerNames'][mi])

	vs = np.array(vs, dtype=np.float32)
	return vs, lbls

