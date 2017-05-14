import math

import numpy as np

import IO
import Interface
import Op
from GCore import Character, SolveIK
from IO import ViconReader, ASFReader
# from Ops import FBXReader


class VSS(Op.Op):
	def __init__(self, name='/Skeleton', locations='', vssFilename='', boneColour=(0.3, 0.42, 0.66, 1.), blockedChannels=''):
		fields = [
			('name', 'name', 'Skeleton name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('vss', 'VSS', 'Solving Skeleton filename', 'filename', vssFilename, {}),
			('boneColour', 'Bone colour', 'Bone colour', 'string', str(boneColour), {}),
			('blockedChannels', 'Blocked channels', 'Blocked channels', 'string', blockedChannels, {})
		]

		super(self.__class__, self).__init__(name, fields)
		self.skelDict = None

	def getSkeleton(self):
		return self.skelDict

	def flush(self):
		self.skelDict = None

	def setup(self, interface, attrs):
		if self.skelDict:
			# Note: Since dirty state is not properly sorted yet we just leave if we have already loaded a skeleton and assume
			#       we want to keep the one in the interface
			return

		if 'vss' in attrs:
			vssFilename = self.resolvePath(attrs['vss'])
			if not vssFilename: return

			try:
				self.skelDict = ViconReader.loadVSS(vssFilename)
			except IOError as e:
				self.logger.error('Could not load skeleton: %s' % str(e))
				return False

			if 'rootMat' not in self.skelDict:
				self.skelDict['rootMat'] = getRootMat()

			if attrs['blockedChannels']:
				blockedChannelNames = attrs['blockedChannels'].split()
				allowedChannels = []
				for ci, channelName in enumerate(self.skelDict['chanNames']):
					if channelName not in blockedChannelNames:
						allowedChannels.append(ci)

				self.skelDict = makeRestrictedSkeleton(self.skelDict, allowedChannels)

			poseSkeleton(self.skelDict)

	def cook(self, location, interface, attrs):
		if self.skelDict is None: return
		skelAttrs = {
			'skelDict': self.skelDict,
			'rootMat': self.skelDict['rootMat'],
			'originalRootMat': self.skelDict['rootMat'],
			'subjectName': self.getName(),
			'boneColour': eval(attrs['boneColour'])
		}
		interface.createChild(interface.name(), 'skeleton', atLocation=interface.parentPath(), attrs=skelAttrs)


class ASF(Op.Op):
	def __init__(self, name='/Skeleton_ASF', locations='', asfFilename='', amcFilename='', offset=0, step=1,
	             boneColour=(0.3, 0.42, 0.66, 1.), allFrames=False, useAnimation=True):
		fields = [
			('name', 'name', 'Skeleton name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('asf', 'ASF', 'Skeleton filename', 'filename', asfFilename, {}),
			('amc', 'AMC', 'Animation filename', 'filename', amcFilename, {}),
			('offset', 'Frame offset', 'Frame offset', 'int', offset, {}),
			('step', 'Frame step', 'Frame step', 'int', step, {}),
			('boneColour', 'Bone colour', 'Bone colour', 'string', str(boneColour), {}),
			('allFrames', 'All frames', 'Make all frames accessible', 'bool', allFrames, {}),
			('useAnimation', 'Use animation', 'Use animation if available', 'bool', useAnimation, {})
		]

		super(self.__class__, self).__init__(name, fields)
		self.flush()
		self.firstFrame, self.lastFrame = 0, 0
		self.cacheManualOverride = True

	def flush(self):
		self.skelDict = None
		self.animDict = None
		self.initialRootMat = None
    
	def setup(self, interface, attrs):
		if self.skelDict: return
		asfFilename = attrs['asf']

		try:
			self.asfDict = ASFReader.read_ASF(asfFilename) #x
			self.skelDict = ASFReader.asfDict_to_skelDict(self.asfDict) #x
		except Exception as e:
			self.logger.error('Unable to load skeleton: %s' % e)
			return

		try:
			if attrs['useAnimation']:
				amcFilename = attrs['amc']
				if not amcFilename: amcFilename = asfFilename.replace('.asf', '.amc')
				self.animDict = ASFReader.read_AMC(amcFilename, self.asfDict) #x

			self.firstFrame = self.animDict['frameNumbers'][0] + attrs['offset']
			self.lastFrame = len(self.animDict['frameNumbers'])
		except Exception as e:
			self.logger.warning('Unable to populate skeleton: %s' % e)

		if 'rootMat' not in self.skelDict:
			self.skelDict['rootMat'] = getRootMat()

		self.initialRootMat = self.skelDict['rootMat'].copy()

	def cook(self, location, interface, attrs):
		if self.skelDict is None: return
		interface.setType('skeleton')

		# TODO: Add timecode support
		frameIdx = max(interface.frame() * attrs['step'] + attrs['offset'], 0)
		if frameIdx < 0: return

		if self.animDict is not None:
			if self.animDict['frameNumbers'][0] <= frameIdx < self.animDict['frameNumbers'][-1]:
				self.skelDict['chanValues'] = self.animDict['dofData'][frameIdx]

		poseSkeleton(self.skelDict)

		skelAttrs = {
			'skelDict': self.skelDict,
			'Gs': self.skelDict['Gs'],
			'rootMat': self.skelDict['rootMat'],
			'originalRootMat': self.initialRootMat,
			'subjectName': self.getName(),
			'boneColour': eval(attrs['boneColour']),
			'frameRange': [self.firstFrame, self.lastFrame],
			'frame': frameIdx
		}

		if attrs['allFrames'] and self.animDict is not None:
			skelAttrs['animDict'] = self.animDict

		interface.createChild(interface.name(), 'skeleton', atLocation=interface.parentPath(), attrs=skelAttrs)

class FBX(Op.Op):
	def __init__(self, name='/Skeleton_FBX', locations='', fbxFilename='', offset=0, step=1,
				 boneColour=(0.3, 0.42, 0.66, 1.), allFrames=False, useAnimation=True):
		fields = [
			('name', 'name', 'Skeleton name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('fbx', 'FBX', 'FBX filename', 'filename', fbxFilename, {}),
			('offset', 'Frame offset', 'Frame offset', 'int', offset, {}),
			('step', 'Frame step', 'Frame step', 'int', step, {}),
			('boneColour', 'Bone colour', 'Bone Colour', 'string', str(boneColour), {}),
			('allFrames', 'All frames', 'Make all frames accessible', 'bool', allFrames, {}),
			('useAnimation', 'Use animation', 'Use animation if available', 'bool', useAnimation, {})
		]

		super(self.__class__, self).__init__(name, fields)
		self.flush()
		self.firstFrame, self.lastFrame = 0, 0
		self.cacheManualOverride = True

	def flush(self):
		self.skelDict = None
		self.animDict = None
		self.initialRootMat = None

	def setup(self, interface, attrs):
		if self.skelDict: return
		fbxFilename = attrs['fbx']

		try:
			self.skelDict, self.animDict = FBXReader.readFile(fbxFilename)
		except Exception as e:
			self.logger.error('Unable to load skeleton: %s' % e)
			return

		self.initialRootMat = self.skelDict['rootMat'].copy()

	def cook(self, location, interface, attrs):
		if self.skelDict is None: return
		interface.setType('skeleton')

		frameIdx = max(interface.frame() * attrs['step'] + attrs['offset'], 0)
		if frameIdx < 0: return

		if self.animDict is not None:
			if frameIdx in self.animDict:
				self.skelDict['Gs'] = self.animDict[frameIdx]

		self.skelDict['Bs'][0][0] = self.skelDict['Gs'][2][:,3]

		skelAttrs = {
			'skelDict': self.skelDict,
			# 'Gs': self.skelDict['Gs'],
			'rootMat': self.skelDict['rootMat'],
			'originalRootMat': self.initialRootMat,
			'subjectName': self.getName(),
			'boneColour': eval(attrs['boneColour']),
			'frameRange': [self.firstFrame, self.lastFrame],
			'frame': frameIdx
		}

		if attrs['allFrames'] and self.animDict is not None:
			skelAttrs['animDict'] = self.animDict

		interface.createChild(interface.name(), 'skeleton', atLocation=interface.parentPath(), attrs=skelAttrs)


class Template(Op.Op):
	def __init__(self, name='/Skeleton', locations='', sklFilename='', boneColour=(0.3, 0.42, 0.66, 1.), retainDownstreamChanges=True,
	             lockOriginal=True, useCurrentLabel=False, useAnimation=True, offset=0, stepSize=1, blockedChannels='',
				 reset=False):
		fields = [
			('name', 'name', 'Skeleton name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('skl', 'SKL', 'Template Skeleton filename', 'filename', sklFilename, {}),
			('boneColour', 'Bone colour', 'Bone colour', 'string', str(boneColour), {}),
			('retainDownstreamChanges', 'Retain downstream changes', 'Retain downstream changes', 'bool', retainDownstreamChanges, {}),
			('lockOriginal', 'Lock original', 'Lock original', 'bool', lockOriginal, {}),
			('useCurrentLabel', 'Use labels as current', 'Use labels as current counter', 'bool', useCurrentLabel, {}),
			('useAnimation', 'Use animation', 'Use animation (.sam) if available', 'bool', useAnimation, {}),
			('offset', 'Frame offset', 'Frame offset', 'int', offset, {}),
			('stepSize', 'Frame step size', 'Frame step size', 'int', stepSize, {}),
			('blockedChannels', 'Blocked channels', 'Blocked channels', 'string', blockedChannels, {}),
			('reset', 'reset', 'reset', 'bool', reset, {})
		]

		super(self.__class__, self).__init__(name, fields)

		self.cacheManualOverride = True
		self.skelDict = None
		self.initialSkelDict = None
		self.initialRootMat = None
		self.animDict = None
		self.useAnimation = False
		self.initialNormals = np.float32([])

	def flush(self):
		self.skelDict = None
		self.initialSkelDict = None
		self.initialRootMat = None
		self.animDict = None

	def getSkeleton(self):
		return self.skelDict

	def setup(self, interface, attrs):
		# Note: This condition needs to go away and instead we only rely on the cache and dirty states (even if frames change)
		#       as we need to respond to e.g. changing the skeleton source
		retainDownstreamChanges = attrs['retainDownstreamChanges'] if 'retainDownstreamChanges' in attrs else True
		if self.skelDict is not None:
			# if retainDownstreamChanges:
			# 	self.skelDict = interface.attr('skelDict')
			# 	if 'lockOriginal' in attrs and not attrs['lockOriginal']:
			# 		self.initialSkelDict = self.skelDict
			# else:
			# 	self.skelDict = self.initialSkelDict

			return

		if 'skl' in attrs and attrs['skl']:
			sklFilename = self.resolvePath(attrs['skl'])
			if not sklFilename: return

			import IO, os
			try:
				_, (self.skelDict) = IO.load(sklFilename)
			except Exception as e:
				self.logger.error('Could not open skeleton: %s' % sklFilename)
				return

			if 'useAnimation' in attrs and attrs['useAnimation']:
				self.useAnimation = True
				animFilename = sklFilename.replace('.skl', '.sam')
				if os.path.isfile(animFilename):
					_, self.animDict = IO.load(animFilename)

			if attrs['blockedChannels']:
				blockedChannelNames = attrs['blockedChannels'].split()
				# Test: Remove certain channels such as T DOFs for the feet
				# blockedChannelNames = [
				# 	'VSS_LeftFoot:tx', 'VSS_LeftFoot:ty', 'VSS_LeftFoot:tz',
				# 	'VSS_RightFoot:tx', 'VSS_RightFoot:ty', 'VSS_RightFoot:tz',
					# 'VSS_LeftShoulderFree:tx', 'VSS_LeftShoulderFree:ty', 'VSS_LeftShoulderFree:tz',
					# 'VSS_RightShoulderFree:tx', 'VSS_RightShoulderFree:ty', 'VSS_RightShoulderFree:tz'
				# ]
				allowedChannels = []
				for ci, channelName in enumerate(self.skelDict['chanNames']):
					if channelName not in blockedChannelNames:
						allowedChannels.append(ci)

				self.skelDict = makeRestrictedSkeleton(self.skelDict, allowedChannels)

			if 'rootMat' not in self.skelDict:
				self.skelDict['rootMat'] = getRootMat()

			self.initialSkelDict = self.skelDict.copy()
			self.initialRootMat = self.skelDict['rootMat'].copy()
			if 'markerNormals' in self.skelDict:
				self.initialNormals = self.skelDict['markerNormals'].copy()

	def cook(self, location, interface, attrs):
		if self.skelDict is None: return

		if attrs['reset']:
			self.skelDict['chanValues'] = self.initialSkelDict['chanValues']
			poseSkeleton(self.skelDict)
			return

		frameIdx = max(interface.frame() * attrs['stepSize'] + attrs['offset'], 0)
		if frameIdx < 0: return

		# if self.animDict['frameNumbers'][0] <= frameIdx < self.animDict['frameNumbers'][-1]:
		if self.useAnimation and self.animDict is not None and frameIdx < len(self.animDict):
			self.skelDict['chanValues'] = self.animDict[frameIdx]
			poseSkeleton(self.skelDict)

		skelAttrs = {
			'skelDict': self.skelDict,
			'Gs': self.skelDict['Gs'],
			'rootMat': self.skelDict['rootMat'],
			'originalRootMat': self.initialRootMat,
			'subjectName': self.getName(),
			'boneColour': eval(attrs['boneColour']),
			'originalNormals': self.initialNormals
		}
		interface.createChild(interface.name(), 'skeleton', atLocation=interface.parentPath(), attrs=skelAttrs)

		if interface.hasAttr('currentLabel', atLocation='/root') and 'useCurrentLabel' in attrs and attrs['useCurrentLabel']:
			try:
				maxLabel = np.max(np.unique(self.skelDict['markerNames']).astype(int))
				interface.setAttr('currentLabel', maxLabel + 1, atLocation='/root')
			except:
				pass


class Transform(Op.Op):
	def __init__(self, name='/Skeleton Transform', locations='', tx=0., ty=0., tz=0., rx=0., ry=0., rz=0.):
		self.fields = [
			('name', 'name', 'Skeleton name', 'string', name, {}),
			('locations', 'locations', 'Skeleton locations', 'string', locations, {}),
			('tx', 'tx', 'local space translation', 'float', tx),
			('ty', 'ty', 'local space translation', 'float', ty),
			('tz', 'tz', 'local space translation', 'float', tz),
			('rx', 'rx', 'local space rotation', 'float', rx),
			('ry', 'ry', 'local space rotation', 'float', ry),
			('rz', 'rz', 'local space rotation', 'float', rz)
		]

		super(self.__class__, self).__init__(name, self.fields)

		self.locations = locations
		self.skeletonFields = False

	def flush(self):
		self.skeletonFields = False

	def _getValue(self, attrName, skelAttrs, location, interface):
		if attrName in skelAttrs: return skelAttrs[attrName]
		value = interface.attr(attrName, atLocation=location)
		if value: return value
		return 0.

	def _updateFields(self, win, skelDict, attrs):
		for aName, aValue in attrs.iteritems():
			self.attrs[aName] = aValue

		# TODO: This should be formalised and moved to runtime behaviour. Ops should just be providing the information
		#       and flag that changes are needed
		for chanName, value in zip(skelDict['chanNames'], skelDict['chanValues']):
			if ':r' in chanName:# and not 'Free' in chanName:
				self.fields.append((chanName, chanName, '', 'float', float(value)))
				self.attrs[chanName] = 0 #float(value)

		self.registerState(win, False)
		self.setAttrs(self.attrs)
		self.skeletonFields = True

	def cook(self, location, interface, attrs):
		#  Check if we have anything to do
		# if not attrs and not interface.isDirty(): return

		# Look for a skeleton dictionary in the interface
		skelDict = interface.attr('skelDict', deep=True)
		if not skelDict: return
		interface.registerCook(location)

		# Update the fields based on the skeleton if it's the first time or if the
		# target changes (not really supported yet)
		if not self.skeletonFields: # or 'locations' in attrs:
			self._updateFields(interface.win(), skelDict, attrs)

		# Get the attributes for the corresponding skeleton (by name)
		for attrName, value in attrs.iteritems():
			if attrName in skelDict['chanNames']:
				chanIndex = skelDict['chanNames'].index(attrName)
				skelDict['chanValues'][chanIndex] += value

		# TODO: Get rid of this shite.. what's wrong with root:xx?
		t = [self._getValue('tx', attrs, location, interface), self._getValue('ty', attrs, location, interface),
			 self._getValue('tz', attrs, location, interface)]
		r = [self._getValue('rx', attrs, location, interface), self._getValue('ry', attrs, location, interface),
			 self._getValue('rz', attrs, location, interface)]

		for aName, aValue in zip(['tx', 'ty', 'tz'], t):
			interface.setAttr(aName, aValue)

		for aName, aValue in zip(['rx', 'ry', 'rz'], r):
			interface.setAttr(aName, aValue)

		priorRootMat = interface.attr('originalRootMat', deep=True)
		rootMat = getRootMat(priorRootMat, t=t, r=r)

		# Get the root changes (this should ideally go away in favour of using the root:[attr]
		# Note: This will override any other transforms at the moment
		skelDict['rootMat'] = rootMat

		# Pose the skeleton and update the corresponding skeleton layer
		poseSkeleton(skelDict)

		# Update normals if we have any (make more efficient)
		if 'markerNormals' in skelDict:
			normals = skelDict['markerNormals']
			for ni, (parent, normal) in enumerate(zip(skelDict['markerParents'], normals)):
				Gs = skelDict['Gs'][parent]
				normals[ni] = np.dot(Gs[:3, :3], normal)

		interface.setAttr('skelDict', skelDict)
		interface.setAttr('Gs', skelDict['Gs'])
		interface.setAttr('rootMat', rootMat)


class Configure(Op.Op):
	def __init__(self, name='/Skeleton Configure', locations='', subjectName='', glMarkerColour=True, createStickPairs=True):
		self.fields = [
			('name', 'name', 'Skeleton name', 'string', name, {}),
			('locations', 'locations', 'Skeleton locations', 'string', locations, {}),
			('subjectName', 'subject', 'Subject name', 'string', subjectName, {}),
			('glMarkerColour', 'glMarkerColour', 'Create GL marker colour', 'bool', glMarkerColour, {}),
			('createStickPairs', 'stickPairs', 'Create stick pairs', 'bool', createStickPairs, {})
		]

		super(self.__class__, self).__init__(name, self.fields)

		self.locations = locations
		self.skeletonFields = False

	def getOpenGlColour(self, colour):
		colourWithAlpha = np.append(colour, 128)  # add Alpha
		glColour = colourWithAlpha.astype(np.float32) / 255.0  # Convert to 0.0~1.0
		return glColour.tolist()

	def configureSkeleton(self, skelDict, subjectName='', glMarkerColour=True, createStickPairs=True):
		if glMarkerColour and 'markerColourGL' not in skelDict and 'markerColour' in skelDict:
			# Convert 8-Bit RGB Colours into 4f for OpenGL
			skelDict['markerColourGL'] = []
			for colour in skelDict['markerColour']:
				skelDict['markerColourGL'].append(self.getOpenGlColour(colour))

		if createStickPairs and 'stickPairs' not in skelDict and 'markerNames' in skelDict:
			# Create stick pairs
			stickPairs = []
			for colour, (leftStick, rightStick) in zip(skelDict['sticksColour'], skelDict['sticks']):
				# A->B and B->A
				li, ri = skelDict['markerNames'].index(leftStick), skelDict['markerNames'].index(rightStick)
				stickPairs.append((li, ri, self.getOpenGlColour(colour)))

			skelDict['stickPairs'] = stickPairs

		if subjectName and 'markerIdxLUT' not in skelDict and 'markerNamesUnq' in skelDict:
			# Make a lut between namespace:markerName and index in the markerColour list
			LUT = {}
			for ni, name in enumerate(skelDict['markerNamesUnq']):
				LUT[subjectName + ':' + name] = ni

			skelDict['markerIdxLUT'] = LUT

			# Add Namespace for matching in C3D
			skelDict['markerNames'] = [subjectName + ':' + n for n in skelDict['markerNames']]

		skelDict['labelNames'] = np.arange(len(skelDict['markerNames']))

	def cook(self, location, interface, attrs):
		#  Check if we have anything to do
		if not attrs and not interface.isDirty(): return

		# Look for a skeleton dictionary in the interface
		skelDict = interface.attr('skelDict')
		if not skelDict: return

		subjectName = attrs['subjectName']
		glMarkerColour = attrs['glMarkerColour']
		createStickPairs = attrs['createStickPairs']
		self.configureSkeleton(skelDict, subjectName, glMarkerColour, createStickPairs)

		interface.setAttr('skelDict', skelDict)
		interface.setAttr('subjectName', subjectName)


class JointCopy(Op.Op):
	def __init__(self, name='/Joint Copy', locations='', sourceSkeleton='', frameRange='', copyRootMat=True, targetPrefix=''):
		self.fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'locations', 'Skeleton locations', 'string', locations, {}),
			('source', 'source', 'Source skeleton', 'string', sourceSkeleton, {}),
			('frameRange', 'Frame range', 'Frame range', 'string', frameRange, {}),
			('copyRootMat', 'Copy root mat', 'Copy root mat', 'bool', copyRootMat, {}),
			('targetPrefix', 'Target prefix', 'Target prefix', 'string', targetPrefix, {})
		]

		super(self.__class__, self).__init__(name, self.fields)

	def cook(self, location, interface, attrs):
		if not self.useFrame(interface.frame(), attrs['frameRange']): return
		if not location or not attrs['source']: return

		# Get the source skeleton dict
		sourceLocation = attrs['source']
		skelDict_source = interface.attr('skelDict', atLocation=sourceLocation)
		if not skelDict_source: return

		# Get the target skeleton dict (the location we are cooking)
		skelDict_target = interface.attr('skelDict')
		if not skelDict_target: return

		if attrs['copyRootMat']:
			skelDict_target['rootMat'] = skelDict_source['rootMat']

		prefix = attrs['targetPrefix'] if 'targetPrefix' in attrs else ''

		# TODO: Hard-coded exclusions (e.g. Hand) should not be here (perhaps add a blacklist)
		for ci, (cv, cn) in enumerate(zip(skelDict_source['chanValues'], skelDict_source['chanNames'])):
			# if "Hand" in cn: continue
			# if 'Leg' in cn or 'Foot' in cn: continue
			if 'root' in cn and not attrs['copyRootMat']: continue
			if cn in skelDict_target['chanNames']:
				idx = skelDict_target['chanNames'].index(cn)
				skelDict_target['chanValues'][idx] = cv
			elif prefix + cn in skelDict_target['chanNames']:
				idx = skelDict_target['chanNames'].index(prefix + cn)
				skelDict_target['chanValues'][idx] = cv

		poseSkeleton(skelDict_target)
		interface.setAttr('skelDict', skelDict_target)


class ResetJoint(Op.Op):
	def __init__(self, name='/Reset Joint', locations='', jointNames='', frameRange=''):
		self.fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'locations', 'Skeleton locations', 'string', locations, {}),
			('jointNames', 'Joint names', 'Joint names (* for all joints)', 'string', jointNames, {}),
			('frameRange', 'Frame range', 'Frame range', 'string', frameRange, {})
		]

		super(self.__class__, self).__init__(name, self.fields)

	def cook(self, location, interface, attrs):
		# Note: For simplicity (for the time being) we only zero joints
		if not self.useFrame(interface.frame(), attrs['frameRange']): return

		# Get the source skeleton dict
		skelDict = interface.attr('skelDict')
		if not skelDict: return

		# Get the target skeleton dict (the location we are cooking)
		jointNames = self.resolveLocations(attrs['jointNames'], prefix='') if attrs['jointNames'] else []
		if not jointNames: return

		for ci, (cv, cn) in enumerate(zip(skelDict['chanValues'], skelDict['chanNames'])):
			for jname in jointNames:
				if jname in cn or jname == '*':
					idx = skelDict['chanNames'].index(cn)
					skelDict['chanValues'][idx] = 0.

		poseSkeleton(skelDict)
		interface.setAttr('skelDict', skelDict)


class GeometryCopy(Op.Op):
	def __init__(self, name='/Geometry Copy', locations='', sourceSkeleton=''):
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

		for k in ('geom_Vs', 'geom_vsplits', 'geom_Gs'): skelDict_target[k] = skelDict_source[k].copy()
		skelDict_target['shape_weights'] = skelDict_source['shape_weights']
		skelDict_target['geom_dict'] = skelDict_source['geom_dict']

		nameMapping = {}
		for jointName in skelDict_source['jointNames']:
			original_source_ji = skelDict_source['jointNames'].index(jointName)
			target_jointName = jointName
			source_ji = original_source_ji
			found = False

			while not found:
				try:
					skelDict_target['jointNames'].index(target_jointName)
					nameMapping[jointName] = target_jointName
					found = True
				except ValueError:
					# Joint not in target, so find the parent.
					source_ji = skelDict_source['jointParents'][source_ji]
					target_jointName = skelDict_source['jointNames'][source_ji]
					continue
		skelDict_target['shape_weights'] = Character.shape_weights_mapping(skelDict_source, skelDict_target, nameMapping)

		interface.setAttr('skelDict', skelDict_target)


class SetMarkerJointMapping(Op.Op):
	def __init__(self, name='/Set Marker Joint Mapping', locations='', useAllWeights=False, x3d_threshold=140.0,
	             allowPartialMatching=False, sourceX3ds='', frameRange=''):
		self.fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'locations', 'Skeleton locations', 'string', locations, {}),
			('useAllWeights', 'Use all weights', 'Use all weights', 'bool', useAllWeights, {}),
			('x3d_threshold', '3D threshold', '3D threshold', 'float', x3d_threshold, {}),
			('allowPartialMatching', 'Allow partial matching', 'Allow partial matching', 'bool', allowPartialMatching, {}),
			('sourceX3ds', 'Source 3Ds', 'Source 3Ds (optional)', 'string', sourceX3ds, {}),
			('frameRange', 'Frame range', 'Frame range', 'string', frameRange, {})
		]

		super(self.__class__, self).__init__(name, self.fields)

	def cook(self, location, interface, attrs):
		if not self.useFrame(interface.frame(), attrs['frameRange']): return
		skelDict = interface.attr('skelDict')
		if not skelDict: return

		# Get the world coordinates for the markers
		x3dsLoc = attrs['sourceX3ds']
		if x3dsLoc:
			vs = interface.attr('x3ds', atLocation=x3dsLoc)
			vs_labels = interface.attr('x3ds_labels', atLocation=x3dsLoc)
		else:
			vs, vs_labels = Interface.getWorldSpaceMarkerPos(skelDict)

		if vs is None or vs_labels is None:
			self.logger.error('Could not find 3D data at: %s' % x3dsLoc)

		linesAttrs = {
			'colour': (0.4, 0.1, 0.4, 0.7),
			'edgeColour': (0.6, 0.1, 0.6, 0.7),
			'pointSize': 2,
			'x3ds_labels': vs_labels
		}

		# Go through the markers and find the nearest vertices on the geometry
		import ISCV
		cloud = ISCV.HashCloud3D(skelDict['geom_Vs'], attrs['x3d_threshold'])
		scores, matches, matches_splits = cloud.score(vs)

		which_matches = np.where(matches_splits[1:] - matches_splits[:-1] >= 1)[0]
		numMarkers = len(vs)
		if len(which_matches) < numMarkers:
			self.logger.error('Not all markers have found a partner: %d/%d' % (len(which_matches), numMarkers))
			if not attrs['allowPartialMatching']:
				notMatching = np.where(matches_splits[1:] - matches_splits[:-1] < 1)[0]
				print 'Not matching:', notMatching
				notMatching_x3ds = vs[notMatching]
				sAttrs = {
					'x3ds': notMatching_x3ds, 'x3ds_colour': (1, 0, 0, 0.8), 'x3ds_pointSize': 12.0
				}
				interface.createChild('notMatching', 'points3d', attrs=sAttrs)
				return

		marker_inds = []
		vert_inds = []
		for i, (s, e) in enumerate(zip(matches_splits[:-1], matches_splits[1:])):
			if s - e == 0: continue
			lowestScoreIdx = matches[s:e][np.argmin(scores[s:e])]
			marker_inds.append(i)
			vert_inds.append(lowestScoreIdx)

		marker_inds = np.array(marker_inds, dtype=np.int32)
		linesAttrs['x0'] = vs[marker_inds]

		vert_inds = np.array(vert_inds, dtype=np.int32)
		linesAttrs['x1'] = skelDict['geom_Vs'][vert_inds]

		interface.createChild('markerToVertex', 'edges', attrs=linesAttrs)

		# Find which markers the joints map to. This will determine the marker parents (parented to joints)
		jointMarkersMap = {}
		weights = skelDict['shape_weights'][0]
		for jointName, idx in weights[1].iteritems():
			if jointName not in skelDict['jointNames']: continue
			jointMarkersMap[jointName] = []
			jointVerts = weights[0][idx][0]
			for i, vi in enumerate(vert_inds):
				if vi in jointVerts:
					jointMarkersMap[jointName].append(marker_inds[i])

		# jointIndsMap = {}
		# for jname, jnum in weights[1].iteritems():
		# 	# if jname in skelDict['jointNames']:
		# 	jointIndsMap[jnum] = {'name': jname, 'index': skelDict['jointIndex'][jname]}
		# 	# else:
		# 	# 	print "Ignoring joint:", jname

		# Find joints to which the vertices map to
		markerJointsMap = {}
		for m_idx, v_idx in zip(marker_inds, vert_inds):
			markerJointsMap[m_idx] = []
			for ji, verts in weights[0].iteritems():
				if v_idx in verts[0]:# and ji in jointIndsMap:
					# Find joint name
					# markerJointsMap[m_idx].append(jointIndsMap[ji]['name'])
					for jname, jnum in weights[1].iteritems():
						if ji == jnum and jname in skelDict['jointNames']:
							markerJointsMap[m_idx].append((jname, verts[1][ji]))

		# Tweak - Find out why we are getting a mapping from the right hand to a marker near the foot!
		# jointMarkersMap['VSS_RightHand'] = jointMarkersMap['VSS_RightHand'][1:]
		# markerJointsMap[378] = [('VSS_RightLeg', np.array([1., 1., 1., 0.88377857], dtype=np.float32))]

		# markerJointsMap[378] = ['VSS_RightLeg']
		# for mi, joints in markerJointsMap.iteritems():
		# 	if 'VSS_Spine2' in joints:
		# 		if 'VSS_LeftArm' in joints:
		# 			markerJointsMap[mi] = [('VSS_LeftArm', 1.)]
		# 		elif 'VSS_RightArm' in joints:
		# 			markerJointsMap[mi] = [('VSS_RightArm', 1.)]

		# interface.setAttr('skelDict', skelDict)
		interface.setAttr('jointMarkersMap', jointMarkersMap)
		interface.setAttr('markerJointsMap', markerJointsMap)

		# Calculate the marker data and set in the skeleton dict
		markerOffsets = []
		markerParents = []
		markerWeights = []
		markerNames = []
		markerColour = []

		if attrs['useAllWeights']:
			# Go through existing known markers and build new marker data
			# for mi, markerName in enumerate(skelDict['markerNames']):
			for mi, joints in markerJointsMap.iteritems():
				for jointName, jointWeight in joints:
					if 'markerNames' in skelDict:
						markerNames.append(skelDict['markerNames'][mi])
					else:
						markerNames.append(str(mi))

					if 'markerColour' in skelDict:
						markerColour.append(skelDict['markerColour'][mi])

					jointIndex = skelDict['jointIndex'][jointName]
					markerParents.append(jointIndex)
					jointGs = skelDict['Gs'][jointIndex].copy()
					jointGs = np.append(jointGs, [[0, 0, 0, 1]], axis=0)
					markerWorldPos = vs[mi].transpose()
					markerWorldPos = np.append(markerWorldPos, [1])
					offset = np.dot(np.linalg.inv(jointGs), markerWorldPos)
					markerOffsets.append(offset[:3])
					markerWeights.append(jointWeight[3])
		else:
			for mi, joints in markerJointsMap.iteritems():
				maxWeight, maxIndex = 0, -1
				for jointName, jointWeight in joints:
					if jointWeight[3] > maxWeight:
						maxWeight = jointWeight[3]
						maxIndex  = skelDict['jointIndex'][jointName]

				if maxIndex != -1:
					# markerNames.append(skelDict['markerNames'][mi])
					# markerColour.append(skelDict['markerColour'][mi])
					if 'markerNames' in skelDict:
						markerNames.append(skelDict['markerNames'][mi])
					else:
						markerNames.append(str(mi))

					if 'markerColour' in skelDict:
						markerColour.append(skelDict['markerColour'][mi])

					markerParents.append(maxIndex)
					jointGs = skelDict['Gs'][maxIndex].copy()
					jointGs = np.append(jointGs, [[0, 0, 0, 1]], axis=0)
					markerWorldPos = vs[mi].transpose()
					markerWorldPos = np.append(markerWorldPos, [1])
					offset = np.dot(np.linalg.inv(jointGs), markerWorldPos)
					markerOffsets.append(offset[:3])
					markerWeights.append(1.0)

		skelDict['numMarkers'] = len(markerNames)
		skelDict['markerNames'] = markerNames
		skelDict['markerParents'] = np.array(markerParents, dtype=np.int32)
		skelDict['markerOffsets'] = np.array(markerOffsets, dtype=np.float32)
		skelDict['markerWeights'] = np.array(markerWeights, dtype=np.float32)
		skelDict['markerColour'] = np.array(markerColour, dtype=np.float32)

		# Renormalise
		if attrs['useAllWeights']:
			effectorLabels = np.array([int(mn) for mn in skelDict['markerNames']], dtype=np.int32)
			labels = np.unique(effectorLabels)
			lengths = np.bincount(skelDict['markerNames'], minlength=labels[-1]+1)[labels]
			splits = Interface.makeSplitBoundaries(lengths)
			for s, e in zip(splits[:-1], splits[1:]):
				weights = skelDict['markerWeights'][s:e]
				weights /= sum(weights)
				skelDict['markerWeights'][s:e] = weights
		else:
			for ji in range(skelDict['numJoints']):
				whichJoints = np.where(skelDict['markerParents'] == ji)[0]
				if not whichJoints.any(): continue
				weightSum = np.sum(skelDict['markerWeights'][whichJoints])
				skelDict['markerWeights'][whichJoints] /= weightSum

		interface.setAttr('skelDict', skelDict)

		# Plot lines - Create lines from the marker to respective joints
		if True:
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
			# 	interface.createChild('jointToMarkers_%s' % skelDict['jointNames'][ji[0]], 'lines', attrs=mappingAttrs)

		# Create lines from the marker to respective joints
		# if True:
		# 	m_x0_inds = []
		# 	m_x1_inds = []
		# 	for markerIdx, jointNames in markerJointsMap.iteritems():
		# 		x0_inds, x1_inds = [], []
		# 		for jname in jointNames:
		# 			x0_inds.append(skelDict['markerNames'][markerIdx])
		# 			x1_inds.append(skelDict['jointIndex'][jname])
		#
		# 		m_x0_inds.append(x0_inds)
		# 		m_x1_inds.append(x1_inds)
		#
		# 	np.random.seed(100)
		# 	colours = np.random.rand(len(m_x0_inds), 3)
		# 	colours = np.hstack((colours, np.ones((colours.shape[0], 1))))
		#
		# 	for i, (x0, x1) in enumerate(zip(m_x0_inds, m_x1_inds)):
		# 		mappingAttrs = {
		# 			'colour': (0.1, 0.4, 0.1, 0.5),
		# 			'edgeColour': colours[i],
		# 			'pointSize': 8
		# 		}
		# 		mappingAttrs['x0'] = vs[x0]
		# 		mappingAttrs['x1'] = skelDict['Gs'][x1][:, 3, 3]
		# 		interface.createChild('markerToJoints_%s' % i, 'edges', attrs=mappingAttrs)

		# interface.createChild('vs', 'points3d', attrs={'x3ds': vs, 'x3ds_labels': vs_labels, 'x3ds_pointSize': 8, 'x3ds_colour': (0, 0, 1, 0.8)})
		# interface.createChild('geomVs', 'points3d', attrs={'x3ds': skelDict['geom_Vs'], 'x3ds_pointSize': 8, 'x3ds_colour': (1, 0, 0, 0.8)})


class ExportTemplate(Op.Op):
	def __init__(self, name='/Export Skeleton Template', locations='', sklFilename='', frameRange=''):
		self.fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'locations', 'Skeleton locations', 'string', locations, {}),
			('saveTo', 'Save to (.skl)', 'Save to (.skl)', 'filename', sklFilename, {}),
			('frameRange', 'Frame range', 'Frame range', 'string', frameRange, {})
		]

		super(self.__class__, self).__init__(name, self.fields)

	def cook(self, location, interface, attrs):
		if not self.useFrame(interface.frame(), attrs['frameRange']): return
		if not attrs['saveTo']: return
		skelDict = interface.attr('skelDict')
		if not skelDict:
			self.logger.error('No skeleton found at location: %s' % location)
			return

		# if 'm_names' in skelDict:
		# 	skelDict['numMarkers'] = len(skelDict['m_names'])
		# 	skelDict['markerOffsets'] = np.array(skelDict['m_offsets'], dtype=np.float32)
		# 	skelDict['markerParents'] = np.array(skelDict['m_parents'], dtype=np.int32)
		# 	skelDict['markerWeights'] = np.array(skelDict['m_weights'], dtype=np.float32)
		# 	skelDict['markerNames'] = skelDict['m_names']
		# 	del skelDict['m_names']
		# 	del skelDict['m_offsets']
		# 	del skelDict['m_parents']
		# 	del skelDict['m_weights']
		# 	del skelDict['markerColour']
		# 	del skelDict['markerNormals']
		# 	del skelDict['markerNamesUnq']

		# if 'm_names' in skelDict: del skelDict['m_names']
		# if 'm_offsets' in skelDict: del skelDict['m_offsets']
		# if 'm_parents' in skelDict: del skelDict['m_parents']
		# if 'm_weights' in skelDict: del skelDict['m_weights']

		saveTo = self.resolvePath(attrs['saveTo'])
		IO.save(saveTo, (skelDict))
		self.logger.info('Skeleton template saved to: %s' % saveTo)


class Generative(Op.Op):
	def __init__(self, name='/Generative Skeleton', locations='', x3ds='', frameRange='', solveRange=''):
		self.fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('x3ds', '3D points', '3D points', 'string', x3ds, {}),
			('frameRange', 'Frame range', 'Frame range', 'string', frameRange, {}),
			('solveRange', 'Solve range', 'Solve range', 'string', solveRange, {})
		]

		super(self.__class__, self).__init__(name, self.fields)
		self.cacheManualOverride = True

		self.frameX3ds, self.frameLabels = [], []
		self.frameMissingData = []

		self.stablePointsGroups = None
		self.displayFrames = None
		self.groupRepresentatives = None
		self.boneEdges = None
		self.stablePoints = None
		self.numJoints = 0
		self.c3d_frames, self.frames = None, None
		self.data = None
		self.goodFrames = None

		self.lastFrame = -1
		self.collect = True

	def flush(self):
		self.frameData = []

	def cook(self, location, interface, attrs):
		if not self.useFrame(interface.frame(), attrs['frameRange']) and self.stablePointsGroups is None: return
		if not attrs['x3ds']: return

		solve = self.useFrame(interface.frame(), attrs['solveRange'])
		if self.lastFrame != -1 and interface.frame() - self.lastFrame < 2 and not solve and self.stablePointsGroups is None: return

		if self.collect:
			x3ds = interface.attr('x3ds', atLocation=attrs['x3ds'])
			if x3ds is None: return

			x3ds_labels = interface.attr('x3ds_labels')
			if x3ds_labels is None:
				x3ds_labels = range(len(x3ds))

			missingDataFlags = interface.attr('missingDataFlags', atLocation=attrs['x3ds'])
			if missingDataFlags is None:
				missingDataFlags = np.zeros((len(x3ds)), dtype=np.float32)

			self.frameX3ds.append(x3ds)
			self.frameLabels.append(x3ds_labels)
			self.frameMissingData.append(missingDataFlags)

			self.lastFrame = interface.frame()

		if not solve and self.stablePointsGroups is None: return
		if self.stablePointsGroups is None:
			self.collect = False

			# c3dFilename = 'D:\\IMS\\ViconDB\\2016_Tests\\R&D\\160621_A1_ChessSuit_Day01\\# ROM #\\Exports\\160621_Toby_Body_Chess_ROM_02.c3d'
			# c3d_dict = C3D.read(c3dFilename)
			# self.c3d_frames, c3d_fps, c3d_labels = c3d_dict['frames'], c3d_dict['fps'], c3d_dict['labels']
			# self.frameMissingData = self.c3d_frames[:, :, 3]

			self.c3d_frames = np.array(self.frameX3ds, dtype=np.float32)
			self.frameMissingData = np.array(self.frameMissingData, dtype=np.float32)
			c3d_labels = np.array(self.frameLabels, dtype=np.int32)

			numFramesVisiblePerPoint = np.sum(self.frameMissingData == 0, axis=0)
			numPointsVisiblePerFrame = np.sum(self.frameMissingData == 0, axis=1)
			print 'Threshold', 0.90 * len(self.c3d_frames)
			goodPoints = np.where(numFramesVisiblePerPoint > 0.80 * len(self.c3d_frames))[0]
			self.goodFrames = np.where(np.sum(self.frameMissingData[:, goodPoints] == 0, axis=1) == len(goodPoints))[0]
			print '# Good points:', len(goodPoints)
			print '# Good frames:', len(self.goodFrames)  # 290 x 6162 (80%), 283 x 8729 (90%), 275x10054 (96%)
			self.frames = self.c3d_frames[self.goodFrames, :, :][:, goodPoints, :][:, :, :3]
			# pointLabels = [c3d_labels[g] for g in goodPoints]

			from IO import ASFReader
			# data = np.array(self.frameData, dtype=np.float32)
			# data = data[::20, :, :]
			# self.data = self.frames[::20, :, :]
			self.data = self.frames
			# M = ASFReader.greedyTriangles(self.data, 30, triangleThreshold=1000., thresholdDistance=10. * 10.)  # only every Nth frame
			M = ASFReader.greedyTriangles(self.data, None, triangleThreshold=100., thresholdDistance=10. * 10.)  # only every Nth frame
			stabilizedPointToGroup, stabilizedPointResiduals, stabilizedFrames = ASFReader.assignAndStabilize(self.data,
			                                                                                                  M['RTs'][M['triIndices'][:28]],
			                                                                                                  thresholdDistance=10. * 10.)

			print '# Frames = %d' % len(stabilizedFrames)
			print '# Labelled points %d' % np.sum(stabilizedPointToGroup != -1)
			print 'RMS of labelled points %fmm' % np.sqrt(np.mean(stabilizedPointResiduals[np.where(stabilizedPointToGroup != -1)]))

			# Tighten fit
			thresh = [10, 10, 9, 9]
			for t in thresh:
				RTs = ASFReader.stabilizeAssignment(self.data, stabilizedPointToGroup)
				stabilizedPointToGroup, stabilizedPointResiduals, stabilizedFrames = ASFReader.assignAndStabilize(self.data, RTs,
				                                                                                                  thresholdDistance=float(t) ** 2)
				print 'number of labelled points %d' % np.sum(stabilizedPointToGroup != -1)
				print 'RMS of labelled points %fmm' % np.sqrt(np.mean(stabilizedPointResiduals[np.where(stabilizedPointToGroup != -1)]))

			stablePointsData = ASFReader.sharedStablePoints(RTs, threshold=8. ** 2)
			self.stablePointsGroups = [sp[0] for sp in stablePointsData]
			self.stablePoints = np.array([sp[2] for sp in stablePointsData], dtype=np.float32)
			# stabilizedPointToGroup,stabilizedPointResiduals,stabilizedFrames = ASFReader.assignAndStabilize(_data, RTs, thresholdDistance = 10.**2)
			print '# Stable points', len(self.stablePoints)

			self.displayFrames = stabilizedFrames
			self.groupRepresentatives = ASFReader.groupRepresentatives(self.data, stabilizedPointToGroup)
			self.numJoints = len(self.stablePoints)
			self.boneEdges = np.array(range(2 * self.numJoints), dtype=np.int32)

		pfr = np.searchsorted(self.goodFrames, interface.frame())

		if pfr >= len(self.displayFrames): return

		verts = self.displayFrames[pfr]
		boneVertices = np.zeros((self.numJoints * 2, 3), dtype=np.float32)
		boneVertices[::2] = self.stablePoints
		boneVertices[1::2] = verts[self.groupRepresentatives[self.stablePointsGroups]]

		boneAttrs = {
			'edges': self.boneEdges,
			'verts': boneVertices
		}
		interface.createChild(interface.name(), 'bones', atLocation=interface.parentPath(), attrs=boneAttrs)

		pAttrs = {
			'x3ds': verts
		}
		interface.createChild('verts', 'points3d', atLocation=interface.parentPath(), attrs=pAttrs)

		# c3dAttrs = {
		# 	'x3ds': self.c3d_frames[interface.name(), :, :3],
		# 	'x3ds_pointSize': 16,
		# 	'x3ds_colour': (0, 1, 0, 0.7)
		# }
		# interface.createChild('c3ds', 'points3d', atLocation=interface.parentPath(), attrs=c3dAttrs)


class PoseSkeletonFromX2ds(Op.Op):
	def __init__(self, name='/PoseSkeletonFromX2ds', locations='', x2ds='', calibration=''):
		self.fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'Skeleton locations', 'Skeleton locations', 'string', locations, {}),
			('x2ds', 'Detections', 'Detections', 'string', x2ds, {}),
			('calibration', 'Calibration', 'Calibration', 'string', calibration, {})
		]

		super(self.__class__, self).__init__(name, self.fields)
		self.trackModel = None

	def cook(self, location, interface, attrs):
		if not attrs['x2ds']: return
		detsLoc = attrs['x2ds']
		x2ds = interface.attr('x2ds', atLocation=detsLoc)
		splits = interface.attr('x2ds_splits', atLocation=detsLoc)
		labels = interface.attr('labels', atLocation=detsLoc)
		if x2ds is None or labels is None:
			self.logger.error('No 2D data found at: %s' % detsLoc)
			return

		skelDict = interface.attr('skelDict')
		if skelDict is None:
			self.logger.error('No skeleton found at: %s' % location)
			return

		if not attrs['calibration']: return
		calLoc = attrs['calibration']
		mats = interface.attr('mats', atLocation=calLoc)
		if mats is None:
			self.logger.error('No mats found at: %s' % calLoc)
			return

		from GCore import Label
		if self.trackModel is None:
			if interface.attr('model') is not None:
				self.trackModel = interface.attr('model')
			else:
				effectorLabels = getEffectorLabels(skelDict)
				self.trackModel = Label.TrackModel(skelDict, effectorLabels, mats)

		self.trackModel.bootPose(x2ds, splits, labels)
		poseSkeleton(self.trackModel.skelDict)
		interface.setAttr('skelDict', self.trackModel.skelDict)
		interface.setAttr('Gs', self.trackModel.skelDict['Gs'].copy())


class ExportMarkerMapping(Op.Op):
	def __init__(self, name='/Marker_Mapping', locations='', sourceSkeleton='', x3d_threshold=140.0, saveTo='',
				 frameRange='', enable=False):
		self.fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'Skeleton locations', 'Skeleton locations', 'string', locations, {}),
			('sourceSkeleton', 'Source skeleton locations', 'Source skeleton locations', 'string', sourceSkeleton, {}),
			('x3d_threshold', '3D threshold', '3D threshold', 'float', x3d_threshold, {}),
			('saveTo', 'Save to', 'Save to', 'string', saveTo, {}),
			('frameRange', 'Frame range', 'Frame range', 'string', frameRange, {}),
			('enable', 'enable', 'enable', 'bool', enable, {})
		]

		super(self.__class__, self).__init__(name, self.fields)

	def cook(self, location, interface, attrs):
		if not attrs['enable']: return
		if not self.useFrame(interface.frame(), attrs['frameRange']): return

		skelDict = interface.attr('skelDict')
		if skelDict is None:
			self.logger.error('No skeleton found at: %s' % location)
			return

		skelDict_src = interface.attr('skelDict', atLocation=attrs['sourceSkeleton'])
		if skelDict_src is None:
			self.logger.error('No source skeleton found at: %s' % attrs['sourceSkeleton'])
			return

		from GCore import SolveIK, Label
		effectorData = SolveIK.make_effectorData(skelDict)
		effectorLabels = range(len(skelDict['markerNames']))
		x3ds_ted, labels_ted = SolveIK.skeleton_marker_positions(skelDict, skelDict['rootMat'], skelDict['chanValues'],
		                                                         effectorLabels, effectorData)

		effectorData_src = SolveIK.make_effectorData(skelDict_src)
		effectorLabels_src = np.array([int(mn) for mn in skelDict_src['markerNames']], dtype=np.int32)
		x3ds_src, labels_src = SolveIK.skeleton_marker_positions(skelDict_src, skelDict_src['rootMat'], skelDict_src['chanValues'],
		                                                         effectorLabels_src, effectorData_src)

		labels = -np.ones(len(x3ds_ted), dtype=np.int32)
		score = Label.match(x3ds_src, x3ds_ted, attrs['x3d_threshold'], None, labels)

		which = np.where(labels != -1)[0]
		print labels
		print zip(which, labels[which])

		from IO import IO
		mapping = {'target': which.astype(np.int32), 'source': labels[which].astype(np.int32)}
		IO.save(attrs['saveTo'], mapping)


class ApplyMarkerMapping(Op.Op):
	def __init__(self, name='/Apply_Marker_Mapping', locations='', sourceSkeleton='', mappingLocation='', frameRange=''):
		self.fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'Skeleton locations', 'Skeleton locations', 'string', locations, {}),
			('sourceSkeleton', 'Source skeleton location', 'Source skeleton location', 'string', sourceSkeleton, {}),
			('mappingLocation', 'Mapping location', 'Mapping location', 'string', mappingLocation, {}),
			('frameRange', 'Frame range', 'Frame range', 'string', frameRange, {})
		]

		super(self.__class__, self).__init__(name, self.fields)
		self.flush()

	def flush(self):
		self.effectorData_source, self.effectorLabels_source = None, None
		self.effectorData_target, self.effectorLabels_target = None, None
		self.targetInds, self.sourceInds = None, None

	def cook(self, location, interface, attrs):
		if not self.useFrame(interface.frame(), attrs['frameRange']): return

		skelDict = interface.attr('skelDict')
		if skelDict is None:
			self.logger.error('No skeleton found at: %s' % location)
			return

		skelDict_src = interface.attr('skelDict', atLocation=attrs['sourceSkeleton'])
		if skelDict_src is None:
			self.logger.error('No source skeleton found at: %s' % attrs['sourceSkeleton'])
			return

		mappingDict = interface.attr('data', atLocation=attrs['mappingLocation'])
		if mappingDict is None:
			self.logger.error('No mapping dictionary found at: %s' % attrs['mappingLocation'])
			return

		if self.effectorData_source is None:
			self.effectorData_target = SolveIK.make_effectorData(skelDict)
			self.effectorLabels_target = range(len(skelDict['markerNames']))

			self.effectorData_source = SolveIK.make_effectorData(skelDict_src)
			self.effectorLabels_source = np.array([int(mn) for mn in skelDict_src['markerNames']], dtype=np.int32)

		x3ds_target, labels_target = SolveIK.skeleton_marker_positions(skelDict, skelDict['rootMat'], skelDict['chanValues'],
		                                                               self.effectorLabels_target, self.effectorData_target)
		x3ds_source, labels_source = SolveIK.skeleton_marker_positions(skelDict_src, skelDict_src['rootMat'], skelDict_src['chanValues'],
		                                                               self.effectorLabels_source, self.effectorData_source)

		if self.targetInds is None:
			self.targetInds, self.sourceInds = mappingDict['target'].astype(np.int32), mappingDict['source'].astype(np.int32)

		x3ds_target[self.targetInds] = x3ds_source[self.sourceInds]
		SolveIK.solve_skeleton_from_3d(x3ds_target, labels_target, self.effectorLabels_target, skelDict, self.effectorData_target,
		                               skelDict['rootMat'])

		interface.setAttr('skelDict', skelDict)


# Utility functions
def getRootMat(rootMat=None, t=[0, 0, 0], r=[0, 0, 0]):
	if rootMat is None:
		mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
	else:
		mat = rootMat.copy()

	Interface.rotateX(mat, math.radians(r[0]))
	Interface.rotateY(mat, math.radians(r[1]))
	Interface.rotateZ(mat, math.radians(r[2]))
	Interface.translate(mat, np.array(t, dtype=np.float32))

	return mat

def poseSkeleton(skelDict):
	from GCore import Character
	Character.pose_skeleton(skelDict['Gs'], skelDict)

def makeRestrictedSkeleton(skelDict, allowedJointChannels=[]):
	restrictedSkel = skelDict.copy()
	jointChans, jointChanSplits, chanNames, chanValues = skelDict['jointChans'], skelDict['jointChanSplits'], \
														 skelDict['chanNames'], skelDict['chanValues']

	tmp = np.zeros(jointChanSplits[-1], dtype=np.bool)
	if allowedJointChannels:
		tmp[allowedJointChannels] = True

	tmpSum = [sum(tmp[:x]) for x in jointChanSplits]
	jointChanSplits_restricted = np.array(tmpSum, dtype=np.int32)
	jointChans = jointChans[allowedJointChannels]
	chanValues = chanValues[allowedJointChannels]

	chanNames = np.array(chanNames, dtype=np.str)
	chanNames = chanNames[allowedJointChannels].tolist()

	restrictedSkel['jointChans'], restrictedSkel['jointChanSplits'], restrictedSkel['chanNames'], \
		restrictedSkel['chanValues'] = jointChans, jointChanSplits_restricted, chanNames, chanValues

	return restrictedSkel

def getEffectorLabels(skelDict):
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


# Register Ops
import Registry
Registry.registerOp('Skeleton VSS', VSS)
Registry.registerOp('Skeleton ASF', ASF)
Registry.registerOp('Skeleton FBX', FBX)
Registry.registerOp('Skeleton Template', Template)
Registry.registerOp('Skeleton Transform', Transform)
Registry.registerOp('Skeleton Configure', Configure)
Registry.registerOp('Skeleton Joint Copy', JointCopy)
Registry.registerOp('Skeleton Joint Reset', ResetJoint)
Registry.registerOp('Skeleton Geometry Copy', GeometryCopy)
Registry.registerOp('Export Skeleton Template', ExportTemplate)
Registry.registerOp('Set Marker Joint Mapping', SetMarkerJointMapping)
Registry.registerOp('Generative Skeleton', Generative)
Registry.registerOp('Pose Skeleton From X2Ds', PoseSkeletonFromX2ds)

