from GCore import State
from UI import QApp

import logging
from os.path import expanduser, expandvars


class Op(object):
	def __init__(self, opName, opFields):
		self.locations = None
		self.attrs = {}
		self.fields = opFields
		for field in self.fields:
			attrName = field[0]
			attrValue = field[4]
			self.attrs[attrName] = attrValue
			if attrName == 'locations':
				self.locations = attrValue

		opName = self.expandLocation(opName)
		self.name = opName

		# QApp.fields[opName] = self.fields
		self.cooked = False
		self.logger = logging.getLogger(self.name)
		self.logger.setLevel(logging.INFO)

		self.lastCookedFrame = -1
		self.cacheManualOverride = False

	def getName(self):
		return self.name

	def setName(self, name):
		self.name = name
		self.logger = logging.getLogger(self.name)

	def getStateAttrs(self):
		return self.attrs

	def registerState(self, win, addToOps=True, type=None):
		if win is None: return
		if addToOps:
			win.addOpItem(self.name, data=self.name)

		if type is None: type = self.name
		QApp.fields[type] = self.fields

	def addField(self, field):
		self.fields.append(field)

	def getFields(self):
		return self.fields

	def isDirty(self, attrName, location=None):
		if location is None: location = self.name
		lookupKey = "/".join([location, "attrs", attrName])
		if not lookupKey.startswith("/"): lookupKey = "/" + lookupKey
		if lookupKey in State.g_dirty:
			return True

		return False

	def getFrame(self):
		return State.getKey("frame")

	def setAttrs(self, attrs, location=None):
		if location is None: location = self.name

		# if State.hasKey(location):
		# 	locationAttrs = State.getKey(location)
		# 	for k, v in attrs.iteritems():
		# 		locationAttrs[k] = v
		# else:
		# 	locationAttrs = attrs

		State.setKey(location, {
			'type': location,
			'attrs': attrs
		})

	def getAttrs(self, location=None, onlyDirty=False):
		if location is None: location = self.name
		op = State.getKey(location)

		if onlyDirty:
			op = State.getKey(location)
			attrs = {name: value for name, value in op['attrs'].iteritems() if self.isDirty(name)}
			return attrs

		return op['attrs']

	def getAttr(self, attrName, defaultValue=None, location=None, onlyDirty=False):
		if location is None: location = self.name
		if not State.hasKey(location): return defaultValue
		op = State.getKey(location)
		if not op or 'attrs' not in op: return defaultValue

		attrs = op['attrs']
		if not attrName in attrs: return defaultValue
		if onlyDirty and not self.isDirty(attrName, location): return defaultValue

		return attrs[attrName]

	def getAttrFromAttrs(self, attrName, defaultValue, attrs):
		if attrName in attrs: return attrs[attrName]
		return defaultValue

	def expandLocation(self, location, prefix='/'):
		# This could interpret an expression language to process /*{attr('tx') == 0. etc.
		# but at the moment it just makes sure the location starts with '/'
		if not location.startswith(prefix): return prefix + location
		return location

	def splitLocations(self, locations, prefix='/'):
		_splitLocations = locations.split(' ' + prefix)
		for li, loc in enumerate(_splitLocations):
			_splitLocations[li] = self.expandLocation(loc, prefix).strip()

		return _splitLocations

	def resolveLocations(self, locations=None, attrs=None, prefix='/'):
		# Note: This could facilitate an expression language etc. but for now
		#       we just split the strings based on '/'
		if locations is None:
			if attrs is None: attrs = self.getAttrs()
			self.locations = self.getAttrFromAttrs('locations', self.locations, attrs)
			if not self.locations: return []
			splits = self.splitLocations(self.locations, prefix)
		else:
			splits = self.splitLocations(locations, prefix)

		return splits

	def isAbsolute(self, locations):
		locs = ' '.join(locations)
		return not '*' in locs

	def getLocations(self):
		return self.locations

	def resolvePath(self, filename):
		filename = filename.strip()
		filename = expandvars(filename)
		filename = expanduser(filename)
		return filename

	def xFrameRange(self, start, stop, step=1):
		if start <= stop:
			stop, step = stop + 1, abs(step)
		else:
			stop, step = stop - 1, -abs(step)

		return (f for f in xrange(start, stop, step))

	def useFrame(self, frame, frameRange):
		import re
		if not frameRange: return True

		frameRangePattern = r'^(-?\d+)(?:-(-?\d+)(?:([:xy]{1})(\d+))?)?$'
		frameRangeRegex = re.compile(frameRangePattern)

		for part in frameRange.split(','):
			if not part: continue

			match = frameRangeRegex.match(part.strip())
			if not match: return False
			start, end, modifier, chunk = match.groups()
			start = int(start)
			end = int(end) if end is not None else start
			chunk = abs(int(chunk)) if chunk is not None else 1
			range = self.xFrameRange(start, end, chunk)
			if frame in range: return True

		return False

	def setLastCookedFrame(self, frame):
		self.lastCookedFrame = frame

	def setup(self, interface, attrs):
		pass

	def cook(self, location, interface, attrs):
		pass

	def flush(self):
		pass