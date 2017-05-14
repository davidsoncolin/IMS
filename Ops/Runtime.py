import numpy as np
import time, sys, os
from GCore import State
import Registry, Interface, RenderCallback


instances = {}
def getInstance(name=None):
	if name is None: name = 'main'
	if name not in instances:
		instances[name] = Instance(name)

	return instances[name]

class Instance:
	def __init__(self, id='main'):
		# The ops could be a tree structure but for now we just define them as a list
		self.id = id
		self.ops = []
		self.interface = Interface.Interface()

		self.tempCache = {}

		self.collect = {}
		self.collectFrames = {}

		self.renderRegistryCbs = {}
		self.win = None

		self.cookIndex = -1
		self.pickedInfo = None

		self.cookRelationships = {}
		self.enableCache = True

	def appendOp(self, op):
		self.ops.append(op)
		return op

	def initialiseOpState(self, op):
		attrs = op.getStateAttrs()
		opName = State.addKey(op.getName(), {
			'type': self.findRegisteredOpName(op),
			'attrs': attrs
		})

		# print self.id, '| Initialise Op State: Op name =', opName
		op.setName(opName)

		self.collect[opName] = []
		self.collectFrames[opName] = []

		if 'locations' in attrs:
			locations = op.resolveLocations(attrs['locations'])
			if opName not in self.cookRelationships: self.cookRelationships[opName] = []
			self.cookRelationships[opName].extend(locations)

	def initialiseOps(self, win=None):
		for op in self.ops:
			self.initialiseOpState(op)

			if win is not None:
				registeredOpName = self.findRegisteredOpName(op)
				win.addOpItem(op.getName(), data=op.getName(), silent=True)
				win.setFields(registeredOpName, op.getFields())

		# Set root attributes
		bootingDefault = 10
		self.interface.setAttr('booting', bootingDefault, '/root')
		self.interface.setAttr('bootingReset', bootingDefault, '/root')
		# print self.id, '| [initialiseOps] Ops =', [n.getName() for n in self.ops]

	def _getOpsByName(self, names):
		# Do something more efficient and clever
		ops = []
		for name in names:
			ops.extend([op for op in self.ops if name == op.getName()])

		return ops

	# Note: Going forward it would be more efficient to not collect, instead do it in a deferred manner where
	#       the locations are only checked and evaluated as they are traversed
	def _collectLocationMatches(self, locations, absolute):
		if absolute: return locations   # Just to simplify the calling code

		matches = []
		for sgLocation in self.interface.scenegraphLocations():
			match = self._matchSgLocation(sgLocation, locations)
			if match is not None:
				matches.append(match)
				continue

		return matches

	def _matchSgLocation(self, sgLocation, locations):
		for location in locations:
			if location == sgLocation: return sgLocation
			elif location.endswith('*'):
				if sgLocation.startswith(location[:-1]):
					return sgLocation

		return None

	def setCookIndex(self, index):
		self.cookIndex = index

	def __changeCookIndex(self, index, flush=True):
		self.win.qnodes.changeCookIndex(index, flush=flush)

	def setUseCache(self, useCache):
		self.enableCache = useCache

	def resetFrame(self, frame):
		self.interface.resetFrame(frame)

	def cookOps(self, win, frame, forceRecook=False, disableCache=False):
		# print self.id, '| cookOps: frame =', frame
		# print self.id, '| interface:', self.interface.locations()
		# Note: For now we force a recook on a new frame
		#       ATM we should cook all downstream ops of one that has changed
		cookFrame = self._isNewFrame(frame)
		# if not forceRecook:
		# 	forceRecook = cookFrame

		# Check if we've cached the cooked data for this frame
		# Note: Perhaps add flag to avoid cache and instead cook every time
		cacheAvailable = self.interface.hasFrameCache(frame)
		# if not self.interface.hasFrameCache(frame):
		# Note: Should we just return here immediately if we are not cooking?

		# if cookFrame:
		# 	# TODO: Improve as this is too simplistic
		# 	skipping = np.abs(frame - self.interface.lastFrame()) > 10
		# 	if skipping:
		# 		booting = self.interface.attr('bootingReset', atLocation='/root')
		# 	else:
		# 		booting = self.interface.attr('booting', atLocation='/root')
		#
		# 	if booting is not None: booting -= 1
		#
		# 	self.interface._setSkipping(skipping)
		# 	self.interface._setLastFrame(frame)
		# 	self.interface.setAttr('booting', booting, atLocation='/root')

		#State.setKey('frame', frame)
		self.interface.setFrame(frame)

		if cookFrame and self.cookIndex != -1:
			self.__changeCookIndex(self.cookIndex, flush=False)
			self.flush()

		if self.win:
			activeCameraIdx = RenderCallback.getActiveCameraIndex()
			if activeCameraIdx: self.interface.setAttr('activeCameraIdx', activeCameraIdx, atLocation=self.interface.root())

		# Check if we can use our temp cache
		# Note: For this we need to have deferred/terminal ops who draw/update based on the cooked interface data
		# if frame in tempCache:

		# print self.id, '| [cookOps] Ops =', [n.getName() for n in self.ops]

		dirtyKeys = State.g_dirty
		dirtyOps = set('/' + k.split('/')[1] for k in dirtyKeys)

		logTime = False

		if logTime:
			startLoop = time.time()
			print('\n######## COOK %d ########' % frame)

		if self.win is not None:
			opNames = self.win.qnodes.getNodes()
			# print ">> opNames =", opNames
			ops = self._getOpsByName(opNames)
		else:
			ops = self.ops

		if not ops: return
		cookRegistry = self.interface.cookRegistry()

		if cacheAvailable and self.enableCache and not disableCache:
			self.interface.fetchFrameCache(frame)

		for op in ops:
			opName = op.getName()
			self.interface.setOpName(opName)

			try:
				locations = op.resolveLocations()
				absolute = op.isAbsolute(locations)
			except Exception as e:
				print self.id, '| Could not resolve locations:', str(e)
				import traceback
				traceback.print_exc()
				return

			# Check if there's any work that needs doing
			opDirty = opName in dirtyOps or op.cacheManualOverride
			inCookRegistry = self._isInCookRegistry(opName, locations, cookRegistry, dirtyOps)
			useCache = cacheAvailable and self.enableCache
			noNeedToCook = useCache and not opDirty and not inCookRegistry

			# If we're not forcing a recook, force a cook on a new frame unless a cache is available
			if not forceRecook: forceRecook = cookFrame and not useCache

			# We want to cook the Op with its name or target locations if a value has changed
			if not forceRecook and noNeedToCook: continue

			# NOTE: This is fine as we are currently using a serial execution model
			self.interface.setLocation(opName)

			if logTime:
				print '> Setup:', op.getName(), '|', frame
				startSetup = time.time()

			attrs = op.getAttrs()
			op.setup(self.interface, attrs)

			if logTime:
				print '> Setup:', op.getName(), '|', frame, '| time:', (time.time() - startSetup)

			# print '> Op:', opName, '|', frame, '|', self.id
			if locations:
				matchedLocations = self._collectLocationMatches(locations, absolute)
				for location in matchedLocations:
					self.interface.setLocation(location)
					if logTime:
						start = time.time()
						print '  > Cook: location =', location

					op.cook(location, self.interface, attrs)
					op.setLastCookedFrame(frame)

					if logTime:
						print '> Cook:', opName, '|', frame, '|', location, '| time:', (time.time() - start)
					# self.collect[opName].append(t)
					# self.collectFrames[opName].append(frame)
			else:
				if logTime:
					start = time.time()
					print '  > Cook: location =', opName

				self.interface.setLocation(opName)
				op.cook(opName, self.interface, attrs)
				op.setLastCookedFrame(frame)

				if logTime:
					print '> Cook:', opName, '|', frame, '|', opName, '| time:', (time.time() - start)
				# self.collect[opName].append(t)
				# self.collectFrames[opName].append(frame)

			# TODO: We want to cook the Op if the locations it refers to have changed

		if logTime:
			endLoop = time.time()
			totalTime = (endLoop - startLoop)
			fps = (1. / (totalTime + 0.0000001))
			print '-- Total (cook):', totalTime, '| ~', fps, 'fps', '--'
			print 'COOK:', frame, ";", len(repr(self.interface))

		# Use temp cache until we do something proper
		# if frame not in tempCache or forceRecook:
		# 	tempCache[frame] = interface

		# print '--------', self.id, '---------'
		# self.interface.printInfo()
		# print '----------------------'

		State.g_dirty.clear()

		# else:
		# 	if cookFrame: self.interface.fetchFrameCache(frame)

		if self.enableCache: self.interface.addFrameToCache()

		# Note: This is too simplistic. We need to take account of what locations are dirty and collect them in a list
		#       Then if we have any locations flagged as dirty we pass them to the processing function which ignores
		#       all locations which have not been flagged.
		cookedLocations = self.interface.cookedLocations()
		if len(cookedLocations) > 0 and len(self.renderRegistryCbs) > 0:
			# print "cookedLocations =", cookedLocations
			self.processLocationsForGui(cookedLocations)

		self.interface.clearCookedLocations()
		self.interface.updateTimeline()

	def refresh(self, cam=None):
		from UI import QApp
		frame = self.getFrame()
		self.interface.clearCookedLocations()
		self.cookOps(QApp.app, frame, forceRecook=True)
		if cam:
			self.win.view().camera = self.win.view().cameras[cam[0]]
			self.win.view().camera.cameraOx = cam[1]
			self.win.view().camera.cameraOy = cam[2]
			self.win.view().camera.cameraZoom = cam[3]

		QApp.app.updateLayers()
		QApp.app.refreshImageData()
		QApp.app.updateGL()

	def getFrame(self):
		return self.interface.frame()
		# if not State.hasKey('/frame'): return 0
		# return State.getKey('/frame')

	def _isNewFrame(self, frame):
		# if not State.hasKey('/frame'): return True
		# if State.getKey('/frame') != frame: return True
		if frame != self.interface.frame(): return True
		return False

	def _isInDirtyKeys(self, location, dirtyKeys):
		for key in dirtyKeys:
			if key.startswith(location + '/'):
				return True

		return False

	def _lookForLocationInRegistry(self, splitLocations, registry):
		for location in splitLocations:
			if location in registry:
				return set(registry[location])

		return None

	def _isInCookRegistry(self, opName, splitLocations, registry, dirtyOps):
		if opName in registry:
			candidates = set(registry[opName])
		else:
			candidates = self._lookForLocationInRegistry(splitLocations, registry)
			if candidates is None: return False

		for dirty in dirtyOps:
			if dirty in candidates: return True

		return False

	def _addOp(self, opType, win, initialiseState=True, name=None):
		if name is None:
			newOp = opType()
		else:
			newOp = opType(name=name)
		self.appendOp(newOp)

		if initialiseState: self.initialiseOpState(newOp)
		newOp.registerState(win, type=self.findRegisteredOpName(newOp))

	def addGuiOp(self, op):
		if self.win is None: return
		self.appendOp(op)
		self.initialiseOpState(op)
		self.win.addOpItem(op.getName(), data=op.getName())
		self.win.setFields(self.findRegisteredOpName(op), op.getFields())
		self.refresh()

	def flush(self, hard=False):
		currentCam = self.win.view().camera
		cam = [
			self.win.view().cameraIndex(), currentCam.cameraOx, currentCam.cameraOy, currentCam.cameraZoom
		]
		self.win.clearLayers(keepGrid=True)
		self.win.view().primitives2D = []
		RenderCallback.flush()
		# self.refresh(cam)

		if hard:
			for op in self.ops: op.flush()
			self.interface.reset()

		self.interface.clearFrameCache()
		self.refresh(cam)

	def handleDroppedFile(self, filename):
		from Ops import Skeleton, Vicon, Data, Mesh#, Video
		from Quad import QuadOps

		head, tail = os.path.split(filename)
		prefix, extension = os.path.splitext(tail)
		fname = filename.lower()

		if fname.endswith('.skl'):
			self.addGuiOp(Skeleton.Template(locations='/root/skeleton', sklFilename=filename))
		elif fname.endswith('.vss') or fname.endswith('.vsk'):
			self.addGuiOp(Skeleton.VSS(locations='/root/skeletons/vss', vssFilename=filename))
		elif fname.endswith('.asf'):
			self.addGuiOp(Skeleton.ASF(locations='/root/skeletons/asf', asfFilename=filename))
		elif fname.endswith('.c3d'):
			self.addGuiOp(Vicon.C3d(locations='/root/track', c3dFilename=filename))
		elif fname.endswith('.c3dio'):
			self.addGuiOp(Vicon.C3dIo(locations='/root/track', c3dFilename=filename))
		elif fname.endswith('.x3d'):
			self.addGuiOp(Vicon.C3dIo(locations='/root/track', c3dFilename=filename))
		elif fname.endswith('.x2d'):
			xcpFilename = filename.replace('.x2d', '.xcp')
			if os.path.isfile(xcpFilename):
				self.addGuiOp(Vicon.XcpAndX2d(locations='/root/vicon', x2dFilename=filename, xcpFilename=xcpFilename))
			else:
				self.addGuiOp(Vicon.X2d(locations='/root/detections', x2dFilename=filename))
		elif fname.endswith('.xcp'):
			self.addGuiOp(Vicon.Xcp(xcpFilename=filename))
		elif fname.endswith('.scal'):
			self.addGuiOp(Vicon.SurreyCal(calFilename=filename))
		elif fname.endswith('.io'):
			self.addGuiOp(Data.Import(filename=filename))
		elif fname.endswith('.mmesh'):
			self.addGuiOp(Mesh.MarkerMesh(locations='/root/mesh', filename=filename))
		elif fname.endswith('.char'):
			self.addGuiOp(Mesh.CharacterFromSkeleton(locations='/root/character', skelFilename=filename))
		elif fname.endswith('.mov'):
			self.addGuiOp(Video.Sequence(locations='/root/cameras', directory=head, prefix=prefix, useCalibration=False,
			                             useTimecode=False))
		elif fname.endswith('.gplvm'):
			self.addGuiOp(QuadOps.ImportGPLVM(locations='/root/model', filename=filename))

	def loadCache(self):
		from PySide import QtGui, QtCore
		filename, _ = QtGui.QFileDialog.getOpenFileName(self.win, 'Load Op Cache (.cache)..', os.environ.get('HOME'), 'Files (*.cache)')
		if not filename: return

		progress = QtGui.QProgressDialog(u'Loading Cache', u'Cancel', 0, 1, self.win)
		progress.setWindowTitle('Please wait...')
		progress.setWindowModality(QtCore.Qt.WindowModal)
		progress.show()
		progress.setValue(0)

		import cPickle as pickle
		with open(filename, 'rb') as file:
			cache = pickle.load(file)

		self.interface.setCache(cache)
		progress.setValue(1)

	def saveCache(self):
		from PySide import QtGui, QtCore
		filename, _ = QtGui.QFileDialog.getSaveFileName(self.win, 'Save Op Cache (.cache)..', os.environ.get('HOME'), 'Files (*.cache)')
		if not filename: return

		progress = QtGui.QProgressDialog(u'Saving Cache', u'Cancel', 0, 1, self.win)
		progress.setWindowTitle('Please wait...')
		progress.setWindowModality(QtCore.Qt.WindowModal)
		progress.show()
		progress.setValue(0)

		import cPickle as pickle
		with open(filename, 'wb') as file:
			pickle.dump(self.interface.getCache(), file)

		progress.setValue(1)

	def addOpMenu(self, win):
		# Add ops to the menu in alphabetical order
		registeredOps = Registry.getRegisteredOps()
		opNames = registeredOps.keys()
		opNames.sort()

		for opName in opNames:
			win.addMenuItem({'menu': '&Add Op', 'item': opName,
			                 'tip': '',
			                 'cmd': self._addOp, 'args': [registeredOps[opName], win]})

		editMenu = win.getOrCreateMenu('&Edit')
		editMenu.addSeparator()

		win.addMenuItem({'menu': '&Edit', 'item': 'Flush',
		                 'shortcut':'Ctrl+F',
		                 'tip': 'Flushes state and interface (cooked) data, and layers',
		                 'cmd': self.flush, 'args': [True]})

		win.addMenuItem({'menu': '&Edit', 'item': 'Soft Flush',
		                 'tip': 'Retains cooked data',
		                 'cmd': self.flush, 'args': [False]})

		win.addMenuItem({'menu': '&Edit', 'item': 'Refresh',
		                 'tip': 'Refresh',
		                 'cmd': self.refresh, 'args': []})

		editMenu.addSeparator()

		win.addMenuItem({'menu': '&Edit', 'item': 'Load Cache..',
		                 'tip': 'Load Op Interface Cache',
		                 'cmd': self.loadCache, 'args': []})

		win.addMenuItem({'menu': '&Edit', 'item': 'Save Cache..',
		                 'tip': 'Save Op Interface Cache',
		                 'cmd': self.saveCache, 'args': []})

		win.addMenuItem({'menu': '&Edit', 'item': 'Disable Cache',
		                 'tip': 'Disable Cache',
		                 'cmd': self.setUseCache, 'args': [False]})

		win.addMenuItem({'menu': '&Edit', 'item': 'Enable Cache',
		                 'tip': 'Enable Cache',
		                 'cmd': self.setUseCache, 'args': [True]})

	def setCallbacks(self, win):
		win.preLoadCB = self.preBuildOps
		win.loadCB = self.buildOps
		win.saveCB = self.saveOps

	def preBuildOps(self, filename, win):
		# Clear existing ops and cooked data
		self.ops = []
		win.clearOpItems()
		win.clearLayers()

		try:
			if win.movies:
				for ci, md in enumerate(win.movies):
					win.view().cameras[ci + 1].invalidateImageData()
					win.view().cameras[ci + 1].clearImage()

			win.camera_ids = None
		except Exception:
			pass

		win.view().cameras = [win.view().cameras[0]]
		win.qview.setCamera(win.view().cameras[0])
		win.new()
		win.refreshImageData()

	def buildOps(self, filename, win):
		# Check if the state contains a recipe as we can't continue without it since the order (graph) is unknown
		if not State.hasKey('/recipe'): return

		# Get the recipe and available ops
		recipe = State.getKey('/recipe')
		registeredOps = Registry.getRegisteredOps()

		# Go through the recipe in order
		for op in recipe:
			opName, opType = op

			# Look up the corresponding State data and check if the type corresponds to a registered Op
			opTypeKey = opName + '/type'
			if not State.hasKey(opName) or not State.hasKey(opTypeKey): return
			opType_state = State.getKey(opTypeKey)[1:]
			if opType_state not in registeredOps: continue

			# Get the Op from the registration factory and create the op (without duplicating the State entry)
			self._addOp(registeredOps[opType_state], win, initialiseState=False, name=opName)

		self.interface.reset()
		self.cookOps(win, self.getFrame(), forceRecook=True)

		if win:
			win.updateGL()
			win.updateLayers()

	def findRegisteredOpName(self, op):
		registeredOps = Registry.getRegisteredOps()
		for name, className in registeredOps.iteritems():
			if isinstance(op, className): return '/' + name

		return op.getName()
		# return ''

	def saveOps(self, filename):
		opNames = [(op.getName(), self.findRegisteredOpName(op)) for op in self.ops]
		State.setKey('recipe', opNames)

	def setWin(self, win):
		self.win = win
		self.win.runtime = self
		self.interface.setWin(win)

	def picked(self, view, data, clearSelection=True, isLabel=False):
		if data is None:
			self.pickedInfo = None
			self.win.clearSelectionLayers()
			RenderCallback.clear2dSelections(self.win)
		else:
			(type, pi, index, depth) = data
			self.pickedInfo = {
				'view': view,
				'type': type,
				'primitiveIndex': pi,
				'index': index,
				'depth': depth,
				'isLabel': isLabel
			}

			if type == '3d':
				primitiveType = self.win.view().primitives[pi].__class__.__name__
				self.pickedInfo['primitiveType'] = primitiveType
				if primitiveType == 'GLCameras':
					self.win.view().camera = self.win.view().cameras[index + 1]
					self.win.updateGL()
					return
				elif primitiveType == 'GLGrid': return
			else:
				# TODO: Finish
				self.pickedInfo['primitiveType'] = 'GLPoints2D'

				# We assume that a perspective camera exists so we decrement the index to match any camera splits/boundaries
				self.pickedInfo['cameraIndex'] = self.win.view().cameraIndex() - 1

			# cookedLocations = self.interface.lastCookedLocations()
			cookedLocations = self.interface.locations()
			if len(cookedLocations) > 0 and len(self.renderRegistryCbs) > 0:
				self.processLocationsForGui(cookedLocations)

		self.win.updateLayers()
		self.win.updateGL()

	def getLayers(self, *args):
		return self.win.view().getLayers(*args)

	def layers(self):
		return self.getLayers()

	def clearLayer(self, layerName):
		# A little temporary hack until we've got some nice UI way of disabling/enabling layers
		from UI import GLPoints3D
		self.win.view().setLayer(layerName, GLPoints3D([]))

	def updateX3dsLayers(self, refresh=True):
		from UI import GLPoints3D
		hitlist = []
		for layerName, layer in self.layers().iteritems():
			if isinstance(layer, GLPoints3D):
				hitlist.append(layerName)

		for layerName in hitlist: del self.layers()[layerName]
		if refresh: self.refresh()

	# Allow users to register their own callback functions
	def registerRenderCallback(self, type, callbackFunction):
		self.renderRegistryCbs[type] = callbackFunction

	def processLocationsForGui(self, cookedLocations):
		# print "#### cookedLocations =", cookedLocations
		for locationName in cookedLocations:
			# print "locationName =", locationName
			location = self.interface.location(locationName)
			if location is None: continue
			# print "> type =", location['type']
			if 'type' not in location: continue
			if location['type'] in self.renderRegistryCbs:
				# print('> Render location [%s] of type [%s]' % (locationName, location['type']))
				# print '  > Attrs', self.interface.attrNames(locationName)
				# if 'visible' in location and not location['visible']: continue
				self.renderRegistryCbs[location['type']](self.win, locationName, location, self.interface, self.pickedInfo)

