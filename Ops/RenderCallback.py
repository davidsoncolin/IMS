import os
import numpy as np
import logging
import Interface

from UI import QApp

logging.basicConfig()
renderLogger = logging.getLogger('Render')

# Keep a map until we can discuss whether or not to use layers for 2D primitives
primitive2D_map = {}
selectionIndices = []
cameraMap = {}


# Note: This is not really a factory
# This encapsulation is just to batch known registered draw callbacks and make
# it easier for users to add their own draw callbacks for given scene graph
# location types
class Factory:
	def __init__(self, win, runtime):
		self.win = win
		self.runtime = runtime
		self.runtime.setWin(win)
		self.__registerCallbacks()

		self.win.cameraPanels = None

	def __registerCallbacks(self):
		self.runtime.registerRenderCallback('cameras', drawCameras)
		self.runtime.registerRenderCallback('points', drawPoints)
		self.runtime.registerRenderCallback('points3d', drawX3ds)
		self.runtime.registerRenderCallback('points2d', drawX2ds)
		self.runtime.registerRenderCallback('detections', drawX2ds)
		self.runtime.registerRenderCallback('mesh', drawMesh)
		self.runtime.registerRenderCallback('character', drawCharacter)
		self.runtime.registerRenderCallback('skeleton', drawSkeleton)
		self.runtime.registerRenderCallback('edges', drawEdges)
		self.runtime.registerRenderCallback('bones', drawBones)
		self.runtime.registerRenderCallback('primitive', drawPrimitive)
		self.runtime.registerRenderCallback('sphere', drawSphere)
		self.runtime.registerRenderCallback('cube', drawCube)

	# Register callback function for a given scenegraph location type
	def registerCallback(self, type, callbackFunction):
		self.runtime.registerRenderCallback(type, callbackFunction)

# **** Convenience functions ****
def getActiveCameraIndex():
	idx = None
	try:
		idx = QApp.view().cameras.index(QApp.view().camera) - 1
	except Exception as e:
		renderLogger.error('Unable to determine active camera index: %s', str(e))

	return idx

def get(dict, key):
	if key not in dict: return None
	return dict[key]

def isVisible(attrs):
	if 'visible' in attrs and not attrs['visible']: return False
	return True

def flush():
	global primitive2D_map, cameraMap
	primitive2D_map = {}
	cameraMap = {}

def get2dPrimitives():
	global primitive2D_map
	return primitive2D_map.keys()

def clear2dSelections(win):
	# Note: As we don't have the layer treatment yet we can't meddle with the order.
	#       For now we resort to clearing the 2D primitives.
	from UI import GLPoints2D
	global selectionIndices
	for i in selectionIndices:
		win.view().primitives2D[i] = GLPoints2D(([], []))

	selectionIndices = []

# **** Draw callback functions for known scenegraph location types ****
def drawCameras(win, locationName, attrs, interface, picked):
	from UI.QGLViewer import Camera
	from UI import GLCameras

	# Get camera ids. That's all we need if we are just updating
	if 'camera_ids' not in attrs:
		print 'Error rendering cameras: No camera_ids found.'
		return

	camera_ids = attrs['camera_ids']
	if camera_ids is None: return

	# Create cameras if we don't have any at this point
	# Get all the gobbin from the cameras
	if 'mats' not in attrs: return
	mats = attrs['mats']
	if mats is None: return

	if 'camera_names' in attrs:
		camera_names = attrs['camera_names']
	else:
		camera_names = [str(n) for n in range(len(camera_ids))]

	# Check if we've got some image data
	imageData = False
	imgs = None
	if 'imgs' in attrs: imgs = attrs['imgs']
	if imgs is not None:
		vheights, vwidths = None, None
		if 'vheight' in attrs: vheights = attrs['vheight']
		if 'vwidth' in attrs: vwidths = attrs['vwidth']
		if vheights is not None and vwidths is not None:
			if len(imgs) == len(vheights) == len(vwidths):
				imageData = True
			else:
				renderLogger.error(
					'Image data not consistent: #imgs[%d] #vheights[%d] #vwidths[%d]' % (len(imgs), len(vheights), len(vwidths)))
		else:
			renderLogger.error('vheights and vwidths for images not found')

	layerExists = win.view().hasLayer(locationName)
	if layerExists:
		camsLayer = win.view().getLayer(locationName)
		if 'colours' in attrs and attrs['colours'] is not None:
			camsLayer.colours = attrs['colours']
		else:
			camsLayer.colours = None

	# TODO : Sort out when images don't exist, so vheights and vwidths don't either
	updateMats = 'updateMats' in attrs and attrs['updateMats']
	if imgs is not None and layerExists and not updateMats:
		if 'updateImage' in attrs and attrs['updateImage']:
			for ci, img, h, w in zip(camera_ids, imgs, vheights, vwidths):
				#cam = QApp.view().cameras[ci + 1]
				cam = QApp.view().cameras[ci + 1]
				cam.setImageData(img, h, w, 3)

		return

	distOverride = {}
	if 'distortion' in attrs: distOverride = attrs['distortion']

	# Go through what we've gathered and create the cameras (with image data if present)
	for ci, (mat, cid, cname) in enumerate(zip(mats, camera_ids, camera_names)):
		P, distortion = mat[2], mat[3]
		if ci in distOverride: distortion = distOverride[ci]
		if cid in cameraMap and updateMats:
			camera = QApp.view().cameras[ci + 1]
			camera.setP(P, distortion=distortion, store=True)
		elif cid not in cameraMap:
			cameraName = "%s | %s" % (cname, cid)
			camera = Camera(cameraName)
			camera.setP(P, distortion=distortion, store=True)
			if imageData and ci < len(imgs):
				img, vheight, vwidth = imgs[ci], vheights[ci], vwidths[ci]
				camera.setImageData(img, vheight, vwidth, 3)

			win.view().addCamera(camera)
			cameraMap[cid] = cname

	cams = GLCameras(camera_ids, mats)
	if 'colour' in attrs and attrs['colour'] is not None:
		cams.colour = attrs['colour']

	if 'colours' in attrs and attrs['colours'] is not None:
		cams.colours = attrs['colours']

	win.setLayer(locationName, cams)

def drawPoints(win, locationName, attrs, interface, picked):
	drawX2ds(win, locationName, attrs, interface, picked)
	drawX3ds(win, locationName, attrs, interface, picked)

def drawX3ds(win, locationName, attrs, interface, picked):
	from UI import GLPoints3D

	if 'x3ds' not in attrs: return

	layerName = locationName
	if attrs['type'] == 'points': layerName += '_x3ds'
	if not win.hasLayer(layerName):
		layer = GLPoints3D([])
		win.setLayer(layerName, layer)

	layer = win.getLayer(layerName)

	if 'x3ds_pointSize' in attrs: layer.pointSize = attrs['x3ds_pointSize']
	else: layer.pointSize = 8

	if 'x3ds_colour' in attrs: layer.colour = attrs['x3ds_colour']
	else: layer.colour = (1, 0.5, 0, 0.7)

	colours = np.array([], dtype=np.float32)
	if 'x3ds_colours' in attrs and attrs['x3ds_colours'].any(): colours = attrs['x3ds_colours']

	if 'drawStyle' in attrs and attrs['drawStyle']: layer.drawStyle = attrs['drawStyle']

	x3ds = np.array(attrs['x3ds'], dtype=np.float32)
	x3ds_labels = None
	if 'x3ds_labels' in attrs:
		x3ds_labels = attrs['x3ds_labels']
		layer.setData(x3ds, names=Interface.getLabelNames(x3ds_labels), colours=colours)
	else:
		layer.setData(x3ds, colours=colours)

	if layer: layer.visible = isVisible(attrs)

	if 'normals' in attrs:
		layer.normals = attrs['normals']

	if 'edges' in attrs and attrs['edges'] is not None:
		layer.edges = attrs['edges']
		# layer.graph = attrs['trackGraph']

	# Check if we want to draw the camera ray contributions
	# Make sure:
	#  - We have a 3D object that has been picked
	#  - The picked object is a 3D point
	#  - The picked layer matches the one we are processing
	if picked is not None and picked['type'] == '3d' and picked['primitiveType'] == 'GLPoints3D' and win.view().layers.keys()[picked['primitiveIndex']] == layerName:
		if 'index' not in picked: return
		if picked['isLabel']:
			li = picked['index']
			if x3ds_labels is not None:
				liIdx = np.where(x3ds_labels == li)[0]
				xi = liIdx[0] if liIdx else li
			else:
				xi = li
		else:
			xi = picked['index']
			li = x3ds_labels[xi]

		if xi >= len(x3ds): return
		selected_x3d = x3ds[xi]

		# Create a yellow highlight around (behind) the picked 3D point
		highlightLayer = GLPoints3D(np.array([selected_x3d], dtype=np.float32), colour=(1, 1, 0, 0.9))
		highlightLayer.pointSize = layer.pointSize + 6.
		win.setLayer('x3d_selected', highlightLayer, selection=True)

		if selected_x3d.any() and 'showCameraContributions' in attrs and attrs['showCameraContributions']:
			if 'camerasLocation' not in attrs: return
			if 'x3ds_labels' not in attrs: return
			if 'labels' not in attrs or 'x2ds_splits' not in attrs: return

			camsLoc = attrs['camerasLocation']
			cams = interface.location(camsLoc)
			if cams is None:
				print 'Render Callback: No cameras found when showing 3D point ray contributions.'
				return

			x2ds_labels = attrs['labels']
			x2ds_splits = attrs['x2ds_splits']
			camIds = [interface.findCameraIdFromRayId(rayId, x2ds_splits) for rayId in np.where(x2ds_labels == li)[0]]

			camNames = [cam.name for ci, cam in enumerate(win.view().cameras[1:]) if ci in camIds]
			camPositions = np.array([m[4] for m in cams['mats']], dtype=np.float32)[camIds]
			print '3D Point', li, '|', 'Cameras:', camIds, camNames
			pts = [selected_x3d]
			pts.extend(camPositions)
			edges = [[0, idx + 1] for idx in range(len(camPositions))]
			selLayer = GLPoints3D(pts, edges=edges, colour=(0, 1, 1, 0.5))
			win.setLayer('x3d_cameraContributions', selLayer, selection=True)

			# Camera panels test
			if False:
				try:
					if win.cameraPanels is None:
						win.cameraPanels = QApp.CameraPanels(mainView=win.view())

					win.cameraPanels.resetCurrPos()
					for c in camIds:
						win.cameraPanels.setCameraLayer(c + 1)

					win.cameraPanels.show()
					win.cameraPanels.update()
				except Exception as e:
					print e

def drawX2ds(win, locationName, attrs, interface, picked):
	from UI import GLPoints2D
	if 'x2ds' in attrs and attrs['x2ds'].any(): dets = attrs['x2ds']
	elif 'rx2ds' in attrs and attrs['rx2ds'].any(): dets = attrs['rx2ds']
	else:
		if locationName in primitive2D_map:
			if primitive2D_map[locationName] in win.view().primitives2D:
				del win.view().primitives2D[primitive2D_map[locationName]]
			del primitive2D_map[locationName]
		return

	if not locationName in primitive2D_map:
		primitiveIndex = len(primitive2D_map)
		win.view().primitives2D.append(GLPoints2D(([], [])))
		primitive2D_map[locationName] = primitiveIndex
	else:
		primitiveIndex = primitive2D_map[locationName]

	# TODO: Needs the layer treatment?
	if primitiveIndex >= len(win.view().primitives2D): return
	primitive2d = win.view().primitives2D[primitiveIndex]
	primitive2d.visible = isVisible(attrs)

	if 'x2ds_pointSize' in attrs: primitive2d.pointSize = attrs['x2ds_pointSize']
	else: primitive2d.pointSize = 8

	if 'x2ds_colour' in attrs: primitive2d.colour = attrs['x2ds_colour']
	else: primitive2d.colour = (0., 1.0, 0., 0.5)

	if 'x2ds_colours' in attrs: primitive2d.colours = attrs['x2ds_colours']

	labelNames = None
	if 'labels' in attrs: labelNames = Interface.getLabelNames(attrs['labels'])
	splits = attrs['x2ds_splits_render'] if 'x2ds_splits_render' in attrs else attrs['x2ds_splits']
	primitive2d.setData(dets, splits, names=labelNames)

	if picked is not None:
		picked3dLayer = win.view().layers.keys()[picked['primitiveIndex']]
		try:
			picked2dLayer = (key for key, value in primitive2D_map.items() if value == picked['primitiveIndex']).next()
		except:
			picked2dLayer = None

		# If we've picked a 3D point, highlight the corresponding detections/rays in the cameras
		if picked['type'] == '3d' and picked['primitiveType'] == 'GLPoints3D' and 'labels' in attrs:
			if 'index' not in picked: return
			prim = win.view().primitives[picked['primitiveIndex']]
			# TODO: Fix bug
			from UI import GLSkel, GLGrid
			if isinstance(prim, GLSkel) or isinstance(prim, GLGrid): return

			x3d_labels = prim.names
			if not x3d_labels or picked['index'] not in x3d_labels: return
			pointLabel = int(x3d_labels[picked['index']])
			matchingLabels = np.where(np.array(attrs['labels'], dtype=np.int32) == pointLabel)[0]
			pointLengths = np.zeros((len(splits) - 1, 1), dtype=np.int32)
			detsCamsFor3dPoint = [interface.findCameraIdFromRayId(l, attrs['x2ds_splits']) for l in matchingLabels]
			pointLengths[detsCamsFor3dPoint] = 1
			camSplits = Interface.makeSplitBoundaries(pointLengths)

			selIndex = len(primitive2D_map)
			win.view().primitives2D.append(GLPoints2D(([], [])))
			primitive2D_map['detsForSelectedX3d'] = selIndex
			selectionLayer = win.view().primitives2D[selIndex]
			selectionLayer.setData(dets[matchingLabels], camSplits)
			selectionLayer.colour = (1, 1, 0, 0.9)
			selectionLayer.pointSize = primitive2d.pointSize + 4
			selectionIndices.append(selIndex)

		elif picked['type'] == '2d' and picked['primitiveType'] == 'GLPoints2D' and picked2dLayer == locationName: # and 'labels' in attrs:
			selectedLabel = picked['index']
			if selectedLabel == -1: return

			# Highlight picked detection (in all cameras)
			selIndex = len(primitive2D_map)
			win.view().primitives2D.append(GLPoints2D(([], [])))
			primitive2D_map['selectedDets'] = selIndex
			selectionLayer = win.view().primitives2D[selIndex]

			pickedPrimitive2d = win.view().primitives2D[picked['primitiveIndex']]
			indexOffset = pickedPrimitive2d.bounds[picked['cameraIndex']]
			absoluteIndex = selectedLabel + indexOffset

			pointLengths = np.zeros((len(splits) - 1, 1), dtype=np.int32)

			if 'labels' in attrs:
				labels = attrs['labels']
				labelIndex = labels[absoluteIndex]

				# Find the same label in all cameras
				matchingLabels = np.where(np.array(labels, dtype=np.int32) == labelIndex)[0]
				detsCams = [interface.findCameraIdFromRayId(l, attrs['x2ds_splits']) for l in matchingLabels]
			else:
				# Note: Assumption the video cams are added first and camera gunk, not great
				detsCams = [picked['cameraIndex']]
				matchingLabels = [selectedLabel + attrs['x2ds_splits'][detsCams[0]]]

			pointLengths[detsCams] = 1
			camSplits = Interface.makeSplitBoundaries(pointLengths)

			selectionLayer.setData(dets[matchingLabels], camSplits)
			selectionLayer.colour = (1, 1, 0, 0.9)
			selectionLayer.pointSize = primitive2d.pointSize + 4
			selectionIndices.append(selIndex)

def drawMesh(win, locationName, attrs, interface, picked):
	from UI import GLMeshes

	# TODO: Add support for multiple meshes. The mesh ops should create lists of vs etc. and we iterate over the lot
	if 'vs' not in attrs or 'tris' not in attrs: return
	vs = attrs['vs']
	tris = attrs['tris']
	vts = attrs['vts'] if 'vts' in attrs else None
	drawStyle = attrs['drawStyle'] if 'drawStyle' in attrs else 'smooth'
	colour = attrs['colour'] if 'colour' in attrs else (0.9, 0.9, 0.9, 1.0)
	names = attrs['names'] if 'names' in attrs else ['default']
	transforms = attrs['transforms'] if 'transforms' in attrs else None

	layerName = locationName + '_mesh'
	if layerName not in win.getLayers():
		mesh = GLMeshes(names=names, verts=vs, faces=tris, drawStyle=drawStyle, colour=colour, vts=vts, transforms=transforms)
		win.setLayer(layerName, mesh)
	else:
		for _vs, _name in zip(vs, names):
			layer = win.getLayer(layerName)
			layer.setVs(_vs, _name)
			layer.colour = colour
			layer.drawStyle = drawStyle

	layer = win.getLayer(layerName)
	if layer: layer.visible = isVisible(attrs)

	if 'texture' in attrs:
		layer.setImage(attrs['texture'])

def drawCharacter(win, locationName, attrs, interface, picked):
	skelLayerName = locationName + '_skel'
	if not win.hasLayer(skelLayerName):
		if 'skeleton' in attrs:
			mesh = attrs['skeleton']
			skelLayer = win.setLayer(skelLayerName, mesh)
			skelLayer.visible = isVisible(attrs)
	else:
		skelLayer = win.getLayer(skelLayerName)
		skelLayer.visible = isVisible(attrs)
		if 'Gs' in attrs:
			skelLayer.setPose(attrs['Gs'])

	geoLayerName = locationName + '_geo'
	if not win.hasLayer(geoLayerName):
		if 'geometry' in attrs:
			geoLayer = win.setLayer(geoLayerName, attrs['geometry'])
			geoLayer.visible = isVisible(attrs)
	else:
		geoLayer = win.getLayer(geoLayerName)
		geoLayer.visible = isVisible(attrs)
		if 'geom_Vs' in attrs:
			geoLayer.setVs(attrs['geom_Vs'])
		if 'geom_Gs' in attrs:
			geoLayer.setPose(attrs['geom_Gs'])

	if 'geo_colour' in attrs and geoLayer:
		geoLayer.colour = attrs['geo_colour']

def drawSkeleton(win, locationName, attrs, interface, picked):
	from UI import GLSkel, COLOURS
	if 'skelDict' not in attrs: return

	boneColour = COLOURS['Bone']
	if 'boneColour' in attrs: boneColour = attrs['boneColour']

	skelDict = attrs['skelDict']
	Gs = attrs['Gs'] if 'Gs' in attrs else skelDict['Gs']

	skelLayer = None
	if locationName not in win.getLayers() or 'override' in attrs:
		skel = GLSkel(skelDict['Bs'], Gs, mvs=get(skelDict, 'markerOffsets'), mvis=get(skelDict, 'markerParents'), bone_colour=boneColour)
		if 'subjectName' in attrs: skel.setName(attrs['subjectName'])
		skelLayer = win.setLayer(locationName, skel)
	else:
		skelLayer = win.getLayer(locationName)
		skelLayer.setPose(Gs)

	if skelLayer: skelLayer.setBoneColour(boneColour)

	layer = win.getLayer(locationName)
	if layer: layer.setVisible(isVisible(attrs))

	# For debugging, plot markers to joint mapping
	if 'jointMarkersMap' in attrs:
		from GCore import SolveIK
		effectorData = SolveIK.make_effectorData(skelDict)
		useMarkers = True
		if useMarkers:
			effectorLabels = np.array([int(mn) for mn in skelDict['markerNames']], dtype=np.int32)
			x3ds, x3ds_labels = SolveIK.skeleton_marker_positions(skelDict, skelDict['rootMat'], skelDict['chanValues'],
			                                                      effectorLabels, effectorData, skelDict['markerWeights'])
		else:
			import ISCV
			effectorTargets = np.zeros_like(effectorData[1])
			numEffectors = len(effectorTargets)
			effectors = np.zeros((numEffectors, 3, 4), dtype=np.float32)
			residual = np.zeros((numEffectors, 3, 4), dtype=np.float32)
			sc = ISCV.pose_effectors(effectors, residual, skelDict['Gs'], effectorData[0], effectorData[1], effectorData[2], effectorTargets)
			x3ds = effectors[:, :3, 3]

		[j_inds, m_inds, colours] = attrs['jointMarkersMap']

		effectorIdx = 0
		for i, (ji, mi) in enumerate(zip(j_inds, m_inds)):
			if not ji or not mi: continue
			mappingAttrs = {
				'colour': (0.1, 0.4, 0.1, 0.5),
				'edgeColour': colours[i],
				'pointSize': 8
			}
			if not useMarkers:
				mi = min(effectorIdx, len(x3ds) - 1)
				effectorIdx += 1
				mappingAttrs['x1'] = np.array([x3ds[mi]], dtype=np.float32)
			else:
				mappingAttrs['x1'] = x3ds[mi]

			mappingAttrs['x0'] = skelDict['Gs'][ji][:, :3, 3]

			childName = os.path.join(locationName, '/', 'jointToMarkers_%s' % skelDict['jointNames'][ji[0]])
			drawEdges(win, childName, mappingAttrs, interface, picked)

def drawEdges(win, locationName, attrs, interface, picked):
	from UI import GLPoints3D
	edgesLayer = None
	if not win.hasLayer(locationName):
		edgesLayer = win.setLayer(locationName, GLPoints3D(np.array([], dtype=np.float32)))
	else:
		edgesLayer = win.getLayer(locationName)

	points_x0 = attrs['x0']
	points_x1 = attrs['x1']

	points = np.concatenate((points_x0, points_x1))
	conns = [[i, i + len(points_x0)] for i in range(len(points_x1))]

	colours = np.array([], dtype=np.float32)
	if 'x3ds_colours' in attrs: colours = attrs['x3ds_colours']

	if 'x3ds_labels' in attrs:
		x3ds_labels = attrs['x3ds_labels']
		edgesLayer.setData(np.array(points, dtype=np.float32), names=Interface.getLabelNames(x3ds_labels), colours=colours)
	else:
		edgesLayer.setData(np.array(points, dtype=np.float32), colours=colours)

	edgesLayer.edges = conns
	edgesLayer.visible = isVisible(attrs)

	if 'colour' in attrs:
		edgesLayer.colour = attrs['colour']
		edgesLayer.edgeColour = attrs['colour']
	else:
		edgesLayer.colour = (0., 0., 0., 1.)
		edgesLayer.edgeColour = (0., 0., 0., 1.)

	if 'edgeColour' in attrs: edgesLayer.edgeColour = attrs['edgeColour']

	if 'pointSize' in attrs: edgesLayer.pointSize = attrs['pointSize']
	else: edgesLayer.pointSize = 6

def drawBones(win, locationName, attrs, interface, picked):
	from UI import GLBones

	boneVertices = attrs['verts']
	boneEdges = attrs['edges']
	if boneVertices is None or boneEdges is None: return

	if not win.hasLayer(locationName):
		win.setLayer(locationName, GLBones(boneVertices, boneEdges))
	else:
		bonesLayer = win.getLayer(locationName)
		bonesLayer.vertices = boneVertices

def drawPrimitive(win, locationName, attrs, interface, picked):
	from UI.GLPrimitives import GLPrimitive

	if not win.hasLayer(locationName):
		win.setLayer(locationName, GLPrimitive(attrs))
	else:
		primLayer = win.getLayer(locationName)
		primLayer.setAttrs(attrs)

def drawSphere(win, locationName, attrs, interface, picked):
	from UI.GLPrimitives import GLPrimitive
	attrs['primitiveType'] = 'sphere'

	if not win.hasLayer(locationName):
		win.setLayer(locationName, GLPrimitive(attrs))
	else:
		primLayer = win.getLayer(locationName)
		primLayer.setAttrs(attrs)

def drawCube(win, locationName, attrs, interface, picked):
	from UI.GLPrimitives import GLPrimitive
	attrs['primitiveType'] = 'cube'

	if not win.hasLayer(locationName):
		win.setLayer(locationName, GLPrimitive(attrs))
	else:
		primLayer = win.getLayer(locationName)
		primLayer.setAttrs(attrs)
