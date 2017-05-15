import Op, Interface
import IO
import numpy as np
import time

from IO import MAReader
import pyopencl as cl
from GCore import Character
from GCore import SolveIK


class MarkerMesh(Op.Op):
	def __init__(self, name='/Marker_Mesh', locations='', filename='', texture='', drawStyle='wire_over_smooth', colour='(1.0, 1.0, 0.0, 0.2)'):
		self.styleOptions = ('wire_over_smooth', 'wire', 'smooth')
		fields = [
			('name', 'name', 'Skeleton name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('filename', 'Filename', 'Mesh filename', 'filename', filename, {}),
			('texture', 'Texture', 'Solving Skeleton filename', 'filename', texture, {}),
			('drawStyle', 'Draw style', 'Draw style', 'select', drawStyle, {'enum': self.styleOptions}),
			('colour', 'Mesh colour', 'Mesh colour', 'string', str(colour), {})
		]

		super(self.__class__, self).__init__(name, fields)

		self.tris, self.verts, self.x3ds_normals = None, None, None
		self.vts, self.texture = None, None

	def flush(self):
		self.tris, self.verts, self.x3ds_normals = None, None, None
		self.vts, self.texture = None, None

	def setup(self, interface, attrs):
		if self.tris is None or self.verts is None:
			if 'filename' in attrs and attrs['filename']:
				filename = self.resolvePath(attrs['filename'])
				try:
					_, (self.tris, self.verts, self.x3ds_normals) = IO.load(filename)
				except Exception as e:
					self.logger.error('Could not load template mesh: %s' % filename)
					return

				tc = np.array([[.5, .1], [0., 1], [1., 0.]])
				self.vts = np.repeat([tc], self.tris.shape[0], axis=0)
				self.tris = np.array(self.tris, dtype=np.int32)

			if 'texture' in attrs and attrs['texture']:
				self.texture = self.resolvePath(attrs['texture'])

	def cook(self, location, interface, attrs):
		if self.verts is None or self.tris is None: return
		# Note: We should have a widget for options such as string as we now have to work around an enum niggle
		style = attrs['drawStyle']
		if isinstance(style, int): style = self.styleOptions[style]

		meshAttrs = {
			'vs': [self.verts],
			'tris': [self.tris],
			'drawStyle': style,
			'colour': [eval(attrs['colour'])],
			'names': ['triangles'],
			'normals': [self.x3ds_normals]
		}
		if self.vts.any():
			meshAttrs['vts'] = [self.vts]

		if self.texture:
			meshAttrs['texture'] = [self.texture]

		interface.createChild(interface.name(), 'mesh', atLocation=interface.parentPath(), attrs=meshAttrs)


class NormalsFromX2ds(Op.Op):
	def __init__(self, name='/NormalsFromX2ds', locations='', x2ds='', calibration='', frameRange=''):
		fields = [
			('name', 'name', 'Skeleton name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('x2ds', 'x2ds', 'x2ds', 'string', x2ds, {}),
			('calibration', 'calibration', 'calibration', 'string', calibration, {}),
			('frameRange', 'Frame range', 'Frame range', 'string', frameRange, {})
		]

		super(self.__class__, self).__init__(name, fields)
		self.tris = None

	def cook(self, location, interface, attrs):
		if not self.useFrame(interface.frame(), attrs['frameRange']): return
		x3ds = interface.attr('x3ds')
		if x3ds is None:
			self.logger.error('No x3ds found at: %s' % location)
			return

		x3ds_labels = interface.attr('x3ds_labels')
		if x3ds_labels is None:
			self.logger.error('No 3D labels found at: %s' % location)
			return

		x2dsLocation = attrs['x2ds']
		x2ds, splits = interface.attr('x2ds', atLocation=x2dsLocation), interface.attr('x2ds_splits', atLocation=x2dsLocation)
		if x2ds is None or splits is None:
			self.logger.error('No detections found at: %s' % x2dsLocation)
			return

		calLocation = attrs['calibration']
		Ps = interface.attr('Ps', atLocation=calLocation)
		if Ps is None:
			self.logger.error('No calibration data found at: %s' % calLocation)
			return

		import ISCV
		x2d_threshold, pred_2d_threshold = 6./2000., 100./2000
		clouds = ISCV.HashCloud2DList(x2ds, splits, max(pred_2d_threshold, x2d_threshold))
		sc, labels, _ = clouds.project_assign(np.ascontiguousarray(x3ds, dtype=np.float32), x3ds_labels, Ps, x2d_threshold)

		mats = interface.attr('mats', atLocation=calLocation)
		camPositions = np.array([m[4] for m in mats], dtype=np.float32)
		normals = np.zeros_like(x3ds)
		for xi, (x3d, label3d) in enumerate(zip(x3ds, x3ds_labels)):
			camIds = [interface.findCameraIdFromRayId(rayId, splits) for rayId in np.where(labels == label3d)[0]]
			if not camIds: continue
			# camPos = Ps[camIds][:, :3, 3]
			camPos = camPositions[camIds]
			rays = camPos - [x3d] * len(camPos)
			rays = np.float32([ray / np.linalg.norm(ray) for ray in rays])
			raysDps = np.dot(rays, rays.T)
			bestRay = np.sum(raysDps > 0, axis=0).argmax()
			# goodRays = np.where(raysDps[bestRay] > 0.05)[0]
			normals[xi] = rays[bestRay]

		interface.setAttr('normals', normals)


class TriangulateMarkerMesh(Op.Op):
	def __init__(self, name='/Triangulate_Marker_Mesh', locations='', drawStyle='wire_over_smooth', colour='(1.0, 1.0, 0.0, 0.2)', frameRange=''):
		fields = [
			('name', 'name', 'Skeleton name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('drawStyle', 'Draw style', 'Draw style', 'string', drawStyle, {}),
			('colour', 'Mesh colour', 'Mesh colour', 'string', str(colour), {}),
			('frameRange', 'Frame range', 'Frame range', 'string', frameRange, {})
		]

		super(self.__class__, self).__init__(name, fields)
		self.tris = None

	def getCylindricalCoords(self, pts, axis_a=1, axis_b=0, d=2):
		r = np.linalg.norm([pts[:, axis_a], pts[:, axis_b]], axis=0)
		theta = np.degrees(np.arctan(pts[:, axis_b] / pts[:, axis_a]))
		depth = pts[:, d]
		return np.array([r, theta, depth], dtype=np.float32).transpose()

	def cook(self, location, interface, attrs):
		if not self.useFrame(interface.frame(), attrs['frameRange']): return
		x3ds = interface.attr('x3ds')
		if x3ds is None:
			self.logger.error('No 3D data found at: %s' % location)
			return

		x3ds_labels = interface.attr('x3ds_labels')
		normals = interface.attr('normals')

		if self.tris is None:
			if False:
				cylCoords = self.getCylindricalCoords(x3ds - np.mean(x3ds, axis=0))
				import matplotlib.pyplot as plt
				plt.plot(cylCoords[:, 0], cylCoords[:, 1], linestyle='None', marker='.')
				plt.show()

				from scipy.spatial import Delaunay
				self.tris = Delaunay(cylCoords[:, :2]).simplices
			else:
				import TrianglesUtils as triUtils
				self.tris = triUtils.createTriangles(x3ds, x3ds_labels, normals)

		mAttrs = {
			'vs': [x3ds],
			'tris': np.int32([self.tris]),
			'names': ['triangles'],
			'drawStyle': attrs['drawStyle'],
			'colour': [eval(attrs['colour'])]
		}
		interface.createChild('mesh', 'mesh', attrs=mAttrs)


class CharacterFromSkeleton(Op.Op):
	def __init__(self, name='/Character', locations='', skelFilename='', geoColour=(0.6, 0.6, 0.6, 1.0)):
		fields = [
			('name', 'name', 'Skeleton name', 'string', name, {}),
			('locations', 'Skeleton locations', 'Skeleton locations', 'string', locations, {}),
			('skelFilename', 'Skeleton filename', 'Skeleton filename', 'filename', skelFilename, {}),
			('geoColour', 'Geometry colour', 'Geometry colour', 'string', str(geoColour), {})
		]

		super(self.__class__, self).__init__(name, fields)

		self.skelDict = None
		self.mesh_dict = None
		self.skel_mesh = None
		self.geom_mesh = None

	def flush(self):
		self.skelDict, self.mesh_dict, self.skel_mesh, self.geom_mesh = None, None, None, None

	def cook(self, location, interface, attrs):
		if self.skelDict is None or self.mesh_dict is None:
			skelFilename = self.resolvePath(attrs['skelFilename'])

			# Use the filename if given to load the skeleton dictionary, otherwise use the cooked skeleton
			if skelFilename:
				try:
					_, self.skelDict = IO.load(skelFilename)

				except Exception as e:
					self.logger.error('Could not open skeleton: \'{}\''.format(skelFilename))
					return
			else:
				self.skelDict = interface.attr('skelDict')

			if self.skelDict is None: return
			rootMat = self.skelDict['rootMat'] if 'rootMat' in self.skelDict else None
			# TODO: This should happen in the render callback
			self.mesh_dict, self.skel_mesh, self.geom_mesh = Character.make_geos(self.skelDict, rootMat)

		# Test updating on the fly (TEMP)
		rootMat = self.skelDict['rootMat'] if 'rootMat' in self.skelDict else None
		Character.updatePose(self.skelDict, x_mat=rootMat)

		interface.setAttr('skelDict', self.skelDict)
		interface.setAttr('meshDict', self.mesh_dict)

		charAttrs = {'skeleton': self.skel_mesh, 'geometry': self.geom_mesh, 'geo_colour': eval(attrs['geoColour'])}
		interface.createChild('meshes', 'character', attrs=charAttrs)


class MayaCharacter(Op.Op):
	def __init__(self, name='/Maya Character', maFilename=''):
		fields = [
			('name', 'name', 'Maya character name', 'string', name, {}),
			('filename', 'filename', 'Maya filename', 'filename', maFilename, {})
		]

		super(self.__class__, self).__init__(name, fields)

	def cook(self, location, interface, attrs):
		if not 'filename' in attrs or not attrs['filename']: return
		filename = self.resolvePath(attrs['filename'])

		try:
			character, _ = MAReader.loadMayaCharacter(filename)
		except IOError as e:
			self.logger.error('Could not load maya character: %s' % str(e))
			return

		if not character: return

		charAttrs = {'skeleton': character.skel_primitive, 'geometry': character.geom_mesh}
		interface.createChild('skeleton', 'character', attrs=charAttrs)


class SkeletonBoundingBox(Op.Op):
	def __init__(self, name='/Skeleton Bounding Box', locations=''):
		fields = [
			('name', 'name', 'Maya character name', 'string', name, {}),
			('locations', 'Skeleton locations', 'Skeleton locations', 'locations', locations, {})
		]

		super(self.__class__, self).__init__(name, fields)

	def cook(self, location, interface, attrs):
		skelDict = interface.attr('skelDict')
		if skelDict is None:
			self.logger.warning('Skeleton dictionary not found at: %s' % location)
			return

		print "..."


class UpdateMarkerMeshFromSkeleton(Op.Op):
	def __init__(self, name='/Mesh Update', locations='', sourceSkeleton='', useWeights=True):
		fields = [
			('name', 'name', 'Skeleton name', 'string', name, {}),
			('locations', 'Mesh locations', 'Mesh locations', 'string', locations, {}),
			('source', 'Source skeleton', 'Source skeleton', 'string', sourceSkeleton, {}),
			('useWeights', 'Use weights', 'Use weights', 'bool', useWeights, {})
		]

		super(self.__class__, self).__init__(name, fields)
		self.effectorData, self.effectorLabels = {}, {}

	def cook(self, location, interface, attrs):
		if not location or not attrs['source']: return

		sourceLocation = attrs['source']
		skelDict_source = interface.attr('skelDict', atLocation=sourceLocation)
		if skelDict_source is None:
			self.logger.error('No source skeleton found at: %s' % sourceLocation)
			return

		if 'markerWeights' not in skelDict_source:
			self.logger.warning('No marker weights found in skeleton at: %s' % sourceLocation)
			return

		markerWeights = skelDict_source['markerWeights'] if attrs['useWeights'] else None

		if location not in self.effectorData: self.effectorData[location] = SolveIK.make_effectorData(skelDict_source)
		if location not in self.effectorLabels: self.effectorLabels[location] = np.array([int(mn) for mn in skelDict_source['markerNames']], dtype=np.int32)
		vs, vs_labels = SolveIK.skeleton_marker_positions(skelDict_source, skelDict_source['rootMat'], skelDict_source['chanValues'],
														self.effectorLabels[location], self.effectorData[location],
														markerWeights=markerWeights)

		interface.setAttr('vs', [vs])
		interface.setAttr('vs_labels', [vs_labels])


class UpdateMarkerMeshFromX3ds(Op.Op):
	def __init__(self, name='/Update_Marker_Mesh_X3ds', locations='', x3ds='', frameRange=''):
		fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'Mesh locations', 'Mesh locations', 'string', locations, {}),
			('source', 'Source X3DS', 'Source X3DS', 'string', x3ds, {}),
			('frameRange', 'Frame range', 'Frame range', 'string', frameRange, {})
		]

		super(self.__class__, self).__init__(name, fields)

	def cook(self, location, interface, attrs):
		if not self.useFrame(interface.frame(), attrs['frameRange']): return
		if not location or not attrs['source']: return

		sourceLocation = attrs['source']
		x3ds = interface.attr('x3ds', atLocation=sourceLocation)
		if x3ds is None:
			self.logger.error('No x3ds found at: %s' % sourceLocation)
			return

		interface.setAttr('vs', [x3ds])


class UpdatePoseAndMeshes(Op.Op):
	def __init__(self, name='/Pose and Mesh Update', locations='', animation=''):
		fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'locations', 'Skeleton and Mesh locations', 'string', locations, {}),
			('animation', 'Skeleton animation', 'Source skeleton animation', 'string', animation, {})
		]

		super(self.__class__, self).__init__(name, fields)

	def cook(self, location, interface, attrs):
		skelDict = interface.attr('skelDict')
		if skelDict is None: return

		animationLocation = attrs['animation']
		if not animationLocation: animationLocation = location
		animDict = interface.attr('animDict', atLocation=animationLocation)
		if animDict is not None:
			# Check if the source animation has indicated that we should be using a particular frame (e.g. due to offsets and step size)
			frame = interface.attr('frame', atLocation=animationLocation)
			if frame is None: frame = interface.frame()

			animData = animDict['dofData'][frame]
			if skelDict['numChans'] != len(animData):
				fle = skelDict['chanValues'].copy()
				animSkelDict = interface.attr('skelDict', atLocation=animationLocation)
				chanIdxs = [skelDict['chanNames'].index(cn) for ci, cn in enumerate(animSkelDict['chanNames']) if cn in skelDict['chanNames']]
				fle[chanIdxs] = animData
				animData = fle

			Character.updatePose(skelDict, animData)

		else:
			rootMat = skelDict['rootMat'] if 'rootMat' in skelDict else None
			Character.updatePose(skelDict, x_mat=rootMat)

		interface.setAttr('Gs', skelDict['Gs'], atLocation=location + '/meshes')
		if 'geom_dict' in skelDict:
			interface.setAttr('geom_Gs', skelDict['geom_Gs'], atLocation=location + '/meshes')
			interface.setAttr('geom_Vs', skelDict['geom_Vs'], atLocation=location + '/meshes')


def createNormalsKernel():
	CL_ctx = cl.create_some_context(False)
	CL_queue = cl.CommandQueue(CL_ctx)

	cl_normalsProgram = cl.Program(CL_ctx, '''
	__kernel void compute_normals(
			__global const float *xs_g,
			__global const int *edgeList_g,
			const int neighbours,
			__global float *res_g
			)
	{

		const int gid = get_global_id(0);
		const int g10 = gid*neighbours;
		const int g3 = gid*3;
		float sx=0,sy=0,sz=0;
		const float x=xs_g[g3],y=xs_g[g3+1],z=xs_g[g3+2];
		int e3 = edgeList_g[g10]*3;
		float ex0 = xs_g[e3]-x, ey0 = xs_g[e3+1]-y, ez0 = xs_g[e3+2]-z;
		for (int i = 1; i < neighbours; ++i) {
			e3 = edgeList_g[g10+i]*3;
			if (xs_g[e3] > 1e10) continue;
			float ex1 = xs_g[e3]-x, ey1 = xs_g[e3+1]-y, ez1 = xs_g[e3+2]-z;
			sx += ey0*ez1-ey1*ez0;
			sy += ez0*ex1-ez1*ex0;
			sz += ex0*ey1-ex1*ey0;
			ex0=ex1; ey0=ey1; ez0=ez1;
		}
		const float sum = sx*sx+sy*sy+sz*sz;
		if (sum < 1e-8) { sx = 0; sy = 0; sz = 0; }
		else {
			const float sc = rsqrt(sum);
			sx *= sc;
			sy *= sc;
			sz *= sc;
		}
		res_g[g3] = sx;
		res_g[g3+1] = sy;
		res_g[g3+2] = sz;
	}
	''').build()

	return CL_ctx, CL_queue, cl_normalsProgram

def trianglesToEdgeList(triangles, numVerts=None, neighbours=10):
	'''Convert a list of triangle indices to an array of neighbouring vertices per vertex (following anticlockwise order).'''
	if numVerts is None: numVerts = np.max(triangles) + 1 if len(triangles) else 1
	if numVerts < 1: numVerts = 1  # avoid empty arrays

	T = [dict() for t in xrange(numVerts)]
	P = [dict() for t in xrange(numVerts)]

	for t0, t1, t2 in triangles:
		T[t0][t1], T[t1][t2], T[t2][t0] = t2, t0, t1
		P[t1][t0], P[t2][t1], P[t0][t2] = t2, t0, t1

	S = np.zeros((numVerts, neighbours), dtype=np.int32)

	for vi, (Si, es, ps) in enumerate(zip(S, T, P)):
		Si[:] = vi
		if not es: continue
		v = es.keys()[0]
		while v in ps: v = ps.pop(v)
		for li in xrange(neighbours):
			Si[li] = v
			if v not in es: break
			v = es.pop(v, vi)

	return S

def calculateNormals(vs, tris, CL_ctx, CL_queue, cl_normalsProgram, reverse=True, neighbours=2):
	# Create edge list (this should be cached in a map for a given triangle list)
	edgeList = trianglesToEdgeList(tris, len(vs), neighbours)
	edgeList_buffer = cl.Buffer(CL_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=edgeList)

	# Calculate normals
	vs_buffer = cl.Buffer(CL_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=vs)
	res_buffer = cl.Buffer(CL_ctx, cl.mem_flags.WRITE_ONLY, vs.nbytes)
	cl_normalsProgram.compute_normals(CL_queue, (vs.shape[0],), None, vs_buffer, edgeList_buffer, np.int32(neighbours), res_buffer)

	# Get the data from the buffer
	normals = np.zeros_like(vs)
	cl.enqueue_copy(CL_queue, normals, res_buffer)

	if reverse:
		normals *= -1

	return normals


class CalculateNormals(Op.Op):
	def __init__(self, name='/Calculate_Normals', locations='', inverse=False):
		fields = [
			('name', 'name', 'Name', 'string', name, {}),
			('locations', 'locations', 'Locations', 'string', locations, {}),
			('inverse', 'Inverse', 'Inverse', 'bool', locations, {})
		]

		super(self.__class__, self).__init__(name, fields)

		self.CL_ctx = None
		self.CL_queue = None
		self.cl_normalsProgram = None

	def setup(self, interface, attrs):
		if self.CL_ctx is None:
			self.CL_ctx, self.CL_queue, self.cl_normalsProgram = createNormalsKernel()

	def cook(self, location, interface, attrs):
		# Get vertices and triangles we are cooking
		vs = interface.attr('vs')
		tris = interface.attr('tris')

		# TODO: Improve this mess
		if vs is not None: vs = vs[0]
		if tris is not None: tris = tris[0]
		try:
			if not vs.any(): return
		except: return

		vs = np.array(vs, dtype=np.float32).reshape(-1, 3)
		tris = np.array(tris, dtype=np.int32).reshape(-1, 3)
		normals = calculateNormals(vs, tris, self.CL_ctx, self.CL_queue, self.cl_normalsProgram)
		interface.setAttr('normals', normals)


'''
Hex and rgb values for Kenneth Kelly's sequence of colours of maximum contrast.
From this paper: https://eleanormaclure.files.wordpress.com/2011/03/colour-coding.pdf
Colours should be used in order.
'''

kelly_colors_hex = [
	0xFFFFFF, # White
	0x000000, # Black
	0xFFB300, # Vivid Yellow
	0x803E75, # Strong Purple
	0xFF6800, # Vivid Orange
	0xA6BDD7, # Very Light Blue
	0xC10020, # Vivid Red
	0xCEA262, # Grayish Yellow
	0x817066, # Medium Gray

	# The following don't work well for people with defective color vision
	0x007D34, # Vivid Green
	0xF6768E, # Strong Purplish Pink
	0x00538A, # Strong Blue
	0xFF7A5C, # Strong Yellowish Pink
	0x53377A, # Strong Violet
	0xFF8E00, # Vivid Orange Yellow
	0xB32851, # Strong Purplish Red
	0xF4C800, # Vivid Greenish Yellow
	0x7F180D, # Strong Reddish Brown
	0x93AA00, # Vivid Yellowish Green
	0x593315, # Deep Yellowish Brown
	0xF13A13, # Vivid Reddish Orange
	0x232C16, # Dark Olive Green
	]

kelly_colors_rgb = [
	[255, 255, 255], # White
	[0, 0, 0], # Black
	[1, 179, 0], # Vivid Yellow
	[128, 62, 117], # Strong Purple
	[255, 104, 0], # Vivid Orange
	[166, 189, 215], # Very Light Blue
	[193, 0, 32], # Vivid Red
	[206, 162, 98], # Grayish Yellow
	[129, 112, 102], # Medium Gray

	# these aren't good for people with defective color vision:
	[0, 125, 52], # Vivid Green
	[246, 118, 142], #Strong Purplish Pink
	[0, 83, 138], # Strong Blue
	[255, 122, 92], # Strong Yellowish Pink
	[83, 55, 122], # Strong Violet
	[255, 142, 0], # Vivid Orange Yellow
	[179, 40, 81], # Strong Purplish Red
	[244, 200, 0], # Vivid Greenish Yellow
	[127, 24, 13], # Strong Reddish Brown
	[147, 170, 0], # Vivid Yellowish Green
	[89, 51, 21], # Deep Yellowish Brown
	[241, 58, 19], # Vivid Reddish Orange
	[35, 44, 22] # Dark Olive Green
]

kelly_colors = dict(white=(255, 255, 255),
					black=(0, 0, 0),
					vivid_yellow=(255, 179, 0),
					strong_purple=(128, 62, 117),
					vivid_orange=(255, 104, 0),
					very_light_blue=(166, 189, 215),
					vivid_red=(193, 0, 32),
					grayish_yellow=(206, 162, 98),
					medium_gray=(129, 112, 102),

					# these aren't good for people with defective color vision:
					vivid_green=(0, 125, 52),
					strong_purplish_pink=(246, 118, 142),
					strong_blue=(0, 83, 138),
					strong_yellowish_pink=(255, 122, 92),
					strong_violet=(83, 55, 122),
					vivid_orange_yellow=(255, 142, 0),
					strong_purplish_red=(179, 40, 81),
					vivid_greenish_yellow=(244, 200, 0),
					strong_reddish_brown=(127, 24, 13),
					vivid_yellowish_green=(147, 170, 0),
					deep_yellowish_brown=(89, 51, 21),
					vivid_reddish_orange=(241, 58, 19),
					dark_olive_green=(35, 44, 22))


class DisplayShapeWeightGroups(Op.Op):
	def __init__(self, name='/Display Shape Weight Groups', locations='', randomColours=False, seed=None):
		fields = [
			('name', 'name', 'Name', 'string', name, {}),
			('locations', 'locations', 'Locations', 'string', locations, {}),
			('randomColours', 'Random colours', 'Random colours', 'bool', randomColours, {}),
			('seed', 'Seed', 'Seed', 'int', seed, {'min': 0})
		]

		super(self.__class__, self).__init__(name, fields)

	def cook(self, location, interface, attrs):

		skelDict = interface.attr('skelDict')
		if skelDict is None: return

		groups = skelDict['shape_weights'][0][0]

		if attrs['randomColours']:
			np.random.seed(attrs['seed'])
			colours = np.random.rand(len(groups.keys()), 3)
		else:
			colours = np.array(kelly_colors_rgb, dtype=np.float32) / 255.

		colours = np.hstack((colours, np.ones((colours.shape[0], 1))))

		for gi, (k, v) in enumerate(groups.iteritems()):
			idxs = v[0]
			gAttrs = {
				'x3ds': skelDict['geom_Vs'][idxs],
				'x3ds_pointSize': 8,
				'x3ds_colour': colours[gi % len(colours)]
			}

			# Find joint name (annoying)
			suffix = str(gi)
			for jname, idx in skelDict['shape_weights'][0][1].iteritems():
				if k == idx:
					suffix = jname
					break

			interface.createChild('geomVs_%s' % suffix, 'points3d', attrs=gAttrs)


class DisplayJointMarkerMapping(Op.Op):
	def __init__(self, name='Display Joint Marker Mapping', locations=''):
		fields = [
			('name', 'name', 'Name', 'string', name, {}),
			('locations', 'locations', 'Locations', 'string', locations, {})
		]

		super(self.__class__, self).__init__(name, fields)

	def cook(self, location, interface, attrs):
		skelDict = interface.attr('skelDict')
		if skelDict is None:
			self.logger.warning('No skeleton found at: %s' % location)
			return

		from Ops import Interface
		vs, vs_labels = Interface.getWorldSpaceMarkerPos(skelDict)

		np.random.seed(100)
		colours = np.random.rand(skelDict['numJoints'], 3)
		colours = np.hstack((colours, np.ones((colours.shape[0], 1))))

		x0s, x1s = [], []
		for mi, ji in enumerate(skelDict['markerParents']):
			x0s.append(vs[mi])
			x1s.append(skelDict['Gs'][ji][:, 3])

		mappingAttrs = {
			'x0': x0s,
			'x1': x1s,
			'colour': (0.1, 0.4, 0.1, 0.5),
			'lineColour': (0.3, 0.2, 0.5, 0.5),
			'pointSize': 8
		}
		interface.createChild('markerToJoints', 'edges', attrs=mappingAttrs)


class ProcLOD(Op.Op):
	def __init__(self, name='/Proc LOD', locations='', colour=(0.6, 0.1, 0.6, 0.4), renderLod=False, displayNormals=False,
				neighbours=2, jointHeightMult=1.1):
		fields = [
			('name', 'name', 'Name', 'string', name, {}),
			('locations', 'Skeleton locations', 'Locations', 'string', locations, {}),
			('colour', 'colour', 'colour', 'string', str(colour), {}),
			('renderLod', 'Render LOD', 'Render LOD', 'bool', renderLod, {}),
			('displayNormals', 'Display normals', 'Display normals', 'bool', displayNormals, {}),
			('neighbours', 'Neighbours', 'Neighbours', 'int', neighbours, {})
		]

		super(self.__class__, self).__init__(name, fields)
		self.tris = np.array([
			[0, 1, 2], [2, 3, 0], [1, 5, 6], [6, 2, 1], [7, 6, 5], [5, 4, 7],
			[4, 0, 3], [3, 7, 4], [4, 5, 1], [1, 0, 4], [3, 2, 6], [6, 7, 3]
		], dtype=np.int32)

		self.verts = np.array([
			[-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
			[-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5]
		], dtype=np.float32)

		self.jointWidth = {
			'root': [380, 300],
			'VSS_Spine': [380, 300], 'VSS_Spine1': [380, 300], 'VSS_Spine2': [400, 300], 'VSS_Chest': [400, 300],
			'VSS_Neck': [100, 100], 'VSS_Head': [180, 200], 'VSS_HeadEnd': [150, 150],
			'VSS_LeftShoulder': [180, 180], 'VSS_LeftArm': [100, 100], 'VSS_LeftForeArm': [90, 90], 'VSS_LeftForeArmRoll': [90, 90],
			'VSS_LeftHand': [50, 50], 'VSS_LeftHand1': [50, 50],
			'VSS_RightShoulder': [180, 180], 'VSS_RightArm': [100, 100], 'VSS_RightForeArm': [90, 90], 'VSS_RightForeArmRoll': [90, 90],
			'VSS_RightHand': [50, 50], 'VSS_RightHand1': [50, 50],
			'VSS_LeftUpLeg': [200, 180], 'VSS_LeftLeg': [140, 140], 'VSS_LeftFoot': [100, 100], 'VSS_LeftFootDummy': [100, 100],
			'VSS_LeftToeBase': [80, 80], 'VSS_LeftToeBaseEnd': [80, 80], 'VSS_LeftAnkle': [100, 100],
			'VSS_RightUpLeg': [200, 180], 'VSS_RightLeg': [140, 140], 'VSS_RightFoot': [100, 100], 'VSS_RightFootDummy': [100, 100],
			'VSS_RightToeBase': [80, 80], 'VSS_RightToeBaseEnd': [80, 80], 'VSS_RightAnkle': [100, 100]
		}

		self.CL_ctx = None
		self.CL_queue = None
		self.cl_normalsProgram = None

	def setup(self, interface, attrs):
		if self.CL_ctx is None:
			self.CL_ctx, self.CL_queue, self.cl_normalsProgram = createNormalsKernel()

	@staticmethod
	def generate_skeleton_lods(skelDict, Gs=None):
		# TODO: First iteration: Improve code and optimise
		# TODO: Contains hard coded values (generalise.. actually probably better to use a callback.. lodgenerator visitor)
		from GCore import Calibrate
		vs, tris, orientation, names = [], [], [], []
		if 'jointWidth' not in skelDict: return
		jointWidth = skelDict['jointWidth']
		jointHeightMultiplier = 1.3

		if Gs is None: Gs = skelDict['Gs']
		Bs = skelDict['Bs']

		lodVerts = skelDict['verts']
		lodTris = skelDict['tris']

		for jointIdx, jointName in enumerate(skelDict['jointNames']):
			if 'Free' in jointName: continue

			jointGs = Gs[jointIdx]
			jointBs = Bs[jointIdx]
			whichAxis = np.where(jointBs == 0)[0]
			R, T, _ = Calibrate.decomposeRT(jointGs, 1, False)
			jointMeshScale = jointBs.copy()
			if jointName == 'root': jointMeshScale = jointMeshScale * 1.4
			elif 'Spine' in jointName: jointMeshScale = jointMeshScale * 1.2

			jointMeshScale[whichAxis] = jointWidth[jointName]

			if jointName == 'VSS_Chest':
				jointMeshScale[0] = jointWidth[jointName][0]
				jointMeshScale[1] = 120.
				jointMeshScale[2] = jointWidth[jointName][1]

			axisToggle = np.array([1, 1, 1], dtype=np.float32)
			axisToggle[whichAxis] = 0.0
			translations = jointMeshScale / 2
			if jointName == 'VSS_Chest': translations[0:1] = 0
			offset = translations * axisToggle

			boneVerts = lodVerts.copy()
			for vi, v in enumerate(boneVerts):
				v = v * jointMeshScale
				if jointName in ['root']:
					v = v - offset
					v = np.dot(Calibrate.composeR(np.array([0, 0, 90], dtype=np.float32)), v.T)
				else:
					v = v + offset

				v = np.dot(jointGs, np.hstack((v, 1)).T)
				boneVerts[vi] = v[:3]

			tris.append(lodTris + len(vs) * 8)
			vs.append(boneVerts)

			boneLength = jointBs[np.where(jointBs != 0)[0]]
			orientation.append(0 if boneLength.any() and boneLength[0] < 0 else 1)
			names.append(jointName)

		v = np.concatenate((vs))
		t = np.concatenate((tris)).tolist()
		lodAttrs = {
			'triangles': v[t],
			'verts': v,
			'tris': t,
			'faces': tris,
			'names': names
		}
		skelDict['visibilityLod'] = lodAttrs
		return v, t, vs, tris, orientation, names

	def cook(self, location, interface, attrs):
		skelDict = interface.attr('skelDict')
		if skelDict is None:
			self.logger.error('Skeleton not found at: %s' % location)
			return

		if 'jointWidth' not in skelDict:
			skelDict['jointWidth'] = self.jointWidth

		if 'verts' not in skelDict:
			skelDict['verts'] = self.verts

		if 'tris' not in skelDict:
			skelDict['tris'] = self.tris

		v, t, vs, tris, orientation, names = ProcLOD.generate_skeleton_lods(skelDict, interface.attr('Gs'))

		normals = calculateNormals(v, t, self.CL_ctx, self.CL_queue, self.cl_normalsProgram, neighbours=attrs['neighbours'])
		orientations = np.repeat(orientation, 8)
		whichToReverse = np.where(orientations == 1)[0]
		normals[whichToReverse] = normals[whichToReverse] * -1
		faceNormals = np.mean(normals[tris, :], axis=2)

		skelDict['visibilityLod']['faceNormals'] = faceNormals
		skelDict['visibilityLod']['generateCb'] = ProcLOD.generate_skeleton_lods
		interface.createChild('visibilityLod', 'group', attrs=skelDict['visibilityLod'])

		if attrs['renderLod']:
			lodAttrs = {
				'vs': [v],
				'tris': [t],
				'colour': [eval(attrs['colour'])]
			}
			interface.createChild('lods', 'mesh', attrs=lodAttrs)

		if attrs['displayNormals']:
			triMeans = np.concatenate((np.mean(v[tris, :], axis=2)))
			norms = np.concatenate((faceNormals))
			normalsTo = triMeans + (norms * 50)
			lineAttrs = {
				'x0': triMeans,
				'x1': normalsTo,
				'colour': (1, 1, 0, 0.5),
				'edgeColour': (1, 1, 0, 1),
				'pointSize': 1
			}
			interface.createChild('normals', 'edges', attrs=lineAttrs)

			faceMeanAttrs = {
				'x3ds': triMeans,
				'x3ds_colour': (0, 1, 1, 1)
			}
			interface.createChild('normalOrigins', 'points3d', attrs=faceMeanAttrs)


class ExportMarkerMesh(Op.Op):
	def __init__(self, name='/Export_Marker_Mesh', locations='', filename='', frameRange=''):
		self.fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'locations', 'Skeleton locations', 'string', locations, {}),
			('saveTo', 'Save to (.mmesh)', 'Save to (.mmesh)', 'filename', filename, {}),
			('frameRange', 'Frame range', 'Frame range', 'string', frameRange, {})
		]

		super(self.__class__, self).__init__(name, self.fields)

	def cook(self, location, interface, attrs):
		if not self.useFrame(interface.frame(), attrs['frameRange']): return
		if not attrs['saveTo']: return

		tris = interface.attr('tris')
		if tris is None:
			self.logger.error('No tris attribute found')
			return

		verts = interface.attr('vs')
		if verts is None:
			self.logger.error('No vs (verts) attribute found')
			return

		normals = interface.attr('normals')
		if normals is None:
			self.logger.error('No normals attribute found')
			return

		# We expect to get a list of meshes so we check if we have at least one
		if not len(tris) or not len(verts) or not len(normals):
			self.logger.error('No sample found')
			return

		# Pick the first one (we only expect one to exist)
		tris, verts, normals = tris[0], verts[0], normals[0]

		saveTo = self.resolvePath(attrs['saveTo'])
		IO.save(saveTo, (tris, verts, normals))


class SphericalPoints(Op.Op):
	def __init__(self, name='/Spherical Points', locations='/root/sphere', numPoints=80, radius=500., seed=0,
				displayNormals=False, pointSize=10., colour=(0.6, 0.1, 0.6, 0.4)):
		fields = [
			('name', 'name', 'Name', 'string', name, {}),
			('locations', 'Skeleton locations', 'Locations', 'string', locations, {}),
			('numPoints', '# Points', 'Number of points', 'int', numPoints, {'min': 4}),
			('radius', 'Radius', 'Radius', 'float', radius, {'min': 1}),
			('seed', 'Seed', 'Seed', 'int', seed, {'min': 0}),
			('displayNormals', 'Display normals', 'Display normals', 'bool', displayNormals, {}),
			('pointSize', 'Point size', '3D Point size', 'float', pointSize, {}),
			('colour', 'colour', 'colour', 'string', str(colour), {})
		]

		super(self.__class__, self).__init__(name, fields)

	def sampleSpherical(self, numPoints, radius):
		phi = np.random.uniform(0., 1., (numPoints, 1)) * np.pi * 2
		theta = np.random.uniform(0., 1., (numPoints, 1)) * np.pi
		x = np.array(radius * np.sin(theta) * np.cos(phi), dtype=np.float32)
		y = np.array(radius * np.sin(theta) * np.sin(phi), dtype=np.float32)
		z = np.array(radius * np.cos(theta), dtype=np.float32)
		return (x, y, z)

	def cook(self, location, interface, attrs):
		np.random.seed(attrs['seed'])

		radius = attrs['radius']
		x, y, z = self.sampleSpherical(attrs['numPoints'], radius)
		pts = np.column_stack((x, y, z))
		labels = np.arange(len(pts))

		pAttrs = {
			'x3ds': pts,
			'x3ds_labels': labels,
			'normals': pts * -1,
			'x3ds_pointSize': attrs['pointSize'],
			'x3ds_colour': eval(attrs['colour'])
		}
		interface.createChild(interface.name(), 'points3d', atLocation=interface.parentPath(), attrs=pAttrs)

		if attrs['displayNormals']:
			n = pts / radius
			normalsTo = pts + (radius * 0.1 * n)
			lineAttrs = {
				'x0': pts,
				'x1': normalsTo,
				'colour': (1, 1, 0, 0.5),
				'lineColour': (1, 1, 0, 1),
				'pointSize': 1
			}
			interface.createChild('normals', 'edges', attrs=lineAttrs)


# Register Ops
import Registry
Registry.registerOp('Marker Mesh', MarkerMesh)
Registry.registerOp('Export Marker Mesh', ExportMarkerMesh)
Registry.registerOp('Character From Skeleton', CharacterFromSkeleton)
Registry.registerOp('Maya Character', MayaCharacter)
Registry.registerOp('Update Marker Mesh From Skeleton', UpdateMarkerMeshFromSkeleton)
Registry.registerOp('Update Marker Mesh From X3DS', UpdateMarkerMeshFromX3ds)
Registry.registerOp('Update Pose And Meshes', UpdatePoseAndMeshes)
Registry.registerOp('Calculate Normals', CalculateNormals)
Registry.registerOp('Display Shape Weight Groups', DisplayShapeWeightGroups)
Registry.registerOp('Skeleton Procedural LOD', ProcLOD)
Registry.registerOp('Display Joint-Marker Mapping', DisplayJointMarkerMapping)
Registry.registerOp('Create Spherical Points', SphericalPoints)
Registry.registerOp('Create Marker Mesh From X3Ds', TriangulateMarkerMesh)
