#!/usr/bin/env python

import os, sys
import numpy as np
import IO

def read_OBJ(filename):
	'''Read an OBJ file from disk. Returns a geom dict.'''
	return decode_OBJ(parse_OBJ(open(filename,'r').readlines()))

def parse_OBJ(obj_strings):
	'''Parse an OBJ file into a dict of group:dict of str:list one of (str,list of dict(str:str),dict(str:str)).'''
	ret = {'#':{}}
	groups = ['#']
	ret['groups'] = groups
	group = ret['#']
	obj_strings = map(str.split, obj_strings)
	for line in obj_strings:
		if len(line) == 0 or line[0].startswith('#'): continue # ignore blank lines and comments
		if line[0] == 'mtllib':
			ret[line[0]] = line[1:]
		elif line[0] == 'g':
			groupname = '/'.join(line[1:])
			while ret.has_key(groupname): groupname += '+'
			groups.append(groupname)
			group = ret[groupname] = {}
		else:
			if not group.has_key(line[0]): group[line[0]] = []
			group[line[0]].append(line[1:])
	return ret

def faces_to_triangles(faces):
	'''given a list of faces (each vertex might contain multiple coordinates), generate a list of triangles and splits for those triangles
	that recreate the faces.'''
	tris,tris_splits = [],[0]
	for f in faces:
		for j,k in zip(f[1:-1],f[2:]): tris.append([f[0],j,k])
		tris_splits.append(len(tris))
	return np.array(tris,dtype=np.int32),np.array(tris_splits,dtype=np.int32)

def decode_OBJ(geomDict):
	'''extract the interesting data from the raw OBJ. also generate some triangles for convenience.'''
	ret = { 'groups':geomDict['groups'], 'v':[[0,0,0]], 'vt':[[0,0]], 'vn':[[0,0,0]], 'f':[], 'v_splits':[1], 'vt_splits':[1], 'vn_splits':[1], 'f_splits':[0] }
	for groupName in geomDict['groups']:
		group = geomDict[groupName]
		for k,v in ret.iteritems():
			if group.has_key(k):
				v.extend(group[k])
			if ret.has_key(k+'_splits'):
				ret[k+'_splits'].append(len(v))
	for k in ['v','vt','vn']: ret[k] = np.array(ret[k],dtype=np.float32)
	for k in ['v_splits','vt_splits','vn_splits']: ret[k] = np.array(ret[k],dtype=np.int32)
	fs = []
	f_splits = ret['f_splits']
	for fi in range(len(f_splits)-1):
		for face in ret['f'][f_splits[fi]:f_splits[fi+1]]:
			f = map(lambda x: map(int,x.split('/')), face)
			fs.append(f)
	ret['fs'] = fs
	ret['tris'],ret['tris_splits'] = faces_to_triangles(fs)
	return ret

def flatten_OBJ_and_x10(geomDict, out=None):
	'''
	convert an OBJ geometry to a flat geometry.
	when the geometry was read, the facets were triangulated into the 'tris' field.
	those triangles index into vertices, texture vertices and normal lists.
	we find all unique vertices (same indices for all three) and flatten the lists.
	this causes an unfortunate renumbering of the vertex indices.
	however, the resulting data can be directly rendered.
	the vertices are also scaled by 10, to convert from cm (obj standard) to mm (our standard).
	'''
	if out is None: out = geomDict
	vs = geomDict['v']*10.# convert to mm
	vts = geomDict['vt']
	vns = geomDict['vn']
	tris = geomDict['tris']
	numVerts = vs.shape[0]
	numTris = tris.shape[0]
	verts = {}
	out_vs = []
	out_vts = []
	out_vns = []
	out_fs = []
	out_fs_splits = [0]
	out_tris = []
	# TODO reserve the first N vertex indices to match the original vertices
	for face in geomDict['fs']:
		for t in face:
			v = tuple(t)
			if v not in verts:
				verts[v] = len(verts)
				out_vs.append(vs[v[0]])
				if len(v)>1: out_vts.append(vts[v[1]]) # we assume there is a texture index
				if len(v)>2: out_vns.append(vns[v[2]]) # we assume there is a normal index
			out_fs.append(verts[v])
		out_fs_splits.append(len(out_fs))
	for tri in tris:
		for t in tri:
			v = tuple(t)
			out_tris.append(verts[v])
	out['v'] = np.array(out_vs,dtype=np.float32).reshape(-1,3)
	out['vt'] = np.array(out_vts,dtype=np.float32).reshape(-1,2)
	out['vn'] = np.array(out_vns,dtype=np.float32).reshape(-1,3)
	out['tris'] = np.array(out_tris,dtype=np.int32).reshape(-1,3)
	out['fs'] = np.array(out_fs,dtype=np.int32)
	out['fs_splits'] = np.array(out_fs_splits,dtype=np.int32)
	return out

def poseGeometry(geom, Gs):
	print geom.keys()
	tv = geom['v'].copy()
	tvn = geom['vn'].copy()
	for ji,ran in geom['gd']:
		ran = np.unique(ran.ravel())
		G = Gs[ji]
		tv[ran] = np.dot(tv[ran], G[:,:3].T) + G[:,3]
		tvn[ran] = np.dot(tvn[ran], G[:,:3].T)
	return {'v':tv,'vt':geom['vt'],'vn':tvn,'gd':geom['gd'],'geomMapping':geom['geomMapping']} #,'tris':np.arange(len(tv),dtype=np.int32).reshape(1,-1,3) }

#@profile
def trianglesToNearVerts(triangles, steps = 25):
	'''Given the triangles, generate the disc of all the vertices reachable in a number of steps.'''
	numVerts = np.max(triangles)+1
	# generate N, the vertices neighbouring a given vertex
	N = [set() for vi in xrange(numVerts)]
	for t in triangles:
		N[t[0]].update(set([t[1],t[2]]))
		N[t[1]].update(set([t[2],t[0]]))
		N[t[2]].update(set([t[0],t[1]]))
	D = [set(Di) for Di in N] # disc of surrounding vertices
	R = [set(Di) for Di in N] # ring of vertices being added
	for step in range(steps):
		print step,'/',steps
		# make R be a one-ring expansion of R (via N)
		R2 = [set() for vi in xrange(numVerts)]
		for Ri,R2i in zip(R,R2):
			for vi in Ri: R2i.update(N[vi])
		R = R2
		# thin out R to not include D; update D
		for Di,Ri in zip(D,R): Ri.difference_update(Di); Di.update(Ri)
	for vi,Di in enumerate(D): Di.discard(vi); D[vi] = np.array(list(Di),dtype=np.int32)
	# corners of mouth are 3382, 6922 I think
	return D

#@profile
def findCloseVerts(xs, threshold = 80.0):
	import ISCV
	cloud = ISCV.HashCloud3D(xs, threshold)
	scores,matches,matches_splits = cloud.score(xs)
	good = (scores < (threshold**2))
	D = [matches[m0:m1][np.where(good[m0:m1])[0]] for m0,m1 in zip(matches_splits[:-1],matches_splits[1:])]
	print 'avg verts',np.mean(map(len,D))
	#for vi,Di in enumerate(D): Di.discard(vi); D[vi] = np.array(list(Di),dtype=np.int32)
	return D

def myNormal(x,y):
	c = np.cross(x,y)
	s = list(c.shape)
	s[-1] = 1
	return c / (1e-8+(np.sum(c*c,axis=-1))**0.25).reshape(s)

#@profile
def computeLocalCoordinates(xs, uvs, nearVerts):
	'''uvs are assumed in correspondence with xs. this won't work at seams (take care).'''
	print xs.shape, uvs.shape
	numVerts = xs.shape[0]
	assert(numVerts == uvs.shape[0])
	assert(len(nearVerts) == numVerts)
	D = np.zeros((numVerts,3,3), dtype=np.float32)

	for vi,nvs in enumerate(nearVerts):
		if len(nvs) < 2: print 'skipping',vi; D[vi] = np.eye(3); continue
		dxs  = xs[nvs] - xs[vi] # N,3
		duvs = uvs[nvs] - uvs[vi] # N,2
		D[vi,:,:2] = np.linalg.lstsq(duvs,dxs)[0].T
		#[[a,b],[c,d]] = np.dot(duvs.T,duvs); scLLT = 1.0/(1e-8+a*d-b*c)
		#D[vi,:,:2] = np.dot(np.dot(dxs.T, duvs), [[d*scLLT,-b*scLLT],[-c*scLLT,a*scLLT]])
		D[vi,:,2] = myNormal(D[vi,:,0],D[vi,:,1])
	return D

#@profile
def computeLocalCoordinates2(xs, x2s, nearVerts):
	'''Compute direct mapping from vertices.'''
	print xs.shape, x2s.shape
	numVerts = xs.shape[0]
	assert(numVerts == x2s.shape[0])
	assert(len(nearVerts) == numVerts)
	D = np.zeros((numVerts,3,3), dtype=np.float32)

	for vi,nvs in enumerate(nearVerts):
		if len(nvs) < 2: print 'skipping',vi; D[vi] = np.eye(3); continue
		dxs  = xs[nvs] - xs[vi] # N,3
		dx2s = x2s[nvs] - x2s[vi] # N,3
		M = np.linalg.lstsq(dxs,dx2s,rcond=1e-6)[0].T # dxs M.T = dx2s; -> M dxs[i] = dx2s[i]
		u,s,vt = np.linalg.svd(M, full_matrices=True)
		s = (s+np.mean(s))*0.5 # try to stabilize the solution
		D[vi] = np.dot(u,np.dot(np.diag(s),vt))
		if np.linalg.det(D[vi]) < 0:
			s[2] *= -1.0
			D[vi] = np.dot(u,np.dot(np.diag(s),vt))
	return D

#@profile
def renderMotion(D, motion):
	numVerts = motion.shape[0]
	ret = np.zeros((numVerts,3),dtype=np.float32)
	for vi in range(numVerts):
		ret[vi] = np.dot(D[vi], motion[vi])
	return ret

#@profile
def computeMotion(xs, D):
	numVerts = xs.shape[0]
	ret = np.zeros((numVerts,3),dtype=np.float32)
	for vi in range(numVerts):
		try:
			ret[vi] = np.linalg.solve(D[vi], xs[vi])
		except:
			print 'singular vertex',vi
	return ret

#@profile
def lunterp2D(M,s):
	'''Given a triangle of uv coordinates, compute the barycenrtic coordinates of the uv coordinate.'''
	[a,b],[c,d],[e,f] = M[1]-M[0],M[2]-M[0],s-M[0]
	det = a*d-b*c
	if not det: return (-10,-10,-10) # triangle is a straight line.. (-inf,inf,inf) maybe
	w1 = (e*d - f*c)/det
	w2 = (f*a - e*b)/det
	return (1-w1-w2,w1,w2)

def lunterp3D(M,s):
	'''Given a triangle of 3D coordinates, compute the barycenrtic coordinates of the 3D vector s.'''
	w1,w2 = np.linalg.lstsq((M[1:]-M[0]).T,s-M[0],rcond=1e-6)[0]
	#[a,b,c],[d,e,f],[g,h,i] = M[1]-M[0],M[2]-M[0],s-M[0]
	## [w1 w2] [a b c] = [g h i]
	##         [d e f]
	## TODO
	#det = a*d-b*c
	#if not det: return (-10,-10,-10) # triangle is a straight line.. (-inf,inf,inf) maybe
	#w1 = (e*d - f*c)/det
	#w2 = (f*a - e*b)/det
	return (1-w1-w2,w1,w2)

def getMapping(hi_geo, triangles, lo_geo, threshold = 20.0):
	'''given a hi-res geometry and topology, and a lo-res geometry, find the triangles and barycentric weights that
	when applied to the hi-res geometry, best fit the lo-res geometry.
	The mapping is returned as a list of weight triples and a list of index triples, per vertex.
	The output vertex is the weighted sum of the extracted indicated source vertices.'''
	is3D = (hi_geo.shape[1] == 3)
	lunterp = lunterp3D if is3D else lunterp2D
	numVertsHi = hi_geo.shape[0]
	numVertsLo = lo_geo.shape[0]
	weights = np.zeros((numVertsLo,3),dtype=np.float32)
	indices = -np.ones((numVertsLo,3),dtype=np.int32)
	import ISCV
	cloud = ISCV.HashCloud3D(hi_geo, threshold) if is3D else ISCV.HashCloud2D(hi_geo, threshold)
	scores,matches,matches_splits = cloud.score(lo_geo.copy())
	# the indices of the closest 3 hi verts to each lo vert
	D = [matches[m0+np.argsort(scores[m0:m1])[:3]] if m0 != m1 else [] for m0,m1 in zip(matches_splits[:-1],matches_splits[1:])]
	# for speed-up, compute all the triangles involving each hi vertex.
	T = [[] for x in xrange(numVertsHi)]
	for ti,tri in enumerate(triangles):
		for tj in tri: T[tj].append(tri)
	bads = []
	for vi,(lo_x,nearIndices,ws,xis) in enumerate(zip(lo_geo,D,weights,indices)):
		best = -10
		for ni in nearIndices:
			for tri in T[ni]:
				xws = lunterp(hi_geo[tri], lo_x)
				sc = np.min(xws)
				if sc > best: # pick the best triangle (the one that it's closest to being inside)
					best = sc
					xis[:] = tri
					ws[:] = xws
					if best >= 0: break
			if best >= 0: break
		# the vertex *might* not be inside any of these triangles
		if best < -0.1:
			bads.append(vi)
			ws[:] = 0.0 # ensure there's no weight
			xis[:] = -1 # and no label
	if len(bads):
		print 'vertices outside',len(bads)
		print bads[:10],'...'
	which = np.where(indices[:,0] != -1)[0]
	print len(which), 'vertices inside'
	return which,weights[which],indices[which]


def getMapping2(hi_geo, triangles, lo_geo, threshold = 20.0):
	'''given a hi-res geometry and topology, and a lo-res geometry, find the triangles and barycentric weights that
	when applied to the hi-res geometry, best fit the lo-res geometry.
	The mapping is returned as a list of weight triples and a list of index triples, per vertex.
	The output vertex is the weighted sum of the extracted indicated source vertices.'''
	is3D = (hi_geo.shape[1] == 3)
	lunterp = lunterp3D if is3D else lunterp2D
	numVertsHi = hi_geo.shape[0]
	which = np.where((lo_geo[:,0] < 1e10) * (lo_geo[:,0] > -1e10))[0]
	numVertsLo = len(lo_geo)
	weights = np.zeros((numVertsLo,3),dtype=np.float32)
	indices = -np.ones((numVertsLo,3),dtype=np.int32)
	import ISCV
	cloud = ISCV.HashCloud3D(hi_geo, threshold) if is3D else ISCV.HashCloud2D(hi_geo, threshold)
	scores,matches,matches_splits = cloud.score(lo_geo[which].copy())
	# the indices of the closest 3 hi verts to each lo vert
	D = [matches[m0+np.argsort(scores[m0:m1])[:3]] if m0 != m1 else [] for m0,m1 in zip(matches_splits[:-1],matches_splits[1:])]
	# for speed-up, compute all the triangles involving each hi vertex.
	T = [[] for x in xrange(numVertsHi)]
	for ti,tri in enumerate(triangles):
		for tj in tri: T[tj].append(tri)
	bads = []
	for (vi,lo_x,nearIndices,ws,xis) in zip(which,lo_geo[which],D,weights[which],indices[which]):
		best = -10
		for ni in nearIndices:
			for tri in T[ni]:
				xws = lunterp(hi_geo[tri], lo_x)
				sc = np.min(xws)
				if sc > best: # pick the best triangle (the one that it's closest to being inside)
					best = sc
					xis[:] = tri
					ws[:] = xws
					if best >= 0: break
			if best >= 0: break
		# the vertex *might* not be inside any of these triangles
		if best < -0.1:
			bads.append(vi)
	if len(bads):
		print 'vertices outside',len(bads)
		print bads[:10],'...'
	return weights,indices


def topologicalMappings(uvs, triangles, uv2s, triangles2, threshold = 0.02):
	a2b = getMapping(uvs, triangles, uv2s, threshold)
	b2a = getMapping(uv2s, triangles2, uvs, threshold)
	return a2b,b2a

def computeTopoMap(ted_obj, tony_obj, ted_vts_copy = None, tony_vts_copy = None):
	'''Compute the direct mapping between geometries. Take into account differences in topology
	using the (possibly adjusted to make them conform better) texture coordinates.'''
	if ted_vts_copy is None: ted_vts_copy = ted_obj['vt']
	if tony_vts_copy is None: tony_vts_copy = tony_obj['vt']
	#assert(np.all(ted_obj['tris'] == tony_obj['tris'])) # TODO WHY? see 633
	(mw,mws,mis),(mw2,mw2s,mi2s) = topologicalMappings(ted_vts_copy, ted_obj['tris'], tony_vts_copy, tony_obj['tris'])
	x2s = renderGeo(tony_obj['v'],mw2s,mi2s)
	nearVerts = findCloseVerts(ted_obj['v'][mw])
	D = computeLocalCoordinates2(ted_obj['v'][mw], x2s, nearVerts)
	return (mw,mws,mis),(mw2,mw2s,mi2s),x2s,D

def renderGeo(xs,weights,indices,out=None):
	numVerts = weights.shape[0]
	if out is None: out = np.zeros((numVerts,3),dtype=np.float32)
	np.sum(weights.reshape(numVerts,-1,1)*xs[indices],axis=1,out=out)
	#for rs,xws,xis in zip(out,weights,indices): np.dot(xws,xs[xis],out=rs) # equivalent code
	return out

'''Given multiple shapes, each vertex can be expressed as some function of its neighbouring vertices.'''
'''Spectral decomposition might help'''


def flipMouth(obj):
	uvs = obj['vt']
	which = np.where((uvs[:,1] < 0.182) * (uvs[:,0] > 0.45) * (uvs[:,0] < 0.55))[0]
	uvs[which,0] = 1.0 - uvs[which,0]

#@profile
def connectedComponents(triangles):
	numVerts = np.max(triangles)+1
	T = np.arange(numVerts,dtype=np.int32)
	while True:
		tmp = T.copy()
		for tri in triangles: T[tri] = np.min(T[tri])
		if np.all(T == tmp): break
	groups = np.unique(T)
	components = []
	for gval in groups:
		components.append(np.where(T == gval)[0])
	reorder = np.argsort(map(len,components))[::-1]
	return [components[i] for i in reorder]

def rotate90(obj, t=90.0):
	t = np.radians(t)
	vs = obj['v']
	vns = obj['vn']
	rot = np.array([[np.cos(t),-np.sin(t),0],[np.sin(t),np.cos(t),0],[0,0,1]],dtype=np.float32)
	vs[:] = np.dot(vs,rot)*0.1
	vns[:] = np.dot(vns,rot)
	#vs[:] *= (np.max(vs[:,1]) - vs[:,1].reshape(-1,1) + 1)/(np.mean(vs[:,1]**2)**0.5)

#@profile
def readFlatObjFlipMouth(obj_filename):
	obj = read_OBJ(obj_filename)
	flatten_OBJ_and_x10(obj)
	flipMouth(obj)
	return obj

def makeLoResShapeMat(hi_rest, hi_mat, weights, indices):
	'''given a hi-res shape matrix and points derived from it by barycentric triangle weights, compute the lo-res shape matrix.'''
	lo_rest = renderGeo(hi_rest, weights, indices)
	lo_mat = np.zeros((hi_mat.shape[0],lo_rest.shape[0],3),dtype=np.float32)
	print 'makeLoResShapeMat shapes',hi_mat.shape,lo_mat.shape
	for hi,lo in zip(hi_mat,lo_mat):
		renderGeo(hi, weights, indices, out=lo)
	print 'done'
	return lo_rest, lo_mat

def fitLoResShapeMat_old(lo_rest, lo_mat, pts, indices=None, bounds=None):
	'''given a lo-res shape matrix and points, find the vector that minimises the rms error.
	to solve only a subset of the points, provide the indices.
	this is simplistic for now!'''
	if indices is None: indices = np.arange((lo_rest.shape[0]),dtype=np.int32)
	M = lo_mat[:,indices,:].reshape(-1,len(indices)*3)
	v = (pts - lo_rest[indices]).reshape(-1)
	#ret = np.linalg.lstsq(M.T,v,rcond = 1e-2)[0]
	ret = np.linalg.lstsq(np.dot(M,M.T)+1e0*np.eye(M.shape[0]),np.dot(M,v),rcond = 1e-2)[0]
	
	print 'stats',np.min(ret),np.mean(ret),np.max(ret)
	return ret
	
def fitLoResShapeMat(lo_rest, lo_mat, pts, Aoffset=10.0, Boffset=3.0, x_0=None, indices=None, bounds=None):
	'''Solve for blendshape weights that give the best constrained fit of model to target.
		M_verts*bshapes x_bshapes + neutral_verts = target_verts.
		M x + (N-T) = 0
		x^T M^T M x + 2 x^T M^T (N-T) + (N-T)^T (N-T) = 0. # ignore third term, which is strictly positive
		Add regularising terms: x ~ x_0 and sum(x).
		Aoffset * (x^T x - 2 x^T x_0 + x_0^T x_0). # ignore third term, which is strictly positive
		Boffset * sum(x).'''
	# Objective function and Jacobian for quadratic solver
	def ObjectiveFunction(x,A,B): return np.dot(x.T, (np.dot(A,x) + B))
	def Jacobian(x,A,B): return np.dot(A,2*x) + B
	
	if x_0 is None or len(x_0) != lo_mat.shape[0]: x_0 = np.zeros(lo_mat.shape[0],dtype=np.float64)
	if indices is None: indices = np.arange((lo_rest.shape[0]),dtype=np.int32)
	if len(indices) == 0: return x_0 # argh!
	M = lo_mat[:,indices,:].reshape(-1,len(indices)*3)
	v = (lo_rest[indices] - pts).reshape(-1)
	
	A = np.dot(M, M.T).astype(np.float64)
	A[np.diag_indices_from(A)] += (Aoffset + 1e-8)
	B = np.array(2*np.dot(M, v) + Boffset - (2*Aoffset)*x_0,dtype=np.float64)
	x = x_0.copy() #np.ones(M.shape[0],dtype=np.float64)*0.5
	from scipy.optimize import minimize
	res = minimize(ObjectiveFunction, x, args=(A,B), method='TNC',bounds=bounds,jac=Jacobian)
	#print '\r',res['fun'],len(np.where(res.x)[0]); sys.stdout.flush()
	return res.x

def animateHead(newFrame):
	global ted_geom,ted_geom2,ted_shape,tony_geom,tony_shape,tony_geom2,tony_obj,ted_obj,diff_geom,c3d_frames
	global tony_shape_vector,tony_shape_mat,ted_lo_rest,ted_lo_mat,ted_lo_which,c3d_points
	global md,movies
	tony_geom.image,tony_geom.bindImage,tony_geom.bindId = ted_geom.image,ted_geom.bindImage,ted_geom.bindId # reuse the texture!
	fo = 55
	MovieReader.readFrame(md, seekFrame=((newFrame+fo)/2))
	view = QApp.view()
	frac = (newFrame % 200) / 100.
	if (frac > 1.0): frac = 2.0 - frac
	fi = newFrame%len(c3d_frames)

	frame = c3d_frames[fi][ted_lo_which]
	which = np.where(frame[:,3] == 0)[0]
	x3ds = frame[which,:3]
	#print which,x3ds.shape,ted_lo_rest.shape,ted_lo_mat.shape
	bnds = np.array([[0,1]]*ted_lo_mat.shape[0],dtype=np.float32)
	#print len(ted_lo_which),len(which),ted_lo_which,which
	tony_shape_vector[:] = fitLoResShapeMat(ted_lo_rest, ted_lo_mat, x3ds, Aoffset=10.0, Boffset=3.0, x_0=tony_shape_vector, indices=which, bounds = bnds)
	#global tony_shape_vectors; tony_shape_vector[:] = tony_shape_vectors[newFrame%len(tony_shape_vectors)]

	#tony_shape_vector *= 0.
	#tony_shape_vector += (np.random.random(len(tony_shape_vector)) - 0.5)*0.2
	if 1:
		ted_shape_v = np.dot(ted_shape_mat_T, tony_shape_vector).reshape(-1,3)
	else:
		import ISCV
		ted_shape_v = np.zeros_like(ted_obj['v'])
		ISCV.dot(ted_shape_mat_T, tony_shape_vector, ted_shape_v.reshape(-1))
	tony_shape_v = ted_shape_v
	#tony_shape_v = tony_shape['v']*frac
	ted_geom.setVs(ted_obj['v'] + ted_shape_v) #ted_shape['v'] * frac)
	tony_geom.setVs(tony_obj['v'] + tony_shape_v - np.array([200,0,0],dtype=np.float32))
	ted_geom2.setVs(ted_obj['v'] * (1.0 - frac) + tony_tedtopo_obj['v'] * frac + np.array([200,0,0],dtype=np.float32))
	#if len(ted_shape_v) == len(tony_shape_v):
	#	tony_geom2.setVs(tony_obj['v'] + ted_shape_v - [400,0,0])
	#	diff_geom.setVs(ted_obj['v'] + tony_shape_v - ted_shape_v - [600,0,0])

	#print [c3d_labels[i] for i in which]
	surface_points.vertices = np.dot(ted_lo_mat.T, tony_shape_vector).T + ted_lo_rest
	surface_points.colour = [0,1,0,1] # green
	c3d_points.vertices = x3ds
	c3d_points.colour = [1,0,0,1] # red
	QApp.app.updateGL()

if __name__ == '__main__':

	import UI
	from UI import QApp, QGLViewer
	from UI import GLMeshes
	import os, sys
	if len(sys.argv) > 1:
		filename = sys.argv[1]
		geom_dict = flatten_OBJ_and_x10(read_OBJ(filename))
		ted_geom = GLMeshes(['ted'],[geom_dict['v']], [geom_dict['tris']], vts = [geom_dict['vt']], transforms = [np.eye(3,4)])
		primitives = [ted_geom]
		QGLViewer.makeViewer(primitives = primitives)
		exit()

	from GCore import Calibrate
	import MovieReader
	import C3D

	global ted_geom,ted_geom2,ted_shape,tony_geom,tony_shape,tony_geom2,tony_obj,ted_obj,diff_geom,c3d_frames
	global tony_shape_vector,tony_shape_mat,ted_lo_rest,ted_lo_mat,c3d_points
	global md,movies
		
	ted_dir = os.path.join(os.environ['GRIP_DATA'],'ted')

	wavFilename = os.path.join(ted_dir,'32T01.WAV')
	md = MovieReader.open_file(wavFilename)

	c3d_filename = os.path.join(ted_dir,'201401211653-4Pico-32_Quad_Dialogue_01_Col_wip_02.c3d')
	c3d_dict = C3D.read(c3d_filename)
	c3d_frames, c3d_fps, c3d_labels = c3d_dict['frames'],c3d_dict['fps'],c3d_dict['labels']
	if False: # only for cleaned-up data
		c3d_subject = 'TedFace'
		which = np.where([s.startswith(c3d_subject) for s in c3d_labels])[0]
		c3d_frames = c3d_frames[:,which,:]
		c3d_labels = [c3d_labels[i] for i in which]
		print c3d_labels
	if False: # this is for the cleaned-up data (don't apply the other offset...)
		offset =  Calibrate.composeRT(Calibrate.composeR( (0.0,0.0, 0)),(0,0,-8),0) # 0.902
		c3d_frames[:,:,:3] = np.dot(c3d_frames[:,:,:3] - offset[:3,3],offset[:3,:3])[:,:,:3]
	offset =  Calibrate.composeRT(Calibrate.composeR( (3.9,-38.7, 0)),(-159.6,188.8,123-12),0) # 0.902
	c3d_frames[:,:,:3] = np.dot(c3d_frames[:,:,:3] - offset[:3,3],offset[:3,:3])[:,:,:3]

	geos = []
	dat_directory = os.path.join(os.environ['GRIP_DATA'],'dat')

	if False: # experiments involving deformation transfer
		geos_filename = 'geos'
		if not os.path.exists(geos_filename):
			ted_dir = os.environ['GRIP_DATA']
			ted_obj = readFlatObjFlipMouth(os.path.join(ted_dir,'ted.obj'))
			ted_shape = readFlatObjFlipMouth(os.path.join(ted_dir,'tedopen.obj'))
			ted_shape['v'] -= ted_obj['v']
			tony_obj = readFlatObjFlipMouth(os.path.join(ted_dir,'tony.obj'))
			nearVerts = trianglesToNearVerts(ted_obj['tris'], steps = 15)
			IO.save(geos_filename,(ted_obj,ted_shape,tony_obj,nearVerts))
		else:
			_,(ted_obj,ted_shape,tony_obj,nearVerts) = IO.load(geos_filename)

		for target in ['ape']: #['andy','avatar','baboon','bigdog','evilelf','fatbat','feline','fishman','kadel','lizardman','mannequin','shaman','ted','tony','troll','wolf']:
			if True:
				#target = 'baboon'
				target_filename = os.path.join(dat_directory,target+'.dat')
				if True: #not os.path.exists(target_filename):
					ted_dir = os.path.join(os.environ['GRIP_DATA'],'ted')
					tony_obj = readFlatObjFlipMouth(os.path.join(ted_dir,target+'.obj'))
					if target == 'ape' or target == 'apenew': flipMouth(tony_obj) # the ape's mouth is already flipped!
					print tony_obj['v'].shape, ted_obj['v'].shape

					print np.mean(map(len,nearVerts))
					vts = ted_obj['vt']

					tony_shape = {'v':0*tony_obj['v']}


					if True:
						print 'computing x-to-x scheme'

						ted_ccs = connectedComponents(ted_obj['tris'])
						print len(ted_ccs),map(len,ted_ccs)
						tony_ccs = connectedComponents(tony_obj['tris'])
						print len(tony_ccs),map(len,tony_ccs)
						for gp in range(7):
							print gp,np.mean(ted_obj['vt'][ted_ccs[gp]],axis=0) - np.mean(tony_obj['vt'][tony_ccs[gp]],axis=0)
						ted_vts_copy = ted_obj['vt'].copy()
						tony_vts_copy = tony_obj['vt'].copy()
						tony_vts_copy[tony_ccs[0]] += np.array([-0.0029, 0],dtype=np.float32)
						tony_vts_copy[tony_ccs[3],0] = 0.715+ tony_vts_copy[tony_ccs[3],0]
						tony_vts_copy[tony_ccs[4],0] = 0.715+ tony_vts_copy[tony_ccs[4],0]

						(mws,mis),(mw2s,mi2s),x2s,D = computeTopoMap(ted_obj, tony_obj, ted_vts_copy, tony_vts_copy)
						print len(np.where(mws > 0.98)[0])
						tony_tedtopo_obj = { 'v':x2s,'vt':ted_obj['vt'], 'tris':ted_obj['tris'] }
						tony_shape = {'v':renderGeo(renderMotion(D, ted_shape['v']), mws,mis)} # reuse everything
					elif True:
						Dsrc = computeLocalCoordinates(ted_obj['v'], vts, nearVerts)
						Dtgt = computeLocalCoordinates(tony_obj['v'], vts, nearVerts)
						localMotion = computeMotion(ted_shape['v'], Dsrc)
						tony_shape['v']= renderMotion(Dtgt, localMotion)
					else:
						steps = 3
						tony_incr = tony_obj['v'].copy()
						ted_incr = ted_obj['v'].copy()
						ted_step = ted_shape['v'] * (1.0/steps)
						for it in xrange(steps):
							Dtgt = computeLocalCoordinates(tony_incr, vts, nearVerts)
							Dsrc = computeLocalCoordinates(ted_incr, vts, nearVerts)
							localMotion = computeMotion(ted_step, Dsrc)
							tony_incr += renderMotion(Dtgt, localMotion)
							ted_incr += ted_step
						tony_shape['v'][:] = tony_incr - tony_obj['v']
					IO.save(target_filename,(tony_obj,tony_shape))
				else:
					_,(tony_obj,tony_shape) = IO.load(target_filename)
			else: #except Exception, e:
				print 'oops',target,e

	if True:
		geos_filename = os.path.join(dat_directory,'ted_new.dat')
		if not os.path.exists(geos_filename):
			ted_obj = readFlatObjFlipMouth(os.path.join(ted_dir,'Ted_NEWRIG_Neutral_Moved.obj'))
		else:
			_,(ted_obj,nearVerts) = IO.load(geos_filename)
		target = 'andy'
		tony_obj = readFlatObjFlipMouth(os.path.join(ted_dir,target+'.obj'))
		_,tony_shapes = IO.load(os.path.join(dat_directory,target+'_shapes.dat'))
		num_shapes = len(tony_shapes)
		print num_shapes
		tony_shape_mat = np.zeros((num_shapes,tony_shapes[0]['v'].shape[0],3),dtype=np.float32)
		for t,ts in zip(tony_shape_mat, tony_shapes): t[:] = ts['v']
		tony_shape_vector = 0.2*np.ones(num_shapes,dtype=np.float32)
		tony_shape_v = np.dot(tony_shape_mat.T, tony_shape_vector).T
		tony_tedtopo_obj = {'v': tony_obj['v'].copy() }
		ted_shape = {'v':tony_shape_v.copy()}
		tony_shape = {'v':tony_shape_v.copy()}


	if True: # ted_shape_mat
		try:
			ted_shape_mat = IO.load('ted_shape_mat')[1]
		except:
			geos_filename = os.path.join(dat_directory,'ted_new.dat')
			if not os.path.exists(geos_filename):
				ted_obj = readFlatObjFlipMouth(os.path.join(ted_dir,'Ted_NEWRIG_Neutral_Moved.obj'))
			else:
				_,(ted_obj,nearVerts) = IO.load(geos_filename)
			_,ted_shapes = IO.load(os.path.join(dat_directory,'ted_shapes.dat'))
			num_shapes = len(ted_shapes)
			ted_shape_mat = np.zeros((num_shapes,ted_shapes[0]['v'].shape[0],3),dtype=np.float32)
			for t,ts in zip(ted_shape_mat, ted_shapes): t[:] = ts['v']
			IO.save('ted_shape_mat',ted_shape_mat)
			# HACK scale ted... it looks like the correct value is 0.90197, which Shridhar introduced
		ted_obj['v'] *= 0.902
		ted_shape_mat *= 0.902
		lo_geo = c3d_frames[0,:,:3]
		ted_lo_which,weights,indices = getMapping(ted_obj['v'], ted_obj['tris'], lo_geo, threshold = 20.0)
		#for it,v in enumerate(zip(weights,indices)): print it,v
		ted_lo_rest, ted_lo_mat = makeLoResShapeMat(ted_obj['v'], ted_shape_mat, weights, indices)
		print np.sum(ted_shape_mat),np.sum(ted_lo_mat)
		ted_shape_mat_T = ted_shape_mat.reshape(ted_shape_mat.shape[0],-1).T.copy()

	if 0: # diagnostic
		tmp = np.sort(np.sum((ted_shape_mat_T!=0), axis=1))
		dtmp = tmp[1:]-tmp[:-1]
		diff = np.where(dtmp)[0]
		print dtmp[diff]
		print 'sort',diff[1:]-diff[:-1]
		u,s,vt = np.linalg.svd(ted_shape_mat_T, full_matrices=False)
		print s/s[0]

	#ted_obj['v'] -= np.mean(ted_obj['v'],axis=0)
	#tony_obj['v'] -= np.mean(tony_obj['v'],axis=0)
	#tony_tedtopo_obj['v'] -= np.mean(tony_tedtopo_obj['v'],axis=0)
	#rotate90(tony_obj,10)
	tony_obj['v'] -= np.array([0, 1750, 0],dtype=np.float32)
	tony_tedtopo_obj['v'] -= np.array([0, 1750, 0],dtype=np.float32)
	display_offset = np.array([0,1000,0],dtype=np.float32) # show above the ground plane
	tony_obj['v'] += display_offset
	tony_tedtopo_obj['v'] += display_offset
	ted_obj['v'] += display_offset
	ted_lo_rest += display_offset
	c3d_frames[:,:,:3] += display_offset
	offset[:3,3] -= np.dot(offset[:3,:3],display_offset)

	draw_normals = False
	if draw_normals:
		geos.append(UI.GLGeometry(vs=zip(tony_obj['v'],tony_obj['v']+Dtgt[:,:,0]*0.005), tris=range(Dtgt.shape[0]*2), drawStyle = 'wire',colour=[1,0,0,1]))
		geos.append(UI.GLGeometry(vs=zip(tony_obj['v'],tony_obj['v']+Dtgt[:,:,1]*0.005), tris=range(Dtgt.shape[0]*2), drawStyle = 'wire',colour=[0,1,0,1]))
		geos.append(UI.GLGeometry(vs=zip(tony_obj['v'],tony_obj['v']+Dtgt[:,:,2]*0.005), tris=range(Dtgt.shape[0]*2), drawStyle = 'wire',colour=[0,0,1,1]))

	#Dsrc = computeLocalCoordinates(ted_obj['v'], vts, nearVerts)
	#Dtgt = computeLocalCoordinates(ted_obj['v'] + ted_shape['v'], vts, nearVerts)
	#localMotion = computeMotion((tony_obj['v'] + [200,0,0])-ted_obj['v'], Dsrc)
	#tony_shape['v']= renderMotion(Dtgt, localMotion)+ (ted_obj['v'] + ted_shape['v']) - (tony_obj['v'] + [200,0,0])

	drawStyle='smooth'#'wire_over_smooth'
	ted_geom = GLMeshes(['ted'],[ted_obj['v']], [ted_obj['tris']], vts = [ted_obj['vt']], transforms = [np.eye(3,4)])
	#ted_geom = UI.GLGeometry(vs = ted_obj['v'], vts = ted_obj['vt'], tris = ted_obj['tris'], transformData=None, drawStyle=drawStyle)
	xspacer = np.array([200,0,0],dtype=np.float32)
	ted_geom2 = UI.GLGeometry(vs = ted_obj['v'] + xspacer, vts = ted_obj['vt'], tris = ted_obj['tris'], transformData=None, drawStyle=drawStyle)
	tony_geom = GLMeshes(['tony'], [tony_obj['v'] - xspacer], [tony_obj['tris']], vts=[tony_obj['vt']], transforms=[np.eye(3,4)]) #UI.GLGeometry(vs = tony_obj['v'] - xspacer, vts = tony_obj['vt'],  tris = tony_obj['tris'], transformData=None, drawStyle=drawStyle)
	#tony_geom2 = UI.GLGeometry(vs = tony_obj['v'] + [-400,0,0], vts = tony_obj['vt'], tris = tony_obj['tris'], transformData=None, drawStyle=drawStyle)
	#diff_geom = UI.GLGeometry(vs = ted_obj['v'] + [-600,0,0], vts = ted_obj['vt'], tris = ted_obj['tris'], transformData=None, drawStyle=drawStyle)

	# the tiara points are defined in an svg file, in units of bogons
	# in the file there is a matrix scale of 0.95723882 (dots per bogon) and a diameter of 14.06605 bogons = 3.8mm
	# 25.4 mmpi / 90 dpi * 0.95723882 dpb_from_svg = 3.8mm / 14.06605 bogon_diameter_from_svg = 0.270 mm_per_bogon
	mm_per_bogon = 0.270154067
	head_pts_bogons = np.array([
		[  23.24843216, -273.46289062],
		[  53.5888443 , -290.25338745],
		[  65.81463623, -341.46832275],
		[ 101.07259369, -361.53491211],
		[ 122.78975677, -391.83300781],
		[ 136.22935486, -352.81604004],
		[ 114.80623627, -318.71374512],
		[ 167.69758606, -335.17553711],
		[ 214.97885132, -337.76928711],
		[ 268.80731201, -338.53485107],
		[ 316.49282837, -331.34683228],
		[ 350.15072632, -349.76928711],
		[ 363.43197632, -315.17553711],
		[ 170.24447632, -407.59741211],
		[ 221.22885132, -405.47241211],
		[ 270.54135132, -409.98803711],
		[ 325.93197632, -398.12866211],
		[ 362.99447632, -379.26928711],
		[ 395.51010132, -350.64428711],
		[ 426.2131958 , -314.92553711],
		[ 417.29135132, -288.36303711],
		[ 447.74447632, -274.65991211]], dtype=np.float32)
	head_pts = np.zeros((head_pts_bogons.shape[0],3),dtype=np.float32)
	head_pts[:,0] = mm_per_bogon*head_pts_bogons[:,0]
	head_pts[:,1] = -mm_per_bogon*head_pts_bogons[:,1] # our y-axis is up
	#print head_pts
	head_pts += [150,-100,0] - np.mean(head_pts,axis=0)
	#head_pts += [85,-193,0]
	head_pts = np.dot(head_pts - offset[:3,3],offset[:3,:3])

	c3d_points = UI.GLPoints3D([])
	surface_points = UI.GLPoints3D([])
	head_points = UI.GLPoints3D(head_pts); head_points.colour = (0,1,1,1.0)

	# generate the animation
	if False:
		tsv_filename = 'tony_shape_vectors6'
		try:
			tony_shape_vectors = IO.load(tsv_filename)[1]
		except:
			tony_shape_vectors = np.zeros((len(c3d_frames), ted_lo_mat.shape[0]),dtype=np.float32)
			bnds = np.array([[0,1]]*ted_lo_mat.shape[0],dtype=np.float32)
			x_0 = np.zeros(ted_lo_mat.shape[0],dtype=np.float32)
			for fi, frame in enumerate(c3d_frames):
				which = np.where(frame[:,3] == 0)[0]
				x3ds = frame[which,:3]
				#print which,x3ds.shape,ted_lo_rest.shape,ted_lo_mat.shape
				x_0[:] = tony_shape_vectors[fi] = fitLoResShapeMat(ted_lo_rest, ted_lo_mat, x3ds, indices=which, bounds=bnds, x_0=x_0)
				print '\rfitting',fi,; sys.stdout.flush()
			IO.save(tsv_filename,tony_shape_vectors)

	primitives = [head_points,c3d_points,surface_points,ted_geom,ted_geom2,tony_geom]
	primitives.extend(geos)

	QGLViewer.makeViewer(timeRange = (0,len(c3d_frames)-1), callback = animateHead, primitives = primitives)
	exit()
