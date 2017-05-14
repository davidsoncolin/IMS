#!/usr/bin/env python

import os, sys, struct
import numpy as np
import ISCV
from IO import IO

import dlib
global g_detector
g_detector = dlib.get_frontal_face_detector()

# cl test code

#import pyopencl as cl

# global CL_ctx
# CL_ctx = None
# global g_tid, g_bid, g_detector
# g_tid, g_bid, g_detector = None,None,dlib.get_frontal_face_detector()

# def init_cl(mats):
	# global CL_ctx, CL_queue, CL_prg, CL_mats_g, CL_mats_shape, CL_out_g
	# if CL_ctx is None:
		# CL_ctx = cl.create_some_context(False)
		# CL_queue = cl.CommandQueue(CL_ctx)
		# CL_prg = cl.Program(CL_ctx, '''
	# __kernel void sum_indices(__global const float *mat_g,  __global const int *indices_g, 
								# int vec_size, int indices_size, __global float *out_g) {
		# const int j = get_global_id(0);
		# __global const float *md = mat_g + j;
		# float sum = 0;
		# for (int i = 0; i < indices_size; ++i) sum += md[vec_size*indices_g[i]];
		# out_g[j] = sum;
	# }
	# ''').build()
		# numEpochs,numLeaves,numVertices,_2 = mats.shape # 15,500*16,68,2
		# import pyopencl.array
		# CL_mats_g = [cl.array.to_device(CL_queue, leaves) for leaves in mats]
		# CL_mat_sshape = mats.shape # 15,500*16,68,2
		# CL_out_g = cl.array.to_device(CL_queue, np.zeros((numVertices,_2), dtype=np.float32))

# def CL_sum_indices(epoch, indices):
	# global CL_ctx, CL_queue, CL_prg, CL_mats_g, CL_mats_shape, CL_out_g
	# CL_indices_g = cl.array.to_device(CL_queue, indices)
	# numEpochs,numLeaves,size,_2 = CL_mats_shape
	# CL_prg.sum_indices(CL_queue, (size*_2,), None, 
					# CL_mats_g[epoch].data, CL_indices_g.data, np.int32(size*_2),np.int32(indices.shape[0]),CL_out_g.data)
	# return CL_out_g.get()


def load_image(fn):
	'''image is rgb and all 2d coordinates are y-down.'''
	from scipy.misc import imread
	img = imread(fn)
	if img is not None and len(img.shape) != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
		ret = np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
		ret[:,:,:3] = img.reshape(img.shape[0],img.shape[1],-1)[:,:,:3]
		img = ret
	return img

def get_boundary(vs, template_vs):
	'''template_vs_N+4x2 mat_2x2= vs_Nx2 ; ret_4x2'''
	size = len(template_vs)
	toff,coff = np.mean(template_vs,axis=0), np.mean(vs,axis=0)
	tvs,cvs = template_vs - toff, vs - coff
	mat = np.linalg.lstsq(tvs[:size-4,:2], cvs[:size-4,:2])[0]
	return np.dot(tvs[size-4:size,:2], mat) + coff

def triangulate_2D(data_vs):
	'''Delaunay triangulation'''
	from scipy.spatial import Delaunay
	return np.int32(Delaunay(data_vs).simplices.tolist())

def test_reboot(img, vs):
	h,w = img.shape[:2]
	border = w*0.5
	if vs is None: return True
	cx,cy = np.mean(vs[:,:2],axis=0)
	vx,vy = (np.var(vs[:,:2],axis=0)**0.5) * 2
	return cx-vx < -border or cy-cy < -border or cx+vx > w+border or cy+vy > h+border or vx < 30 or vy < 30 or min(vx,vy)/max(vx,vy) < 0.6

#@profile
def eval_shape_predictor(img, pred, init, cutLo=0):
	ref_pinv,ref_shape,forest_splits,forest_leaves,pix_anchors,pix_deltas = pred['ref_pinv'],pred['ref_shape'],pred['splits'],pred['leaves'],pred['pix_anchors'],pred['pix_deltas']
	forest_splits,forest_leaves,pix_anchors,pix_deltas = forest_splits[cutLo:],forest_leaves[cutLo:],pix_anchors[cutLo:],pix_deltas[cutLo:]
	current_shape = np.dot(ref_shape, np.dot(ref_pinv,init)) + np.mean(init,axis=0)
	#current_shape = (init*0.5 + current_shape*0.5)
	if 1: # all-in-one C 1.8ms
		forest_leaves2 = forest_leaves.reshape(forest_leaves.shape[0],forest_leaves.shape[1],forest_leaves.shape[2],-1)
		ISCV.eval_shape_predictor(ref_pinv, current_shape, forest_splits, forest_leaves2, pix_anchors, pix_deltas, img, current_shape, True)
		return current_shape
	if 1: # single python loop C
		pixels = np.zeros(pix_anchors.shape[1], dtype=np.uint8)
		leaf_indices = np.zeros(forest_leaves.shape[1], dtype=np.int32)
		forest_leaves2 = forest_leaves.reshape(forest_leaves.shape[0],-1,forest_leaves.shape[3],forest_leaves.shape[4])
		for ei,(splits,leaves,rpas,rpds) in enumerate(zip(forest_splits,forest_leaves2,pix_anchors,pix_deltas)):
			ISCV.sample_pixels(current_shape, ref_pinv, rpas, rpds, img, pixels, True)
			ISCV.traverse_forest(splits, pixels, leaf_indices)
			tmp = ISCV.sum_indices(leaves.reshape(leaves.shape[0],-1),leaf_indices).reshape(leaves.shape[1:])
			A = np.dot(ref_pinv,current_shape)
			current_shape += np.dot(tmp,A)
		return current_shape
	if 0: # pure python implementation (>100xC)
		clip_min,clip_max = np.array([[0,0],[img.shape[1]-1,img.shape[0]-1]], dtype=np.int32)
		for (tree_splits,leaves,rpas,rpds) in zip(forest_splits,forest_leaves,pix_anchors,pix_deltas):
			A = np.dot(ref_pinv,current_shape)
			p = (np.dot(rpds,A) + current_shape[rpas]).astype(np.int32) # test coordinates as pixels
			if True: p[:,1] = img.shape[0]-p[:,1]
			np.clip(p,clip_min,clip_max,out=p) # clip to image
			pixels = img[p[:,1],p[:,0],1] # and sample
			for (splits,leaf_values) in zip(tree_splits,leaves):
				idx1,idx2,thresh = splits[0]
				i = (2 if (pixels[idx1] <= thresh + pixels[idx2]) else 1)
				idx1,idx2,thresh = splits[i]
				i = (2*i+2 if (pixels[idx1] <= thresh + pixels[idx2]) else 2*i+1)
				idx1,idx2,thresh = splits[i]
				i = (2*i+2 if (pixels[idx1] <= thresh + pixels[idx2]) else 2*i+1)
				idx1,idx2,thresh = splits[i]
				i = (2*i+2 if (pixels[idx1] <= thresh + pixels[idx2]) else 2*i+1)
				current_shape += np.dot(leaf_values[i-len(splits)],A)
		return current_shape
	if 1: # pure python vectorized (3xC)
		clip_min,clip_max = np.array([[0,0],[img.shape[1]-1,img.shape[0]-1]], dtype=np.int32)
		numEpochs,numTrees,treeDepth,_3 = forest_splits.shape # 15,500,15,3
		splits_off = np.arange(0,numTrees, dtype=np.int32)*treeDepth - 1
		forest_splits = forest_splits.reshape(numEpochs,-1,3)
		leaves_off = np.arange(0,numTrees, dtype=np.int32)*forest_leaves.shape[2] - 16
		forest_leaves = forest_leaves.reshape(numEpochs,-1,current_shape.shape[0],2)
		#pixels = np.zeros(pix_anchors.shape[1], dtype=np.uint8)
		for (splits,leaves,rpas,rpds) in zip(forest_splits,forest_leaves,pix_anchors,pix_deltas):
			A = np.dot(ref_pinv,current_shape)
			p = (np.dot(rpds,A) + current_shape[rpas]).astype(np.int32) # test coordinates as pixels
			np.clip(p,clip_min,clip_max,out=p) # clip to image
			if True: p[:,1] = img.shape[0]-p[:,1]
			pixels = img[p[:,1],p[:,0],1] # and sample
			idx = np.ones(numTrees, dtype=np.int32)
			for x in range(4):
				sp1 = splits[idx+splits_off]
				idx *= 2
				idx += (pixels[sp1[:,0]] < pixels[sp1[:,1]] + sp1[:,2])
			current_shape += np.dot(np.sum(leaves[idx+leaves_off],axis=0),A)
		return current_shape

def detect_face(img, predictor, speedup=1, rotation=0):
	'''given a predictor model, detect a face and return a correctly positioned and scaled shape'''
	global g_detector
	rects = None
	try:
		if not rects and (rotation == 0 or rotation == -1):
			transposed = False
			flipped = 1
			if speedup == 1:
				rects = g_detector(img,0)  # run the face detector; 2nd parameter is number of upsamples (for tiny faces)
			else:
				rects = g_detector(img[::speedup,::speedup].copy(),0)  # run the face detector; 2nd parameter is number of upsamples (for tiny faces)
		if not rects and (rotation == 1 or rotation == -1): # try again with rotated image
			transposed = True
			flipped = 0
			rects = g_detector(img[::-speedup,::speedup].transpose(1,0,2).copy(),0)  # run the face detector; 2nd parameter is number of upsamples (for tiny faces)
		if not rects and (rotation == 2 or rotation == -1):
			transposed = False
			flipped = 2
			rects = g_detector(img[::-speedup,::-speedup].copy(),0)  # run the face detector; 2nd parameter is number of upsamples (for tiny faces)
		if not rects and (rotation == 3 or rotation == -1): # try again with rotated image
			transposed = True
			flipped = 3
			rects = g_detector(img[::speedup,::-speedup].transpose(1,0,2).copy(),0)  # run the face detector; 2nd parameter is number of upsamples (for tiny faces)
		if not rects: return None
		rect = rects[0]
		# rect is y-down; set img_scl[1] < 0 to invert the y-up ref_shape
		l,r,t,b = rect.left(),rect.right(),rect.top(),rect.bottom()
		img_scl = np.float32([(r-l)*speedup,(t-b)*speedup])
		img_off = np.float32([(l+r)*0.5*speedup,(t+b)*0.5*speedup])
		ret = predictor['ref_shape'] * img_scl + img_off
		if transposed: ret = ret[:,::-1]
		if flipped & 1: ret[:,1] = img.shape[0]-ret[:,1]
		if flipped & 2: ret[:,0] = img.shape[1]-ret[:,0]
		print 'det',l,r,t,b,img_scl,img_off,transposed,flipped
		return ret
	except Exception as e:
		import traceback
		print 'detect_face exception',e,traceback.format_exc()
		return None

def boot_rotate(predictor, (h,w), rotate):
	if rotate == -1: rotate = 0 # ARGH
	transposed = [False, True, False, True][rotate]
	flipped = [1,0,2,3][rotate]
	# simulate values from the booter (negative y-scale)
	img_scl = np.float32([w*0.5,-h*0.5])
	img_off = np.float32([w*0.5,h*0.5])
	ret = predictor['ref_shape'] * img_scl + img_off
	if transposed: ret = ret[:,::-1]
	if flipped & 1: ret[:,1] = h-ret[:,1]
	if flipped & 2: ret[:,0] = w-ret[:,0]
	return ret

#@profile
def track_face(img, predictor, out=None, numIts=5, rotate=0):
	'''out are 2d points, in y-up coordinates. img is y-down coordinates. generate an x2d data array in y-up coordinates.'''
	if out is None:
		out = boot_rotate(predictor, img.shape[:2], rotate)
	sum_out = np.zeros_like(out)
	for it in range(numIts):
		sum_out += eval_shape_predictor(img, predictor, out) #, cutLo=4*(it>0))
		out[:] = sum_out/(it+1)
	return out

def fix_image(img, vs=None, max_size=640):
	if img.shape[2] == 1:
		img = np.hstack((img,img,img))
	if img.shape[0] > max_size or img.shape[1] > max_size:
		step = (int(img.shape[1] / (max_size/2))+1)/2
		img = img[step/2::step,step/2::step,:]
		if vs is not None: vs *= (1.0/step)
	if img.shape[0]%8!=0 or img.shape[1]%8!=0:
		img = img[img.shape[0]%8:,:img.shape[1]/8*8,:] # crop from bottom left (doesn't affect vs)
	return img.copy()

def show_image(img=None, shape=None, shape2=None, height=None):
	# show a y-down image, and a y-up shape (for debugging)
	import pylab as pl
	if img is not None: pl.imshow(img); height = img.shape[0]
	if shape is not None:
		if height is not None: pl.plot(shape[:,0], height-shape[:,1], 'o-r')
		else: pl.plot(shape[:,0], shape[:,1], 'o-r')
	if shape2 is not None:
		if height is not None: pl.plot(shape2[:,0], height-shape2[:,1], 'o-b')
		else: pl.plot(shape2[:,0], shape2[:,1], 'o-b')
	pl.show()

def normalized_aam_coords(model, shape_u, tex_u):
	'''put the shape and texture coordinates into a Mahalanobis space where their norm represents the log likelihood.'''
	shapes_s,texture_s = model['shapes_s'],model['texture_s']
	return (shape_u/shapes_s)*(len(shape_u)**0.5), (tex_u/texture_s)*(len(tex_u)**0.5)

def convert_aam(model):
	'''Legacy: convert old-style model into new-style.'''
	if isinstance(model, dict): return model
	(ref_shape,ref_pinv,model_indices,model_weights,shapes,textures,\
	shapes_u,shapes_s,shapes_vt,texture_mean,texture_u,texture_s,texture_vt,order) = model
	return {'ref_shape':ref_shape,'ref_pinv':ref_pinv,'model_indices':model_indices,'model_weights':model_weights,\
		'shapes':shapes,'textures':textures,'shapes_u':shapes_u,'shapes_s':shapes_s,'shapes_vt':shapes_vt,\
		'texture_mean':texture_mean,'texture_u':texture_u,'texture_s':texture_s,'texture_vt':texture_vt}

def aam_residual(model, shp, img):
	'''compute the norm of the texture residual.'''
	texture_mean, texture_vt = model['texture_mean'],model['texture_vt']
	tex = extract_texture(img, shp, model['model_indices'], model['model_weights'])
	tex_u = np.dot((tex - texture_mean).reshape(-1), texture_vt.T)
	tex2 = texture_mean + np.dot(tex_u, texture_vt).reshape(-1,3)
	return np.linalg.norm((tex-tex2)/(len(tex)**0.5))/(np.linalg.norm(tex-np.mean(tex))/(len(tex)**0.5)+1e-8)

def fit_aam_shape_image_weights(model, shp):
	'''given an aam and a shape, extract the pose and image space coordinates.'''
	A = np.dot(model['ref_pinv'], shp)
	A_inv = np.linalg.inv(A)
	mn = np.mean(shp, axis=0)
	shape_u = np.dot((np.dot(shp - mn, A_inv) - model['ref_shape']).reshape(-1), model['shapes_vt'].T)
	#shape_u = aam_clip_shape_weights(shape_u, shapes_s)
	image_u = np.dot(model['shapes_u'], shape_u) #*shapes_s)
	return image_u, A, mn

def reconstruct_aam_shape_from_image_weights(model, A, mn, image_u):
	'''given an aam and pose and images space coordinates, generate the shape'''
	shape_u = np.dot(model['shapes_u'].T, image_u) #/shapes_s
	shp = ref_shape + np.dot(shape_u, model['shapes_vt']).reshape(-1,2)
	shp = np.dot(shp, A) + mn
	return shp

def aam_clip_shape_weights(shape_u, shapes_s):
	vals = ((9.0/len(shape_u))**0.5)*shapes_s
	#print shape_u/vals
	return np.minimum(np.maximum(shape_u, -vals), vals, out=shape_u)

def safe_inv(mat, threshold=1e-3):
	u,s,vt = np.linalg.svd(mat,full_matrices = 0)
	zero = np.where(s < threshold*s[0])
	s = 1.0/s
	s[zero] = 0
	return np.dot(vt.T, u.T * s.reshape(-1,1))

def weighted_fit_aam_shape(model, shp, weights):
	'''given an aam and a shape and weights per vertex, fit to the model and return the shape.'''
	weights = weights / np.mean(weights)
	wts = weights.reshape(-1,1)
	ref_shape = model['ref_shape']
	m_3x2 = np.linalg.lstsq(np.hstack((ref_shape*wts, wts)), shp*wts)[0]
	# (ref | 1)*wts m_3x2 = shp*wts
	A,mn = m_3x2[:2],m_3x2[2]
	A_inv = safe_inv(A)
	w = np.hstack((wts,wts)).reshape(1,-1)
	u2,s2,vt2 = np.linalg.svd(model['shapes_s'].reshape(-1, 1) * model['shapes_vt'] * w, full_matrices=0)
	#print shapes_s/s2
	rank = np.sum(s2 > s2[0]*0.001)
	u2,s2,vt2 = u2[:,:rank],s2[:rank],vt2[:rank,:]
	#print 'cutoff',rank
	shape_u = np.dot(((np.dot(shp-mn, A_inv) - ref_shape)*wts).reshape(-1), vt2.T)
	aam_clip_shape_weights(shape_u, s2)
	shp = np.dot(shape_u, vt2).reshape(-1, 2)/wts + ref_shape
	shp = np.dot(shp, A) + mn
	return shp

# Capitals for matrices, lower case for vectors. _NxM indicates a matrix of N rows and M columns. ' indicates a lossy result.
# PCA model:
# I = numImages(3241), M = numModes(130), V = 2*numVerts(138)
# Given a matrix A of mean-centred shape vectors per image: A_IxV
# A^T_VxI i_I = v_V          # A^T generates a weighted sum of example shapes
# i is the image weights vector, which is independent of the vertices
# we form the SVD : A' = U.diag(S).VT and cut off at M by setting the low values in S to zero.
# U_IxM S_M V^T_MxV = A'_IxV # this is PCA: A' is an M-modes (or M-principal components) approximation of A.
# U^T U = V^T V = I_MxM      # identity matrices
# U are the weighted sum of image shapes that form each mode; i = U m; m' = U^T i (lossy since M < V)
# V S are the modes; v = V S m; m' = 1/S V^T v (lossy since M < V)
# m is the coordinate in PCA space (Mahalanobis)
# normal coordinates of m are in the range +-sqrt(9/V)
# m' = 1/S V^T v             # given a shape, m is its coordinate in PCA space (lossy since M < V)
# V S m' = v'                # v' is a reconstruction of v
# V V^T v = v'               # v' is the nearest point to v in the span of the modes; let's prove it
# VS(m'+d)-v = v'-v + VSd    # suppose that m'+d is closer to v', ie norm < (v'-v)^2
# norm = (v'-v)^2 + (VSd)^2 + 2*(v'-v).VSd
# observe that norm >= (v'-v)^2 if (v'-v).VSd = 0
# =dSV^T(v'-v)=dSSm'-dSSm'=0 QED
# i' = U 1/S V^T v           # putting the equations together, we can convert from a vector to image coordinates (lossy)
# A' i' = v'                 # and from image coordinates back to a vector
# what we want to do is fit to a partial shape
# suppose we built the model from a subset of vertices
# A^T[which] i = v[which]
# A^T[which] = U S V[which]^T
# V[which]^T V[which] = I is no longer true
# U2_MxM2 S2_M2 VT2_M2xV2


# d = v' - v = (V V^T - I) v # let's prove it: get the delta
# 2 d grad d = 0
# Given a partial shape, the model can generate a complete shape

# err = sum((M^T M - I) s*w)**2

def weighted_reconstruct_aam_shape(model, A_inv, mn, shape_u, w):
	'''given an aam and pose and shape space coordinates, generate the shape'''
	print w
	shp = model['ref_shape'] + np.dot(shape_u, model['shapes_vt']).reshape(-1,2)/w
	shp = np.dot(shp,A_inv) + mn
	return shp

def fit_aam_shape(model, shp):
	'''given an aam and a shape, extract the pose and shape space coordinates.'''
	shp,A,mn = normalize(shp, model['ref_pinv'])
	shape_u = np.dot((shp - model['ref_shape']).reshape(-1), model['shapes_vt'].T)
	return shape_u, A, mn

def reconstruct_aam_shape(model, A_inv, mn, shape_u):
	'''given an aam and pose and shape space coordinates, generate the shape'''
	shp = model['ref_shape'] + np.dot(shape_u, model['shapes_vt']).reshape(-1,2)
	shp = np.dot(shp, A_inv) + mn
	return shp

def fit_aam(model, shp, img):
	'''given an aam, a shape and an image, extract the shape and texture coordinates.'''
	tex = extract_texture(img, shp, model['model_indices'], model['model_weights'])
	tex_u = np.dot((tex - model['texture_mean']).reshape(-1), model['texture_vt'].T)
	A = np.dot(model['ref_pinv'], shp)
	A_inv = np.linalg.inv(A)
	mn = np.mean(shp,axis=0)
	shape_u = np.dot((np.dot(shp - mn, A_inv) - model['ref_shape']).reshape(-1), model['shapes_vt'].T)
	return shape_u, tex_u, A, mn, 

def render_aam(model, A_inv, mn, shape_u, tex_u, img):
	'''given an aam and shape and texture coordinates, render an image.'''
	shp = reconstruct_aam_shape(model, A_inv, mn, shape_u)
	tex = model['texture_mean'] + np.dot(tex_u, model['texture_vt']).reshape(-1,3)
	np.clip(tex,0,255,out=tex)
	render_texture(tex, img, shp, model['model_indices'], model['model_weights'])

# Support for reading dlib-format files (legacy)

def dlib_convert_to_grip(directory):
	print 'converting predictor data file, this will take a couple of minutes'
	data = open(os.path.join(directory,'shape_predictor_68_face_landmarks.dat'),'rb').read()
	data,shape_predictor = dlib_read_shape_predictor((data,0))
	assert data[1] == len(data[0])
	version,ref_shape,forests,pix_anchors,pix_deltas = shape_predictor
	splits = [[f[0] for f in forest] for forest in forests]
	leaves = [[np.array(f[1], dtype=np.float32).reshape(len(f[1]),-1,2) for f in forest] for forest in forests]
	splits,leaves = np.array(splits, dtype=np.int32),np.array(leaves, dtype=np.float32)
	pix_anchors = np.array(pix_anchors, dtype=np.int32)
	pix_deltas = np.array(pix_deltas, dtype=np.float32)
	ref_shape.shape=(-1,2)
	ref_shape -= np.mean(ref_shape, axis=0)
	ref_pinv = np.linalg.pinv(ref_shape)
	return {'ref_pinv':ref_pinv,'ref_shape':ref_shape,'splits':splits,'leaves':leaves,'pix_anchors':pix_anchors,'pix_deltas':pix_deltas}
	
def dlib_read_int(d):
	data,o = d
	l = ord(data[o])
	s = (l>=0x80)
	l = (l & 0x7f)
	v = struct.unpack_from('<Q',data[o+1:o+l+1]+chr(0)*(8-l))[0]
	return (data,o+l+1),(-v if s else v)

def dlib_read_float(data):
	data,m = dlib_read_int(data)
	data,e = dlib_read_int(data)
	f = m * (2.0**e)
	return data,f

def dlib_read_split_feature(data):
	data,idx1 = dlib_read_int(data)
	data,idx2 = dlib_read_int(data)
	data,thresh = dlib_read_float(data)
	return data,(idx1,idx2,thresh)

def dlib_read_regression_tree(data):
	data,splits = dlib_read_vector(data,dlib_read_split_feature)
	data,leaf_values = dlib_read_vector(data,dlib_read_matrix_float)
	return data,(splits,leaf_values)

def dlib_read_vector(data, func):
	data,size = dlib_read_int(data)
	ret = []
	for s in range(size):
		data,vi = func(data)
		ret.append(vi)
	return data,ret

def dlib_read_matrix_float(data):
	data,rows = dlib_read_int(data)
	data,cols = dlib_read_int(data)
	rows,cols = abs(rows),abs(cols)
	ret = []
	for r in range(rows):
		v = np.zeros(cols, dtype=np.float32)
		for c in range(cols): data,v[c] = dlib_read_float(data)
		ret.append(v)
	return data,np.array(ret, dtype=np.float32)

def dlib_read_dlib_vector_float_2(data):
	data,f1 = dlib_read_float(data)
	data,f2 = dlib_read_float(data)
	return data,(f1,f2)

def dlib_read_shape_predictor(data):
	data,version = dlib_read_int(data)
	assert version==1,'bad version'
	print 'initial_shape',data[1],'/',len(data[0])
	data,initial_shape = dlib_read_matrix_float(data)
	print 'forests',data[1],'/',len(data[0])
	data,forests = dlib_read_vector(data,lambda d:dlib_read_vector(d,dlib_read_regression_tree))
	#forests = np.array(forests, dtype=np.float32)
	print 'pix_anchors',data[1],'/',len(data[0])
	data,pix_anchors = dlib_read_vector(data,lambda d:dlib_read_vector(d,dlib_read_int))
	pix_anchors = np.array(pix_anchors, dtype=np.int32)
	print 'pix_deltas',data[1],'/',len(data[0])
	data,pix_deltas = dlib_read_vector(data,lambda d:dlib_read_vector(d,dlib_read_dlib_vector_float_2))
	pix_deltas = np.array(pix_deltas, dtype=np.float32)
	return data,(version,initial_shape,forests,pix_anchors,pix_deltas)

# support for reading dlib-format database
	
def dlib_load_images_shapes(directories, cache_fn=None):
	try:
		print 'cache',cache_fn
		images,shapes = IO.load(cache_fn)[1]
	except Exception as e:
		print 'err',e
		images,shapes = [],[]
		for dn in directories:
			print dn
			listdir = os.listdir(dn)
			filenames = sorted([x for x in listdir if x.endswith('.jpg') or x.endswith('.png')])
			for fn in filenames:
				if 'mirror' in fn: continue
				print fn
				img = load_image(os.path.join(dn,fn))
				if len(img.shape) != 3: print 'skipping',fn; continue
				if fn[:-4]+'.land' in listdir:
					data = open(os.path.join(dn,fn[:-4]+'.land')).readlines() # y-up coordinates
					nl = int(data[0])
					assert(nl == 74)
					data_vs = np.array(map(lambda x:x.split(),data[1:75]), dtype=np.float32)
				elif fn[:-4]+'.pts' in listdir:
					data = open(os.path.join(dn,fn[:-4]+'.pts')).readlines() # y-up coordinates
					nl = int(data[1].split()[1])
					assert(nl == 68)
					data_vs = np.array(map(lambda x:x.split(),data[3:71]), dtype=np.float32)
					data_vs[:,1] = img.shape[0]-data_vs[:,1]
				else:
					print '?????'
					continue
				mn = np.min(data_vs,axis=0)
				mx = np.max(data_vs,axis=0)
				md = mx-mn
				mn = np.max(([0,0],mn-md*0.5),axis=0)
				mx = np.min(([img.shape[1],img.shape[0]],mx+md*0.5),axis=0)
				img = img[img.shape[0]-mx[1]:img.shape[0]-mn[1],mn[0]:mx[0]].copy() # crop to roi
				data_vs -= mn
				img = fix_image(img, data_vs)
				#show_image(img, data_vs)
				images.append(img)
				shapes.append(data_vs)
		IO.save(cache_fn,(images,shapes))
	if isinstance(shapes[0],tuple): shapes = list(zip(*shapes)[0])
	print 'successfully loaded',len(images),'images and markups'
	for it,(i,o) in enumerate(zip(images,shapes)):
		h,w,_ = i.shape
		mn = np.min(o,axis=0)
		mx = np.max(o,axis=0)
		if not (mn[0] > -10 and mn[1] > -10 and mx[0] < w+10 and mx[1] < h+10):
			print it,'warn',mn,mx,i.shape
			show_image(i,o)
	print 'checked ok',len(images)
	return images,shapes

def dlib_test(predictor):
	grip_dir = os.environ['GRIP_DATA']
	directory = os.path.join(grip_dir,'ibug_300W_large_face_landmark_dataset')
	directories = [os.path.join(directory,d) for d in ['lfpw/testset','helen/testset']]
	fn = directory+'in-test.train'
	images,shapes = dlib_load_images_shapes(directories, fn)
	double_data(images,shapes)
	#show_image(None,shapes[0],(shapes[0]+shapes[0][flip_order]*[-1,1])*0.5)
	return test_shape_predictor(predictor, images, shapes)
	
def dlib_train():
	grip_dir = os.environ['GRIP_DATA']
	directory = os.path.join(grip_dir,'ibug_300W_large_face_landmark_dataset')
	directories = [os.path.join(directory,d) for d in ['lfpw/trainset','ibug','helen/trainset','afw']]
	fn = directory+'in-train.train'
	images,shapes = dlib_load_images_shapes(directories, fn)
	double_data(images,shapes)
	return train_shape_predictor(images, shapes)


# ### Training

def normalizing_A_square(shape, ref_pinv):
	u,s,vt = np.linalg.svd(np.dot(ref_pinv, shape), full_matrices=False)
	return np.dot(u,vt).T * ((s[0]*s[1])**-0.5)

def normalizing_A(shape, ref_pinv):
	return np.linalg.inv(np.dot(ref_pinv, shape))

def normalize_shape(shape, ref_pinv):
	return np.dot(shape - np.mean(shape,axis=0), normalizing_A(shape, ref_pinv))

def fit_shape(src, tgt, ref_pinv):
	return np.dot(normalize_shape(src, ref_pinv), np.dot(ref_pinv, tgt)) + np.mean(tgt,axis=0)

def regularize(shape, padding=0.05):
	'''scale and translate a shape to fit in the unit square'''
	shape = np.float32(shape)
	mn = np.min(shape, axis=0)
	sc = np.max(shape, axis=0)-mn
	mn -= sc * padding
	sc *= 1+2*padding
	return (shape-mn)/sc, sc, mn

def normalize(shape, ref_pinv):
	'''affine transform a shape to match the reference'''
	A = np.dot(ref_pinv, shape)
	A_inv = np.linalg.inv(A)
	mn = np.mean(shape, axis=0)
	shp = np.dot(shape-mn, A_inv)
	return shp, A, mn

def normalized_shapes(shapes):
	shapes = np.float32(zip(*map(regularize,shapes))[0]) # regularize shapes to unit square before averaging them
	ref_shape = regularize(np.sum(shapes, axis=0))[0] # regularize to fit in unit square
	ref_shape -= np.mean(ref_shape, axis=0) # ref_shape is mean centered
	ref_pinv = np.linalg.pinv(ref_shape)
	shapes = np.float32([normalize_shape(shape, ref_pinv) for shape in shapes]) # now straighten shapes
	ref_shape = np.mean(shapes, axis=0) # and average again
	ref_shape -= np.mean(ref_shape, axis=0) # ref_shape is mean centered
	ref_pinv = np.linalg.pinv(ref_shape)
	return shapes, ref_shape, ref_pinv

def generate_shape_samples(shapes, ref_shape, ref_pinv, samples_per_image, images=None):
	# images are just for debugging
	samples = [] # generate some samples
	for img_index,shape in enumerate(shapes):
		norm_shape,A,mn = normalize(shape, ref_pinv)
		samples.append((img_index,shape,shape))
		if samples_per_image > 1:
			samples.append((img_index,shape,np.dot(ref_shape,A)+mn))
			for rands in np.random.random_sample((samples_per_image-2,3,2))-0.5:
				current_shape = np.zeros_like(norm_shape)
				for rand_idx in np.random.randint(len(shapes),size=4):
					rand_shape = normalize(shapes[rand_idx], ref_pinv)[0]
					current_shape += rand_shape*(np.random.random_sample()+0.1)
				current_shape = normalize(current_shape, ref_pinv)[0]
				cs_mn = np.mean(current_shape, axis=0) # I think this should be zero
				current_shape -= cs_mn
				current_shape = np.dot(current_shape, np.float32(rands[:2]*0.3+np.eye(2)))  # random distort 0.85:1.15
				current_shape += np.float32(rands[2]*0.3)  # random offset -0.15:0.15
				current_shape += cs_mn
				samples.append((img_index,shape,np.dot(current_shape,A)+mn))
				#if images is not None: show_image(images[img_index], np.dot(current_shape,A)+mn, np.dot(norm_shape,A)+mn)
	image_indices,target_shapes,current_shapes = zip(*samples)
	return np.int32(image_indices),np.float32(target_shapes),np.float32(current_shapes)

def test_shape_predictor(pred, images, shapes):
	# images are in y-down bitmap order, shapes have y-up coordinates
	ref_pinv,ref_shape,forest_splits,forest_leaves,pix_anchors,pix_deltas = pred['ref_pinv'],pred['ref_shape'],pred['splits'],pred['leaves'],pred['pix_anchors'],pred['pix_deltas']
	image_indices,target_shapes,current_shapes = generate_shape_samples(shapes, ref_shape, ref_pinv, 5, images)
	delta_shapes = np.zeros_like(current_shapes)
	for image_index,target_shape,current_shape,delta_shape in zip(image_indices,target_shapes,current_shapes,delta_shapes):
		delta_shape[:] = np.dot(target_shape-current_shape, normalizing_A(current_shape, ref_pinv)) # store the delta_shape
	leaf_indices = np.zeros(forest_leaves.shape[1], dtype=np.int32)
	forest_leaves = forest_leaves.reshape((forest_leaves.shape[0],-1,forest_leaves.shape[3],forest_leaves.shape[4]))
	pixels = np.zeros(pix_anchors.shape[1], dtype=np.uint8)
	true_shapes = np.float32([normalize(shape, ref_pinv)[0] for shape in target_shapes])
	dsc = ((delta_shapes.size*0.5)**-0.5)/0.4
	print np.linalg.norm(delta_shapes)*dsc
	ret = []
	for it2 in range(5):
		delta_shapes = 0.5 * delta_shapes + 0.5 * (true_shapes-ref_shape)
		for it,(splits,leaves,rpas,rpds) in enumerate(zip(forest_splits, forest_leaves, pix_anchors, pix_deltas)):
			#if it2 != 0 and it < 4: continue
			for image_index,target_shape,current_shape,delta_shape in zip(image_indices,target_shapes,current_shapes,delta_shapes):
				ISCV.sample_pixels(current_shape, ref_pinv, rpas, rpds, images[image_index], pixels, True)
				ISCV.traverse_forest(splits, pixels, leaf_indices)
				tmp = ISCV.sum_indices(leaves.reshape(leaves.shape[0],-1),leaf_indices).reshape(leaves.shape[1:])
				current_shape += np.dot(tmp,np.dot(ref_pinv,current_shape))
				A = np.dot(ref_pinv,current_shape)
				np.dot(target_shape-current_shape, normalizing_A(current_shape, ref_pinv), out=delta_shape) # store the delta_shape
		print 'it2 %d/%d' % (it2+1,5),np.linalg.norm(delta_shapes)*dsc
		ret.append(delta_shapes.copy())
		delta_shapes = np.mean(ret,axis=0)
		print 'it2 %d/%d' % (it2+1,5),np.linalg.norm(delta_shapes)*dsc

#@profile
def train_shape_predictor(images, shapes, samples_per_image=20, num_forests=15, pixels_per_forest=500, pixel_lambda=0.15, trees_per_forest=500, tree_depth=4, test_splits=20, learn_rate=0.2):
	# images are in y-down bitmap order, shapes are shapes with y-up coordinates
	assert len(images)==len(shapes)
	print 'training from %d images' % len(images)
	norm_shapes, ref_shape, ref_pinv = normalized_shapes(shapes)
	forest_splits = np.zeros((num_forests,trees_per_forest,(2**tree_depth)-1,3), dtype=np.int32)
	forest_leaves = np.zeros((num_forests,trees_per_forest,2**tree_depth,ref_shape.shape[0],ref_shape.shape[1]), dtype=np.float32)
	pix_anchors   = np.random.randint(ref_shape.shape[0],size=(num_forests,pixels_per_forest)).astype(np.int32)
	pix_deltas    = (np.random.random((num_forests,pixels_per_forest,2))*0.2-0.1).astype(np.float32)
	pixels        = np.zeros((len(images)*samples_per_image,pixels_per_forest), dtype=np.uint8)
	feat_probs    = np.zeros((pixels_per_forest,pixels_per_forest), dtype=np.float32)
	leaf_indices  = np.zeros(trees_per_forest, dtype=np.int32)
	fls = forest_leaves.reshape((forest_leaves.shape[0],-1,forest_leaves.shape[3],forest_leaves.shape[4]))
	for it,(tree_splits,test_leaves,rpas,rpds) in enumerate(zip(forest_splits, forest_leaves, pix_anchors, pix_deltas)):
		image_indices,target_shapes,current_shapes = generate_shape_samples(shapes, ref_shape, ref_pinv, samples_per_image, images)
		delta_shapes = np.zeros_like(current_shapes)
		for target_shape,current_shape,delta_shape in zip(target_shapes,current_shapes,delta_shapes):
			delta_shape[:] = np.dot(target_shape-current_shape, normalizing_A(current_shape, ref_pinv)) # store the delta_shape
		ds,dss,dsc = delta_shapes.reshape(delta_shapes.shape[0],-1),delta_shapes.shape[1:],((delta_shapes.size*0.5)**-0.5)/0.4 # prepare
		for it2,(splits,leaves,rpas2,rpds2) in enumerate(zip(forest_splits[:it], fls[:it], pix_anchors[:it], pix_deltas[:it])):
			print 'it2',it2+1,np.linalg.norm(delta_shapes)*dsc
			for image_index,target_shape,current_shape,delta_shape in zip(image_indices, target_shapes,current_shapes,delta_shapes):
				ISCV.sample_pixels(current_shape, ref_pinv, rpas2, rpds2, images[image_index], pixels[0], True)
				ISCV.traverse_forest(splits, pixels[0], leaf_indices)
				tmp = ISCV.sum_indices(leaves.reshape(leaves.shape[0],-1),leaf_indices).reshape(leaves.shape[1:])
				current_shape += np.dot(tmp,np.dot(ref_pinv, current_shape))
				np.dot(target_shape-current_shape, normalizing_A(current_shape, ref_pinv), out=delta_shape) # store the delta_shape
		print 'it %d/%d' % (it+1,num_forests),np.linalg.norm(delta_shapes)*dsc
		pix_coords = ref_shape[rpas]+rpds
		for i,(pci,fpi) in enumerate(zip(pix_coords, feat_probs)):
			fpi[:] = np.exp(np.linalg.norm(pix_coords-pci, axis=1) * (-1.0/pixel_lambda))
			fpi[i] = 0.0
		feat_probs_cumsum = np.cumsum(feat_probs, dtype=np.float32)
		feat_probs_cumsum = feat_probs_cumsum[:-1] /feat_probs_cumsum[-1] # searchsorted can return n+1 if rnd exceeds final val
		for image_index,current_shape,pixs in zip(image_indices, current_shapes, pixels):
			ISCV.sample_pixels(current_shape, ref_pinv, rpas, rpds, images[image_index], pixs, True)
		for it2,(splits,leaves) in enumerate(zip(tree_splits, test_leaves)):
			heap = [(np.arange(len(delta_shapes), dtype=np.int32),np.sum(delta_shapes, axis=0))]
			for split in splits:
				sample_indices,sum_deltas = heap.pop(0)
				sample_pixels = pixels[sample_indices]
				best,feat_rans = -1.0,np.float32(np.random.random(test_splits))
				for idx1,idx2 in zip(*np.unravel_index(np.searchsorted(feat_probs_cumsum, feat_rans), feat_probs.shape)):
					vals = np.diff(sample_pixels[:,[idx2,idx1]].astype(np.int32)).reshape(-1)
					thresh = int(np.mean(np.random.choice(vals,3))) if len(vals) else 0
					L_indices,R_indices = sample_indices[vals > thresh],sample_indices[vals <= thresh]
					if len(L_indices) < len(R_indices): # swap to keep left branch the bigger part
						idx1,idx2,thresh,L_indices,R_indices = idx2,idx1,-thresh,R_indices,L_indices
					R_delta = ISCV.sum_indices(ds, R_indices).reshape(dss)
					L_delta = sum_deltas-R_delta
					score = np.linalg.norm(L_delta) + np.linalg.norm(R_delta)
					if score > best: best,split[:],he = score,(idx1,idx2,thresh),[(L_indices,L_delta),(R_indices,R_delta)]
				heap.extend(he)
			for (sample_indices,sum_deltas),leaf in zip(heap, leaves):
				leaf[:] = sum_deltas*np.float32(learn_rate/(len(sample_indices)+1e-8))
				delta_shapes[sample_indices] -= leaf # actually modify the deltas
			if (trees_per_forest-1-it2)%10 == 0: print 'x',(it2+1),np.linalg.norm(delta_shapes)*dsc
		ret = {'ref_pinv':ref_pinv,'ref_shape':ref_shape,'splits':forest_splits[:it+1],'leaves':forest_leaves[:it+1],'pix_anchors':pix_anchors[:it+1],'pix_deltas':pix_deltas[:it+1]}
		IO.save('tmp%d.io'%(it+1),ret)
	return ret

def retrain_shape_predictor(pred, images, shapes, samples_per_image=20, pixel_lambda=0.15, learn_rate=0.2):
	'''Given a previous predictor, rebuild the structure with potentially updated images (about 20x faster).'''
	ref_pinv,ref_shape,forest_splits,forest_leaves,pix_anchors,pix_deltas = pred['ref_pinv'],pred['ref_shape'],pred['splits'],pred['leaves'],pred['pix_anchors'],pred['pix_deltas']
	# images are in y-down bitmap order, shapes are shapes with y-up coordinates
	assert len(images)==len(shapes)
	print 'training from %d images' % len(images)
	norm_shapes, ref_shape, ref_pinv = normalized_shapes(shapes)
	num_forests = forest_splits.shape[0]
	trees_per_forest = forest_splits.shape[1]
	tree_depth = int(np.log2(forest_leaves.shape[2]))
	pixels_per_forest = pix_anchors.shape[1]
	assert(forest_splits.shape == (num_forests,trees_per_forest,(2**tree_depth)-1,3))
	assert(forest_leaves.shape == (num_forests,trees_per_forest,2**tree_depth,ref_shape.shape[0],ref_shape.shape[1]))
	assert(pix_anchors.shape == (num_forests,pixels_per_forest))
	assert(pix_deltas.shape == (num_forests,pixels_per_forest,2))
	pixels        = np.zeros((len(images)*samples_per_image,pixels_per_forest), dtype=np.uint8)
	leaf_indices  = np.zeros(trees_per_forest, dtype=np.int32)
	fls = forest_leaves.reshape((forest_leaves.shape[0],-1,forest_leaves.shape[3],forest_leaves.shape[4]))
	for it,(tree_splits,test_leaves,rpas,rpds) in enumerate(zip(forest_splits, forest_leaves, pix_anchors, pix_deltas)):
		image_indices,target_shapes,current_shapes = generate_shape_samples(shapes, ref_shape, ref_pinv, samples_per_image, images)
		delta_shapes = np.zeros_like(current_shapes)
		for target_shape,current_shape,delta_shape in zip(target_shapes,current_shapes,delta_shapes):
			delta_shape[:] = np.dot(target_shape-current_shape, normalizing_A(current_shape, ref_pinv)) # store the delta_shape
		ds,dss,dsc = delta_shapes.reshape(delta_shapes.shape[0],-1),delta_shapes.shape[1:],((delta_shapes.size*0.5)**-0.5)/0.4 # prepare
		for it2,(splits,leaves,rpas2,rpds2) in enumerate(zip(forest_splits[:it], fls[:it], pix_anchors[:it], pix_deltas[:it])):
			print 'it2',it2+1,np.linalg.norm(delta_shapes)*dsc
			for image_index,target_shape,current_shape,delta_shape in zip(image_indices, target_shapes,current_shapes,delta_shapes):
				ISCV.sample_pixels(current_shape, ref_pinv, rpas2, rpds2, images[image_index], pixels[0], True)
				ISCV.traverse_forest(splits, pixels[0], leaf_indices)
				tmp = ISCV.sum_indices(leaves.reshape(leaves.shape[0],-1),leaf_indices).reshape(leaves.shape[1:])
				current_shape += np.dot(tmp,np.dot(ref_pinv, current_shape))
				np.dot(target_shape-current_shape, normalizing_A(current_shape, ref_pinv), out=delta_shape) # store the delta_shape
		print 'it %d/%d' % (it+1,num_forests),np.linalg.norm(delta_shapes)*dsc
		for image_index,current_shape,pixs in zip(image_indices, current_shapes, pixels):
			ISCV.sample_pixels(current_shape, ref_pinv, rpas, rpds, images[image_index], pixs, True)
		for it2,(splits,leaves) in enumerate(zip(tree_splits, test_leaves)):
			heap = [(np.arange(len(delta_shapes), dtype=np.int32),np.sum(delta_shapes, axis=0))]
			for idx1,idx2,thresh in splits:
				sample_indices,sum_deltas = heap.pop(0)
				sample_pixels = pixels[sample_indices]
				vals = np.diff(sample_pixels[:,[idx2,idx1]].astype(np.int32)).reshape(-1)
				L_indices,R_indices = sample_indices[vals > thresh],sample_indices[vals <= thresh]
				R_delta = ISCV.sum_indices(ds, R_indices).reshape(dss)
				L_delta = sum_deltas-R_delta
				heap.extend([(L_indices,L_delta),(R_indices,R_delta)])
			for (sample_indices,sum_deltas),leaf in zip(heap, leaves):
				leaf[:] = sum_deltas*np.float32(learn_rate/(len(sample_indices)+1e-8))
				delta_shapes[sample_indices] -= leaf # actually modify the deltas
			if (trees_per_forest-1-it2)%10 == 0: print 'x',(it2+1),np.linalg.norm(delta_shapes)*dsc
		ret = {'ref_pinv':ref_pinv,'ref_shape':ref_shape,'splits':forest_splits[:it+1],'leaves':forest_leaves[:it+1],'pix_anchors':pix_anchors[:it+1],'pix_deltas':pix_deltas[:it+1]}
		IO.save('tmp%d.io'%(it+1),ret)
	return ret

def double_data(images,shapes,flip_order=None,double_images=True):
	if flip_order is None:
		flip_order = [16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0, 26,25,24,23,22,21,20,19,18,17, 27,28,29,30, 35,34,33,32,31, \
					  45,44,43,42, 47,46, 39,38,37,36, 41,40, 54,53,52,51,50,49,48, 59,58,57,56,55, 64,63,62,61,60, 67,66,65]
	if len(flip_order) != len(shapes[0]):
		print 'WARNING: flip_order[%d] != shapes[%d]' % (len(flip_order),len(shapes[0]))
		return
	images2,shapes2 = [],[]
	for i,d in zip(images,shapes):
		d2 = np.float32((d[flip_order])*[-1,1]+[i.shape[1],0])
		if double_images: images2.append(i[:,::-1].copy())
		shapes2.append(d2)
		#show_image(i2,d2)
	images.extend(images2)
	shapes.extend(shapes2)
	print 'doubled',len(shapes)

def load_predictor(fn, cutOff=None):
	print 'loading predictor',fn
	pred = IO.load(fn)[1]
	# repair old, bad files
	if isinstance(pred, tuple):
		if len(pred) == 6:
			ref_pinv,ref_shape,splits,leaves,pix_anchors,pix_deltas = pred
			ref_shape[:,1] *= -1
			ref_pinv[1,:] *= -1
			assert np.allclose(np.linalg.pinv(ref_shape),ref_pinv)
			leaves *= [1,-1]
			pix_deltas *= [1,-1]
		else:
			yup,ref_pinv,ref_shape,splits,leaves,pix_anchors,pix_deltas = pred
			assert(yup)
	else:
		ref_pinv,ref_shape,splits,leaves,pix_anchors,pix_deltas = pred['ref_pinv'],pred['ref_shape'],pred['splits'],pred['leaves'],pred['pix_anchors'],pred['pix_deltas']
	ref_shape -= np.mean(ref_shape, axis=0) # ref_shape must be mean-centred now
	print splits.shape
	size = min(len(splits),len(pix_anchors))
	splits = splits[:size].copy()
	leaves = leaves[:size].copy()
	pix_anchors = pix_anchors[:size].copy()
	pix_deltas = pix_deltas[:size].copy()
	if cutOff is not None:
		splits,leaves,pix_anchors,pix_deltas = map(lambda x:x[:cutOff],(splits,leaves,pix_anchors,pix_deltas))
	pred = {'ref_pinv':ref_pinv,'ref_shape':ref_shape,'splits':splits,'leaves':leaves,'pix_anchors':pix_anchors,'pix_deltas':pix_deltas}
	return pred

	
# AAM stuff

def lunterp2D(M,s):
	'''Given a triangle of uv coordinates, compute the barycenrtic coordinates of the uv coordinate.'''
	[a,b],[c,d],[e,f] = M[1]-M[0],M[2]-M[0],s-M[0]
	det = a*d-b*c
	if not det: return (-10,-10,-10) # triangle is a straight line.. (-inf,inf,inf) maybe
	w1 = (e*d - f*c)/det
	w2 = (f*a - e*b)/det
	return np.float32([1-w1-w2,w1,w2])
	
def make_sample_model(ref_mesh, ref_tris, grid_size):
	# given a mesh and triangles, generate a set of pixel samples from that mesh
	# returns model_indices Nx3 and model_weights Nx3
	mn,mx = np.min(ref_mesh,axis=0),np.max(ref_mesh,axis=0)
	model_indices, model_weights = [],[]
	T = {} # map from vertex to list of triangles
	N = {} # map from vertex to list of neighbouring vertices
	for tri in ref_tris:
		for t in tri:
			T.setdefault(t,[]).append(tri)
			N.setdefault(t,set()).update(tri)
	for y in np.arange(mn[1],mx[1],grid_size):
		for x in np.arange(mn[0],mx[0],grid_size):
			# HACK: triangle must involve a vertex connected to the nearest vertex
			vi = np.argmin(np.sum((ref_mesh - [x,y])**2,axis=1))
			for vi in N[vi]:
				for tri in T[vi]:
					wts = lunterp2D(ref_mesh[tri], [x,y])
					if np.all(wts >= 0) and np.all(wts <= 1):
						model_indices.append(tri)
						model_weights.append(wts)
						break
	return np.int32(model_indices),np.float32(model_weights)

def compute_texture_coordinates(vertices, height, width, model_indices, model_weights):
	vs = vertices*np.float32([1,-1]) + np.float32([0,height])
	p = np.int32(np.sum(vs[model_indices]*model_weights.reshape(-1,3,1), axis=1))
	np.clip(p, np.int32([0,0]), np.int32([width-1,height-1]), out=p) # clip to image
	return p

def extract_texture(img, vertices, model_indices, model_weights):
	# extract a vector from an image
	p = compute_texture_coordinates(vertices, img.shape[0], img.shape[1], model_indices, model_weights)
	return img[p[:,1],p[:,0]] # and sample

def render_texture(pixels, img, vertices, model_indices, model_weights):
	# render a vector into an image
	p = compute_texture_coordinates(vertices, img.shape[0], img.shape[1], model_indices, model_weights)
	img[p[:,1],p[:,0]] = pixels

def powerfactorise(M, rank, debug=True):
	'''
	Calculate the svd of M, first 'rank' eigenvectors.
	We calculate the principal eigenvector by the power method, which is finding the stable point of 
	this iteration equation (starting at a random value):
	v = norm(M^T M v)
	Then Mv = s u; M^T u = s v and M^T M v = s^2 v describes the principal eigenvector (s is the eigenvalue, u is left e'vec, v is right).
	norm(M_ij - s u_i v_j) is minimised. 
	This method modifies M to calculate subsequent eigevectors.
	'''
	numFrames, numVerts = M.shape
	UT = np.zeros((rank,numFrames), dtype=np.float32)
	VT = np.zeros((rank,numVerts), dtype=np.float32)
	VT[:] = np.random.random(VT.shape)
	S = np.zeros(rank, dtype=np.float32)
	lu = lv = 1e-6
	bads = []
	if 0:
		_,(U,S,VT) = IO.load('USV_raw'); UT = U.T
	else:
		for ei in xrange(rank):
			u,v = UT[ei],VT[ei]
			#v -= np.dot(np.dot(VT[:ei],v),VT[:ei])
			v /= (np.linalg.norm(v)+1e-8)
			l = 0.0
			for it in range(20):
				stride = max(0,8-it)*3+1
				M1,v1,u1 = M[:,::stride],v[::stride],u
				u1[:] = np.dot(M1  , v1) / (np.sum(v1**2)+lu) # powerfactorisation
				su = np.linalg.norm(u1)+1e-8; u1 *= 1.0/su # normalise
				v1[:] = np.dot(M1.T, u1) / (np.sum(u1**2)+lv) # powerfactorisation
				sv = np.linalg.norm(v1)+1e-8; v1 *= 1.0/sv # normalise
				prev_l,l = l,(su*sv)**0.5 # eigenvalue
				if l == prev_l: break # convergence
			if debug: print it, np.linalg.norm(u)*np.linalg.norm(v), l, prev_l
			S[ei] = l
			if debug: ui_max,vj_max = np.argmax(u*u), np.argmax(v*v); print ei,l, u[ui_max],ui_max,v[vj_max],vj_max
			for Mi,ui in zip(M,u): Mi -= (ui*l)*v # Modify M
			if debug: print 'err',np.linalg.norm(M)*(np.prod(M.shape)**-0.5)
		if debug: IO.save('USV_raw',(UT.T,S,VT))
	# turn the result into something we'll recognise TODO faster method
	U,S,VT2 = np.linalg.svd((UT.T)*S, full_matrices=False)
	VT = np.dot((VT2.T * S).T, VT)
	U2,S,VT = np.linalg.svd(VT, full_matrices=False)
	U = np.dot(U,U2)
	return U,S,VT

def load_aam(fn = None):
	grip_dir = os.environ['GRIP_DATA']
	if fn is None: fn = os.path.join(grip_dir,'aam.io')
	print 'loading',fn
	ret = IO.load(fn)[1]
	if isinstance(ret,tuple): ret = convert_aam(ret)
	return ret

def train_aam_model(images, shapes, flip_order=None, texture_rank=20):
	double_data(images,shapes,flip_order)
	assert len(images)==len(shapes)
	print 'training aam from %d images' % len(images)
	s0, ref_shape, ref_pinv = normalized_shapes(shapes)
	s0 -= ref_shape
	ref_fs = triangulate_2D(ref_shape)
	model_indices,model_weights = make_sample_model(ref_shape, ref_fs, grid_size=0.01)
	print 'made model'
	#shapes
	shapes_u,shapes_s,shapes_vt = np.linalg.svd(s0.reshape(s0.shape[0],-1), full_matrices=0)
	rank = np.sum(shapes_s > shapes_s[0]*0.001)
	print 'shape rank',rank
	shapes_u,shapes_s,shapes_vt = shapes_u[:,:rank],shapes_s[:rank],shapes_vt[:rank,:]
	# textures
	textures = []
	for it,(img,obj) in enumerate(zip(images,shapes)):
		if ((it+1)%100) == (len(images)%100): print it+1,'/',len(images)
		tex = extract_texture(img, obj, model_indices, model_weights)
		textures.append(tex)
		if it > 0: assert tex.shape == textures[0].shape, repr(it)+','+repr(tex.shape)+','+repr(textures[0].shape)
	textures = np.uint8(textures)
	print 'extracted textures'
	texture_mean = np.mean(textures, axis=0, dtype=np.float32)
	t0 = np.float32(textures) - texture_mean
	texture_u,texture_s,texture_vt = powerfactorise(t0.reshape(t0.shape[0],-1), texture_rank)
	ret = {'shapes':shapes,'ref_shape':ref_shape,'ref_pinv':ref_pinv,\
		'shapes_u':shapes_u,'shapes_s':shapes_s,'shapes_vt':shapes_vt,\
		'textures':textures,'texture_mean':texture_mean,'model_indices':model_indices,'model_weights':model_weights,\
		'texture_u':texture_u,'texture_s':texture_s,'texture_vt':texture_vt}
	return ret

if __name__ == '__main__':
	# TODO: test code
	pass
