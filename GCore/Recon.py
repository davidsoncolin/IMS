#!/usr/bin/env python

"""
GCore/Recon.py

Requires:
	sys
	numpy
	
	GRIP
		ISCV(compute_E, HashCloud2DList)
"""
import sys
import numpy as np
import ISCV

def linsolveN3(E, dis, out):
	"""
	Solve E0 x + e0 = 0 via (E0^T E0)x = -E0^T e0. E is Nx2x3
	
	Args:
		E (float[][2][3]):
		dis (?):
		out (float[3]): 
		
	Returns:
		float[3]: 
	"""
	#Equivalent code:
	# E0, e0 = E[dis,:,:3].reshape(-1,3),E[dis,:,3].reshape(-1)
	# out[:] = np.linalg.solve(np.dot(E0.T,E0)+np.eye(3)*1e-8,-np.dot(E0.T,e0))
	E0, e0 = E[dis,:,:3].reshape(-1,3),E[dis,:,3].reshape(-1)
	# Now form this equation and solve:
	#|a b c|[x y z] = |g|
	#|b d e|          |h|
	#|c e f|          |i|
	a,b,c = np.dot(E0[:,0],E0)
	d,e = np.dot(E0[:,1],E0[:,1:])
	f = np.dot(E0[:,2],E0[:,2])
	a += 1e-8;d += 1e-8;f += 1e-8 # force strictly positive det
	g,h,i = -np.dot(E0.T,e0)
	be_cd = b*e-c*d # 3
	bc_ae = b*c-a*e # 3
	ce_bf = c*e-b*f # 2
	ad_bb = a*d-b*b # 2
	af_cc = a*f-c*c # 1
	df_ee = d*f-e*e # 1
	sc = 1.0/(c*be_cd + e*bc_ae + f*ad_bb)
	out[0] = (g*df_ee + h*ce_bf + i*be_cd)*sc
	out[1] = (g*ce_bf + h*af_cc + i*bc_ae)*sc
	out[2] = (g*be_cd + h*bc_ae + i*ad_bb)*sc
	return out

def dets_to_rays(x2ds, splits, mats):
	"""
	Convert 2D detections into normalized Rays
	
	Args:
		x2ds (float[][2]): 2D Detections.
		splits (int[]): Indices of ranges of 2Ds per camera.
		mats (GcameraMat[]): Matrices of the cameras

	Returns:
		float[][3]: "rays" - Unit vector from the PP of it's viewing camera??
	"""
	# (px,py,-1) = K R (a(x, y, z) - T)
	# (px - ox,py - oy,-f)^T R = a(x,y,z)-T
	rays = np.zeros((len(x2ds),3),dtype=np.float32)
	for c0,c1,m in zip(splits[:-1],splits[1:],mats):
		K,RT,T = m[0],m[1],m[4]
		crays = rays[c0:c1]
		np.dot(x2ds[c0:c1],RT[:2,:3],out=crays) # ray directions (unnormalized)
		crays -= np.dot([-K[0,2],-K[1,2],K[0,0]],RT[:3,:3])
		#print project((T + 1000*crays[0]).reshape(1,3),np.array([0],dtype=np.int32),[m[2]])[0], x2ds[c0]
	rays /= (np.sum(rays*rays,axis=1)**0.5).reshape(-1,1) # normalized ray directions
	return rays
	
def solve_x3ds(x2ds, splits, labels, Ps, robust=True):
	"""
	Given some labelled 2d points, generate labelled 3d positions for every multiply-labelled point and equations for
	every single-labelled point.
	
	Args:
		x2ds (float[][2]): 2D Detections.
		splits (int[]): Indices of ranges of 2Ds per camera.
		labels (int[]: Labels of x2ds.
		Ps (?): Projection matrices of the cameras?
		robust (bool): Robustness Flag (requires more Rays to reconstruct). Default = True

	Returns:
		float[][3]: "x3ds" - the resulting 3D reconstructions.
		int[]: "x3d_labels" - the labels for the 3D points.
		??: "E[singles]" - Equations describing 2D detections not born of the 3D yet.
		int[] "singles_labels" - labels for the 2D contributions.
		
	Requires:
		ISCV.solve_x3ds
	"""
	return ISCV.solve_x3ds(x2ds, splits, labels, Ps, robust) # x3ds, x3d_labels, E[single_rays], single_ray_labels

def solve_x3ds_normals(x2ds, splits, labels, Ps, rays, robust=True):
	x3ds, x3ds_labels, E, x2ds_labels = ISCV.solve_x3ds(x2ds, splits, labels, Ps, robust)

	x3ds_normals = np.ones_like(x3ds)
	for xi, label in enumerate(x3ds_labels):
		rayIndices = np.where(labels == label)[0]
		x3ds_normals[xi] = np.sum(rays[rayIndices], axis=0)

	# Normalise the ray directions
	x3ds_normals /= (np.sum(x3ds_normals * x3ds_normals, axis=1) ** 0.5).reshape(-1, 1)

	return x3ds, x3ds_labels, x3ds_normals, E, x2ds_labels

def test_solve_x3ds(x2ds, splits, labels, Ps):
	"""
	Given some labelled 2d points, generate labelled 3d positions for every multiply-labelled point and equations for
	every single-labelled point.
	
	Is this the Python code before translation into C?
	
	Args:
		x2ds (float[][2]): 2D Detections.
		splits (int[]): Indices of ranges of 2Ds per camera.
		labels (int[]: Labels of x2ds.
		Ps (?): Projection matrices of the cameras?

	Returns:
		float[][3]: "x3ds" - the resulting 3D reconstructions.
		int[]: "x3d_labels" - the labels for the 3D points.
		??: "E[singles]" - Equations describing 2D detections not born of the 3D yet.
		int[] "singles_labels" - labels for the 2D contributions.
		
	Requires:
		ISCV.compute_E
		linsolveN3
	"""
	x3ds2, x3d_labels2, E2, x2d_labels2 = solve_x3ds(x2ds, splits, labels, Ps)

	numLabels = max(labels)+1
	counts = np.zeros(numLabels,dtype=np.int32)
	# take care to throw away unlabelled rays
	counts[:] = np.bincount(labels+1, minlength = numLabels+1)[1:numLabels+1]
	# find all the 2+ ray labels
	x3d_labels = np.array(np.where(counts >= 2)[0],dtype=np.int32)
	# find all the single ray labels
	x2d_labels = np.array(np.where(counts==1)[0],dtype=np.int32)
	E = ISCV.compute_E(x2ds, splits, Ps)
	label_dis    = -np.ones(len(x2ds),dtype=np.int32) # the indices of the detection for each label
	label_splits = np.zeros(numLabels+1,dtype=np.int32)
	np.cumsum(counts,out=label_splits[1:],dtype=np.int32)
	index = label_splits.copy()
	for c0,c1 in zip(splits[:-1],splits[1:]):
		ls = labels[c0:c1]
		label_dis[index[ls]] = range(c0,c1)
		index[ls] += 1
	# compute the 3d points
	x3ds = np.zeros((len(x3d_labels),3),dtype=np.float32)
	for li,x in zip(x3d_labels, x3ds):
		dis=label_dis[label_splits[li]:label_splits[li+1]]
		linsolveN3(E,dis,x)
		err = np.dot(E[dis,:,:3].reshape(-1,3),x) + E[dis,:,3].reshape(-1)
		err = err.reshape(-1,2)
		err = np.sum(err**2,axis=1)
		err = 50.0/(50.0+err)
		#print err
		E[dis,:] *= err.reshape(-1,1,1)
		linsolveN3(E,dis,x)

	#print 'diff',np.sum((x3ds-x3ds2)**2,axis=1)
	assert(np.allclose(x3ds,x3ds2,1e-6,1e-3))
	assert(np.allclose(x3d_labels,x3d_labels2))
	#assert(np.allclose(E[label_splits[x2d_labels]],E2,1e-6,1e-6))
	assert(np.allclose(x2d_labels,x2d_labels2))
	return x3ds, x3d_labels2, E2, x2d_labels2
	
def intersect_rays(x2ds, splits, Ps, mats, seed_x3ds=None, tilt_threshold=0.0002, x2d_threshold=0.01, x3d_threshold=30.0, min_rays=3,
                   numPolishIts=3, forceRayAgreement=False, visibility=None):
	"""
	Given 2D detections, we would like to find bundles of rays from different cameras that have a common solution.
	For each pair of rays, we can solve for a 3D point. Each such solve has a residual: we want to find low residual pairs.

	Closer together camera pairs and cameras with more unlabelled markers should have more matches.
	Visit the camera pairs by order of distance-per-unlabelled-marker score (lower is better).

	For a given camera pair, each ray can be given an order which is the tilt (angle between the ray from the camera to
	that ray and a line perpendicular to a reference plain containing both camera centres).

	tilt = asin(norm(raydir^(c2-c1)).ocdir))
	TODO: compare atan2(raydir^(c2-c1).ocdir,|raydir^(c2-c1)^ocdir|)

	Precisely the rays with the same tilt (within tolerance) intersect.
	This fails only if the first camera is looking directly at the second.

	For each pair of cameras, sort the unassigned rays by tilt and read off the matches.
	(DON'T match if there are two candidates with the same tilt on the same camera.)
	For each match, solve the 3D point.
	Naively, this costs ~NumDetections^2.
	However, if we project the point in all the cameras and assign rays then we can soak up all the rays in the other cameras.
	The maximum number of matches should be ~NumPoints.
	So the dominant cost becomes project assign (NumPoints * NumCameras using hash).

	Polish all the 3D points.
	Check for any 3D merges (DON'T merge if there are two rays from the same camera).
	Project all the points in all the cameras and reassign.
	Cull any points with fewer than 2 rays.
	Potentially repeat for the remaining unassigned rays.

	Args:
		x2ds (float[][2]): 2D Detections.
		splits (int): Indices of ranges of 2Ds per camera.
		Ps (?): Projection matrices of the cameras?
		mats (GcameraMat[]): Camera Matrices.
		seed_x3ds (float[][3]): existing 3D data? Default = None.
		tilt_threshold (float): Slack factor for tilt pairing = 0.0002
		x2d_threshold (float): What's this? Default = 0.01
		x3d_threshold (float): What's this? = 30.0
		min_rays (int): Min number of rays to reconstruct. Default = 3.

	Returns:
		float[][3]: (x3ds_ret) List of 3D points produced as a result of intersecting the 2Ds
		int[]: (labels) List of labels corresponding to the x3ds.

	Requires:
		ISCV.compute_E
		ISCV.HashCloud2DList
		ISCV.HashCloud3D
		clouds.project_assign

	"""
	Ks = np.array(zip(*mats)[0],dtype=np.float32)
	RTs = np.array(zip(*mats)[1],dtype=np.float32)
	Ts = np.array(zip(*mats)[4],dtype=np.float32)
	if visibility is not None:
		ret2 = ISCV.intersect_rays_base(x2ds, splits, Ps, Ks, RTs, Ts, seed_x3ds, tilt_threshold, x2d_threshold, x3d_threshold,
		                                min_rays, numPolishIts, forceRayAgreement, visibility)
	else:
		ret2 = ISCV.intersect_rays2(x2ds, splits, Ps, Ks, RTs, Ts, seed_x3ds, tilt_threshold, x2d_threshold, x3d_threshold,
		                            min_rays, numPolishIts, forceRayAgreement)
	return ret2

	import itertools
	numCameras = len(splits)-1
	numDets = splits[-1]
	labels = -np.ones(numDets,dtype=np.int32)
	E = ISCV.compute_E(x2ds, splits, Ps)
	rays = dets_to_rays(x2ds, splits, mats)
	Ts = np.array([m[4] for m in mats],dtype=np.float32)

	def norm(a): return a / (np.sum(a**2)**0.5)

	tilt_axes = np.array([norm(np.dot([-m[0][0,2],-m[0][1,2],m[0][0,0]],m[1][:3,:3])) for m in mats],dtype=np.float32)
	corder = np.array(list(itertools.combinations(range(numCameras),2)),dtype=np.int32) # all combinations ci < cj
	#corder = np.array(np.concatenate([zip(range(ci),[ci]*ci) for ci in xrange(1,numCameras)]),dtype=np.int32)
	clouds = ISCV.HashCloud2DList(x2ds, splits, x2d_threshold)
	x3ds_ret = []
	if seed_x3ds is not None:
		x3ds_ret = list(seed_x3ds)
		# initialise labels from seed_x3ds
		_,labels,_ = clouds.project_assign_visibility(seed_x3ds, np.arange(len(x3ds_ret),dtype=np.int32), Ps, x2d_threshold, visibility)
	# visit the camera pairs by distance-per-unlabelledmarker
	#camDists = np.array([np.sum((Ts - Ti)**2, axis=1) for Ti in Ts],dtype=np.float32)
	#for oit in range(10):
		#if len(corder) == 0: break
		#urcs = np.array([1.0/(np.sum(labels[splits[ci]:splits[ci+1]]==-1)+1e-10) for ci in xrange(numCameras)],dtype=np.float32)
		#scmat = camDists*np.array([np.maximum(urcs,uci) for uci in urcs],dtype=np.float32)
		#scores = scmat[corder[:,0],corder[:,1]]
		#so = np.argsort(scores)
		#corder = corder[so]
		#for it in range(10):
			#if len(corder) == 0: break
			#ci,cj = corder[0]
			#corder = corder[1:]
	for ci in xrange(numCameras):
		for cj in xrange(ci+1,numCameras):
			ui,uj = np.where(labels[splits[ci]:splits[ci+1]]==-1)[0],np.where(labels[splits[cj]:splits[cj+1]]==-1)[0]
			if len(ui) == 0 or len(uj) == 0: continue
			ui += splits[ci]; uj += splits[cj]
			axis = Ts[cj] - Ts[ci]
			tilt_i = np.dot(map(norm,np.cross(rays[ui], axis)), tilt_axes[ci])
			tilt_j = np.dot(map(norm,np.cross(rays[uj], axis)), tilt_axes[ci]) # NB tilt_axes[ci] not a bug
			io = np.argsort(tilt_i)
			jo = np.argsort(tilt_j)
			ii,ji = 0,0
			data = []
			while ii < len(io) and ji < len(jo):
				d0,d1 = tilt_i[io[ii]], tilt_j[jo[ji]]
				diff = d0 - d1
				if abs(diff) < tilt_threshold:
					# test for colliding pairs
					# if ii+1 < len(io) and tilt_i[io[ii+1]]-d0 < tilt_threshold: ii+=2; continue
					# if ji+1 < len(jo) and tilt_j[jo[ji+1]]-d1 < tilt_threshold: ji+=2; continue
					# test for colliding triples
					# if ii > 0 and d0-tilt_i[io[ii-1]] < tilt_threshold: ii+=1; continue
					# if ji > 0 and d1-tilt_j[jo[ji-1]] < tilt_threshold: ji+=1; continue
					d = [ui[io[ii]],uj[jo[ji]]]
					data.append(d)
					ii += 1
					ji += 1
				elif diff < 0: ii += 1
				else:          ji += 1
			if len(data) != 0:
				# intersect rays
				for d in data:
					E0, e0 = E[d,:,:3].reshape(-1,3),E[d,:,3].reshape(-1)
					x3d = np.linalg.solve(np.dot(E0.T,E0)+np.eye(3)*1e-7,-np.dot(E0.T,e0))
					sc,labels_out,_ = clouds.project_assign_visibility(np.array([x3d],dtype=np.float32), np.array([0],dtype=np.int32), Ps, x2d_threshold, visibility)
					tmp = np.where(labels_out==0)[0]
					if len(tmp) >= min_rays:
						tls_empty = np.where(labels[tmp] == -1)[0]
						if len(tls_empty) >= min_rays:
							labels[tmp[tls_empty]] = len(x3ds_ret)
							x3ds_ret.append(x3d)
	# TODO: polish, merge, reassign, cull, repeat
	# merge
	if False:
		x3ds_ret = np.array(x3ds_ret,dtype=np.float32).reshape(-1,3)
		cloud = ISCV.HashCloud3D(x3ds_ret, x3d_threshold)
		scores,matches,matches_splits = cloud.score(x3ds_ret)
		mergers = np.where(matches_splits[1:] - matches_splits[:-1] > 1)[0]
		for li in mergers:
			i0,i1=matches_splits[li:li+2]
			collisions = np.where(scores[i0:i1] < x3d_threshold**2)[0]
			if len(collisions) > 1:
				collisions += i0
				#print 'merger',li,i0,i1,scores[i0:i1] # TODO merge these (frame 7854)

	# now cull the seed_x3ds, because they could confuse matters
	if seed_x3ds is not None:
		labels[np.where(labels < len(seed_x3ds))] = -1

	minNumRays1 = np.min([len(np.where(labels == l)[0]) for l in np.unique(labels)])
	maxNumRays1 = np.max([len(np.where(labels == l)[0]) for l in np.unique(labels) if l != -1])

	# final polish
	x3ds_ret, x3ds_labels, E_x2ds_single, x2ds_single_labels = solve_x3ds(x2ds, splits, labels, Ps)
	# throw away the single rays and their 3d points by renumbering the generated 3d points
	# _,labels,_ = clouds.project_assign_visibility(x3ds_ret, None, Ps, x2d_threshold, visibility)
	minNumRays3 = np.min([len(np.where(labels == l)[0]) for l in np.unique(labels)])
	maxNumRays3 = np.max([len(np.where(labels == l)[0]) for l in np.unique(labels) if l != -1])
	_,labels,_ = clouds.project_assign(x3ds_ret, None, Ps, x2d_threshold)
	minNumRays2 = np.min([len(np.where(labels == l)[0]) for l in np.unique(labels)])
	maxNumRays2 = np.max([len(np.where(labels == l)[0]) for l in np.unique(labels) if l != -1])
	x3ds_ret, x3ds_labels, E_x2ds_single, x2ds_single_labels = solve_x3ds(x2ds, splits, labels, Ps)
	ret = x3ds_ret,labels
	return ret

	# TODO the results seem to be the same, but the labels are not... but why?
	# print (len(ret[1]),len(ret2[1]), ret[1][:10],ret2[1][:10])
	# print (np.where(ret[1]==ret[1][0])[0])
	# print (np.where(ret2[1]==ret2[1][0])[0])
	# #assert np.allclose(ret[0],ret2[0])
	# #assert np.all(ret[1] == ret2[1])
	# return ret2
