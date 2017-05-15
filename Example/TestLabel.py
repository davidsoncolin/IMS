#!/usr/bin/env python

import sys, os
import numpy as np
from GCore import Label, Retarget, SolveIK, Calibrate, Recon, Character
from IO import IO, ViconReader, ASFReader, C3D
import ISCV

def test_label_2d_from_2d_with_vel(prev_x2ds, prev_vels, prev_splits, prev_labels, x2ds, splits, x2d_threshold):
	'''Track 2d points from one frame to the next in all the cameras.'''
	labels = np.zeros(len(x2ds),dtype=np.int32)
	vels = np.zeros_like(x2ds)
	assert(len(prev_splits) == len(splits))
	pred_x2ds = prev_x2ds + prev_vels
	prev_has_predictions = (prev_vels[:,0]!=0)
	sc = 0
	for s0,s1,p0,p1 in zip(splits[:-1],splits[1:],prev_splits[:-1],prev_splits[1:]):
		label = labels[s0:s1]
		x2d = x2ds[s0:s1]
		vel = vels[s0:s1]
		prev_label = prev_labels[p0:p1]
		prev_x2d = prev_x2ds[p0:p1]
		pred_x2d = pred_x2ds[p0:p1]
		sc += Label.match(pred_x2d, x2d, x2d_threshold, prev_has_predictions[p0:p1], label)
		labelled = np.array(np.where(label!=-1)[0],dtype=np.int32)
		if len(labelled) != 0:
			prev_labelled = label[labelled] # a copy, not a reference to a subset of label
			vel[labelled] = x2d[labelled]-prev_x2d[prev_labelled]
			label[labelled] = prev_label[prev_labelled]
	clouds = ISCV.HashCloud2DList(x2ds, splits, x2d_threshold)
	sc2,labels2,vels2 = clouds.assign_with_vel(prev_x2ds, prev_vels, prev_splits, prev_labels, x2d_threshold)
	assert(np.all(labels == labels2))

def test_label_2d_from_2d(prev_x2ds, prev_splits, prev_labels, x2ds, splits, x2d_threshold):
	'''Track 2d points from one frame to the next in all the cameras.'''
	labels = np.zeros(len(x2ds),dtype=np.int32)
	vels = np.zeros_like(x2ds)
	assert(len(prev_splits) == len(splits))
	sc = 0
	for s0,s1,p0,p1 in zip(splits[:-1],splits[1:],prev_splits[:-1],prev_splits[1:]):
		label = labels[s0:s1]
		x2d = x2ds[s0:s1]
		vel = vels[s0:s1]
		prev_label = prev_labels[p0:p1]
		prev_x2d = prev_x2ds[p0:p1]
		sc += Label.match(prev_x2d, x2d, x2d_threshold, None, label)
		labelled = np.array(np.where(label!=-1)[0],dtype=np.int32)
		if len(labelled) != 0:
			prev_labelled = label[labelled] # a copy, not a reference to a subset of label
			label[labelled] = prev_label[prev_labelled]
			vel[labelled] = x2d[labelled]-prev_x2d[prev_labelled]

	clouds = ISCV.HashCloud2DList(x2ds, splits, x2d_threshold)
	sc2,labels2,vels2 = clouds.assign(prev_x2ds, prev_splits, prev_labels, x2d_threshold)
	assert(np.all(labels == labels2))

def test_project_assign(x2ds, splits, x3ds, x3ds_labels, Ps, x2d_threshold):
	clouds = ISCV.HashCloud2DList(x2ds,splits,x2d_threshold)
	clouds2 = [ISCV.HashCloud2D(x2ds[c0:c1], x2d_threshold) for c0,c1 in zip(splits[:-1],splits[1:])]
	proj_x2ds, proj_splits, proj_labels = ISCV.project(x3ds, x3ds_labels, Ps)
	labels_out = -np.ones(len(x2ds),dtype=np.int32)
	vels = np.zeros((len(x2ds),2),dtype=np.float32)
	labels_tmp = np.zeros(len(proj_x2ds),dtype=np.int32)
	sc = 0
	for ci,cloud in enumerate(clouds2):
		x2d		= x2ds[splits[ci]:splits[ci+1]]
		label	= labels_out[splits[ci]:splits[ci+1]]
		vel		= vels[splits[ci]:splits[ci+1]]
		px2d	= proj_x2ds[proj_splits[ci]:proj_splits[ci+1]]
		lc		= labels_tmp[proj_splits[ci]:proj_splits[ci+1]]
		pl		= proj_labels[proj_splits[ci]:proj_splits[ci+1]]
		scores,matches,matches_splits = cloud.score(px2d)
		sc += ISCV.min_assignment_sparse(scores, matches, matches_splits, x2d_threshold**2, lc)
		labelled = np.where(lc!=-1)[0]
		if len(labelled) != 0:
			which_x2ds = lc[labelled]
			label[which_x2ds] = pl[labelled]
			vel[which_x2ds] = x2d[which_x2ds]-px2d[labelled]

	sc2,labels_out2,vels2 = Label.project_assign(clouds, x3ds, x3ds_labels, Ps, x2d_threshold)
	assert(np.allclose(sc, sc2))
	assert(np.all(labels_out == labels_out2))
	assert(np.allclose(vels, vels2))

def test_project(x3ds, labels_x3d, Ps):
	'''Project all the 3d points in all the cameras. Clip to ensure that the point is in front of the camera and in the frame.'''
	numPoints = len(x3ds)
	numCameras = len(Ps)
	proj = np.zeros((numCameras,numPoints,3),dtype=np.float32)
	for ci,P in enumerate(Ps):
		np.dot(x3ds,P[:3,:3].T,out=proj[ci])
		proj[ci] += P[:,3]
	proj[:,:,:2] /= -proj[:,:,2].reshape(numCameras,numPoints,1)
	ind = (proj[:,:,0]<1.01) * (proj[:,:,0]>-1.01) * (proj[:,:,1]<1.01) * (proj[:,:,1]>-1.01) * (proj[:,:,2] < 0)
	counts = np.sum(ind,axis=1) # num kept per camera
	splits = np.zeros(numCameras+1,dtype=np.int32)
	np.cumsum(counts,out=splits[1:],dtype=np.int32)
	labels = labels_x3d[np.where(ind)[1]]
	if ind.shape[1] == 0: return np.zeros((0,2),dtype=np.float32),splits,labels
	x2ds = np.array(proj[ind,:2],dtype=np.float32)
	x2ds2, splits2, labels2 = ISCV.project(x3ds,labels_x3d, Ps)
	assert(np.allclose(x2ds2, x2ds))
	assert(np.all(labels == labels2))
	assert(np.all(splits == splits2))

def test_compute_E(x2ds, splits, Ps):
	'''Form this equation: E x = e by concatenating the constraints (two rows per ray) and solve for x (remember to divide by -z).
	[P00 + P20 px, P01 + P21 px, P02 + P22 px][x; y; z] = -	[ P03 + P23 px ]
	[P10 + P20 py, P11 + P21 py, P12 + P22 py]				[ P13 + P23 py ]
	If the projection matrices are divided through by the focal length then the errors should be 3D-like (ok to mix with 3d equations)
	Use the same equations to add single ray constraints to IK: derr(x)/dc = E dx/dc; residual = E x - e.'''
	E = np.zeros((len(x2ds),2,4),dtype=np.float32)
	# populate the ray equations for each camera
	for P,c0,c1 in zip(Ps,splits[:-1],splits[1:]):
		E[c0:c1] = P[:2] + P[2]*x2ds[c0:c1].reshape(-1,2,1)
	E2 = ISCV.compute_E(x2ds, splits, Ps)
	assert(np.allclose(E,E2))

def test_intersect_rays(x3ds_ret, labels, true_x3ds, true_labels):
	# score
	if true_labels is not None:
		label_to_true_label = np.zeros((max(labels)+1,max(max(true_labels),max(labels))+1),dtype=np.int32)
		if label_to_true_label.shape[0]*label_to_true_label.shape[1]:
			for li,ti in zip(labels,true_labels):
				label_to_true_label[li,ti] += 1
			import Assign
			label_to_true_label = Assign.maxAssignmentDense(label_to_true_label)[0]
			label_to_true_label = list(label_to_true_label)
			label_to_true_label.append(-1)
			label_to_true_label = np.array(label_to_true_label,dtype=np.int32)
			extras = np.where(label_to_true_label == -1)[0]
			label_to_true_label[extras] = np.array(range(len(extras)))+1000
			label_to_true_label[-1] = -1
			labels = label_to_true_label[labels]
			print 'labels',labels

			print max(labels)
			ver = [(li,ti) for li,ti in zip(labels,true_labels)]

			tmp = np.where([li != ti for li,ti in ver])[0]
			print tmp,np.array(ver)[tmp]
			print sum([li == ti for li,ti in ver])
			print len([li == ti for li,ti in ver])
			print len(true_labels)
			import QGLViewer; QGLViewer.makeViewer(x3ds_ret, altVertices = true_x3ds)

def solve_camera_from_3d(x3ds, x2ds, P, solve_distortion = False, solve_principal_point = False, solve_focal_length = True):
	'''Initialise a camera from 6+ corresponding points.'''
	assert(False) # Don't use this, it's just a toy for comparison with opencv
	# P (x y z 1)_i = a_i (px py -1)_i
	# P0 X_i + P03 + (P2 X_i + P23) px = 0
	# P1 X_i + P13 + (P2 X_i + P23) py = 0
	# This gives two equations in P per point.
	# x_i P00 + y_i P01 + z_i P02 + P03 + x_i px_i P20 + y_i px_i P21 + z_i px_i P22 + px_i P23 = 0
	# x_i P10 + y_i P11 + z_i P12 + P13 + x_i py_i P20 + y_i py_i P21 + z_i py_i P22 + py_i P23 = 0
	numPoints = len(x3ds)
	#assert(numPoints >= 6)
	M = np.zeros((numPoints*2,12),dtype=np.float32)
	for ci,((x,y,z),(px,py)) in enumerate(zip(x3ds, x2ds)):
		M[2*ci+0] = [x,y,z,1,0,0,0,0,x*px,y*px,z*px,px]
		M[2*ci+1] = [0,0,0,0,x,y,z,1,x*py,y*py,z*py,py]
	#M /= np.sum(M*M,axis = -1).reshape(-1,1)**0.5 # for numerics?
	u,s,vt = np.linalg.svd(M) # svd is [2N,S]*diag(S)*[S,12]
	P = vt[-1].reshape(3,4) # the evec with smallest eval gives the solution
	P /= np.sum(P[2,:3]**2)**0.5 # normalize P
	# P is only defined up to sign -- the sign could be wrong
	if np.linalg.det(P[:3,:3]) < 0.0: P = -P
	K,RT = Calibrate.decomposeKRT(P)
	# impose skew = 0 and fx=fy
	K[0,1] = 0.0
	K[0,0] = K[1,1] = (K[0,0]*K[1,1])**0.5
	k1,k2 = 0.0,0.0
	P = np.dot(K,RT)
	# now we want to solve np.dot(M,P.reshape(12)) = 0 by multiplying P on the right by an RT
	res = np.dot(M,P.reshape(12))
	print 'pre',(np.mean(res**2)*2)**0.5,
	for it in range(3):
		grad_RT = np.array([[0,0,0,0, 0,0,1,0, 0,-1,0,0], [0,0,-1,0, 0,0,0,0, 1,0,0,0], [0,1,0,0, -1,0,0,0, 0,0,0,0],\
				[0,0,0,P[0,0], 0,0,0,P[1,0], 0,0,0,P[2,0]], [0,0,0,P[0,1], 0,0,0,P[1,1], 0,0,0,P[2,1]],\
				[0,0,0,P[0,2], 0,0,0,P[1,2], 0,0,0,P[2,2]] ],dtype=np.float32) # 6x12
		dRT = -0.5 * (np.linalg.lstsq(np.dot(M,grad_RT.T), np.dot(M,P.reshape(12)))[0])
		(s1,s0,s2),(c1,c0,c2) = np.sin(dRT[:3]),np.cos(dRT[:3])
		cc,cs,sc,ss = c1*c2,c1*s2,s1*c2,s1*s2
		mat = np.array([[ss*s0+cc,s2*c0,cs*s0-sc,dRT[3]],[sc*s0-cs,c2*c0,cc*s0+ss,dRT[4]],[c0*s1,-s0,c0*c1,dRT[5]],[0,0,0,1]],dtype=np.float32)
		P = np.dot(P,mat)
		res = np.dot(M,P.reshape(12))
		rms = project_and_compute_rms(x3ds, x2ds, P, [0,0])
		print it,(np.mean(res**2)*2)**0.5,rms

	print solve_distortion
	if solve_distortion:
		ks = Calibrate.solve_camera_distortion_from_3d(x3ds, x2ds, P)
		#print k1,k2,ks
		if not np.allclose(ks,0,0.5,0.5): print 'WARNING bad ks',ks,[k1,k2] #; ks = np.zeros(2,dtype=np.float32)
	else: ks = np.zeros(2,dtype=np.float32)
	#print 'testTest',np.dot(M, P.reshape(12))
	# TODO initialise distortion params k1,k2
	# TODO nonlinearly optimise 11 parameter model: f,ox,oy,rx,ry,rz,tx,ty,tz,k1,k2
	rms = project_and_compute_rms(x3ds, x2ds, P, ks)
	return P,ks,rms

def solve_camera_distortion_from_3d_k1_only(x3s, x2s, P):
	'''find distortion parameters given assignments'''
	# (px,py) -> (1 + k1 r2) (px-ox,py-oy) + (ox,oy)
	# r2 = (px-ox)**2+(py-oy)**2
	# s = r + k1 r^3
	# Solve [r^3] [k1] = [s - r]
	# Solve [r^4] [k1] = [sr - rr]
	numPoints = len(x2s)
	K,RT = Calibrate.decomposeKRT(P)
	ox,oy = float(-K[0,2]), float(-K[1,2])
	sr = np.zeros(numPoints,dtype=np.float32)
	tmp = np.dot(x3s,P[:,:3].T) + P[:,3].reshape(1,3)
	ss = np.sum((tmp[:,:2]/-tmp[:,2].reshape(-1,1) - [ox,oy])**2,axis=-1)**0.5
	rs = np.sum((x2s - [ox,oy])**2,axis=-1)**0.5
	r35 = rs**4
	sr[:] = (ss-rs)*rs
	x = np.dot(r35.T,sr)/np.dot(r35.T,r35)
	return x

def score(true_labels, labels):
	sc = sum([ti == li for ti,li in zip(true_labels, labels)])
	return sc/float(len(true_labels))

def track_anim(labels_filename, skelDict, rootMat = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]],dtype=np.float32), pred_2d_threshold = 50./2000., x2d_threshold = 25./2000., x3d_threshold = 30., first_frame = None, last_frame = None):
	'''Track a skeleton through a sequence by propagating the labels from the first frame, solving IK, then picking up labels by projection.'''
	labels_frames,mats,effectorLabels,cids = loadLabels(labels_filename, skelDict['markerNames'])

	if first_frame is None: first_frame = 1
	if last_frame is None: last_frame = len(labels_frames[1])+1

	prev_x2ds, _prev_labels, prev_splits = Label.extract_label_frame(first_frame-1, labels_frames)
	prev_vels = np.zeros_like(prev_x2ds)
	Ps = np.array([m[2]/(np.sum(m[2][0,:3]**2)**0.5) for m in mats],dtype=np.float32)
	numCameras = len(prev_splits)-1

	effectorData = SolveIK.make_effectorData(skelDict)
	effectorTargets = np.zeros_like(effectorData[1])
	numTargets = len(effectorTargets)

	numDofs = len(skelDict['chanValues'])
	frameNumbers = np.array(range(first_frame, last_frame,[1,-1][first_frame>last_frame]),dtype=np.int32)
	numFrames = len(frameNumbers)
	dofData = np.zeros((numFrames,numDofs), dtype=np.float32)

	# initialise the labels from the pose
	clouds = ISCV.HashCloud2DList(prev_x2ds, prev_splits, x2d_threshold)
	sc,prev_labels,_ = Label.label_2d_from_skeleton_and_3ds(clouds, prev_x2ds, prev_splits, effectorLabels, skelDict, effectorData, rootMat, Ps, x2d_threshold)

	for fni,fi in enumerate(frameNumbers):
		u2ds,_lbls,splits = Label.extract_label_frame(fi-1, labels_frames)
		print '\rtracking %d' %(fi),fps(),; sys.stdout.flush()
		clouds = ISCV.HashCloud2DList(u2ds,splits,max(pred_2d_threshold,x2d_threshold))
		Ls_old = SolveIK.bake_ball_joints(skelDict)
		sc,labels,vels = clouds.assign_with_vel(prev_u2ds, prev_vels, prev_splits, prev_labels, pred_2d_threshold)
		# TODO, some of the labels here for the same 3d point don't intersect; should try to enforce consensus from predicted 3ds
		#x3ds, x3d_labels, E, x2d_labels = Recon.solve_x3ds(u2ds, splits, labels, Ps)
		for it in range(1):
			x3ds, x3d_labels, E, x2d_labels = SolveIK.solve_skeleton_from_2d(u2ds, splits, labels, effectorLabels, Ps, skelDict, effectorData, rootMat, outerIts=1)
			sc,labels,resids = Label.label_2d_from_skeleton_and_3ds(clouds, u2ds, splits, effectorLabels, skelDict, effectorData, rootMat, Ps, x2d_threshold, x3ds, x3d_threshold)
		ISCV.update_vels(u2ds,splits,labels,prev_u2ds,prev_splits,prev_labels,vels)
		prev_u2ds,prev_splits,prev_labels,prev_vels = u2ds,splits,labels,vels
		SolveIK.unbake_ball_joints(skelDict, Ls_old)
		dofData[fni,:] = skelDict['chanValues']
	return { 'dofData' : dofData, 'frameNumbers' : frameNumbers }

def loadLabels(labels_filename, markerNames, enforce_complete = False):
	print 'loading labels'
	labels_dict = IO.load(labels_filename)[1]
	print labels_dict.keys()
	labels_frames,mats,cids,label_names = labels_dict['frames'],labels_dict['mats'],labels_dict['cids'],labels_dict['label_names']
	print 'num frames', len(labels_frames[1])
	effectorLabels = np.array([label_names.index(ln) if ln in label_names else -1 for ln in markerNames],dtype=np.int32)
	print np.where(effectorLabels==-1)[0]
	if enforce_complete: assert(len(np.where(effectorLabels==-1)[0]) == 0) # no missing data!
	else: effectorLabels[np.where(effectorLabels==-1)] = len(label_names)
	return labels_frames,mats,effectorLabels,cids

def score_anim(labels_filename, skelDict, anim_dict, rootMat = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]],dtype=np.float32), x2d_threshold = 25./2000., x3d_threshold = 60., first_frame = None, last_frame = None):
	'''Score an animation by comparing the project-assignments with the true assignments.'''
	labels_frames,mats,effectorLabels,cids = loadLabels(labels_filename, skelDict['markerNames'])

	if first_frame is None: first_frame = max(1,min(anim_dict['frameNumbers']))
	if last_frame is None: last_frame = min(len(labels_frames[1])+1,max(anim_dict['frameNumbers']))
	Ps = np.array([m[2]/(np.sum(m[2][0,:3]**2)**0.5) for m in mats],dtype=np.float32)

	effectorData = SolveIK.make_effectorData(skelDict)
	effectorTargets = np.zeros_like(effectorData[1])
	#effectorWeights = effectorData[2]
	#skelDict['chanValues'][:] = animData[0]
	print effectorLabels

	frameNumbers = np.array(range(first_frame, last_frame,[1,-1][first_frame>last_frame]),dtype=np.int32)
	numFrames = len(frameNumbers)

	tmp = list(anim_dict['frameNumbers'])
	fns = [tmp.index(fi) for fi in frameNumbers]

	good_score,bad_score,total_score = 0,0,0
	cases = [0,0,0,0,0,0,0]
	for fni,fi in enumerate(frameNumbers):
		u2ds, true_labels, splits = Label.extract_label_frame(fi-1, labels_frames)
		clouds = ISCV.HashCloud2DList(u2ds,splits,x2d_threshold)
		skelDict['chanValues'][:] = anim_dict['dofData'][fns[fni]]
		x3ds, x3d_labels, E, x2d_labels = Recon.solve_x3ds(u2ds, splits, true_labels, Ps) # WARNING using true_labels here; these should be precisely the c3ds (including points not assigned to markers)
		#sc2,labels2,vels2 = Label.project_assign(clouds, x3ds, x3d_labels, Ps, x2d_threshold) # these are true_labels, but with the same threshold
		sc,labels,resids = Label.label_2d_from_skeleton_and_3ds(clouds, u2ds, splits, effectorLabels, skelDict, effectorData, rootMat, Ps, x2d_threshold, x3ds, x3d_threshold)
		good_score += sum(labels == true_labels)
		bad_score += sum(labels != true_labels)
		total_score += len(labels)
		#bads = np.where(labels != true_labels)[0]
		#print zip(labels[bads], true_labels[bads])
		print '\r',fi,good_score/float(total_score),fps(),;sys.stdout.flush()
	print


def test3D(c3d_filename, x3d_threshold = 150.):
	'''Test the labelling of a 3d point sequence by propagating the labels to the next frame.'''
	print 'loading 3d'
	c3d_dict = C3D.read(c3d_filename)
	c3d_frames, c3d_fps, c3d_labels = c3d_dict['frames'],c3d_dict['fps'],c3d_dict['labels']
	print 'num frames', len(c3d_frames)
	c3ds = c3d_frames[0]
	prev_labels = np.array(np.where(c3ds[:,3]==0)[0],dtype=np.int32)
	prev_x3ds = np.array(c3ds[prev_labels,:3],dtype=np.float32)
	prev_vels = np.zeros_like(prev_x3ds)
	for fi,c3ds in enumerate(c3d_frames):
		true_labels = np.array(np.where(c3ds[:,3]==0)[0],dtype=np.int32)
		x3ds = c3ds[true_labels,:3]
		sc,labels,vels = Label.label_3d_from_3d(prev_x3ds, prev_labels, prev_vels, x3ds, x3d_threshold)
		if not np.all(labels==test_x3d_labels):
			for li in np.where(labels!=test_x3d_labels)[0]:
				tli = labels[li]
				pli = test_x3d_labels[li]
				if pli == -1:
					print 'on frame', fi, 'label', tli,'aka',c3d_labels[tli],'was unlabelled'
					if tli in prev_labels: print 'THIS IS A FAIL'
				else:
					print 'on frame', fi, 'label', tli,'aka',c3d_labels[tli],'was seen as',pli,'aka',c3d_labels[pli]
		prev_x3ds,prev_labels,prev_vels = x3ds,labels,vels

def fps(cache = {}):
	import time
	if not cache.has_key('t'):
		cache['t'] = [time.time()]
	c = cache['t']
	c.append(time.time())
	while len(c) > 3 and c[-1] > c[0]+0.1: c.pop(0) # keep only 1/10th second
	if len(c) > 1000: c.pop(0)
	return '%3.3ffps' % ((len(c)-1)/(c[-1] - c[0] + 1e-10))

def test2D(labels_filename, x2d_threshold = 50./2000.):
	'''Test the labelling of a 2d point sequence by propagating the labels to the next frame.'''
	print 'loading 2d'
	labels_dict = IO.load(labels_filename)[1]
	print labels_dict.keys()
	labels_frames,mats,cids,label_names = labels_dict['frames'],labels_dict['mats'],labels_dict['cids'],labels_dict['label_names']
	print 'num frames', len(labels_frames)
	prev_frame = labels_frames[0]
	prev_labels = prev_frame['labels']
	prev_x2ds = prev_frame['u2ds']
	prev_vels = np.zeros_like(prev_x2ds)
	prev_splits = prev_frame['splits']
	numCams = len(mats)
	#label_names = ['#'+str(x) for x in range(60)]
	#print labels_frames

	good_score,bad_score = 0,0
	cases = [0,0,0,0,0,0,0]
	for fi,frame in enumerate(labels_frames):
		true_labels = frame['labels']
		x2ds,splits = frame['u2ds'],frame['splits']

		clouds = ISCV.HashCloud2DList(x2ds, splits, x2d_threshold)
		sc,labels,vels = clouds.assign_with_vel(prev_x2ds, prev_vels, prev_splits, prev_labels, x2d_threshold)

		for ci in range(numCams):
			test_x2d_labels = labels[splits[ci]:splits[ci+1]]
			true_lbl = true_labels[splits[ci]:splits[ci+1]]
			prev_lbl = prev_labels[prev_splits[ci]:prev_splits[ci+1]]
			bads = np.where(true_lbl!=test_x2d_labels)[0]
			good_score += len(true_lbl)-len(bads)
			bad_score += len(bads)
			for li in bads:
				tli = true_lbl[li]
				pli = test_x2d_labels[li]
				if pli == -1: # tli != -1 (we didn't give a label)
					if tli not in prev_lbl: # ~90% of the misses are here (the marker wasn't seen in that camera on the previous frame)
						bad_score -= 1 # these shouldn't be counted as errors because 'unlabelled' is the right result
						cases[0] += 1
					else: # (it should have been labelled, but wasn't)
						cases[1] += 1 # (35%) false negatives
				else: # pli != -1; (65%) (we labelled a point, but wrongly)
					if tli == -1: # (it should not have been labelled)
						cases[2] += 1 # (16.5%) false positives
					else: # (it should have been labelled)
						if tli not in prev_lbl: # (34%) (it couldn't have been labelled, but we stole someone's label)
							#print fi, ci, cids[ci], tli, pli, label_names[tli], label_names[pli]
							cases[3] += 1
							continue
						else: # it could have been labelled ... who stole our label?
							if tli in test_x2d_labels: # someone
								cases[4] += 1 # (2.5%)
							else: # no-one!
								cases[5] += 1 # (12%)
					#print 'on frame', fi, 'label', tli,'aka',label_names[tli],'was seen as',pli,'aka',label_names[pli]
		prev_labels,prev_x2ds,prev_splits,prev_vels = true_labels,x2ds,splits,vels
		print '\r%d : %f (%d/%s)' % (fi,float(good_score)/(good_score+bad_score),bad_score,cases),
	print


def boot_skel_from_cheat(skelDict, rootMat, amc_dict, fi):
	skelDict['chanValues'][:] = amc_dict['dofData'][(fi-amc_dict['frameNumbers'][0])%len(amc_dict['dofData'])]
	Character.pose_skeleton(skelDict['Gs'], skelDict, x_mat=rootMat)

def setFrame(fi):
	from UI import QApp
	drawing_skel = True
	drawing_2d = True
	drawing_3d = True
	global anim_dict, amc_dict, skelDict, primitives, primitives2D, x2d_frames, x3d_frames
	rootMat = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]],dtype=np.float32)
	dofs = anim_dict['dofData'][(fi-anim_dict['frameNumbers'][0])%len(anim_dict['frameNumbers'])]
	Character.pose_skeleton(skelDict['Gs'], skelDict, dofs)
	primitives[1].setPose(skelDict['Gs'])
	if drawing_skel:
		dofs = amc_dict['dofData'][(fi-amc_dict['frameNumbers'][0])%len(amc_dict['frameNumbers'])]
		Character.pose_skeleton(skelDict['Gs'], skelDict, dofs)
		primitives[2].setPose(skelDict['Gs'])
	if drawing_2d:
		x2ds_data,x2ds_splits = x2d_frames[(fi-1)%len(x2d_frames)]
		primitives2D[0].setData(x2ds_data,x2ds_splits)
	if drawing_3d:
		x3ds_data = x3d_frames[(fi-1)%len(x3d_frames),:,:3]
		primitives[0].setData(x3ds_data)
	QApp.app.updateGL()

def setFrameCalibrate(fi):
	from UI import QApp
	drawing_skel = True
	drawing_2d = True
	global anim_dict,skelDict
	global primitives,primitives2D
	dofs = anim_dict['dofData'][(fi-anim_dict['frameNumbers'][0])%len(anim_dict['frameNumbers'])]
	Character.pose_skeleton(skelDict['Gs'], skelDict, dofs)
	primitives[1].setPose(skelDict['Gs'])
	if drawing_2d:
		global wand_frames, camera_solved, Ps, mats
		x2ds_data,x2ds_splits = ViconReader.frameCentroidsToDets(wand_frames[(fi-1)%len(wand_frames)],mats)
		primitives2D[0].setData(x2ds_data,x2ds_splits)
		x2ds_labels = -np.ones(x2ds_data.shape[0],dtype=np.int32)
		ISCV.label_T_wand(x2ds_data, x2ds_splits, x2ds_labels, 2.0, 0.5, 0.01, 0.07)
		for ci,(c0,c1) in enumerate(zip(x2ds_splits[:-1],x2ds_splits[1:])): # remove labels from unsolved cameras
			if not camera_solved[ci]: x2ds_labels[c0:c1] = -1
		x3ds, x3ds_labels, E_x2ds_single, x2ds_single_labels = Recon.solve_x3ds(x2ds_data, x2ds_splits, x2ds_labels, Ps)
		primitives[0].setData(x3ds)
		#global graph, mats
		#l2x = [list(x3ds_labels).index(x) if x in list(x3ds_labels) else -1 for x in range(5)]
		#sc = 0
		#print x2ds_labels
		#x3ds,x2ds_labels = Recon.intersect_rays(x2ds_data, x2ds_splits, Ps, mats, seed_x3ds = None)
		#l2x = np.zeros(len(graph[0]),dtype=np.int32)
		#sc = ISCV.label_from_graph(x3ds, graph[0],graph[1],graph[2],graph[3], 500, 30.0, l2x)
		#print sc,l2x
		#if l2x[0] != -1:
		#	clouds = ISCV.HashCloud2DList(x2ds_data, x2ds_splits, 0.01)
		#	sc2,labels_out2,vels2 = Label.project_assign(clouds, x3ds[l2x], np.array(range(5),dtype=np.int32), Ps, 0.01)
		#	print labels_out2
	if drawing_skel:
		dofs = anim_dict['dofData'][(fi-anim_dict['frameNumbers'][0])%len(anim_dict['frameNumbers'])]
		Character.pose_skeleton(skelDict['Gs'], skelDict, dofs)
		primitives[2].setPose(skelDict['Gs'])
	QApp.app.updateGL()

def test_label_T_wand(x2ds_data, x2ds_splits, ratio = 2.0, x2d_threshold = 0.5, straightness_threshold = 0.01, match_threshold = 0.07):
	x2ds_labels = -np.ones(x2ds_data.shape[0],dtype=np.int32)
	for ci,(c0,c1) in enumerate(zip(x2ds_splits[:-1],x2ds_splits[1:])):
		x2ds = x2ds_data[c0:c1]
		ret,scores = Label.find_T_wand_2d(x2ds, ratio, x2d_threshold, straightness_threshold, match_threshold)
		if len(ret) == 1:
			x2ds_labels[c0:][ret[0]] = range(5)
	x2ds_labels2 = -np.ones_like(x2ds_labels)
	ISCV.label_T_wand(x2ds_data, x2ds_splits, x2ds_labels2, ratio, x2d_threshold, straightness_threshold, match_threshold)
	assert(np.all(x2ds_labels == x2ds_labels2))
	return x2ds_labels

def test_2d_3d(x2ds, x3ds):
	'''Given a 2d point seen over multiple frames and a 3d point seen over the same frames, determine how compatible the points are.'''
	# If they are the same point, then some linear combination of x, y, z, px x, px y, px z, px, py x, py y, py z, py is zero...
	# P[:2,:4][x y z 1]_4x1 = -[px; py]_2x1[x y z 1]_1x4 P[2,:4].T

	# Let's try an affine model. Each frame of the 2d points is a projection of the same 3d points: x2d = A x3d.
	# First, mean-centre the 2d and 3d data.
	# (x2d - mean(x2d))_2FxN = A_2Fx3 (x3d - mean(x3d))_3xN
	# So that matrix has rank 3. We can yield the decomposition from SVD, but only up to a 3x3 transform.
	# We can solve the correspondence by RANSAC assigning 4 points and scoring the position of the 5th.
	# The ratio of distances along a line is preserved... look for 3 points in a line
	# Since the wand is planar, the matrix has rank 2. The wand is bounded by a triangle. Central point is closest to the origin. The other point is half way between two bounding vertices.
	# TODO
	return

def match_2d_3d(x2ds, x3ds):
	'''Given some 3d points and some 2d projections of those points in an unknown camera, come up with an assignment.
	x2ds is (num2DPoints, numFrames, 2); x3ds is (num3DPoints, numFrames, 3)
	We can form this equation for P:
	[x, y, z, 1, 0, 0, 0, 0, px x, px y, px z, px] [P00, P01, P02, P03, P10, P11, P12, P13, P20, P21, P22, P23] = 0
	[0, 0, 0, 0, x, y, z, 1, py x, py y, py z, py]
	'''
	# [x2ds;1]_3,N X_N,M = P_3,4 [x3ds;1]_4,M where X is a reordering and P is a projection matrix
	# TODO
	for x2i,x2d in enumerate(x2ds):
		for x3i,x3d in enumerate(x3ds):
			M = np.dot(x2d.T, x3d) # 2x3 matrix
			u,s,vt = np.linalg.svd(M)
			print x2i,x3i,s

def track_anim2(labels_filename, skelDict, rootMat = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]],dtype=np.float32), pred_2d_threshold = 50./2000., x2d_threshold = 25./2000., x3d_threshold = 30., first_frame = None, last_frame = None):
	'''Track a skeleton through a sequence by propagating the labels from the first frame, solving IK, then picking up labels by projection.'''
	labels_frames,mats,effectorLabels,cids = loadLabels(labels_filename, skelDict['markerNames'])

	if first_frame is None: first_frame = 1
	if last_frame is None: last_frame = len(labels_frames[1])+1

	frameNumbers = np.array(range(first_frame, last_frame,[1,-1][first_frame>last_frame]),dtype=np.int32)
	numDofs = len(skelDict['chanValues'])
	dofData = np.zeros((len(frameNumbers),numDofs), dtype=np.float32)
	u2ds, _, splits = Label.extract_label_frame(frameNumbers[0], labels_frames)
	model = Label.TrackModel(skelDict, effectorLabels, mats, x2d_threshold, pred_2d_threshold, x3d_threshold)
	model.bootLabels(u2ds,splits)
	for fni,fi in enumerate(frameNumbers):
		u2ds, _, splits = Label.extract_label_frame(fi-1, labels_frames)
		print '\rtracking %d' %(fi),fps(),; sys.stdout.flush()
		model.push(u2ds,splits)
		dofData[fni,:] = skelDict['chanValues']
	return { 'dofData':dofData, 'frameNumbers':frameNumbers }

def test_dm_from_l3ds(l3ds, ws):
	'''Given l3ds, a numFrames x numLabels x 3 data matrix of animating labelled 3d points, and ws a numFrames x numLabels weights matrix, compute the distance matrix.'''
	numLabels = l3ds.shape[1]
	M = np.zeros((numLabels,numLabels),dtype=np.float32)
	W = np.zeros((numLabels,numLabels),dtype=np.float32)
	for li in xrange(numLabels):
		for lj in xrange(li):
			w = ws[:,li]*ws[:,lj]
			sw = np.sum(w)
			sw = (1.0/sw if sw > 0.0 else 0.0)
			d = (l3ds[:,li,:] - l3ds[:,lj,:])*w.reshape(-1,1)
			d2 = np.sum(d*d,axis=1)**0.5
			d2_mean = np.sum(d2)*sw
			d2 -= d2_mean
			d2_var = np.sum(d2*d2)*sw
			M[li,lj] = M[lj,li] = d2_mean
			W[li,lj] = W[lj,li] = ((1.0/(d2_var + 1.0))**0.5)
	M2 = np.zeros((numLabels,numLabels),dtype=np.float32)
	W2 = np.zeros((numLabels,numLabels),dtype=np.float32)
	ISCV.dm_from_l3ds(l3ds, ws, M2, W2)
	print M-M2,W-W2
	return M, W

def score_graph(x3ds, l2x, graph, penalty):
	g2l, graphSplits, backlinks, DM = graph
	score = 0
	for gi,li in enumerate(g2l):
		b0,b1 = graphSplits[gi],graphSplits[gi+1]
		for gj,MW in zip(backlinks[b0:b1],DM[b0:b1]):
			lj = g2l[gj]
			if l2x[li] == -1 or l2x[lj] == -1: score += penalty
			else:
				D = np.linalg.norm(x3ds[l2x[li]] - x3ds[l2x[lj]])
				score += ((D-MW[0])*MW[1])**2
	return score

def test_boot(c3d_filename, skelDict):
	import time
	times = []
	times.append(time.time())

	print 'loading 3d'
	c3d_dict = C3D.read(c3d_filename)
	c3d_frames, c3d_fps, c3d_labels = c3d_dict['frames'],c3d_dict['fps'],c3d_dict['labels']
	print 'num frames', len(c3d_frames)
	times.append(time.time())

	graph = Label.graph_from_c3ds(skelDict, c3d_labels, c3d_frames) if True else Label.graph_from_skel(skelDict, c3d_labels)

	times.append(time.time())

	markerNames = np.unique(skelDict['markerNames'])
	effectorLabels = np.array(sorted([c3d_labels.index(ln) for ln in markerNames if ln in c3d_labels]),dtype=np.int32)
	numLabels = len(graph[0])
	l2x = -np.ones(numLabels,dtype=np.int32)
	penalty = 30.0
	print graph
	if False:
		l3ds_2 = np.array(c3d_frames[:,effectorLabels,:3], dtype=np.float32)
		ws_2 = np.array(1.0 + c3d_frames[:,effectorLabels,3], dtype=np.float32) # 0 = good, -1 = bad
		graph2 = Label.make_dm_graph_from_l3ds(l3ds_2, ws_2, threshold = 50)
		print graph,graph2
		tmp = np.where(W2 > 0.02)
		print tmp
		print np.mean(M[tmp]-M2[tmp])
		print np.mean(W[tmp]),np.mean(W2[tmp])
		exit()

	goods, bads, total = 0,0,0
	#for fi in range(27700,27900): # james spader finger flip
	for fi in range(1,len(c3d_frames),10):
		c3ds = c3d_frames[fi]
		times.append(time.time())
		x2l = np.array(np.where(c3ds[:,3]==0)[0],dtype=np.int32)
		x3ds = c3ds[x2l,:3]
		sc = ISCV.label_from_graph(x3ds, graph[0],graph[1],graph[2],graph[3], 500, penalty, l2x)
		tmp = np.where(x2l[l2x] != effectorLabels)[0]
		print fi,'c',x2l[l2x][tmp],effectorLabels[tmp], sc
		goods += np.sum(x2l[l2x] == effectorLabels)
		total += len(np.where(c3ds[effectorLabels,3]==0)[0])
		print 'true',goods/float(total), total-goods
		#assert(np.allclose(sc,score_graph(x3ds, l2x, graph, penalty)))
		#print '\r',fi,;sys.stdout.flush()
	times.append(time.time())
	times = np.array(times)
	print np.around((times[1:] - times[:-1])*1000.)

def test_label_from_graph(x3ds, M, W, graph, keepHypotheses, penalty, l2x):
	'''A hypothesis is a g2p labelling: for each graph node, the assignment of which point.
	Grow a list of hypotheses by adding each node of the graph. Update the hypotheses to include that node (for every assignment).
	Keep the hypotheses sorted by score and limit the size and worst score.
	Returns the labels for the best hypothesis and the score.
	'''
	numLabels = M.shape[0]
	l2x_c = -np.ones(numLabels,dtype=np.int32)
	sc_c = ISCV.label_from_graph(x3ds, M, W, graph, 500, 9.0, l2x_c)
	print 'c',l2x_c,sc_c
	numPoints = x3ds.shape[0]
	D = np.zeros((numPoints,numPoints),dtype=np.float32)
	for d,x in zip(D,x3ds): d[:] = np.sum((x3ds - x)**2,axis=1)
	numLabels = M.shape[0]; assert(M.shape[1] == M.shape[0])
	assert(M.shape == W.shape)
	#if graph is None: graph = [[li,range(li)] for li in range(numLabels)] # fully-connected graph
	#assert(len(graph) == numLabels)
	maxHypotheses = keepHypotheses+numPoints+1
	hypotheses_in_size = 1
	hypotheses_in = np.zeros((hypotheses_in_size, 0),dtype=np.int32)
	hypotheses_in_scores = np.zeros(hypotheses_in_size,dtype=np.float32)
	in_h = np.zeros(numPoints+1,dtype=np.int32) # in_h[-1] is always false
	g2l = graph[:numLabels]
	penalty2 = penalty*2
	for gi,li in enumerate(g2l):
		backEdges = graph[numLabels*2+1+graph[numLabels+gi]:numLabels*2+1+graph[numLabels+gi+1]]
		M_li, W_li = M[li],W[li]
		hypotheses_out = np.zeros((maxHypotheses, gi+1),dtype=np.int32)
		hypotheses_out_scores = np.zeros(maxHypotheses,dtype=np.float32)
		hypotheses_out_size = 0
		thresholdScore = hypotheses_in_scores[0]+penalty*len(backEdges)+penalty2
		print 'py it', gi, 'thresh', thresholdScore, 'size', hypotheses_in_size
		for h,hsc in zip(hypotheses_in[:hypotheses_in_size],hypotheses_in_scores[:hypotheses_in_size]): # h is g2p
			if hsc > thresholdScore: break
			in_h[:] = 0
			in_h[h] = 1
			in_h[-1] = 0
			for pi in xrange(-1,numPoints):
				if in_h[pi]: continue # already assigned
				sc = float(hsc)
				D_pi = D[pi]
				for gj in backEdges:
					lj = g2l[gj]
					pj = h[gj]
					if pi == -1 or pj == -1: sc += penalty
					else: sc += ((D_pi[pj]-M_li[lj])**2)*W_li[lj]
				if sc > thresholdScore: continue
				hypotheses_out_scores[hypotheses_out_size] = sc
				hout = hypotheses_out[hypotheses_out_size]
				hout[:gi] = h
				hout[gi] = pi
				hypotheses_out_size += 1
			if hypotheses_out_size > keepHypotheses:
				order = np.argsort(hypotheses_out_scores[:hypotheses_out_size])
				hypotheses_out_scores[:keepHypotheses] = hypotheses_out_scores[order[:keepHypotheses]]
				hypotheses_out[:keepHypotheses] = hypotheses_out[order[:keepHypotheses]]
				hypotheses_out_size = keepHypotheses
				thresholdScore = min(hypotheses_out_scores[0]+penalty2, hypotheses_out_scores[keepHypotheses])
		order = np.argsort(hypotheses_out_scores[:hypotheses_out_size])
		hypotheses_out_size = min(keepHypotheses,hypotheses_out_size)
		hypotheses_in_scores = hypotheses_out_scores[order[:hypotheses_out_size]]
		hypotheses_in = hypotheses_out[order[:hypotheses_out_size]]
		hypotheses_in_size = hypotheses_out_size
	l2g = np.zeros(numLabels,dtype=np.int32)
	l2g[g2l] = range(numLabels)
	l2x[:] = hypotheses_in[0][l2g]
	sc = hypotheses_in_scores[0]
	print 'py',l2x,sc
	return l2x,sc

def project_and_compute_rms(x3ds, raw_x2s, P, (k1,k2)):
	'''Compute the rms error of 3d points compared with 2d points in one camera.'''
	K,RT = Calibrate.decomposeKRT(P)
	return ISCV.project_and_compute_rms(x3ds, raw_x2s, P, float(-K[0,2]), float(-K[1,2]), float(k1), float(k2))

def score_wand_reprojection(x3ds_data, x2ds_frames, frame_numbers, mats):
	'''Project the wands in all the cameras. Compute the rms error of all cameras where all 5 points can be seen.'''
	def get_order(labels):
		"""
		Return the x2d index of the five points of the T Wand
		
		Args:
			labels (int[]): 
			
		Returns:
			int[5]: "order" label indexes
			
		"""
		try:
			l = list(labels)
			order = [l.index(x) for x in xrange(5)]
			return order
		except:
			return None
	Ps = np.array([m[2]/(np.sum(m[2][0,:3]**2)**0.5) for m in mats],dtype=np.float32)
	x3ds_labels = np.array(range(5),dtype=np.int32)
	sses = [[] for x in xrange(numCameras)]
	worst_error,worst_frame = 0,(-1,-1)
	x2ds_labels_frames = []
	for fi in frame_numbers:
		x3ds, (x2ds_raw_data, x2ds_splits) = x3ds_data[fi][:,:3].copy(), x2ds_frames[fi]
		clouds = ISCV.HashCloud2DList(x2ds_raw_data, x2ds_splits, 6./2000.)
		sc2,x2ds_labels,_ = Label.project_assign(clouds, x3ds, x3ds_labels, Ps, 6./2000.)
		for ci,(c0,c1) in enumerate(zip(x2ds_splits[:-1],x2ds_splits[1:])):
			x2l = x2ds_labels[c0:c1]
			order = get_order(x2l)
			if order is not None:
				rms = project_and_compute_rms(x3ds, x2ds_raw_data[c0:c1][order], mats[ci][2], mats[ci][3])
				if rms > worst_error: worst_error,worst_frame = rms,(fi,ci)
				sses[ci].append(rms**2)
			else:
				x2l[:] = -1
		x2ds_labels_frames.append(x2ds_labels)
	rmss = np.array(map(np.mean,sses))**0.5
	tmp = [r * m[5][0]*0.5 for r,m in zip(rmss,mats) if r is not None]
	return tmp,worst_error,worst_frame,x2ds_labels_frames

def test_cam(m,(P,ks,rms),lo_focal_threshold, hi_focal_threshold,cv_2d_threshold):
	f = m[0][0,0]
	o = m[0][:2,2]
	if f < lo_focal_threshold or f > hi_focal_threshold: # impossible focal
		return 'crazy focal length %f (should be in %f:%f)'%(f,lo_focal_threshold,hi_focal_threshold)
	if rms > cv_2d_threshold: # bad rms
		return 'crazy 2d rms %f'%(rms)
	if not np.allclose(ks,0,0.5,0.5): # bad distortion
		return 'crazy ks %s'%(ks)
	if not np.allclose(o,0,0.05,0.05): # bad principal point
		return 'due to crazy principal point %s'%(o)
	return None

def score_and_solve_wands(wand_frames, mats2, camera_solved, rigid_filter = True, solve_cameras = True, error_thresholds = None, 
							solve_distortion = False, solve_principal_point = False,
							lo_focal_threshold = 0.5, hi_focal_threshold = 4.0, cv_2d_threshold = 0.01):
	Ps2 = np.array([m[2]/m[0][0,0] for m in mats2],dtype=np.float32)
	x2s_cameras,x3s_cameras,frames_cameras,num_kept_frames = Calibrate.generate_wand_correspondences(wand_frames, mats2, camera_solved, error_thresholds)
	print 'kept %d/%d frames'%(num_kept_frames,len(wand_frames))
	# TODO: count number of detections before/after killing frames. if we killed nearly all the frames for one camera,
	# we should reset that camera
	rmss,rmss2 = [],[]
	for ci,(x2s,x3s,which_frames) in enumerate(zip(x2s_cameras,x3s_cameras,frames_cameras)):
		x2d_to_pixels_scale = mats2[ci][5][0]*0.5
		if solve_cameras:
			if len(which_frames) < 2:
				print 'killing camera %d with %d frames %s'%(ci,len(which_frames),which_frames)
				mats2[ci] = Calibrate.makeUninitialisedMat(ci,mats2[ci][5])
				camera_solved[ci] = False
				continue
			cv2_mat = Calibrate.cv2_solve_camera_from_3d(x3s, x2s, Kin=mats2[ci][0], solve_distortion=solve_distortion, solve_principal_point=solve_principal_point)
			rms = cv2_mat[2]
			mats2[ci] = Calibrate.makeMat(cv2_mat[0],cv2_mat[1],mats2[ci][5])
			if solve_principal_point and not np.allclose(mats2[ci][0][:2,2],0,0.05,0.05): # bad principal point; try solving without
				print 'wandering principal point',ci,mats2[ci][0][:2,2]
				mats2[ci][0][:2,2] = 0
				cv2_mat = Calibrate.cv2_solve_camera_from_3d(x3s, x2s, Kin=mats2[ci][0], solve_distortion=solve_distortion, solve_principal_point=False)
				rms = cv2_mat[2]
				mats2[ci] = Calibrate.makeMat(cv2_mat[0],cv2_mat[1],mats2[ci][5])
			camera_solved[ci] = True
			msg = test_cam(mats2[ci],cv2_mat,lo_focal_threshold, hi_focal_threshold,cv_2d_threshold)
			if msg is not None:
				print 'killed camera %d due to %s'%(ci,msg)
				mats2[ci] = Calibrate.makeUninitialisedMat(ci,mats2[ci][5])
				camera_solved[ci] = False
				continue
			rmss.append((ci, rms * x2d_to_pixels_scale))
		if camera_solved[ci]:
			rmss2.append((ci, project_and_compute_rms(x3s, x2s, mats2[ci][2], mats2[ci][3]) * x2d_to_pixels_scale))
	def unzip(rmss): return ([r[0] for r in rmss],[r[1] for r in rmss])
	return unzip(rmss), unzip(rmss2)

def refine_cameras(wand_frames, mats2, camera_solved, its=100, lo_focal_threshold=0.5, hi_focal_threshold=4.0, cv_2d_threshold=0.02):
	# in a series of passes, update the cameras to agree with their intersections
	error_thresholds = 24./2000.
	numFrames = len(wand_frames)
	fs = xrange(0, min(100, numFrames))
	old_score = 1e10
	max_sample_rate = int(numFrames / 400)+1
	min_sample_rate = int(numFrames / 2000)+1
	sample_rate = max_sample_rate
	for it in xrange(its):
		if it > its/5: error_thresholds = 6./2000.

		print 'it',it,sample_rate
		(sc_cams,scores),(sc2_cams,scores2) = score_and_solve_wands([wand_frames[fi] for fi in fs], mats2, camera_solved, \
													rigid_filter = True, solve_cameras = True,\
													error_thresholds=error_thresholds, solve_distortion = (it > its/5), \
													solve_principal_point = (it > its/2), lo_focal_threshold=lo_focal_threshold, \
													hi_focal_threshold=hi_focal_threshold, cv_2d_threshold=cv_2d_threshold)
		if len(scores) == 0 or len(scores2) == 0:
			print 'this shouldn\'t happen???'
			continue
		print 'scores',np.sort(scores2),np.min(scores2),np.max(scores2),np.mean(scores2),np.median(scores2)
		new_score = np.mean(scores2)
		if it > its/2:
			if new_score > old_score - 0.002:
				sample_rate = max(min_sample_rate, sample_rate / 2)
			else:
				sample_rate = min(max_sample_rate, sample_rate * 2)
				#sample_rate = max_sample_rate # TESTING
			sample_rate = min(sample_rate, int(2**((its-1)-(it+1))))
		if it < its/2: sample_rate = max(sample_rate,min_sample_rate)
		old_score = new_score
		fs = xrange(it, numFrames, max(sample_rate,1))

def transform_cameras(mats2, camera_solved, vicon_mats, vicon_solved):
	which = np.where(np.array(camera_solved)*np.array(vicon_solved))[0]
	print camera_solved
	print vicon_solved
	print 'transforming based on',which
	#K,RT,P,[k1,k2],T,[width,height]
	t1s = np.array([m[4] for m in mats2],dtype=np.float32)[which]
	t2s = np.array([m[4] for m in vicon_mats],dtype=np.float32)[which]
	mat,inliers = Calibrate.rigid_align_points_inliers(t1s, t2s)
	def apply_mat(m, mat):
		K,RT,P,k,T,wh = m
		T[:] = np.dot(T,mat[:3,:3].T)+mat[:,3]
		RT[:3,:3] = np.dot(RT[:3,:3], mat[:3,:3].T)
		#RT[:3,3] = np.dot(RT[:3,3] + mat[:3,3], mat[:3,:3].T)
		#T[:] = -np.dot(RT[:,:3].T,RT[:,3])
		RT[:,3] = -np.dot(RT[:3,:3],T)
		np.dot(K,RT,out=P)
		return m
	for m,cs in zip(mats2,camera_solved):
		if cs: apply_mat(m, mat)


def generate_geometry(x3s_cameras, Ps, Pmat, x3d_size):
	'''Given 3d points and calibrated cameras, generate a geometry of the boundaries of empty space.'''
	# TODO this generates only the bounding volume (the empty space is assumed to be compact)
	# Knock up an opengl rendering context...
	width,height = 1080,1080
	from OpenGL import GL, GLU, GLUT
	GLUT.glutInit()
	GLUT.glutInitWindowSize(width,height)
	GLUT.glutCreateWindow("Context_is_all")
	GLUT.glutIconifyWindow()
	
	rbid = GL.glGenRenderbuffers(1)
	GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, rbid)
	GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT, width, height)
	GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, 0)

	fbid = GL.glGenFramebuffers(1)
	GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbid)
	GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER, rbid)

	# strange rendering option: we want to render the furthest points, so we clear depth to 0.0 and render 'furthest depth'
	GL.glClearDepth(0.0)
	GL.glDepthFunc(GL.GL_GREATER)
	GL.glEnable(GL.GL_DEPTH_TEST)
	GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
	GL.glDisable(GL.GL_CULL_FACE)
	#GL.glFrontFace(GL.GL_CCW)
	GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
	GL.glMatrixMode(GL.GL_PROJECTION)
	cameraInterest = 1000 # =1m; render from 1cm=0.0 to 1km=1.0
	GL.glLoadMatrixf(np.array([[1,0,0,0],[0,width/float(height),0,0],[0,0,-1.00002,-1],[0,0,cameraInterest*-0.0200002,0]],dtype=np.float32))
	GL.glMatrixMode(GL.GL_MODELVIEW)
	glm = np.eye(4,dtype=np.float32); glm[:4,:3] = Pmat[:3,:4].T
	GL.glLoadMatrixd(glm)

	#K,RT=Calibrate.decomposeKRT(Pmat)
	#T0 = -np.dot(RT[:3,:3].T,RT[:,3])
	#axis = Pmat[2,:3]

	from OpenGL.arrays import vbo
	triangles = np.array([0,1,2, 0,2,3, 0,3,4, 0,4,1],dtype=np.int32)
	triangles = vbo.VBO(triangles, target=GL.GL_ELEMENT_ARRAY_BUFFER, usage='GL_STATIC_DRAW_ARB')
	triangles.bind()
	x3d_size = float(x3d_size)
	for ci,(P,x3s) in enumerate(zip(Ps,x3s_cameras)):
		K,RT=Calibrate.decomposeKRT(P)
		T = -np.dot(RT[:3,:3].T,RT[:,3])
		for x3d in x3s:
			#if np.dot(x3d-T,axis) > 0: continue
			x,y = np.cross(x3d-T,P[1,:3]), np.cross(x3d-T,P[0,:3])
			x *= x3d_size/(np.sum(x**2)**0.5)
			y *= x3d_size/(np.sum(y**2)**0.5)
			vertices = np.array([T,x3d-x-y,x3d-x+y,x3d+x+y,x3d+x-y],dtype=np.float32)
			GL.glVertexPointerf(vertices)
			GL.glDrawElementsui(GL.GL_TRIANGLES, triangles)
	triangles.unbind()

	# read the results
	field = np.zeros((height,width),dtype=np.float32)
	GL.glReadPixels(0, 0, width, height, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT, array = field)

	# undo all the gl state
	GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
	GL.glClearDepth(1.0)
	GL.glDepthFunc(GL.GL_LESS)
	GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
	GL.glDeleteRenderbuffers(1,[rbid])
	GL.glDeleteFramebuffers(1,[fbid])

	return field

def test_calibration(vsk_filename=None, x2d_filename=None, xcp_filename=None):
	directory = os.environ['GRIP_DATA']
	if 1: # 6 mins; 0.41/1.27/0.65/0.59
		directory = os.path.join(directory,'140519')
		vsk_filename = os.path.join(directory,'Combined Wand and L-Frame[Y-Up] (5 Markers).vsk')
		x2d_filename = os.path.join(directory,'183500_CAL292_Wandp_01.x2d')
		xcp_filename = os.path.join(directory,'183500_CAL292_Wandp_Floor_Final.xcp')

	global wand_frames, camera_solved, mats, Ps, anim_dict, skelDict
	skelDict = ViconReader.loadVSS(vsk_filename)

	vicon_Ps, vicon_mats, x2d_cids, camera_names, wand_frames, x2d_header = ViconReader.load_xcp_and_x2d(xcp_filename, x2d_filename, raw=True)

	numFrames = len(wand_frames)
	lo_focal_threshold, hi_focal_threshold, cv_2d_threshold = 0.5, 4.0, 0.01
	numCameras = len(camera_names)

	if False:
		print 'loading 3d'
		c3d_dict = C3D.read(c3d_filename)
		c3d_frames, c3d_fps, c3d_labels = c3d_dict['frames'],c3d_dict['fps'],c3d_dict['labels']
		print 'num frames', len(c3d_frames)
		print c3d_labels[-5:]
		print c3d_frames.shape
		skipFrames = 40
		x3ds_data = c3d_frames[:,-5:,:]
		print x3ds_data.shape
		x3ds_frames = np.where(np.sum(x3ds_data[::skipFrames,:,3],axis=1) == 0)[0] * skipFrames
		#print x3ds_frames
		tmp,worst_error,worst_frame,true_labels = score_wand_reprojection(x3ds_data, wand_frames, x3ds_frames, vicon_mats)
		print tmp,np.min(tmp),np.max(tmp),np.mean(tmp)
		print 'worst',worst_error,worst_frame[0]+1,camera_names[worst_frame[1]]
		#exit()
	

	vicon_solved = [not (m[1][1,3] == 0.0 and m[1][2,3] == 0.0 and m[1][0,3] != 0.0) for m in vicon_mats] # uninitialised cameras are on the positive x-axis
	vicon_uninitialised_cameras = np.where([not x for x in vicon_solved])[0] # list of uninitialised cameras
	print 'uninitialised cameras',[camera_names[ci] for ci in vicon_uninitialised_cameras]
	(scores_cams,scores),(scores2_cams,scores2) = score_and_solve_wands(wand_frames[::20], vicon_mats, vicon_solved, solve_cameras=False, rigid_filter=True)
	print 'scores2',np.sort(scores2),np.min(scores2),np.max(scores2),np.mean(scores2),np.median(scores2)

	if 0:
		try:
			field = IO.load('field')[1]
		except:
			x2s_cameras,x3s_cameras,frames_cameras,num_kept_frames = Calibrate.generate_wand_correspondences(wand_frames, vicon_mats, vicon_solved)
			hero_camera = 0
			frames = frames_cameras[hero_camera]
			numFrames = len(wand_frames)
			which0 = np.zeros(numFrames,dtype=np.bool)
			which0[frames] = True
			x3s = []
			for xs,fs in zip(x3s_cameras,frames_cameras):
				sel = np.where(which0[fs])[0]
				x3s.append(xs.reshape(-1,15).take(sel,axis=0).reshape(-1,3))
			field = generate_geometry(x3s, vicon_Ps, vicon_Ps[hero_camera], 6.0)
			IO.save('field',field)
		print np.mean(field)

		import pylab as pl
		field = IO.load('field')[1]
		field[np.where(field==0)] = np.min(field[np.where(field!=0)])
		pl.imshow(field[::-1])
		pl.show()

	mns = list(np.unique(skelDict['markerNames']))
	print mns
	if False:
		tracker = Label.Track2D(numCameras, 0.01)
		graph = Label.graph_from_skel(skelDict, mns)
		for x2ds_data,x2ds_splits in wand_frames[:100]:
			tracker.push(x2ds_data, x2ds_splits)
			#for ci,(m0,m1) in enumerate(zip(x2ds_splits[:-1],x2ds_splits[1:])):
			#	x2ds = x2ds_data[m0:m1]
			#	ret = Label.find_T_wand_2d(x2ds, ratio = 2., x2d_threshold = 0.2)
			#	if ret != []: print ci,ret

		print len(tracker.tracks), map(len, tracker.tracks.values())

	mats, camera_solved = Calibrate.boot_cameras_from_wand(wand_frames, zip(*vicon_mats)[5])
	refine_cameras(wand_frames, mats, camera_solved, its=110)
	IO.save('tmp',(mats,camera_solved,x2d_cids))

	#(mats,camera_solved,x2d_cids) = IO.load('tmp')[1]

	transform_cameras(mats, camera_solved, vicon_mats, vicon_solved)
	Ps = np.array([m[2]/(np.sum(m[2][0,:3]**2)**0.5) for m in mats],dtype=np.float32)

	if False:
		x3ds_frames = []
		tmp,worst_error,worst_frame,true_labels = score_wand_reprojection(x3ds_data, wand_frames, x3ds_frames, vicon_mats)
		print tmp,np.min(tmp),np.max(tmp),np.mean(tmp),np.median(tmp)
		print 'worst',worst_error,worst_frame[0]+1,camera_names[worst_frame[1]]

	if False: # write this properly, would be interesting to do a direct comparison
		print 'loading 3d'
		c3d_dict = C3D.read(c3d_filename)
		c3d_frames, c3d_fps, c3d_labels = c3d_dict['frames'],c3d_dict['fps'],c3d_dict['labels']
		print 'num frames', len(c3d_frames)
		print c3d_labels[-5:]
		print c3d_frames.shape
		skipFrames = 40
		x3ds_data = c3d_frames[:,-5:,:]
		print x3ds_data.shape
		x3ds_frames = np.where(np.sum(x3ds_data[::skipFrames,:,3],axis=1) == 0)[0] * skipFrames
		#print x3ds_frames
		tmp,worst_error,worst_frame,true_labels = score_wand_reprojection(x3ds_data, wand_frames, x3ds_frames, mats)
		print tmp,np.min(tmp),np.max(tmp),np.mean(tmp),np.median(tmp)
		print 'worst',worst_error,worst_frame[0]+1,camera_names[worst_frame[1]]
		#exit()

	IO.save('mats.mats',(mats,x2d_cids))

	amc_dict = anim_dict = { 'dofData':np.zeros((numFrames,6),dtype=np.float32),'frameNumbers':np.array(range(numFrames),dtype=np.int32)}
	from UI import QGLViewer
	from UI import GLPoints2D
	global primitives,primitives2D,glPoints
	primitives = QGLViewer.makePrimitives(vertices=[],skelDict=skelDict, altSkelDict=skelDict)
	glPoints = GLPoints2D(([],[0]))
	primitives2D = [glPoints]
	QGLViewer.makeViewer(timeRange = (anim_dict['frameNumbers'][0],anim_dict['frameNumbers'][-1]), callback=setFrameCalibrate, mats=mats+vicon_mats, camera_ids=camera_names+['VICON'+cn for cn in camera_names], primitives = primitives, primitives2D = primitives2D)


def test_skel_tracking(generating_labels=True, generating_anim=True, scoring_anim=True):
	global anim_dict, amc_dict, skelDict, primitives,primitives2D, x2d_frames, mats, x3d_frames
	
	trackFirstFrame = 1
	amcFrameOffset = 0
	x2dFrameOffset = 0
	directory = os.path.join(os.environ['GRIP_DATA'],'151110')
	rootMat = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]],dtype=np.float32)
	root = os.path.join(directory,'Orn')
	x2d_filename = os.path.join(directory,'50_Grip_RoomCont_AA_02.x2d')
	xcp_filename = os.path.join(directory,'50_Grip_RoomCont_AA_02.xcp')
	skelPrefix = 'Orn'
	amc_filename = '%s.amc'%root
	c3d_filename = '%s.c3d'%root
	labels_filename = '%s.labels'%root
	anim_filename = '%s.anim'%root
	vss_filename = os.path.join(directory,skelPrefix+'.vss')

	print 'loading VSS' # use vss not asf because the asf doesn't have markers and has clipped joints
	skelDict = ViconReader.loadVSS(vss_filename)
	print 'loading AMC'
	amc_dict = ASFReader.read_AMC(amc_filename, skelDict)
	amc_dict['frameNumbers']+=amcFrameOffset
	skelDict['markerNames'] = [skelPrefix+':'+n for n in skelDict['markerNames']]

	if 1:
		print ('loading C3D')
		c3d_dict = C3D.read(c3d_filename)
		x3d_frames, c3d_fps, c3d_labels = c3d_dict['frames'],c3d_dict['fps'],c3d_dict['labels']
		which_labels = [c3d_labels.index(s) for s in skelDict['markerNames']]
		x3d_frames = x3d_frames[:,which_labels,:]
		c3d_labels = [c3d_labels[x] for x in which_labels]
	if generating_labels:
		Ps, mats, x2d_cids, camera_names, x2d_frames, x2d_header = ViconReader.load_xcp_and_x2d(xcp_filename, x2d_filename)
		print (len(x3d_frames),len(x2d_frames))
		x2d_frames = x2d_frames[x2dFrameOffset:]
		ret = Label.label_2d_from_3ds(x2d_frames, x3d_frames, Ps, x2d_threshold = 20./2000.)
		labels_dict = {'frames':ret,'mats':mats,'cids':camera_names,'label_names':c3d_labels}
		print ('saving labels')
		IO.save(labels_filename, labels_dict)
		del c3d_dict,c3d_fps,c3d_labels
		del Ps,mats,x2d_cids,camera_names,x2d_frames
		del ret,labels_dict

	if generating_anim:
		t2d = 20./2000.
		t3d = 60.
		boot_skel_from_cheat(skelDict, rootMat, amc_dict, trackFirstFrame)
		anim_dict = track_anim2(labels_filename, skelDict, x2d_threshold=t2d, x3d_threshold=t3d, first_frame=trackFirstFrame, last_frame=None)
		if trackFirstFrame != 1:
			boot_skel_from_cheat(skelDict, rootMat, amc_dict, trackFirstFrame)
			anim_dict2 = track_anim2(labels_filename, skelDict, x2d_threshold=t2d, x3d_threshold=t3d, first_frame=trackFirstFrame, last_frame=1)
			anim_dict = { 'dofData': np.concatenate((anim_dict2['dofData'][:0:-1],anim_dict['dofData'])), 'frameNumbers' : np.concatenate((anim_dict2['frameNumbers'][:0:-1],anim_dict['frameNumbers'])) }
			del anim_dict2
		print 'saving animation'
		IO.save(anim_filename,anim_dict)
	else:
		print 'loading animation'
		anim_dict = IO.load(anim_filename)[1]

	if scoring_anim:
		t2d = 20./2000.
		t3d = 60.
		score_anim(labels_filename, skelDict, anim_dict, x2d_threshold=t2d, x3d_threshold=t3d, first_frame=trackFirstFrame, last_frame=None)

	Ps, mats, x2d_cids, camera_names, x2d_frames, x2d_header = ViconReader.load_xcp_and_x2d(xcp_filename, x2d_filename)
	x2d_frames = x2d_frames[x2dFrameOffset:]

	from UI import QGLViewer
	from UI import GLPoints2D
	primitives = QGLViewer.makePrimitives(vertices=[], skelDict=skelDict, altSkelDict=skelDict)
	primitives2D = QGLViewer.makePrimitives2D(([],[0]))
	QGLViewer.makeViewer(timeRange = (anim_dict['frameNumbers'][0],anim_dict['frameNumbers'][-1]), callback=setFrame, mats=mats, camera_ids=camera_names, primitives=primitives, primitives2D=primitives2D)

def test_hashcloud():
	# a synthetic test of the hashcloud
	import Hash
	x2ds = np.array([[0,0],[1,0],[2,0],[1,1],[0.5,0.5],[0,1]], dtype=np.float32)+7.5
	x2ds_2 = np.array(15-x2ds,dtype=np.float32)
	splits = np.array([0,len(x2ds),2*len(x2ds)],dtype=np.int32)
	labels = np.array(list(range(len(x2ds))+list(range(len(x2ds)))),dtype=np.int32)
	threshold = 7.5
	hs, horder, h_splits = Hash.hash2d_cloud(x2ds, threshold)
	print hs,horder,Hash.hash2d_10_offsets()
	scores,matches,matches_splits = Hash.hash2d_score(x2ds_2, threshold, x2ds, horder, h_splits)
	print 'v1',scores, matches, matches_splits
	import Assign
	print 1.5**2 * 2 * len(x2ds)
	print Assign.min_assignment_sparse(scores, matches, matches_splits, threshold*threshold)

	cloud = ISCV.HashCloud2D(x2ds, threshold)
	scores,matches,matches_splits = cloud.score(x2ds_2)
	print 'v2',scores, matches, matches_splits
	labels_out = -np.ones(6,dtype=np.int32)
	sc = ISCV.min_assignment_sparse(scores, matches, matches_splits, threshold**2, labels_out)
	print sc,labels_out
	clouds = ISCV.HashCloud2DList(np.array(np.concatenate((x2ds,x2ds)),dtype=np.float32),splits,threshold)
	ret = clouds.assign(np.array(np.concatenate((x2ds_2,x2ds)),dtype=np.float32), splits, labels, threshold)
	print 'v3', ret

def test_single_ray():
	# a synthetic experiment to test/debug the single ray code
	skelDict = {
			'name'           : 'test',
			'numJoints'      : 2,
			'jointNames'     : ['root','joint'],  # list of strings
			'jointIndex'     : {'root':0,'joint':1}, # dict of string:int
			'jointParents'   : np.array([-1,0],dtype=np.int32),
			'jointChans'     : np.array([0,1,2,3,4,5],dtype=np.int32), # 0 to 5 : tx,ty,tz,rx,ry,rz
			'jointChanSplits': np.array([0,3,6,6,6],dtype=np.int32),
			'chanNames'      : ['tx','ty','tz','rx','ry','rz'],   # list of strings
			'chanValues'     : np.zeros(6,dtype=np.float32),
			'numChans'       : 6,
			'Bs'             : np.array([[1000,0,0],[1000,0,0]], dtype=np.float32),
			'Ls'             : np.array([[[1,0,0,0],[0,1,0,0],[0,0,1,0]],[[1,0,0,1000],[0,1,0,0],[0,0,1,0]]], dtype=np.float32),
			'Gs'             : np.array([[[1,0,0,0],[0,1,0,0],[0,0,1,0]],[[1,0,0,1000],[0,1,0,0],[0,0,1,0]]], dtype=np.float32),
			'markerOffsets'  : np.array([[500,0,0]], dtype=np.float32),
			'markerParents'  : np.array([1],dtype=np.int32),
			'markerNames'    : ['marker'],
			'markerWeights'  : np.array([1],dtype=np.float32),
			'numMarkers'     : 1,
			}
	effectorData = SolveIK.make_effectorData(skelDict)
	x3ds = np.array([[1000,0,-2000]],dtype=np.float32)
	x3d_labels = np.array([0],dtype=np.int32)
	E = np.array([[[1,0,0,-10],[0,1,0,-20]]],dtype=np.float32)
	E *= 0.001
	x2d_labels = np.array([0],dtype=np.int32)
	residual = np.zeros((1,3),dtype=np.float32)
	residual2 = np.zeros((1,2),dtype=np.float32)
	#residual_old = np.zeros((1,3,4),dtype=np.float32)
	effectors = np.zeros((1,3),dtype=np.float32)
	#effectors_old = np.zeros((1,3,4),dtype=np.float32)
	effectorTargets = np.zeros((1,3,4),dtype=np.float32)
	effectorTargets[:,:,3] = x3ds
	effectorOffsets = np.zeros((1,3),dtype=np.float32)
	effectorOffsets[:] = effectorData[1][:,:,3]
	effectorWeights = np.zeros(1, dtype=np.float32)
	effectorWeights[x3d_labels] = 1
	effectorWeights[x2d_labels] = 1
	#x3ds,x3d_labels,residual = x3ds[:0],x3d_labels[:0],residual[:0] # ! disable 3d !
	sc = ISCV.pose_effectors_single_ray(effectors, residual, residual2, skelDict['Gs'], effectorData[0], effectorOffsets, effectorWeights, x3ds, x3d_labels, E, x2d_labels)
	print sc,residual,residual2
	SolveIK.solveIK1Ray(skelDict, effectorData, x3ds, x3d_labels, E, x2d_labels, outerIts = 15, rootMat = rootMat)
	Character.pose_skeleton(skelDict['Gs'], skelDict)
	#sc = ISCV.pose_effectors(effectors_old, residual_old, skelDict['Gs'], effectorData[0], effectorData[1], effectorData[2], effectorTargets)
	#print sc, skelDict['chanValues'], effectorTargets, effectors_old, residual_old
	sc = ISCV.pose_effectors_single_ray(effectors, residual, residual2, skelDict['Gs'], effectorData[0], effectorOffsets, effectorWeights, x3ds, x3d_labels, E, x2d_labels)
	animDict = {'dofData':np.array([skelDict['chanValues']],dtype=np.float32),'frameNumbers':np.zeros(1,dtype=np.int32)}
	print sc,residual,residual2
	exit()
	boot_skel_from_cheat(skelDict, rootMat, amc_dict, trackFirstFrame)
	animDict = track_anim(labels_filename, skelDict, first_frame = trackFirstFrame, last_frame = trackFirstFrame+100)
	print 'saving animation'
	IO.save(anim_filename,animDict)


def setFrameResection(fi):
	from UI import QApp
	drawing_2d = True
	drawing_3d = True
	global x3ds_frames, x2ds_frames, vicon_mats,primitives,primitives2D, x3ds_labels, x2ds_data, x2ds_labels, x2ds_splits
	(x3ds,x3ds_labels,x2ds_labels) = x3ds_frames[fi%len(x3ds_frames)]
	if drawing_2d:
		x2ds_data,x2ds_splits = x2ds_frames[fi%len(x2ds_frames)]
		primitives2D[0].setData(x2ds_data,x2ds_splits)#,x2ds_labels)
		primitives[0].setData(x3ds)#,x3ds_labels)
	QApp.app.updateGL()

def picked(view,data,clearSelection=True):
	global primitives,primitives2D, x2d_frames, x3ds_frames, x2ds_data, x2ds_labels, x2ds_splits
	if data is None or clearSelection: primitives[1].setData(np.zeros((0,3),dtype=np.float32));primitives2D[1].setData([],[])
	if data is None: view.updateGL(); return
	print data
	(type,pi,index,depth) = data
	if type == '3d' and pi == 1:
		primitives[1].setData(np.concatenate((primitives[1].vertices,primitives[0].vertices[index:index+1])))
		li = x3ds_labels[index]
		print '3d label is',li
		mis = list(np.where(x2ds_labels == li)[0])
		print 'ray indices are',mis
		cameras = [np.any([ci in mis for ci in range(c0,c1)]) for c0,c1 in zip(x2ds_splits[:-1],x2ds_splits[1:])]
		print 'cameras',np.where(cameras)[0]
		primitives2D[1].setData(x2ds_data[mis], np.array([np.sum(cameras[:ci]) for ci in xrange(len(x2ds_splits))],dtype=np.int32))
	view.updateGL()
	
def solve_camera_from_single_point(fin_2d,t2d,fin_3d,t3d,K,test_x2ds,test_x3ds,cloud,x2d_threshold=20./2000.,solve_intrinsics=False):
	'''Given a 2d track and a 3d track that it is assumed to correspond with and a camera intrinsic matrix,
	find the overlapping frames and use them to solve the camera. Given the camera, project the test 3d points
	and compare them with the cloud of test 2d points. The score is the reprojection error (lower is better).'''
	fin,fend = max(fin_2d, fin_3d),min(fin_2d+len(t2d),fin_3d+len(t3d))
	if fend <= fin + 10: return 1e10,(None,None)
	t2d_crop = t2d[fin-fin_2d:fend-fin_2d]
	t3d_crop = t3d[fin-fin_3d:fend-fin_3d]
	assert(len(t2d_crop) == len(t3d_crop))
	P2, ks, rms = Calibrate.cv2_solve_camera_from_3d(t3d_crop, t2d_crop, Kin=K, solve_focal_length=solve_intrinsics, solve_distortion=False)
	#if rms > 0.1: return 1e10,(None,None) # disregard horrible solutions!
	proj_x2ds, proj_splits, proj_labels = ISCV.project(test_x3ds, np.arange(len(test_x3ds),dtype=np.int32), P2.reshape(1,3,4))
	scores,matches,matches_splits = cloud.score(proj_x2ds)
	labels = -np.ones(len(proj_x2ds),dtype=np.int32)
	sc = ISCV.min_assignment_sparse(scores, matches, matches_splits, x2d_threshold**2, labels)
	sc -= (x2d_threshold**2)*len(proj_x2ds)
	which = np.where(labels != -1)[0]
	if len(which) < 10:
		#print 'bad rms',rms
		return 1e10,(None,None)
	x2s,x3s = np.concatenate((t2d_crop,test_x2ds[labels[which]])),np.concatenate((t3d_crop,test_x3ds[proj_labels[which]]))
	P2, ks, rms2 = Calibrate.cv2_solve_camera_from_3d(x3s, x2s, Kin=K, solve_focal_length=solve_intrinsics, solve_distortion=False)
	#print 'second it',rms,rms2
	proj_x2ds, proj_splits, proj_labels = ISCV.project(test_x3ds, np.arange(len(test_x3ds),dtype=np.int32), P2.reshape(1,3,4))
	scores,matches,matches_splits = cloud.score(proj_x2ds)
	labels = -np.ones(len(proj_x2ds),dtype=np.int32)
	sc = ISCV.min_assignment_sparse(scores, matches, matches_splits, x2d_threshold**2, labels)
	sc -= (x2d_threshold**2)*len(proj_x2ds)
	#print 'good rms',rms, sc
	return sc,(P2,zip(labels,proj_labels))

def test_camera_calibration(vicon_mats, x3ds_frames, x2ds_frames, x2d_threshold, camera_names, tracks_2d, tracks_3d):
	'''just some test code to boot cameras from known-correct labels.'''
	for ci in xrange(numCameras):
		Kin = vicon_mats[ci][0]
		vicon_mats[ci] = Calibrate.makeUninitialisedMat(ci,vicon_mats[ci][5],K=Kin)
		# the test data are the 2D detections / 3D points on the first frame
		test_x3ds = x3ds_frames[0][0]
		test_x2ds = x2ds_frames[0][0][x2ds_frames[0][1][ci]:x2ds_frames[0][1][ci+1]]
		cloud = ISCV.HashCloud2D(test_x2ds, x2d_threshold)
		print 'testing',camera_names[ci]
		# select all the camera tracks more than 10 frames and where it's moved more than 0.01
		camera_tracks_2d = [t[1:] for t in tracks_2d if t[0] == ci and len(t) > 20 and np.sum((t[2] - t[-1])**2) > 0.01**2]
		if not len(camera_tracks_2d):
			print 'no tracks for camera',camera_names[ci]
			continue
		# score the tracks by how far they move from the initial position
		tmp = [np.sum((np.array(t[1:])-t[1])**2,axis=1) for t in camera_tracks_2d]
		track_scores = map(max,tmp)
		ti_2d = np.argmax(track_scores)
		t2d = camera_tracks_2d[ti_2d]
		fin_2d,t2d = t2d[0],np.array(t2d[1:],dtype=np.float32)
		# we need to find the label of this 2d track...
		x2ds_data,x2ds_splits = x2ds_frames[fin_2d]
		x3ds,x3ds_labels,x2ds_labels = x3ds_frames[fin_2d]
		x2ds = x2ds_data[x2ds_splits[ci]:x2ds_splits[ci+1]]
		labels = x2ds_labels[x2ds_splits[ci]:x2ds_splits[ci+1]]
		res = np.sum((x2ds - t2d[0])**2,axis=1)
		ri = np.argmin(res)
		if res[ri] != 0.0:
			print 'should be 0.0 =',res[ri],'fin',fin_2d,x2ds_splits[ci-1:ci+2],x2ds_data[x2ds_splits[ci]-10:x2ds_splits[ci+1]+10],x2ds,t2d[0]
			assert(res[ri] == 0.0)
		ti_3d = labels[ri]
		t3d = tracks_3d[ti_3d]
		fin_3d,t3d = t3d[0],np.array(t3d[1:],dtype=np.float32)
		sc,(P2,labels) = solve_camera_from_single_point(fin_2d,t2d,fin_3d,t3d,Kin,test_x2ds,test_x3ds,cloud,x2d_threshold)
		if P2 is None:
			print 'failed'
			continue
		best = sc,ti_2d,ti_3d,labels,P2
		K,RT = Calibrate.decomposeKRT(P2)
		print sc,K[0,0],Calibrate.decomposeR(RT[:3,:3]),-np.dot(RT[:3,:3].T,RT[:3,3]),'cf',true_Ts[ci]
		vicon_mats[ci] = Calibrate.makeMat(P2,vicon_mats[ci][3],vicon_mats[ci][5])

def test_resection(wand_vsk_filename, x2d_filename, xcp_filename, frame_slice=[0,-1]):
	'''resection is fancy speak for solving cameras from data. if the calibration is good enough that we can get some 3d points
	then we ought to be able to fix any bumped cameras.'''
	global x3ds_frames, x2ds_frames, vicon_mats,primitives,primitives2D
	from UI import QGLViewer
	from UI import GLPoints2D

	Ps, vicon_mats, x2d_cids, camera_names, x2ds_frames, x2d_header = ViconReader.load_xcp_and_x2d(xcp_filename, x2d_filename)
	x2ds_frames = x2ds_frames[frame_slice[0]:frame_slice[1]]
	numCameras = len(Ps)
	for it in range(1):
		tracker_3d = Label.Track3D(vicon_mats)
		tracker_2d = Label.Track2D(numCameras, 0.01)
		x2ds_data, x2ds_splits = x2ds_frames[0]
		x3ds,x3ds_labels = tracker_3d.boot(x2ds_data, x2ds_splits)
		x3ds_frames = [] # TODO get rid of 'x3ds_frames'; make the code use tracks_3d
		for fi,(x2ds_data, x2ds_splits) in enumerate(x2ds_frames):
			x3ds,x3ds_labels = tracker_3d.push(x2ds_data, x2ds_splits)
			tracker_2d.push(x2ds_data, x2ds_splits)
			x2ds_labels = tracker_3d.x2ds_labels
			print '\rframe %d max label %d ratio %f' % (fi,tracker_3d.next_id,tracker_3d.next_id/float(fi+1)),len(x3ds_labels),fps(),; sys.stdout.flush()
			x3ds_frames.append((x3ds,x3ds_labels,x2ds_labels))

		# tracks_3d[li] = [frame_index, x3d_0, x3d_1, ...]
		tracks_3d = tracker_3d.tracks
		# tracks_2d[li] = [camera_index, frame_index, x2d_0, x2d_1, ...]
		tracks_2d = tracker_2d.tracks

		# good tracks are longer than 20 frames, aren't stationary and are supported by the most cameras.
		good_tracks_3d = [t for t in tracks_3d.itervalues() if (len(t) > 20 and np.max(np.sum((np.array(t[2:]) - t[1])**2,axis=1)) > 50**2)]
		print 'good 3d tracks',len(good_tracks_3d),map(len,good_tracks_3d)

		# health check; we want to find (1) dead cameras (2) noisy cameras (3) knocked cameras (4) moving cameras
		counts = np.array([f[1] for f in x2ds_frames], dtype=np.int32) # this is nframes x ncams+1
		counts = np.sum(counts[:,1:] - counts[:,:-1],axis=0) # this is num detections per camera
		order = np.argsort(counts)
		#print counts[order]
		#print [camera_names[o] for o in order]
		dead_cameras = np.where(counts == 0)[0]
		if len(dead_cameras):
			print 'WARNING: cameras are not providing data:',[camera_names[o] for o in dead_cameras]
		median = counts[order[(len(order) - len(dead_cameras))/2 + len(dead_cameras)]]
		noisy_cameras = np.where(counts > 3*median)[0]
		if len(noisy_cameras):
			print 'WARNING: cameras are providing much more data than the others:',[camera_names[o] for o in noisy_cameras]

		unlabelled = np.zeros_like(counts)
		for (x2ds_data, x2ds_splits),(x3ds,x3ds_labels,x2ds_labels) in zip(x2ds_frames,x3ds_frames):
			for ci,(c0,c1) in enumerate(zip(x2ds_splits[:-1],x2ds_splits[1:])):
				unlabelled[ci] += np.sum(x2ds_labels[c0:c1] == -1)

		fractions = unlabelled / (counts+1.0)
		order = np.argsort(fractions)
		#print fractions[order]
		#print [camera_names[o] for o in order]
		knocked_cameras = np.where(fractions > 0.5)[0]
		if len(knocked_cameras):
			print 'WARNING: cameras were knocked:',[camera_names[o] for o in knocked_cameras]
			print 'with fractions > 0.5 :', fractions[knocked_cameras]

		x2d_threshold = 40./2000.

		true_Ts = [v[4] for v in vicon_mats]

		#test_camera_calibration(vicon_mats, x3ds_frames, x2ds_frames, x2d_threshold, camera_names, tracks_2d, tracks_3d)

		if True: # repair knocked_cameras
			# we would like to repair these cameras
			for ci in knocked_cameras:
				Kin = vicon_mats[ci][0]
				#vicon_mats[ci] = Calibrate.makeUninitialisedMat(ci,vicon_mats[ci][5],K=Kin)
				
				# try to choose a good test frame to evaluate the success of calibration
				count_2ds = np.array([f[1][ci+1]-f[1][ci] for f in x2ds_frames], dtype=np.int32)
				count_3ds = np.array([len(f[0]) for f in x3ds_frames], dtype=np.int32)
				fi = np.argmax(count_2ds * (count_2ds <= count_3ds))
				print 'picked frame',fi,'=orig',fi+frame_slice[0]
				test_x3ds = x3ds_frames[fi][0]
				test_x2ds = x2ds_frames[fi][0][x2ds_frames[fi][1][ci]:x2ds_frames[fi][1][ci+1]]
				cloud = ISCV.HashCloud2D(test_x2ds, x2d_threshold)
				print 'repairing',camera_names[ci]
				# select all the camera tracks more than 10 frames and where it's moved
				camera_tracks_2d = [t[1:] for t in tracks_2d if t[0] == ci and len(t) > 20 and np.sum((t[2] - t[-1])**2) > 0.01**2]
				if not len(camera_tracks_2d):
					print 'no tracks for camera',camera_names[ci]
					continue
				print 'good 2d tracks',len(camera_tracks_2d),map(len,camera_tracks_2d)
				tmp = [np.sum((np.array(t[1:])-t[1])**2,axis=1) for t in camera_tracks_2d]
				order_2d = np.argsort(map(max,tmp))[::-1]
				# just choose the longest track_2d
				best = 1e10,None
				for ti_2d in order_2d[:20]:#[:5]:
					t2d = camera_tracks_2d[ti_2d]
					fin_2d,t2d = t2d[0],np.array(t2d[1:],dtype=np.float32)
					# this 2d track could match any 3d track
					for ti_3d,t3d in enumerate(good_tracks_3d):
						fin_3d,t3d = t3d[0],np.array(t3d[1:],dtype=np.float32)
						sc,(P2,labels) = solve_camera_from_single_point(fin_2d,t2d,fin_3d,t3d,Kin,test_x2ds,test_x3ds,cloud,x2d_threshold)
						if sc < best[0]:
							best = sc,ti_2d,ti_3d,labels,P2
							K,RT = Calibrate.decomposeKRT(P2)
				if best[1] is None: print 'FAILED'; continue
				sc,ti_2d,ti_3d,labels,P2 = best
				K,RT = Calibrate.decomposeKRT(P2)
				print sc,K[0,0],Calibrate.decomposeR(RT[:3,:3]),-np.dot(RT[:3,:3].T,RT[:3,3]),'cf',true_Ts[ci]
				vicon_mats[ci] = Calibrate.makeMat(P2,vicon_mats[ci][3],vicon_mats[ci][5])

	ViconReader.saveXCP(xcp_filename + '+', vicon_mats, x2d_cids)
	
	primitives = QGLViewer.makePrimitives(vertices=[], altVertices=[])
	primitives2D = QGLViewer.makePrimitives2D(p1 = ([],[0]), p2 = ([],[0]))
	#+['boot'+n for n in camera_names]
	QGLViewer.makeViewer(timeRange = (0,len(x2ds_frames)-1), callback=setFrameResection, mats=vicon_mats, camera_ids=camera_names, primitives=primitives, primitives2D=primitives2D, pickCallback=picked)

if __name__=='__main__':

	if False: # test booting
		skelDict = ViconReader.loadVSS(vss_filename)
		skelDict['markerNames'] = [skelPrefix+n for n in skelDict['markerNames']]
		test_boot(c3d_filename, skelDict)
		exit()

	if 0:
		test_resection()
		exit()

	if 0: # test calibration
		test_calibration() 
	
	if False: # test 2D tracking
		test2D(filename)

	if True: # test skeleton tracking
		test_skel_tracking()
