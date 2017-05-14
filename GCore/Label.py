#!/usr/bin/env python

"""
GCore/Label.py

Requires:
	sys
	numpy

	Grip
		ISCV (project, cloud, )
	
"""
import sys
import numpy as np
import ISCV
from GCore import SolveIK, Recon, Retarget, list_of_lists_to_splits, Character

class PushSettings:
	def __init__(self):
		self.visibility = None
		self.x3ds_normals = None
		self.useWeights = False

		self.useVisibility = False
		self.triangleNormals = None
		self.cameraPositions = None
		self.intersect_threshold = 100.
		self.generateNormals = True
		self.generateVisibilityLodsCb = None

		self.numPolishIts = 3
		self.forceRayAgreement = False
		self.x2d_thresholdOverride = None

def project_assign(clouds, x3ds, x3ds_labels, Ps, x2d_threshold):
	"""
	Project 3D points & their labels into all cameras, then assign to 2Ds held in the "clouds" object.
	
	A clouds object results from the ISCV 2D and 3D hashing functions.
	
	Args:
		clouds (ISCV.cloud): Hash cloud of 2D detections... I think.
		x3ds (float[][3]): list of 3D reconstructions.
		x3ds_labels (int[]): list of 3D labellings for x3ds.
		Ps (float[][3][4]): Projection matrices of the cameras.
		x2d_threshold (float): Max 2D distance.
		
	Returns:
		result of clouds.assign - score, x2d_labels, splits ????
		
	Requires:
		ISCV.project
		
	"""
	proj_x2ds, proj_splits, proj_labels = ISCV.project(x3ds, x3ds_labels, Ps)
	ret = clouds.assign(proj_x2ds, proj_splits, proj_labels, x2d_threshold)

	ret2 = clouds.project_assign(x3ds, x3ds_labels, Ps, x2d_threshold)
	assert ret[0] == ret2[0]
	assert np.all(ret[1] == ret2[1])
	assert np.all(ret[2] == ret2[2])
	return ret
	#proj_x2ds, proj_splits, proj_labels = ISCV.project(x3ds, x3ds_labels, Ps)
	#return clouds.assign(proj_x2ds, proj_splits, proj_labels, x2d_threshold)

def project_assign_visibility(clouds, x3ds, x3ds_labels, Ps, x2d_threshold, visibility):
	proj_x2ds, proj_splits, proj_labels = ISCV.project_visibility(x3ds, x3ds_labels, Ps, visibility)
	sc, labels, residuals = clouds.assign(proj_x2ds, proj_splits, proj_labels, x2d_threshold)
	return sc, labels, residuals, proj_x2ds, proj_splits, proj_labels

def match(prev_xs, xs, threshold, prev_has_prediction, labels_out):
	"""
	Match a list of 2d or 3d points against a reference set. Threshold gives the maximum distance between matches.
	Returns the sum-of-squares distance and the assignment (index of prev_xs per item of xs, -1 for unassigned).
	prev_has_prediction is a list of True/False values that indicate whether each point is actively being tracked.'''
		
	Args:
		prev_xs (float[][2|3]): Previous or Reference Detections or Positions.
		xs (float[][2|3]): Detections or Positions under test.
		threshold (float): Max distance to assign.
		prev_has_prediction (bool[]): Tracking status of prev_xs elements
		labels_out (int[]): What's this?
		
	Returns:
		float[][2]: for each xs element under test,
			a list of assignment cost and it's assignment with reference to prev_xs. I think.
			
	Requires:
		ISCV.HashCloud2D
		ISCV.HashCloud3D
		ISCV.min_assignment_sparse
		
	"""
	if prev_xs.shape[-1] == 2: cloud = ISCV.HashCloud2D(prev_xs, threshold)
	else:                      cloud = ISCV.HashCloud3D(prev_xs, threshold)
	scores,matches,matches_splits = cloud.score(xs)
	# double the error (4x the score) for points that have predictions
	if prev_has_prediction is not None: scores[np.where((matches != -1) * prev_has_prediction[matches])] *= 4.0
	sc = ISCV.min_assignment_sparse(scores, matches, matches_splits, threshold**2, labels_out)
	return sc

def label_3d_from_3d(prev_x3ds, prev_labels, prev_vels, x3ds, x3d_threshold):
	"""
	Track 3d points from one frame to the next.
	
	Args:
		prev_x3ds (float[3]): Previous frame's 3D data.
		prev_labels (int[]): Labelling of the previous frame.
		prev_vels (float[][3]): Velocities of previous frame's 3D data.
		x3ds (float[][3]): Current frame's 3D data with no labelling data.
		x3d_threshold (float): Max 3D proximity.
		
	Returns:
		(??): Match Score(s)?
		int[]: labels
		float[]: vels

	"""
	labels = np.zeros(len(x3ds),dtype=np.int32)
	vels = np.zeros_like(x3ds)
	pred_x3ds = prev_x3ds
	prev_has_prediction = None
	if prev_vels is not None:
		pred_x3ds = prev_x3ds + prev_vels
		prev_has_prediction = prev_vels[:,0]!=0
	sc = match(pred_x3ds, x3ds, x3d_threshold, prev_has_prediction, labels)
	# copy the labels from the previous frame, dealing with unlabelled points
	labelled = np.where(labels!=-1)[0]
	if len(labelled) != 0:
		prev_labelled = labels[labelled] # a copy, not a reference to a subset of label
		labels[labelled] = prev_labels[prev_labelled]
		vels[labelled] = x3ds[labelled]-prev_x3ds[prev_labelled]
	return sc,labels,vels

def fps(cache = {}):
	'''Handy function that calculates the fps.'''
	import time
	if not cache.has_key('t'):
		cache['t'] = [time.time()]
	c = cache['t']
	c.append(time.time())
	while len(c) > 3 and c[-1] > c[0]+0.1: c.pop(0) # keep only 1/10th second
	if len(c) > 1000: c.pop(0)
	return '%3.3ffps' % ((len(c)-1)/(c[-1] - c[0] + 1e-10))

def label_2d_from_3ds(u2d_frames, c3d_frames, Ps, x2d_threshold = 6./2000.):
	'''Label a 2d sequence by assigning 3D points to 2D points via projection and thresholding.'''
	u2ds,splits,labels = [],[],[]
	for fi,((u2ds_data,u2ds_splits),c3ds) in enumerate(zip(u2d_frames,c3d_frames)):
		print '\r%d/%d ' % (fi,len(c3d_frames)),fps(),; sys.stdout.flush()
		true_labels = np.int32(np.where(c3ds[:,3]==0)[0])
		x3ds_true = c3ds[true_labels,:3]
		clouds = ISCV.HashCloud2DList(u2ds_data, u2ds_splits, x2d_threshold)
		sc,true_x2ds_labels,x2ds_vels = project_assign(clouds, x3ds_true, true_labels, Ps, x2d_threshold)
		u2ds.append(u2ds_data)
		splits.append(u2ds_splits)
		labels.append(true_x2ds_labels)
		#ix3ds, ilabels, iE, ix2ds = Recon.solve_x3ds(u2ds_data, u2ds_splits, x2ds_labels, Ps)
		#print 'x3ds',np.max(np.sum((x3ds - ix3ds)**2, axis=1))
	u2ds_data,u2ds_splits = list_of_lists_to_splits(u2ds,dtype=np.float32)
	labels_data,labels_splits = list_of_lists_to_splits(labels)
	assert np.all(u2ds_splits == labels_splits)
	splits = np.int32(splits)
	return (u2ds_data,labels_data,u2ds_splits),splits

def extract_label_frame(fi, labels):
	(u2ds_data,labels_data,u2ds_splits),splits = labels
	u0,u1 = u2ds_splits[fi:fi+2]
	return u2ds_data[u0:u1],labels_data[u0:u1],splits[fi]

def label_2d_from_skeleton_and_3ds(clouds, x2ds, splits, effectorLabels, skelDict, effectorData, rootMat, Ps, x2d_threshold,
								   prev_x3ds=None, x3d_threshold=None, settings=None):
	"""
	Based on the pose of the skeleton in skelDict, determine where (approx) the effector targets
	(AKA the Reals, or Markers) would be in order for the skeleton to be in this pose.
	
	Project/Assign these 3D positions into the 2D detections.
	
	If available, use the previous 3D labelling to help.
	
	Args:
		clouds
		x2ds (float[][2]): 2d Detections from all cameras
		splits (int[]): list of camera indices
		effectorLabels (?): For each effector, which label it depends on.
			Joints may be effected by a number of labellings.
		skelDict (GskelDict): The Skeleton to process
		effectorData (?): What's this?
		effectorTargets
		rootMat (float[3][4]): reference frame of the Skeleton.
		Ps (float[][3][4]): Projection matrices of the cameras.
		x2d_threshold (float): Max 2D distance.
		prev_x3ds (float[3]): Previous frame's 3D data. Default = None.
		x3d_threshold (float): Max 3D proximity. Default = None
	
	Returns:
		result of clouds.assign - score, x2d_labels, splits ????
		
	Requires:
		SolveIK.skeleton_marker_positions
		
	"""
	if settings is None: settings = PushSettings() # Default settings
	markerWeights = skelDict['markerWeights'] if settings.useWeights else None
	x3ds, x3ds_labels = SolveIK.skeleton_marker_positions(skelDict, rootMat, skelDict['chanValues'], 
													      effectorLabels, effectorData, markerWeights)
	#x3ds2 = np.array([np.mean(effectors[np.where(effectorLabels==li)[0],:,3],axis=0) for li in labels],dtype=np.float32)
	#assert(np.allclose(x3ds,x3ds2))
	if prev_x3ds is not None:
		_,prev_labels,prev_vels = label_3d_from_3d(x3ds, np.array(range(len(x3ds)),dtype=np.int32), None, prev_x3ds, x3d_threshold)
		keepers = np.where(prev_labels != -1)[0]
		if len(keepers) != 0: x3ds[prev_labels[keepers]] = prev_x3ds[keepers]

	if settings.visibility is not None:
		return clouds.project_assign_visibility(x3ds, x3ds_labels, Ps, x2d_threshold, settings.visibility)
		# sc, labels, residuals, _, _, _ = project_assign_visibility(clouds, x3ds, x3ds_labels, Ps, x2d_threshold, settings.visibility)
		# return sc, labels, residuals

	# if settings.x3ds_normals is not None:
	# 	if settings.useVisibility and 'visibilityLod' in skelDict:
	# 		if settings.generateVisibilityLodsCb: settings.generateVisibilityLodsCb(skelDict)
	# 		lods = skelDict['visibilityLod']
	# 		proj_x2ds, proj_splits, proj_labels = ISCV.project_visibility_normals(x3ds, x3ds_labels, settings.x3ds_normals, Ps, lods['triangles'], settings.cameraPositions,
	# 		                                                                      settings.triangleNormals, settings.intersect_threshold, settings.generateNormals)
	# 		return clouds.assign(proj_x2ds, proj_splits, proj_labels, x2d_threshold)
	# 	else:
	# 		sc, labels, residuals, proj_x2ds, proj_splits, proj_labels = project_assign_visibility(clouds, x3ds, x3ds_labels, Ps, x2d_threshold, settings.visibility)
	# 		return sc, labels, residuals

	return clouds.project_assign(x3ds, x3ds_labels, Ps, x2d_threshold)

def label_3d_from_skeleton(prev_x3ds, effectorLabels, skelDict, effectorData, effectorTargets, rootMat, x3d_threshold):
	"""
	Given a posed skeleton and some 3d points, label the 3d points from the effectors position's.
	
	Args:
		prev_x3ds
		effectorLabels
		skelDict
		effectorData
		effectorTargets
		rootMat
		x3d_threshold
		
	Returns: ( result of label_3d_from_3d )
		(??): Match Score(s)?
		int[]: labels
		float[]: vels
		
	"""
	Character.pose_skeleton(skelDict['Gs'], skelDict, x_mat=rootMat)
	numEffectors = len(effectorTargets)
	effectors    = np.zeros((numEffectors,3,4),dtype=np.float32)
	labels       = np.array(range(numEffectors),dtype=np.int32)
	residual     = np.zeros((numEffectors,3,4),dtype=np.float32)
	sc = ISCV.pose_effectors(effectors, residual, skelDict['Gs'], effectorData[0], effectorData[1], effectorData[2], effectorTargets)
	labels = np.unique(effectorLabels)
	x3ds = np.array([np.mean(effectors[np.where(effectorLabels==li)[0],:,3],axis=0) for li in labels],dtype=np.float32)
	return label_3d_from_3d(x3ds, labels, None, prev_x3ds, x3d_threshold)

def find_T_wand_2d(x2ds, ratio=2.0, x2d_threshold=0.5, straightness_threshold=0.01, match_threshold=0.07):
	"""
	A T-wand is defined to be a 5-point wand where three points are unevenly spaced on a line according to the given ratio;
	and the central of these three points is at one end of a perpendicular line with the other two, which are evenly spaced.
	
	relative_marker_positions = [[160, 0, 0],[0, 0, 0],[-80, 0, 0],[0, 0, -120],[0, 0, -240]] # in mm of idealised wand
	
	Beautiful ASCII art:
		 80mm 160mm
		C---B------A
		    | 120mm
		    |
		    D
		    | 120mm
		    |
		    E
			
	This function hopes to identify this pattern of markers in a given 2D frame.  Each marker is suggested as "B", and this
	theory is tested by assessing the other close markers for their compliance to the ratios 1:2 and 1:1 shown above.
	
	Due to an ambiguity there are sometimes two candidate labels that satisfy the criteria.
	
	Args:
		x2ds (float[][2]): 2D detections on this frame.
		ratio (float): CB-BA Ratio. default = 2.0.
		x2d_threshold (float): 2D max distance. Default = 0.5.
		straightness_threshold (float): Slack factor for wand straightness. Default = 0.01.
		match_threshold (float): Slack factor for acuracey of proposition to truth???. Default = 0.07.
		
	Returns:
		int[][5]: "ret" - List of possible labels for all 2Ds that meet the requirements of a T wand.
		string[]: "ret2" - Description of hypothesis and scoring for this match, lock-step with ret.
	
	Requires:
		ISCV.HashCloud2D
	"""
	ret = []
	ret2 = []
	if len(x2ds) < 5: return ret,ret2 # can't hope to label less than 5 points!
	cloud = ISCV.HashCloud2D(x2ds, x2d_threshold)
	scores,matches,matches_splits = cloud.score(x2ds)
	for xi,(m0,m1) in enumerate(zip(matches_splits[:-1],matches_splits[1:])):
		if m1 < m0+5: continue # must have 5 neighbours (including itself)
		su,mu = scores[m0:m1],matches[m0:m1]
		order = np.argsort(su)[:5]
		ss,ms = su[order]**0.5,mu[order]
		assert(ms[0] == xi) # we assume xi is the central point and ms are the other four points
		xs = x2ds[ms]-x2ds[xi]
		# two of the points should be on a straight line with xi, having the correct ratio and opposite directions
		# the other two point should be on a straight line with xi, having ratio 2.0 and same direction
		# since BC<BA and BD<BE, the closest point is either C or D; and the furthest point is either E or A
		# find which point is most straight with the closest
		perp = np.array([[-xs[1,1]/ss[1]],[xs[1,0]/ss[1]]],dtype=np.float32) # perpendicular unit vector
		tmp = (np.dot(xs[2:],perp).reshape(3)/ss[2:])**2
		straightOrder = np.argsort(tmp)
		if tmp[straightOrder[0]] > straightness_threshold: continue # not straight enough
		if tmp[straightOrder[1]] < straightness_threshold*2: continue # too straight (four points in a row)
		xj = 2+straightOrder[0] # pick the straightest
		sense = np.dot(xs[1],xs[xj])/(ss[1]*ss[xj])
		rat = ss[xj]/ss[1]
		xk,xl = 2+(xj==2),4-(xj==4)
		sense2 = np.dot(xs[xk],xs[xl])/(ss[xk]*ss[xl])
		rat2 = ss[xl]/ss[xk]
		# Due to Necker cube ambiguity, there are two possible answers representing a true camera position and
		# a false camera position which is the true position mirrored in the wand.
		if np.allclose([sense,sense2],[1.0,-1.0],straightness_threshold*2) and np.allclose([rat,rat2],[2.0,ratio],match_threshold):
			ret.append(ms[[xl,0,xk,1,xj]])
			ret2.append('case 1 %f %f %f %f' % (sense,sense2, rat,rat2) + str(tmp[straightOrder]))
		elif np.allclose([sense2,sense],[1.0,-1.0],straightness_threshold*2) and np.allclose([rat2,rat],[2.0,ratio],match_threshold):
			ret.append(ms[[xj,0,1,xk,xl]])
			ret2.append('case 2 %f %f %f %f' % (sense2,sense, rat2,rat) + str(tmp[straightOrder]))
	return ret,ret2

def graph_from_dm(mean_distance, inverse_variance, threshold):
	"""
	Given a matrix of mean distance and inverse variance of that, form a graph from the stiff edges.
	The graph is encoded as a single list: first the labels for each node of the graph, then splits
	of the backlinks, then the backlinks.
	
	Args:
		mean_distance
		inverse_variance
		threshold
		
	Returns: ('graph' as)
		g2l
		graphSplits
		backlinks
		DM
		
	Requires:
	
	"""

	numLabels = mean_distance.shape[0]
	g = np.where(inverse_variance > 1.0/threshold) # W is the inverse variance
	ingraph = np.zeros(numLabels,dtype=np.int32)
	g2l = -np.ones(numLabels,dtype=np.int32)
	l2g = -np.ones(numLabels,dtype=np.int32)
	graphSplits = np.zeros(numLabels+1,dtype=np.int32)

	for gi in xrange(numLabels):
		# ig[li] is 1 where label g[0][li] is not in the graph and label g[1][li] is in the graph
		ig = (1-ingraph[g[0]])*(ingraph[g[1]])

		# counts[li] is 0 (for already-added labels) and (for not-yet-added labels), the number of
		# already-added neighbours PLUS one more than the number of selected edges
		counts = [np.sum(ig * (g[0] == li)) + (1-ingraph[li]) * (1+np.sum(g[0] == li)) for li in range(numLabels)]

		# pick the label that:
		#  (1) is unpicked
		#  (2) has the most already-added neighbours
		#  (3) has the most neighbours
		li = np.argmax(counts)

		assert(l2g[li] == -1)
		l2g[li] = gi
		g2l[gi] = li
		ingraph[li] = 1

	backlinks = [[] for gi in range(numLabels)]
	for li,lj in zip(*g):
		gi = l2g[li]
		gj = l2g[lj]
		if gj < gi: backlinks[gi].append(gj)

	graph = []
	for gi,b in enumerate(backlinks):
		graph.extend(sorted(b))
		graphSplits[gi+1] = len(graph)

	DM = np.zeros((len(graph),2),dtype=np.float32)
	for gi,(b0,b1) in enumerate(zip(graphSplits[:-1],graphSplits[1:])):
		li = g2l[gi]
		for bi in range(b0,b1):
			lj = g2l[graph[bi]]
			DM[bi,0] = mean_distance[li,lj]
			DM[bi,1] = inverse_variance[li,lj]

	backlinks = np.array(graph,dtype=np.int32)
	return g2l, graphSplits, backlinks, DM

def make_dm_graph_from_l3ds(l3ds, ws, threshold):
	"""
	Given 3d points, generate a dm and graph.
	
	Args:
		l3ds (float[numframes][trajectories][3]): 3d data
		ws (int[numframes][trajectories]): "Go/No-Go" metric for given l3ds 0=good, -1=bad.
		threshold
		
	Returns:
		result of "graph_from_dm"
		
	Requires:
		ISCV.dm_from_l3ds
		
	"""
	# limit to 1000 frames for speed - maybe a paramiter?
	stride = max(int(l3ds.shape[0]/1000),1)
	l3ds_tmp = np.array(l3ds[::stride], dtype=np.float32, copy=True, order='C')
	ws_tmp = np.array(ws[::stride], dtype=np.float32, copy=True, order='C')
	numLabels = l3ds_tmp.shape[1]
	M = np.zeros((numLabels,numLabels),dtype=np.float32)
	W = np.zeros((numLabels,numLabels),dtype=np.float32)
	ISCV.dm_from_l3ds(l3ds_tmp, ws_tmp, M, W)
	#print (M[0])
	#print (W[0])
	return graph_from_dm(M, W, threshold)

def graph_from_c3ds(skelDict, c3d_labels, c3d_frames, threshold=50):
	"""
	
	Args:
		skelDict (structure): a skelDict structure for the subject to be graphed
		c3d_labels (string[trajectories]): List of human readable labels
		c3d_frames (float[frames][trajectories][4]): labeled 3d data [x, y, z, good/bad]
		threshold (float): Default = 50.
		
	Return:
		"graph"
	"""
	markerNames = np.unique(skelDict['markerNames'])
	effectorLabels = np.array(sorted([c3d_labels.index(ln) for ln in markerNames if ln in c3d_labels]),dtype=np.int32)
	l3ds = np.array(c3d_frames[:,effectorLabels,:3], dtype=np.float32)
	ws = np.array(1.0 + c3d_frames[:,effectorLabels,3], dtype=np.float32) # 0 = good, -1 = bad
	graph = make_dm_graph_from_l3ds(l3ds, ws, threshold)
	return graph

def find_graph_edges_for_labels(graph, whichLabels):
	numLabels = len(graph[0])
	l2x = -np.ones(numLabels + 1, dtype=np.int32)
	l2x[whichLabels] = np.arange(len(whichLabels))

	edges = set()
	graphSplitPairs = zip(graph[1][:-1], graph[1][1:])

	# Go through graph labels and construct edges from the label to its
	# respective back-links
	for gi, (backlinksStart, backlinksEnd) in enumerate(graphSplitPairs):
		links = graph[2][backlinksStart:backlinksEnd].tolist()
		e0 = l2x[graph[0][gi]]
		if e0 == -1: continue
		for link in links:
			e1 = l2x[graph[0][link]]
			if e1 == -1: continue
			edges.add((e0, e1))

	return np.array(list(edges), dtype=np.int32)

def make_l3ds_from_anim(anim, effectorLabels, skelDict, effectorData, rootMat):
	"""
	Make a structure of labeled 3D positions representing the effector positions needed to solve the given animation
	
	Args:
		anim
		effectorLabels
		skelDict
		effectorData
		rootMat
		
	Returns:
		float[][][3]: "l3ds"
		float[][]: ones
		
	Requires:
		SolveIK.skeleton_marker_positions
		
	"""
	numFrames = len(anim)
	labels = np.unique(effectorLabels)
	numLabels = len(labels)
	l3ds = np.zeros((numFrames, numLabels, 3),dtype=np.float32)
	ws = np.ones((numFrames,numLabels),dtype=np.float32)
	for fi,chans in enumerate(anim):
		l3ds[fi],_ = SolveIK.skeleton_marker_positions(skelDict, rootMat, chans, effectorLabels, effectorData)
	return l3ds, ws

def make_random_anim(skelDict, numFrames=1000):
	"""
	Given a skelDict, synthesise an animation.
	
	Args:
		skelDict (skelDict): "skelDict" defining the skeleton to animate.
		numFrames (int): Duration of animation in frames. Default = 1000.
		
	Returns:
		? "anim": animation on channels of the skeldict.
		
	Requires:
		
	"""
	rootMat = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]],dtype=np.float32)
	jointChans = skelDict['jointChans'] # 0,1,2 = translation; 3,4,5 = rotation
	numDofs = len(jointChans)
	scales = np.ones(numDofs,dtype=np.float32)
	scales[np.where(jointChans < 3)]  *= 10.0 # 10mm of translation
	scales[np.where(jointChans >= 3)] *= 0.2 # 20 degrees of rotation
	anim = np.array(np.random.randn(numFrames, numDofs),dtype=np.float32) * scales
	return anim

def graph_from_skel(skelDict, c3d_labels, threshold=50):
	"""
		
	Args:
		skelDict
		c3d_labels
		threshold = 50
		
	Returns:
		graph
		
	Requires:
		make_effectorData
		
	"""
	markerNames = skelDict['markerNames']
	effectorLabels = np.array([c3d_labels.index(ln) if ln in c3d_labels else -1 for ln in markerNames],dtype=np.int32)
	effectorData = SolveIK.make_effectorData(skelDict)
	rootMat = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]],dtype=np.float32)
	anim = make_random_anim(skelDict)
	l3ds,ws = make_l3ds_from_anim(anim, effectorLabels, skelDict, effectorData, rootMat)
	l3ds += np.random.randn(*l3ds.shape)*1.0
	graph = make_dm_graph_from_l3ds(l3ds, ws, threshold)
	return graph

def label_x3ds_from_x2ds_labels(x3ds, x2ds_labels, clouds, Ps, x2d_threshold):
	import collections
	# some of the x2ds_labels are wrong; others are correct
	# we have a list of all the 3d points
	# we want to find rays that DON'T intersect their 3d point and unlabel them
	num_x3ds = len(x3ds)
	x2ds_labels_2 = clouds.project_assign(x3ds, np.arange(num_x3ds, dtype=np.int32), Ps, x2d_threshold)[1]

	T = {}
	# for each label in x2ds_labels_2 [ ie range(num_x3ds) ], make a list of all the corresponding detections
	for (li, li2) in zip(x2ds_labels, x2ds_labels_2):
		if li != -1: T.setdefault(li2, []).append(li)

	x3ds_labels = np.arange(num_x3ds, dtype=np.int32)
	for xi in range(num_x3ds):
		if xi in T:
			common = collections.Counter(T[xi]).most_common(3)
			li, numHits = common[0]
			if numHits < 2: li = -1
		else:
			li = -1

		x3ds_labels[xi] = li

	# TODO there is a bug here; two x3ds could be assigned the same label; an x3d could be assigned a label because only one of its rays agreed with a label and all the others missed
	return x3ds_labels

def test(x3ds, x2ds_labels, clouds, Ps, x2d_threshold):
	import collections
	# some of the x2ds_labels are wrong; others are correct
	# we have a list of all the 3d points
	# we want to find rays that DON'T intersect their 3d point and unlabel them
	num_x3ds = len(x3ds)
	x2ds_labels_2 = clouds.project_assign(x3ds, np.arange(num_x3ds, dtype=np.int32), Ps, x2d_threshold)[1]

	T = {}
	# for each label in x2ds_labels_2 [ ie range(num_x3ds) ], make a list of all the corresponding detections
	for (li, li2) in zip(x2ds_labels, x2ds_labels_2):
		if li != -1: T.setdefault(li2, []).append(li)

	# x3ds_labels = np.arange(num_x3ds, dtype=np.int32)
	for xi in range(num_x3ds):
		if xi in T:
			common = collections.Counter(T[xi]).most_common(3)
			li, numHits = common[0]
			if numHits < 3 and len(common) > 1: li = -1
		else:
			np.where(x2ds_labels == xi)
			li = -1

		# x3ds_labels[xi] = li

	# TODO there is a bug here; two x3ds could be assigned the same label; an x3d could be assigned a label because only one of its rays agreed with a label and all the others missed
	# return x3ds_labels


class TrackModel:
	"""
	Model-based tracking object.
	
	Requires:
		numpy
		Grip ISCV (HashCloud2DList, HashCloud2DList, update_vels)
		SolveIK (make_effectorData, solve_skeleton_from_2d_bake, solve_skeleton_from_3d_and_single_rays)
		Labelling (label_2d_from_skeleton_and_3ds, )
		
	Attributes:
		numCameras (int): Number of cameras in system
		skelDict (GskelDict): The skeleton to track in Grip "skelDict" format
		mats (GcameraMat[]): Matrices of the cameras (P, KRT, RT, (k1, k2), (w, h) ???
		effectorLabels (?): What's this?
		x2d_threshold (float): 2D max distance for hash.
		pred_2d_threshold (float): 2D prediction max distance.
		x3d_threshold (int): What's this?
		x2ds (float[]): 2D detections on the frame being processed.
		vels (float[]): velocities corresponding to 2D detections.
		splits (int[]): splits into x2ds, vels.
		labels (int[]): labels of x2ds, in x2d order.
		frame (int): Frame count since boot.
		Ps (float[][3][4]): List of the camera's Projection matrices as 3x4 mats.
		rootMat (float[3][4]): Skeleton's reference mat (?)
		effectorData (?): What's this?
		effectorTargets (?): What's this?
	"""
	
	def __init__(self, skelDict, effectorLabels, mats, x2d_threshold=25./2000, pred_2d_threshold=50./2000, x3d_threshold=30):
		"""
		Initialise the tracking module with the skeleton to track, and the camera matrices
		
		Args:
			skelDict (GskelDict): skeletal model to track
			effectorLabels (?): What's this?
			mats (GcameraMat[]): Matrices of the cameras in the system.
			x2d_threshold (float): 2D max distance for hash. Default = 25./2000.
			pred_2d_threshold (float): 2D prediction max distance. Default = 50./2000.
			x3d_threshold (int): What's this? Default = 30
			
		Requires:
			make_effectorData
			
		"""
		self.numCameras = len(mats)
		self.skelDict = skelDict
		self.mats = mats
		self.effectorLabels = effectorLabels
		self.x2d_threshold = x2d_threshold
		self.pred_2d_threshold = pred_2d_threshold
		self.x3d_threshold = x3d_threshold
		self.x2ds = np.zeros((0,2),dtype=np.float32)
		self.vels = np.zeros((0,2),dtype=np.float32)
		self.splits = np.zeros(self.numCameras+1,dtype=np.int32)
		self.labels = np.zeros(0,dtype=np.int32)
		self.frame = 0
		self.Ps = np.array([m[2]/(np.sum(m[2][0,:3]**2)**0.5) for m in mats],dtype=np.float32)
		self.rootMat = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]],dtype=np.float32)
		self.effectorData = SolveIK.make_effectorData(self.skelDict)
		self.effectorTargets = np.zeros_like(self.effectorData[1])
		self.x3ds = np.zeros((0, 3), dtype=np.float32)
		self.x3d_labels = np.zeros(0, dtype=np.int32)

		self.settings = PushSettings()
		self.settings.numPolishIts = 1
		self.settings.forceRayAgreement = True
		self.track3d = None #Track3D(mats, 6./2000., 100./2000., 30., boot_interval=1)

	def bootLabels(self, x2ds, splits):
		"""
		Initialise the labels from the pose.
		
		Args:
			x2ds (float[]): 2D detections in "splits" structure.
			splits (int[]): index of each camera's portion of the x2ds.
			
		Returns:
			float: ?? score of skeleton's labelling.
			
		Requires:
			ISCV.HashCloud2DList
		"""
		# if self.track3d is not None: self.track3d.boot(x2ds, splits, settings=self.settings)
		self.x2ds,self.splits = x2ds,splits
		self.vels = np.zeros_like(self.x2ds)
		clouds = ISCV.HashCloud2DList(x2ds, splits, self.x2d_threshold)
		sc,self.labels,_ = label_2d_from_skeleton_and_3ds(clouds, self.x2ds, self.splits, self.effectorLabels, self.skelDict, self.effectorData, self.rootMat, self.Ps, self.x2d_threshold)
		return sc

	def bootPose(self, x2ds, splits, x2d_labels, its=5):
		"""
		Initialise the pose from labels.
		
		Args:
			x2ds (float[]): 2D detections in "splits" structure.
			splits (int[]): index of each camera's portion of the x2ds.
			x2d_labels (int[]): label of each 2D detection - same order as x2ds.
			its (int): IK iterations to solve the skeleton for Booting. Default = 5.
			
		Returns:
			float: ?? score of skeleton's labelling.
		
		Requires:
			SolveIK.solve_skeleton_from_2d_bake
		"""
		chan_values = self.skelDict['chanValues']
		jps,jcs = self.skelDict['jointParents'],self.skelDict['jointChanSplits']
		ring = -np.ones(chan_values.shape[0],dtype=np.int32)
		depth = -np.ones(jps.shape[0],dtype=np.int32)
		for ji,jp in enumerate(jps): depth[ji] = depth[jp]+1 # NB this works because ring[-1]=-1
		for (jp,j0,j1) in zip(jps,jcs[:-1],jcs[1:]):
			ring[j0:j1] = depth[jp]
		for it in range(np.max(ring)):
			self.skelDict['chanValues'][np.where(ring >= it)] = 0.0 # reset lower joints
			SolveIK.solve_skeleton_from_2d_bake(x2ds, splits, x2d_labels, self.effectorLabels, self.Ps, self.skelDict, self.effectorData, self.rootMat, outerIts=its)
		return self.bootLabels(x2ds,splits)

	def push(self, x2ds, splits, its=1, settings=None):
		"""
		Update the model with the latest set of 2D detections, solving the skeleton.
		
		Args:
			x2ds (float[]): undistorted 2D detections per camera
			splits (int[]): index of start of each camera's x2ds
			its (int): IK iterations to solve the skeleton for tracking. Default = 1.
			
		Returns:
			float[]: Resulting skeletal pose expressed as skelDict['chanValues'].
		
		Requires:
			ISCV.HashCloud2DList
			ISCV.update_vels
			SolveIK.bake_ball_joints
			SolveIK.solve_skeleton_from_2d
			SolveIK.unbake_ball_joints
		"""
		
		assert(len(splits) == self.numCameras+1)
		if settings is None: settings = self.settings
		x2d_threshold = settings.x2d_thresholdOverride if settings.x2d_thresholdOverride else self.x2d_threshold
		clouds = ISCV.HashCloud2DList(x2ds, splits, max(self.pred_2d_threshold, x2d_threshold))
		SolveIK.bake_ball_joints(self.skelDict)
		sc,labels,vels = clouds.assign_with_vel(self.x2ds, self.vels, self.splits, self.labels, self.pred_2d_threshold)

		if self.track3d is not None:
			# Some of the labels here for the same 3d point don't intersect; we try to enforce consensus from tracked 3d points
			# x3ds, _ = self.track3d.push(x2ds, splits, settings=self.settings)
			x3ds = self.track3d.x3ds
			threshold = x2d_threshold
			x3ds_labels = label_x3ds_from_x2ds_labels(x3ds, labels, clouds, self.Ps, threshold)
			x2ds_labels_2 = clouds.project_assign(x3ds, x3ds_labels, self.Ps, threshold)[1]
			_x3ds, _x3d_labels, _, singleRayLabels = Recon.solve_x3ds(x2ds, splits, labels, self.Ps)
			singleRayLabels = set(singleRayLabels)
			whichLabels = (labels != -1)
			exclude = [labelIndex for labelIndex, label in enumerate(labels) if label in singleRayLabels]
			whichLabels[exclude] = 0
			labels[whichLabels] = x2ds_labels_2[whichLabels]
			labels = x2ds_labels_2

		inner_iterations = 1
		for it in range(its):
			self.x3ds, self.x3d_labels, E, x2d_labels = \
				SolveIK.solve_skeleton_from_2d(x2ds, splits, labels, self.effectorLabels, self.Ps, self.skelDict,
											   self.effectorData, self.rootMat, outerIts=inner_iterations)

			sc, labels, residuals = \
				label_2d_from_skeleton_and_3ds(clouds, x2ds, splits, self.effectorLabels, self.skelDict, self.effectorData,
											   self.rootMat, self.Ps, x2d_threshold, self.x3ds, self.x3d_threshold, settings)

		ISCV.update_vels(x2ds,splits,labels,self.x2ds,self.splits,self.labels,vels)
		self.x2ds,self.splits,self.labels,self.vels = x2ds,splits,labels,vels
		SolveIK.unbake_ball_joints(self.skelDict)
		self.frame += 1
		return self.skelDict['chanValues']

	def rebuildEffectorData(self, skelDict, effectorLabels):
		self.skelDict = skelDict
		self.effectorLabels = effectorLabels
		self.effectorData = SolveIK.make_effectorData(self.skelDict)
		self.effectorTargets = np.zeros_like(self.effectorData[1])


class TrackGraph:
	"""
	Graph-based tracking object. Given a sequence of x3ds, we can...
	  1) find relationships (stiff distances) between pairs of markers
	  2) label the markers more consistently (eg deal with flicker)
	  3) generate a graph for display purposes.
	
	Requires:
		numpy
		Grip ISCV (HashCloud3D)
		
	Attributes:
		x3d_threshold (int): What's this?
		nearestN (int): number of nearest Neighbours to assess
		frame (int): Frame count since boot.
		x3ds (float[]): 3D reconstructions
		x3ds_labels (int[]): Labels of x2ds, in x2d order.
		graph (graphStructure[]): Graph of closest stiffest markers
			Graph structure is:
			[ label_index, means, variance, counts, stiffies ]
			[ int,         float, float,     int,    float    ]

	"""
	
	def __init__(self, x3d_threshold=300., nearestN=4):
		"""
		Initialise the graph tracker.
		
		Args:
			x3d_threshold (float): Max 3D distance (hash). Default = 300.0.
			nearestN (int): number of nearest Neighbours to assess. Default = 4.	
		"""
		self.x3d_threshold = x3d_threshold
		self.nearestN = nearestN
		self.frame = 0
		self.x3ds = np.zeros((0, 3), dtype=np.float32)
		self.x3ds_vels = np.zeros((0, 3), dtype=np.float32)
		self.x3ds_labels = np.zeros(0, dtype=np.int32)
		self.graph = {}

		self.x3ds_data = {}
		self.vels_data = {}

	def drawing_graph(self, threshold=1e10):
		"""
		Get pairs of x3d indices representing stiff and close 3D points in the Marker cloud.
		
		Args:
			threshold (float): Stiffness threshold Default = 1e10.
			
		Returns:
			int[]: Unique pairs of indices into the current x3ds that satisfy the stiff/close requirements.
			
		"""
		ret = set()
		ltox = -np.ones(max(len(self.graph)-1, np.max(list(self.x3ds_labels)+[0])) + 1, dtype=np.int32)
		ltox[self.x3ds_labels] = np.arange(len(self.x3ds_labels))
		ltox[-1] = -1
		for li in self.x3ds_labels:
			if li not in self.graph: continue
			g = self.graph[li]
			if li >= len(ltox): break
			xi = ltox[li]
			if xi == -1: continue
			for lj, mean, var, count, stiff, _ in zip(*g):
				if lj != -1 and lj < len(ltox) and stiff < threshold:
					xj = ltox[lj]
					if xj != -1:
						ret.add((min(xi, xj), max(xi, xj)))

		return np.array(list(ret), dtype=np.int32)

	def calculateGraphPropertiesForLabel(self, currIndex, li, x3d, m0, m1, matches, x3ds, x3ds_labels, x3ds_joints=None):
		if li in self.graph: g = self.graph[li]
		else:
			g = [
				-np.ones(self.nearestN, dtype=np.int32),            # Neighbour labels
				np.zeros(self.nearestN, dtype=np.float32),          # Mean distances
				np.zeros(self.nearestN, dtype=np.float32),          # Variance
				np.zeros(self.nearestN, dtype=np.int32),            # Count
				1e10 * np.ones(self.nearestN, dtype=np.float32),    # Stiffness
				np.zeros((self.nearestN, 3), dtype=np.float32)      # Direction vectors
			]

		ljs = x3ds_labels[matches[m0:m1]]
		ds = np.sum((x3ds[matches[m0:m1]] - x3d)**2, axis=1)**0.5
		dirs = x3ds[matches[m0:m1]] - x3d
		# keep g sorted by stiffest-closest = mean * variance
		stiffies = ds**3
		order = np.argsort(stiffies)
		ds = ds[order]
		ljs = ljs[order]

		if x3ds_joints is not None:
			joints_js = x3ds_joints[matches[m0:m1]]
		else:
			joints_js = None

		for j, (d, lj, dir) in enumerate(zip(ds, ljs, dirs)): # in order of stiffest-closest
			if lj == -1 or lj == li: continue
			if x3ds_joints is not None:
				joint_i = x3ds_joints[currIndex]
				joint_j= joints_js[j]
				if (joint_i != -1 or joint_j != -1) and (joint_i != joint_j): continue

			if lj in list(g[0]):
				gi = list(g[0]).index(lj)
				g[1][gi] += d
				g[2][gi] += d*d
				g[3][gi] += 1
				mean = g[1][gi]/g[3][gi]
				var = g[2][gi]/g[3][gi] - mean**2
				g[4][gi] = mean * var
				g[5][gi] = (g[5][gi] + dir) / g[3][gi]
				#g[5] = self.frame
			else: # add this measurement to the graph; get an insert index
				if np.any(g[0] == -1): # if there's an empty slot, fill it first
					ii = list(g[0]).index(-1)
				else: # replace the worst slot, if we're better
					order = np.argsort(g[4])
					ii = order[-1]
				if g[4][ii] > d**3:
					g[0][ii] = lj
					g[1][ii] = d
					g[2][ii] = d*d + 10*10
					g[3][ii] = 1
					g[4][ii] = d*(10*10)
					g[5][ii] = dir
					#g[5] = self.frame

		return g

	def push(self, x3ds, x3ds_labels, updateGraph=True, x3ds_joints=None):
		"""
		Update the model

		Args:
			x3ds (float[]): 3D reconstructions.
			x3ds_labels (int[]): Labels of 3D reconstructions, in order.

		Require:
			ISCV.HashCloud3D
		"""
		cloud = ISCV.HashCloud3D(x3ds, self.x3d_threshold)
		scores, matches, matches_splits = cloud.score(x3ds)

		_, labels, vels = label_3d_from_3d(self.x3ds, self.x3ds_labels, self.x3ds_vels, x3ds, self.x3d_threshold)
		matchingLabels = np.where(labels != -1)[0]
		if len(matchingLabels) != 0:
			x3ds_labels[matchingLabels] = labels[matchingLabels]

		# For each labelled point, we're going to keep track of the nearest N=6 neighbours
		labelUpdates = {}
		lastGraph = self.graph.copy()
		if updateGraph:
			for li, (lbl, x3d, m0, m1) in enumerate(zip(x3ds_labels, x3ds, matches_splits[:-1], matches_splits[1:])):
				if lbl == -1: continue # wtf?
				if lbl not in lastGraph:
					# print 'Generating graph for:', li
					# graphLabel: [ lis, means, vars, counts, stiffies ]
					graphLabel = self.calculateGraphPropertiesForLabel(li, lbl, x3d, m0, m1, matches, x3ds, x3ds_labels, x3ds_joints)
					useLabel, minDist, numShared = -1, np.inf, 0

					# Check if a long lost friend has reappeared in which case we prefer our old label to the new label
					# Our criteria for relabelling a new label is that it shares at least X neighbours and that their
					# relative location is roughly the same
					_buddies, _means, _vars, _, _, _offsets = graphLabel
					for label, g in lastGraph.iteritems():
						if len(np.where(x3ds_labels == label)[0]) != 0: continue
						sharedLabels = np.intersect1d(g[0], _buddies)
						shared = np.where(sharedLabels != -1)[0]
						if len(shared) >= 3 and len(shared) >= numShared:
							whichLabelsG = np.where(sharedLabels[shared] == g[0].reshape(-1, 1))[0]
							whichLabelsNew = np.where(sharedLabels[shared] == _buddies.reshape(-1, 1))[0]
							meanDist = np.mean(np.sqrt((g[1][whichLabelsG] - _means[whichLabelsNew])**2))
							if meanDist < 5. and meanDist < minDist:
								useLabel = label
								minDist = meanDist
								numShared = len(shared)

					if useLabel != -1:
						self.graph[useLabel] = graphLabel
						labelUpdates[li] = useLabel
					else:
						self.graph[lbl] = graphLabel

		for li, lblTo in labelUpdates.iteritems():
			# print 'Change 3D label from %d to %d' % (x3ds_labels[li], lblTo)
			x3ds_labels[li] = lblTo

		self.frame += 1
		self.x3ds, self.x3ds_labels = x3ds, x3ds_labels
		self.x3ds_vels = vels


class Track3D:
	"""
	3D point tracking object
	
	Requires:
		numpy
		Grip ISCV (HashCloud2DList, update_vels)
		
	Attributes:
		numCameras (int): number of cameras in system.
		mats  (GcameraMat[]): list of matrices of the cameras in the system.
		x2d_threshold (float): 2D max distance.
		pred_2d_threshold (float): What's this?
		x3d_threshold  (int): What's this?
		x2ds (float[][2]): 2D detections on frame.
		vels (float[][2]): 2D velocities of the x2ds.
		splits (int[]): Camera Splits, index of ranges of x2ds for each camera.
		x2ds_labels (int[]): labels of the x2ds, in order.
		frame (int): count of frames since object booted.
		Ps (float[][3][4]): Projection matrices of the cameras.
		x3ds (float[][3]): Resulting 3D points.
		x3ds_vels (float[][3]): Velocities of x3ds.
		x3ds_labels (int[]): Labels of x3ds.
		next_id (int): Next ID assignable to a 3D reconstruction
		tracks (float[][[3]]): list of track IDs containing a "tuple" of frame No, x3d for which that ID has data.
		min_rays (int): Min number of rays to intersect to create 3D reconstruction.
		tilt_threshold (float): Slack factor for Tilt-based paring.
		boot_interval (int): frequency of booting attempts, in frames.

	"""
	def __init__(self, mats, x2d_threshold = 6./2000, pred_2d_threshold = 100./2000, x3d_threshold = 30, tilt_threshold = 0.0002,
	             min_rays = 3, boot_interval = 10):
		"""
		Initialise Point tracking object.
		
		Args:
			mats (GcameraMat[]): Matrices of the cameras.
			x2d_threshold (float): 2D max distance. Default = 6./2000.
			pred_2d_threshold (float): What's this? Default = 100./2000.
			x3d_threshold (int): What's this? Default = 30.
			tilt_threshold (float): Slack factor for Tilt paring. Default = 0.0002.
			min_rays (int): Min number of rays required for 3D. Default = 3.
			boot_interval (int): Frequency of boot attempts. Default = 10.
		"""
		self.numCameras = len(mats)
		self.mats = mats
		self.x2d_threshold = x2d_threshold
		self.pred_2d_threshold = pred_2d_threshold
		self.x3d_threshold = x3d_threshold
		self.x2ds = np.zeros((0,2),dtype=np.float32)
		self.vels = np.zeros((0,2),dtype=np.float32)
		self.splits = np.zeros(self.numCameras+1,dtype=np.int32)
		self.x2ds_labels = np.zeros(0,dtype=np.int32)
		self.frame = 0
		self.Ps = np.array([m[2]/(np.sum(m[2][0,:3]**2)**0.5) for m in mats],dtype=np.float32)
		self.x3ds = np.zeros((0,3),dtype=np.float32)
		self.x3ds_vels = np.zeros((0,3),dtype=np.float32)
		self.x3ds_labels = np.zeros(0,dtype=np.int32)
		self.prev_x3ds = np.zeros((0,3),dtype=np.float32)
		self.next_id = 0
		self.tracks = {}
		self.min_rays = min_rays
		self.tilt_threshold = tilt_threshold
		self.boot_interval = boot_interval

		self.tracks_lastSeen = {}
		self.tracks_vels = {}
		self.predictedLabels3d = []
		self.labelFrom3ds = True

	def fin(self):
		"""
		Finalise a frame by updating tracks
		
		Returns:
			float[][3]: The resulting x3ds
			int[]: Labels of the x3ds
		"""
		for li,x3d in zip(self.x3ds_labels,self.x3ds):
			if li != -1:
				self.tracks.setdefault(li,[self.frame]).append(x3d)
				if li not in self.predictedLabels3d: self.tracks_lastSeen[li] = self.frame
		self.frame += 1
		return self.x3ds, self.x2ds_labels

	def info(self):
		'''Give a return value similar to solve_x3ds(...)'''
		return self.x3ds, self.x3ds_labels, self.E, self.E_labels

	def boot(self, x2ds, splits, settings=None):
		"""
		Boot the tracker with this frame's x2d splits
		
		Args:
			x2ds (float[][2]): 2d Detections from all cameras
			splits (int[]): list of camera indices
			
		Returns:
			float[][3]: The resulting x3ds
			int[]: Labels of the x3ds
			
		Requires:
			Recon.intersect_rays
		"""
		self.next_id = 0
		self.tracks = {}
		if settings is None: settings = PushSettings()
		self.x2ds, self.splits = x2ds.copy(),splits.copy()
		self.vels = np.zeros_like(self.x2ds,dtype=np.float32)
		self.x3ds, self.x3ds_labels, self.E, self.E_labels = Recon.intersect_rays(self.x2ds, self.splits, self.Ps, self.mats, tilt_threshold=self.tilt_threshold,
		                                                   x2d_threshold=self.x2d_threshold, x3d_threshold=self.x3d_threshold,
		                                                   min_rays=self.min_rays, visibility=settings.visibility)
		self.x3ds_labels = np.arange(self.next_id,self.next_id+self.x3ds.shape[0],dtype=np.int32)
		clouds = ISCV.HashCloud2DList(self.x2ds, self.splits, max(self.pred_2d_threshold, self.x2d_threshold))
		sc2, self.x2ds_labels, _ = 	clouds.project_assign(self.x3ds, self.x3ds_labels, self.Ps, self.x2d_threshold)
		self.x3ds_vels = np.zeros_like(self.x3ds,dtype=np.float32)
		self.next_id += len(self.x3ds)
		return self.fin()

	def push(self, x2ds, splits, its=1, settings=None):
		"""
		Update the tracker with this frame's x2d splits
		
		Args:
			x2ds (float[][2]): 2d Detections from all cameras
			splits (int[]): list of camera indices
			its (int): Number of iterations of a function (not used in this method). Default = 1.
			
		Returns:
			float[][3]: The resulting x3ds
			int[]: Labels of the x3ds
			
		Requirements:
			ISCV.HashCloud2DList
			ISCV.update_vels
			ISCV.solve_x3ds
			Recon.intersect_rays
			ISCV.solve_x3ds
			
		"""
		# assert(len(splits) == self.numCameras+1)
		if len(splits) != self.numCameras + 1:
			print 'Track3D:', len(splits), '!=', self.numCameras + 1
			return

		clouds = ISCV.HashCloud2DList(x2ds, splits, max(self.pred_2d_threshold, self.x2d_threshold))
		sc, labels, vels = clouds.assign_with_vel(self.x2ds, self.vels, self.splits, self.x2ds_labels, self.pred_2d_threshold)
		x3ds, x3ds_labels, E, x2d_labels = ISCV.solve_x3ds_rays(x2ds, splits, labels, self.Ps, True, self.min_rays)

		# pick up new tracking points by intersecting unused rays
		if settings is None: settings = PushSettings()
		if (self.frame % self.boot_interval) == 0:
			extra_x3ds, extra_x3ds_labels, _, _ = Recon.intersect_rays(x2ds, splits, self.Ps, self.mats, seed_x3ds=x3ds, tilt_threshold=self.tilt_threshold,
			                                                     x2d_threshold=self.x2d_threshold, x3d_threshold=self.x3d_threshold, min_rays=self.min_rays,
			                                                     numPolishIts=settings.numPolishIts, forceRayAgreement=settings.forceRayAgreement,
			                                                     visibility=settings.visibility)
			sc2, extra_x2ds_labels, _ = clouds.project_assign(extra_x3ds, extra_x3ds_labels, self.Ps, self.x2d_threshold)

			assert len(extra_x2ds_labels) == len(labels),'What?'+repr(extra_x2ds_labels.shape)+','+repr(labels.shape)

			new_labels = np.where(extra_x2ds_labels != -1)[0]
			if len(new_labels):
				labels[new_labels] = extra_x2ds_labels[new_labels] + self.next_id
				self.next_id += np.max(extra_x2ds_labels)+1
				x3ds, x3ds_labels, E, x2d_labels = ISCV.solve_x3ds_rays(x2ds, splits, labels, self.Ps, True, self.min_rays)

		if self.labelFrom3ds:
			# So, we might have picked up some new points from unused rays and given them new labels.
			_, _labels, _vels = label_3d_from_3d(self.x3ds, self.x3ds_labels, self.x3ds_vels, x3ds, self.x3d_threshold)
			matchedLabels = np.where(_labels != -1)[0]
			if len(matchedLabels) != 0:
				x3ds_labels[matchedLabels] = _labels[matchedLabels]
			self.x3ds_vels = _vels

		if settings.visibility is not None:
			sc2, labels, _ = clouds.project_assign_visibility(x3ds, x3ds_labels, self.Ps, self.x2d_threshold, settings.visibility)
		else:
			sc2, labels, _ = clouds.project_assign(x3ds, x3ds_labels, self.Ps, self.x2d_threshold)

		self.x3ds, self.x3ds_labels, self.E, self.x2d_labels = x3ds, x3ds_labels, E, x2d_labels

		# Update detections and their velocity
		ISCV.update_vels(x2ds,splits,labels,self.x2ds,self.splits,self.x2ds_labels,vels)
		self.x2ds,self.splits,self.x2ds_labels,self.vels = x2ds.copy(),splits.copy(),labels,vels

		# Update 3D tracks
		return self.fin()


class Track2D:
	"""
	2D detection tracking object
	
	Requires:
		numpy
		Grip ISCV (HashCloud2DList)
		
	Attributes:
		numCameras (int): number of cameras in system.
		x2d_threshold (float): 2D max distance.
		x2ds (float[][2]): List of 2D detections
		vels (float[][2]): Velocities of the detections
		splits (int[]): Indexes into x2ds/vels representing slice produced by a given camera
		labels (int[]): Assigned labels of the x2ds, in order
		tracks ([][]): list of tracked detections headed with camera and frame number for every label.
		frame (int): Count of frames since boot.
	"""
	def __init__(self, numCameras, x2d_threshold=6./2000):
		"""
		Initialise 2D Detection tracking object.
		
		Args:
			numCameras (int): Number of cameras.
			x2d_threshold (float): 2D max distance. Default = 6./2000.
		"""
		self.numCameras = numCameras
		self.x2d_threshold = x2d_threshold
		self.x2ds = np.zeros((0,2),dtype=np.float32)
		self.vels = np.zeros((0,2),dtype=np.float32)
		self.splits = np.zeros(numCameras+1,dtype=np.int32)
		self.labels = np.zeros(0,dtype=np.int32)
		self.tracks = []
		self.frame = 0

	def push(self, x2ds, splits):
		"""
		Update the tracker with this frame's x2d splits
		
		Args:
			x2ds (float[][2]): 2d Detections from all cameras
			splits (int[]): list of camera indices
			
		Requires:
			ISCV.HashCloud2DList
		"""
		assert(len(splits) == self.numCameras+1)
		clouds = ISCV.HashCloud2DList(x2ds, splits, self.x2d_threshold)
		sc2,labels2,vels2 = clouds.assign_with_vel(self.x2ds, self.vels, self.splits, self.labels, self.x2d_threshold)
		self.x2ds,self.vels,self.splits,self.labels = x2ds,vels2,splits,labels2
		for ci,(c0,c1) in enumerate(zip(splits[:-1],splits[1:])):
			for di,(x2d,li) in enumerate(zip(self.x2ds[c0:c1],self.labels[c0:c1])):
				if li == -1:
					li = len(self.tracks)
					self.labels[c0+di] = li
					self.tracks.append([ci, self.frame])
				self.tracks[li].append(x2d)
		self.frame += 1

