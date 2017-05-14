#!/usr/bin/env python

import math

import numpy as np
import Character
import ISCV
import IO

'''
Given a source skeleton S = { SboneNames_i, Sdofs_i_j, Sdoftypes_i_j, Sparents_i, SJoints_i }, and a target skeleton T


The local matrix for a joint i is SL_i = prod_j { Sdoftypes_i_j(Sdofs_i_j) }
The global matrix for a joint i is SG_i = SG_Sparents_i * SJoints_i * SL_i, SG_Sparents_root = I
There are two constraint types: global position & orientation


1) Copy the joint angles from the source to the target. Take into account differences in rotation order (make the local matrices agree as best as possible).
2) For each end effector, set a position constraint and solve to the root [or spine] (allow stiffness controls for each dof to direct the solver, default is equal stiffness).
3) The position of the target constraint can be offset by various user-controllable variables and expressions.
4) The variables can be dialled interactively at capture time.
5) The target space can be scaled, offset, rotated or otherwise deformed [height map, path deformer] relative to the source space.
6) Changes to the variables should be timestamped (ie keys), so that the effect can be recorded and to facilitate ease in/ease out
7) Undo button!
8) Add policing
9) ...
10) Profit!

copyJoint(srcJoint, tgtJoint)
solveToRoot(srcJoint, pos = tgtPos, orient = tgtOrient, weights = {dof:weight})
setTargetScale(s)
setTargetOffset(tx,ty,tz)
setTargetOrientation(Rzxy(rx,ry,rz))
Rzxy(rx,ry,rz) = rotation matrix
getPos(srcJoint, offset = None)
getOrient(srcJoint, offset = None)
easeIn(variable, expression, offset, frameCount, curve = lambda x:0.5*(1-cos(x*pi))) = expression(a*variable(lastKey) + (1-a)*variable(t))+offset, where a = curve(clip((lastKey+frameCount-t)/frameCount))
'''

def findChildBone(skel_dict, ji):
	# Finds the first bone with a parent being the given joint
	# TODO this is mostly wrong. should find a better way to do this
	for ci, parent in enumerate(skel_dict['jointParents']):
		if parent == ji:
			vec = skel_dict['Ls'][ci][:,3]
			norm = np.linalg.norm(vec)
			return vec / norm if norm > 0 else None
	return None

def rotateBetweenVectors(A,B, eps=1e-5):
	# Returns a rotation matrix of A onto B
	# http://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/476311#476311
	def skewSymmetricCross(v):
		return np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]], dtype=v.dtype)
	a = A / np.linalg.norm(A)
	b = B / np.linalg.norm(B)
	d_type = A.dtype
	v = np.cross(a,b)
	v_norm = np.linalg.norm(v)
	if v_norm < eps:
		i = np.argmax(np.abs(a))
		return np.eye(3) * np.sign(a[i]) * np.sign(b[i])
	v_x = skewSymmetricCross(v)
	return np.eye(3, dtype=d_type) + v_x + np.dot(v_x,v_x) * (1 - np.dot(a,b.T)) / (v_norm**2)

def copyJointPose(rtg_S, rtg_T, s, t, offsets=None):
	'''
	Inputs: Source Skeleton, Target Skeleton, Source Joint Name, Target Joint Name
	Optional Input: current joint channels to pose from

	Returns: Updated Joint Channels

	We seek a rotation, R, such that G_t R b_t = G_s b_s where G is the global matrix of a joint and b the child bone
	As such R is the rotation from b_t to G_t.T G_s b_s. Calculate using the cross product method.

	setChannelsFromMatrix starts by doing L.T M where L is the local matrix of the joint and M the given matrix so
	we pre-multiply by L_t to offset this.
	'''

	jS = rtg_S['jointIndex'][s]
	jT = rtg_T['jointIndex'][t]
	if offsets is None:
		offsets = np.zeros(rtg_T['numChans'], dtype=np.float32)
	Character.pose_skeleton(rtg_S['Gs'], rtg_S, np.zeros(rtg_S.numChans, dtype=np.float32))
	Character.pose_skeleton(rtg_T['Gs'], rtg_T, offsets)
	Gs = rtg_S['Gs'][jS]
	Gt = rtg_T['Gs'][jT]
	Lt = rtg_T['Ls'][jT]
	bS = findChildBone(rtg_S, jS)
	bT = findChildBone(rtg_T, jT)
	print ("Source: {} Target: {}".format(bS, bT))
	if bS is None or bT is None: return offsets
	bS = np.dot(Gt[:,:3].T, np.dot(Gs[:,:3], bS))
	R = np.eye(3,4,dtype=np.float32)
	r = rotateBetweenVectors(bT, bS)
	R[:,:3] = np.dot(Lt[:,:3], r)
	setChannelsFromMatrix(R, rtg_T, jT)
	where = np.where(rtg_T.chanValues != 0)[0]
	offsets[where] = rtg_T.chanValues[where]
	return offsets

def test_derror_dchannel(ret, channelMats, usedChannels, usedCAEs, usedCAEsSplits, jointChans, effectors, effectorWeights):
	'''Compute the derivative of the position of the effector with respect to each channel.'''
	funcMap = [d_tx, d_ty, d_tz, d_rx, d_ry, d_rz]
	for ci,(err,cmat,ct) in enumerate(zip(ret, channelMats[usedChannels], jointChans[usedChannels])):
		affectedEffectors = usedCAEs[usedCAEsSplits[ci]:usedCAEsSplits[ci+1]]
		func = funcMap[ct]
		for ei in affectedEffectors:
			err[ei,:] = effectors[ei]
			func(cmat, err[ei])
	ret *= effectorWeights
	return ret

def test_pose_effectors(effectors, residual,
					  Gs, effectorJoints,
					  effectorOffsets, effectorWeights,
					  effectorTargets):

	Gs.reshape(-1,3,4)
	numJoints = Gs.shape[0]
	effectorJoints.reshape(-1)
	numEffectors = effectorJoints.shape[0]
	for i in range(numEffectors):
		ji = effectorJoints[i]
		assert 0 <= ji < numJoints
		effectors[i] = Gs[ji]
		off = effectorOffsets[i]
		tgt = effectorTargets[i]
		wts = effectorWeights[i]
		tgt[:,3] = np.dot(tgt[:,:3], off[:,3]) + tgt[:,3]
		tgt[:,:3] = np.dot(tgt[:,:3], off[:,:3])
		residual[i] = wts * (tgt - effectors[i])
	total_sum = np.sum(residual)
	return total_sum, effectors, residual

def copyJoint(src, tgt, srcJoint, tgtJoint, swizzle = None, offset = None):
	si = src.jointIndex[srcJoint]
	ti = tgt.jointIndex[tgtJoint]
	copyJointIndex(src, tgt, si, ti, swizzle=swizzle, offset=offset)

def copy_joints(src, tgt, copyData):
	src_Ls			  = src.Ls
	src_jointChans	  = src.jointChans
	src_jointChanSplits = src.jointChanSplits
	src_chanValues	  = src.chanValues
	tgt_Ls			  = tgt.Ls
	tgt_jointChans	  = tgt.jointChans
	tgt_jointChanSplits = tgt.jointChanSplits
	tgt_chanValues	  = tgt.chanValues
	jointMapping, jointSwizzles, jointOffsets = copyData
	ISCV.copy_joints(src_Ls, src_jointChans, src_jointChanSplits, src_chanValues, tgt_Ls, tgt_jointChans, tgt_jointChanSplits, tgt_chanValues, jointMapping, jointSwizzles, jointOffsets)

def copyJoints2(src, tgt, copyData, positionOffsets=None):
	src_Ls			  = src['Ls']
	jointMapping, jointSwizzles, jointOffsets = copyData
	if positionOffsets is not None:
		tgt.chanValues[:] = positionOffsets
		Character.pose_skeleton(tgt.Gs, tgt)
	for (si,ti,swi,ofi) in jointMapping:
		swizzle, offset = None,None
		if swi != -1: swizzle = jointSwizzles[swi]
		if ofi != -1: offset = jointOffsets[ofi]
		M = src_Ls[si].copy()
		M = matrixFromChannels(M, src, si)
		# M[:,:3] = np.dot(tgt['Ls'][ti][:,:3], np.dot(src_Ls[si][:,:3], M[:,:3]))
		if swizzle is not None:
			M[:,:3] = np.dot(M[:,:3],swizzle)
			M[:,:3] = np.dot(swizzle.T, M[:,:3])
		if offset is not None:
			M[:,:3] = np.dot(M[:,:3], offset[:,:3])
			M[:,3] += np.dot(M[:,:3], offset[:,3])
		if positionOffsets is not None:
			positionOffset = matrixFromChannels(np.eye(3,4,dtype=np.float32), tgt, ti)
			M[:,:3] = np.dot(positionOffset[:,:3], M[:,:3])
			M[:,3] += positionOffset[:,3]
		# M = T L R; M_T = T + L_T; M_R = L_R R_R
		setChannelsFromMatrix(M, tgt, ti)

def copyJoints3(src, tgt, copyData):
	jointMapping, jointSwizzles, jointOffsets = copyData
	for (si,ti,swi,ofi) in jointMapping:
		swizzle, offset = None,None
		if swi != -1: swizzle = jointSwizzles[swi]
		if ofi != -1: offset = jointOffsets[ofi]
		tran_from,rot_from,rot_end = src.jointChanSplits[2*si:2*si+3]
		chans = src['jointChans'][rot_from:rot_end]
		chan_vals = np.zeros((3, 1), dtype=np.float32)
		chan_vals[chans-3,0] = src['chanValues'][rot_from:rot_end].copy()
		if swizzle is not None:
			chan_vals = np.dot(swizzle, chan_vals)
		tran_from,rot_from,rot_end = tgt.jointChanSplits[2*ti:2*ti+3]
		tgt_chans = tgt['jointChans'][rot_from:rot_end]
		tgt_vals = tgt['chanValues'][rot_from:rot_end]
		stop = min(rot_end-rot_from, len(chans))
		tgt_vals[:stop] = chan_vals[:stop, 0]
		print ("Src Vals: {}\nSrc Chans:{}\nTgt Vals:{}\nTgt Chans:{}".format(chan_vals, chans, tgt_vals, tgt_chans))
		tgt['chanValues'][rot_from:rot_end] = tgt_vals

def copyJointIndex(src, tgt, si, ti, swizzle = None, offset = None):
	M = localMatrix(src, si)
	if swizzle is not None:
		M[:,:3] = np.dot(np.dot(swizzle.T, M[:,:3]),swizzle)
	if offset is not None:
		M[:,3] += np.dot(M[:,:3], offset[:,3])
		M[:,:3] = np.dot(M[:,:3], offset[:,:3])
	setChannelsFromMatrix(M, tgt, ti)

def localMatrix(src, si):
	'''Yields the matrix transform for the joint, in its current state.'''
	M = src.Ls[si].copy()
	return matrixFromChannels(M, src, si)

def matrixFromChannels(M, src, si):
	'''Yields the matrix transform for the joint, in its current state.'''
	tran_from,rot_from,rot_end = src.jointChanSplits[2*si:2*si+3]
	M[src.jointChans[tran_from:rot_from],3] += src.chanValues[tran_from:rot_from]
	# TODO fixme, make the order of rotations be logical
	for c,v in zip(src.jointChans[rot_end-1:rot_from-1:-1], src.chanValues[rot_end-1:rot_from-1:-1]):
		cv,sv=math.cos(v),math.sin(v)
		cj,ck = (c+1)%3,(c+2)%3
		Rj,Rk = M[:,cj],M[:,ck]
		Rj[:],Rk[:] = Rj*cv+Rk*sv, Rk*cv-Rj*sv
	return M

#@profile
def setChannelsFromMatrix(M, tgt, ti):
	# solve t L r = M : r = L_R^T M_R, t = M_T - L_T
	L = tgt.Ls[ti]
	M[:,3] -= L[:,3]
	M[:,:3] = np.dot(L[:,:3].T, M[:,:3])
	tran_from,rot_from,rot_end = tgt.jointChanSplits[2*ti:2*ti+3]
	tgt.chanValues[tran_from:rot_from] = M[tgt.jointChans[tran_from:rot_from],3]
	if rot_end > rot_from:
		out = tgt.chanValues[rot_from:rot_end]
		axes = tgt.jointChans[rot_from:rot_end]
		i = (axes[0]%3)
		if len(axes) == 1: parity = 1 # single channel
		else:			  parity = (axes[1]-axes[0]+3)
		j,k = (i+parity)%3,(i+2*parity)%3
		cj = math.sqrt(M[i,i]*M[i,i] + M[j,i]*M[j,i])
		if cj > 1e-10: out[:] = [math.atan2(M[k,j],M[k,k]),math.atan2(-M[k,i],cj),math.atan2(M[j,i],M[i,i])][:len(out)]
		else:		  out[:] = [math.atan2(-M[j,k],M[j,j]),math.atan2(-M[k,i],cj),0.0][:len(out)]
		if ((parity%3) == 2): out[:] = -out

def globalMatrix(src, si):
	'''Slow method for getting the matrix for a joint.'''
	#SG_Sparents_i * SJoints_i * SL_i, SG_Sparents_root = I
	R = np.zeros((3,4), dtype=np.float32)
	pi = src.parents[si]
	if pi != -1: R[:] = getGMat(src, pi)
	L = localMatrix(src, si)
	R[:,3] += np.dot(R[:,:3], L[:,3])
	R[:,:3] = np.dot(R[:,:3], L[:,:3])
	return R

def getPos(src, srcJoint, offset = None):
	si = src.jointIndex[srcJoint]
	ret = globalMatrix(src, si)[:,3]
	if offset is not None: ret += offset
	return ret

def getOrient(src, srcJoint, offset = None):
	si = src.jointIndex[srcJoint]
	ret = globalMatrix(src, si)[:,:3]
	if offset is not None: return np.dot(ret, offset)
	return ret


def testDerivatives(skelDict, chanValues, effectorData, effectorTargets):
	''' skelDict specifies a skeleton.
		Given an initial skeleton pose (chanValues), effectors (ie constraints: joint, offset, weight, target), solve for the skeleton pose.
		Effector weights and targets are 3x4 matrices.
		* Setting 1 in the weight's 4th column makes a position constraint.
		* Setting 100 in the weight's first 3 columns makes an orientation constraint.
	'''
	rootMat = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]],dtype=np.float32)
	effectorJoints, effectorOffsets, effectorWeights, usedChannels, usedCAEs, usedCAEsSplits = effectorData
	jointParents	= skelDict['jointParents']
	Gs			  = skelDict['Gs']
	Ls			  = skelDict['Ls']
	jointChans	  = skelDict['jointChans']
	jointChanSplits = skelDict['jointChanSplits']
	numChannels	 = jointChanSplits[-1]
	numEffectors	= len(effectorJoints)
	numUsedChannels = len(usedChannels)
	channelMats	 = np.zeros((numChannels,3,4), dtype=np.float32)
	usedEffectors   = np.where(effectorWeights.reshape(-1) != 0)[0]
	numUsedEffectors= len(usedEffectors)
	effectors	   = np.zeros((numEffectors,3,4),dtype=np.float32)
	residual		= np.zeros((numEffectors,3,4),dtype=np.float32)
	derrors		 = np.zeros((numUsedChannels,numEffectors,3,4), dtype=np.float32)
	steps		   = np.ones((numUsedChannels),dtype=np.float32)*1.0
	steps[np.where(jointChans[usedChannels] < 3)[0]] = 30.
	delta		   = np.zeros((numUsedChannels),dtype=np.float32)
	JJTB			= np.zeros((numEffectors*12),dtype=np.float32)
	Character.pose_skeleton_with_chan_mats(channelMats, Gs, chanValues, rootMat)
	bestScore = ISCV.pose_effectors(effectors, residual, Gs, effectorJoints, effectorOffsets, effectorWeights, effectorTargets)
	print (bestScore, chanValues[usedChannels])
	ISCV.derror_dchannel(derrors, channelMats, usedChannels, usedCAEs, usedCAEsSplits, jointChans, effectors, effectorWeights)
	JT = derrors.reshape(numUsedChannels,-1)
	B = residual.reshape(-1)
	residual2	   = np.zeros(residual.shape,dtype=np.float32)
	# JT[ci] = dresidual_dci
	for uci in xrange(numUsedChannels):
		ci = usedChannels[uci]
		tmp = chanValues[ci]
		d_ci = max(0.001,abs(chanValues[ci]*0.001))
		chanValues[ci] += d_ci
		Character.pose_skeleton(Gs, skelDict, chanValues, rootMat)
		bestScore = ISCV.pose_effectors(effectors, residual2, Gs, effectorJoints, effectorOffsets, effectorWeights, effectorTargets)
		#print (bestScore)
		diff = (residual2.reshape(-1) - B)/-d_ci
		if np.dot(JT[uci], diff)/np.sqrt(1e-10+np.dot(JT[uci],JT[uci])*np.dot(diff,diff)) < 0.99:
			print (uci, ci, np.dot(JT[uci], diff)/np.sqrt(1e-10+np.dot(JT[uci],JT[uci])*np.dot(diff,diff)))
		chanValues[ci] = tmp


def computeRBF(x, c, beta):
	# takes an input point and the centre of the RBF and returns the value of
	# p(||x - c||)
	if not isinstance(x,np.ndarray):
		x = np.array([[x]])
	elif len(x.shape) == 1:
		x.shape = (-1, 1)
	return np.exp(-(beta * np.linalg.norm(x - c, axis=1)**2))

def normalizedRBFN(X, C, Beta, normalise=True):
	tiny = 1e-10
	out = np.zeros((X.shape[0], C.shape[0]), dtype=X.dtype)
	for ci, c in enumerate(C):
		out[:, ci] = computeRBF(X, c, Beta[ci])
	if normalise:
		row_sums = np.sum(out, axis=1)
		out /= (row_sums[:, np.newaxis] + tiny)
	return out

def evaluateRBFN( W, C, Beta, input, normalise=True):
	# Evaluates an RBFN for a given set of Weights, Centres on a given input
	num_dims = W.shape[1]
	output = np.zeros((input.shape[0], num_dims), dtype=input.dtype)
	RBFs = normalizedRBFN(input, C, Beta, normalise)
	output[:] = np.dot(RBFs, W[:,:])
	return output, (RBFs / np.sum(RBFs))[0,:]

def rNearestNeighbourBetaCalculation(C, r):
	if len(C.shape) == 1:
		C = C.reshape(-1,1)
	r = min((C.shape[0] - 1), r)
	Beta = np.zeros((C.shape[0], 1), dtype=np.float32)
	dists = np.zeros((C.shape[0], C.shape[0]))
	neighbours = np.zeros((C.shape[0], r), dtype=np.float32)
	for i in xrange(C.shape[0]):
		for j in xrange(C.shape[0]):
			dists[i,j] = np.linalg.norm(C[i] - C[j])
		neighbours[i,:] = np.sort(dists[i,:])[1:r+1]
		Beta[i] = (np.sqrt(np.sum(neighbours[i,:]**2)/ r))
	return Beta

def totalBetaCalc(C):
	if len(C.shape) == 1:
		C = C.reshape(-1,1)
	d_max = 0
	for i in xrange(C.shape[0]):
		for j in xrange(i+1, C.shape[0]):
			dist = np.linalg.norm(C[i] - C[j])
			if dist > d_max:
				d_max = dist
	Beta = np.array([np.sqrt(2 * C.shape[0]) / d_max] * C.shape[0]).reshape(-1,1)
	return Beta

def trainRBFN(X, Y, normalise=True):
	# Trains a RBFN given a set of known points
	# returns the weights and centres
	if not isinstance(Y, np.ndarray):
		Y = np.array([[Y]])
	elif len(Y.shape) == 1:
		Y.shape = (-1, 1)
	Beta = rNearestNeighbourBetaCalculation(X, 5)
	Beta = (totalBetaCalc(X))*(1/Beta)
	x_size, y_dim = X.shape[0], Y.shape[1]
	G = np.zeros((x_size, x_size), dtype=X.dtype)
	G[:,:] = normalizedRBFN(X, X, Beta, normalise)
	W = np.zeros((x_size, y_dim), dtype=X.dtype)
	error = np.zeros(y_dim, dtype=np.float32)
	for dim in xrange(y_dim):
		W[:, dim] = np.linalg.lstsq(G, Y[:,dim])[0]
		error[dim] = np.linalg.norm(Y[:,dim] - np.dot(G, W[:,dim]).T)
	print "Error: {}".format(np.mean(error))
	return W, X, Beta

def pointsToEdges(points, mapping_list=None):
	# mapping_list is such that i is mapped to mapping_list[i]
	from scipy.spatial import Delaunay
	tris = Delaunay(points).simplices
	edges = ISCV.trianglesToEdgeList(np.max(tris)+1 if len(tris) else 1, tris) # (numVerts,10)
	edgeList = set()
	for vi, el in enumerate(edges):
		which = np.where(el > vi)[0]
		edgeList.update(zip(which,el[which]))
	edgeList = np.int32(list(edgeList))
	if mapping_list is not None: edgeList = np.int32(mapping_list)[edgeList]
	return edgeList
