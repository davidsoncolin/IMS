"""
GCore/SolveIK.py

Requires:
	sys
	numpy
	scipy
	
	Grip
		ISCV (project, cloud, )
		
	
"""
import sys
import numpy as np
import ISCV
from GCore import Character, Recon, list_of_lists_to_splits
import scipy.linalg.lapack as LAPACK

def computeChannelAffectedEffectors(jointCutOff, jointParents, jointChanSplits, effectorJoints):
	'''Returns for each channel the list of effectors that are affected (their joint is a child of the channel's joint).'''
	# first compute for each joint a list of all the channels back to the root (or jointCutOff)
	numJoints = len(jointParents)
	assert(len(jointChanSplits) == numJoints*2+1)
	numChannels = jointChanSplits[-1]
	j2pjs = [[x] for x in xrange(numJoints)] # for each joint, a list of all the parent joints
	for ji,pi in enumerate(jointParents):
		if ji != jointCutOff and pi != -1: j2pjs[ji].extend(j2pjs[pi])
	jcs = [[range(jointChanSplits[2*ji],jointChanSplits[2*ji+2]) for ji in js] for js in j2pjs] # turn joints into lists of channels
	jcs = [[di for dis in jc for di in dis] for jc in jcs] # flatten the inner lists...
	channelAffectedEffectors = [[] for x in xrange(numChannels)]
	for ei,pi in enumerate(effectorJoints):
		assert(pi != -1)
		for ci in jcs[pi]: channelAffectedEffectors[ci].append(ei)
	usedChannels = np.where(np.array(map(len, channelAffectedEffectors))!=0)[0]
	usedChannels = np.array(list(usedChannels),dtype=np.int32)
	usedCAEs = [channelAffectedEffectors[ci] for ci in usedChannels]
	return usedChannels, list_of_lists_to_splits(usedCAEs)

def make_effectorData(skelDict, jointCutOff=-1, p_o_w = None):
	"""
	effectorData is a structure that holds all the information for computing positions and derivatives of effectors (markers)
	when varying channels UNDER the jointCutOff.
	
	Args:
		skelDict (GskelDict): The Skeleton to process.
		
	Returns:
		structure: "effectorData" containing:
			effectorJoints, effectorOffsets, effectorWeights:
				to compute the position of the effector, get the global matrix of the joint and apply it to the offset
				the weight may control the IK
			usedChannels:
				the list of channels that might be involved (they lie between an effector and the jointCutOff)
			usedChannelWeights:
				the weights for each channel: by default, all ones (this might affect the stiffness of a joint in IK)
			usedCAEs, usedCAEsSplits:
				"used channel affected effectors". for each channel, the list of effectors that are affected by varying that channel.
	Requires:
		computeChannelAffectedEffectors
	"""
	if p_o_w is None:
		markerParents,markerOffsets,markerWeights = skelDict['markerParents'],skelDict['markerOffsets'],skelDict['markerWeights']
	else:
		markerParents,markerOffsets,markerWeights = p_o_w
	effectorJoints = markerParents
	numMarkers = len(effectorJoints)
	effectorOffsets = np.zeros((numMarkers,3,4),dtype=np.float32)
	effectorWeights = np.zeros((numMarkers,3,4),dtype=np.float32)
	effectorOffsets[:] = np.eye(3,4,dtype=np.float32)
	effectorOffsets[:,:,3] = markerOffsets
	effectorWeights[:,:,3] = markerWeights.reshape(-1,1)
	usedChannels, (usedCAEs, usedCAEsSplits) = computeChannelAffectedEffectors(jointCutOff, skelDict['jointParents'], skelDict['jointChanSplits'], effectorJoints)
	usedChannelWeights = np.ones(len(usedChannels),dtype=np.float32)
	effectorData = (effectorJoints, effectorOffsets, effectorWeights, usedChannels, usedChannelWeights, usedCAEs, usedCAEsSplits)
	return effectorData

def skeleton_marker_positions(skelDict, rootMat, chanValues, effectorLabels, effectorData, markerWeights=None):
	"""
	Based on the pose implied by the chanValues and rootMat, compute the 3D world-space
	positions of the markers.
	
	Multiple effectors may determine the position of the marker. effectorLabels provides this mapping.
	
	The weights for the markers, if any, are set by markerWeights.
	
	Args:
		skelDict (GskelDict): the skeleton
		rootMat (float[3][4]): reference frame of the Skeleton.
		chanValues (float[]) List of channel values to pose the skeleton
		effectorLabels : the marker that each effector determines
		effectorData : (effectorJoints, effectorOffsets, ...)
		markerWeights : the weight that each effector has on its marker
		
	Returns:
		int[]: Labels for the 3D positions of the markers.
		float[][3]: 3D positions of where the target would be in the pose.
		
	Requires:
		Character.pose_skeleton
		ISCV.marker_positions
		
	"""
	Character.pose_skeleton(skelDict['Gs'], skelDict, chanValues, rootMat)
	labels = np.unique(effectorLabels)
	els2 = np.int32([list(labels).index(x) for x in effectorLabels])
	x3ds = ISCV.marker_positions(skelDict['Gs'], effectorData[0], effectorData[1], els2, markerWeights)
	return x3ds, labels


def solveIK(skelDict, chanValues, effectorData, effectorTargets, outerIts=10, rootMat=None):
	"""
	Given an initial skeleton pose (chanValues), effectors (ie constraints: joint, offset, weight, target), solve for the skeleton pose.
	Effector weights and targets are 3x4 matrices.
		* Setting 1 in the weight's 4th column makes a position constraint.
		* Setting 100 in the weight's first 3 columns makes an orientation constraint.
		
	Args:
		skelDict (GskelDict): The Skeleton to process
		chanValues (float[]): Initial pose of the skeleton as Translation and many rotations applied to joints in the skelDict.
		effectorData (big o'l structure!):
			effectorJoints, effectorOffsets, effectorWeights, usedChannels, usedChannelWeights, usedCAEs, usedCAEsSplits
		effectorTargets (?): What's this?
		outerIts (int): IK Iterations to solve the skeleton. Default = 10
		rootMat (float[3][4]): reference frame of the Skeleton. Default = None
		
	Returns:
		None: The result is an update of the skelDict to the solution - chanValues, channelMats, and Gs.
		
	Requires:
		Character.pose_skeleton_with_chan_mats
		ISCV.pose_effectors
		ISCV.derror_dchannel
		ISCV.JTJ
	"""
	effectorJoints, effectorOffsets, effectorWeights, usedChannels, usedChannelWeights, usedCAEs, usedCAEsSplits = effectorData
	jointParents    = skelDict['jointParents']
	Gs              = skelDict['Gs']
	Ls              = skelDict['Ls']
	jointChans      = skelDict['jointChans']
	jointChanSplits = skelDict['jointChanSplits']
	numChannels     = jointChanSplits[-1]
	numEffectors    = len(effectorJoints)
	numUsedChannels = len(usedChannels)
	channelMats     = np.zeros((numChannels,3,4), dtype=np.float32)
	#usedEffectors   = np.array(np.where(np.sum(effectorWeights,axis=(1,2)) != 0)[0], dtype=np.int32)
	usedEffectors   = np.array(np.where(effectorWeights.reshape(-1) != 0)[0], dtype=np.int32)
	# numUsedEffectors= len(usedEffectors)
	effectors       = np.zeros((numEffectors,3,4),dtype=np.float32)
	residual        = np.zeros((numEffectors,3,4),dtype=np.float32)
	derrors         = np.zeros((numUsedChannels,numEffectors,3,4), dtype=np.float32)
	# steps           = np.ones((numUsedChannels),dtype=np.float32)*0.2
	# steps[np.where(jointChans[usedChannels] < 3)[0]] = 30.
	# steps = 1.0/steps
	delta           = np.zeros((numUsedChannels),dtype=np.float32)
	# JJTB            = np.zeros((numEffectors*12),dtype=np.float32)
	JTJ             = np.zeros((numUsedChannels, numUsedChannels),dtype=np.float32)
	JTB             = np.zeros((numUsedChannels),dtype=np.float32)
	JT = derrors.reshape(numUsedChannels,-1)
	JTJdiag         = np.diag_indices_from(JTJ)
	B = residual.reshape(-1)
	# TODO, calculate the exact requirements on the tolerance
	B_len = len(B)
	tolerance = 0.00001
	it_eps = (B_len**0.5)*tolerance
	for it in xrange(outerIts):
		# TODO, only usedChannels are changing, only update the matrices that have changed after the first iteration.
		# TODO Look into damping, possibly clip residuals?
		# updates the channelMats and Gs
		Character.pose_skeleton_with_chan_mats(channelMats, Gs, skelDict, chanValues, rootMat)
		bestScore = ISCV.pose_effectors(effectors, residual, Gs, effectorJoints, effectorOffsets, effectorWeights, effectorTargets)
		if np.linalg.norm(B) < it_eps: break # early termination
		ISCV.derror_dchannel(derrors, channelMats, usedChannels, usedChannelWeights, usedCAEs, usedCAEsSplits, jointChans, effectors, effectorWeights)
		# if True: # DLS method : solve (JTJ + k^2 I) delta = JTB
		ISCV.JTJ(JTJ,JTB,JT,B,usedEffectors) #np.dot(JT, B, out=JTB); np.dot(JT, JT.T, out=JTJ)
		JTJ[JTJdiag] += 1
		JTJ[JTJdiag] *= 1.1
		_, delta[:], _ = LAPACK.dposv(JTJ,JTB) # Use Positive Definite Solver
		# Use General Solver
		# delta[:] = np.linalg.solve(JTJ, JTB)
		# elif it==0: # SVD method: solve J delta = B
		# 	delta[:] = np.linalg.lstsq(JT.T[usedEffectors], B[usedEffectors], rcond=0.0001)[0].reshape(-1)
		# else:     # J transpose method
		# 	testScale = ISCV.J_transpose(delta, JJTB, JT, B)
		# 	#np.dot(JT, B, out=delta); np.dot(JT.T,delta,out=JJTB); delta *= np.dot(B,JJTB)/(np.dot(JJTB,JJTB)+1.0)
		#scale = np.max(np.abs(delta*steps))
		#if scale > 1.0: delta *= 1.0/scale
		#np.clip(delta,-steps,steps,out=delta)
		chanValues[usedChannels] += delta
		# TODO: add channel limits
		#bestScore = ISCV.lineSearch(chanValues, usedChannels, delta, Gs, Ls, jointParents, jointChans, jointChanSplits,
		#							rootMat, effectorJoints, effectorOffsets, effectorWeights, effectorTargets, innerIts, bestScore)
	#print np.mean(B*B)
	Character.pose_skeleton(Gs, skelDict, chanValues, rootMat)

def solveIK1Ray(skelDict, effectorData, x3ds, effectorIndices_3d, E, effectorIndices_2d, outerIts=10, rootMat=None):
	"""
	solveIK routine form Label.py - Has Single ray constraint equations enables

	Given effectors (joint, offset, weight) and constraints for those (3d and 2d), solve for the skeleton pose.
	Effector offsets, weights and targets are 3-vectors
		
	Args:
		skelDict (GskelDict): The Skeleton to process
		effectorData (big o'l structure!):
			effectorJoints, effectorOffsets, effectorWeights, usedChannels, usedChannelWeights, usedCAEs, usedCAEsSplits
		x3ds (float[][3]): 3D Reconstructions
		effectorIndices_3d (?): What's this?
		E (): Equations for 1-Ray constraints, or MDMA.
		effectorIndices_2d (?): What's this?
		outerIts (int): IK Iterations to solve the skeleton. Default = 10
		rootMat (float[3][4]): reference frame of the Skeleton. Default = None
		
	Returns:
		None: The result is an update of the skelDict to the solution - chanValues, channelMats, and Gs.
		
	Requires:
		Character.pose_skeleton_with_chan_mats
		ISCV.derror_dchannel_single_ray
		ISCV.JTJ_single_ray
	"""
	if rootMat is None: rootMat = np.eye(3,4,dtype=np.float32)
	effectorJoints, effectorOffsets, effectorWeightsOld, usedChannels, usedChannelWeights, usedCAEs, usedCAEsSplits = effectorData
	chanValues      = skelDict['chanValues']
	jointParents    = skelDict['jointParents']
	Gs              = skelDict['Gs']
	Ls              = skelDict['Ls']
	jointChans      = skelDict['jointChans']
	jointChanSplits = skelDict['jointChanSplits']
	numChannels     = jointChanSplits[-1]
	numEffectors    = len(effectorJoints)
	num3ds          = len(effectorIndices_3d)
	num2ds          = len(effectorIndices_2d)
	effectorOffsets = np.copy(effectorOffsets[:,:,3])
	effectorWeights = np.zeros(numEffectors, dtype=np.float32)
	effectorWeights[effectorIndices_3d] = 1 # TODO Why does this fail? effectorWeightsOld[effectorIndices_3d,0,3]
	effectorWeights[effectorIndices_2d] = 1 # effectorWeightsOld[effectorIndices_2d,0,3]
	numUsedChannels = len(usedChannels)
	channelMats     = np.zeros((numChannels,3,4), dtype=np.float32)
	effectors       = np.zeros((numEffectors,3),dtype=np.float32)
	residual        = np.zeros((num3ds,3),dtype=np.float32)
	residual2       = np.zeros((num2ds,2),dtype=np.float32)
	derrors         = np.zeros((numUsedChannels,numEffectors,3), dtype=np.float32)
	delta           = np.zeros((numUsedChannels),dtype=np.float32)
	JTJ             = np.zeros((numUsedChannels, numUsedChannels),dtype=np.float32)
	JTB             = np.zeros((numUsedChannels),dtype=np.float32)
	JT              = derrors.reshape(numUsedChannels,-1)
	JTJdiag         = np.diag_indices_from(JTJ)
	for it in xrange(outerIts):
		# TODO, only usedChannels are changing, only update the matrices that have changed after the first iteration.
		# updates the channelMats and Gs
		Character.pose_skeleton_with_chan_mats(channelMats, Gs, skelDict, chanValues, rootMat)
		bestScore = ISCV.pose_effectors_single_ray(effectors, residual, residual2, Gs, effectorJoints, effectorOffsets, effectorWeights, x3ds, effectorIndices_3d, E, effectorIndices_2d)
		if np.sum(residual*residual)+np.sum(residual2*residual2) <= 1e-5*(num3ds+num2ds): break # early termination
		ISCV.derror_dchannel_single_ray(derrors, channelMats, usedChannels, usedChannelWeights, usedCAEs, usedCAEsSplits, jointChans, effectors, effectorWeights)
		# J = d_effectors/dc
		# err(c) = x3ds - effectors[effectorIndices_3d], e0 + E effectors[effectorIndices_2d]; err(c+delta) = x3ds - effectors[effectorIndices_3d] - J[effectorIndices_3d] delta, e0 + E effectors[effectorIndices_2d] + E J[effectorIndices_2d] delta  = 0
		# J dc = B; (J[effectorIndices_3d] ; E J[effectorIndices_2d]) dc = B ; e0
		# DLS method : solve (JTJ + k^2 I) delta = JTB
		ISCV.JTJ_single_ray(JTJ,JTB,JT,residual,effectorIndices_3d,E,effectorIndices_2d,residual2) #np.dot(JT, B, out=JTB); np.dot(JT, JT.T, out=JTJ)
		JTJ[JTJdiag] += 1
		JTJ[JTJdiag] *= 1.1
		# delta[:] = np.linalg.solve(JTJ, JTB)
		_, delta[:], _ = LAPACK.dposv(JTJ,JTB) # Use Positive Definite Solver
		chanValues[usedChannels] += delta
		# TODO: add channel limits

		# # J_transpose method, 3d only: scaling problems with translation
		#JT = derrors[:,effectorIndices_3d,:].reshape(numUsedChannels,-1)
		#np.dot(JT, B, out=delta)
		#np.dot(JT.T,delta,out=JJTB)
		#delta *= np.dot(B,JJTB)/(np.dot(JJTB,JJTB)+1)
		#delta[:3] *= 100000.
		#testScale = ISCV.Jtranspose_SR(delta, JJTB, JT, residual,effectorIndices_3d,residual2,effectorIndices_2d)
	Character.pose_skeleton(Gs, skelDict, chanValues, rootMat)


def scoreIK(skelDict, chanValues, effectorData, effectorTargets, rootMat=None):
	"""
	Args:
		skelDict (GskelDict): The Skeleton to process

	Returns:
		?

	Requires:
		Character.pose_skeleton
		ISCV.score_effectors
	"""
	Character.pose_skeleton(skelDict['Gs'], skelDict, chanValues, rootMat)
	return (ISCV.score_effectors(skelDict['Gs'], effectorData[0], effectorData[1], effectorData[2], effectorTargets)/np.sum(effectorData[1]))**0.5

def bake_ball_joints(skelDict):
	"""
	For every 3 DoF joint, multiply in matrices to reduce gimbal lock.
	Includes pre-conversion python code.
	
	Args:
		skelDict (GskelDict): The Skeleton to process.
		
	Requires:
		ISCV.bake_ball_joints
	"""
	Ls              = skelDict['Ls']
	jointChans      = skelDict['jointChans']
	jointChanSplits = skelDict['jointChanSplits']
	chanValues      = skelDict['chanValues']
	if not skelDict.has_key('Ls_orig'): skelDict['Ls_orig'] = Ls.copy()
	Ls_orig         = skelDict['Ls_orig']
	ISCV.bake_ball_joints(Ls, jointChans, jointChanSplits, chanValues)

def unbake_ball_joints(skelDict):
	"""
		
	Args:
		skelDict (GskelDict): The Skeleton to process.
		Ls_orig (float[?}): Unbaked arrangement of Local Matrices of the skeleton's 3-DoF joints.
		
	Returns:
		None: Results are a transformation of the skelDict
		
	Requires:
		ISCV.unbake_ball_joints
	"""
	Ls              = skelDict['Ls']
	jointChans      = skelDict['jointChans']
	jointChanSplits = skelDict['jointChanSplits']
	chanValues      = skelDict['chanValues']
	if not skelDict.has_key('Ls_orig'): skelDict['Ls_orig'] = Ls.copy()
	Ls_orig         = skelDict['Ls_orig']
	ISCV.unbake_ball_joints(Ls, jointChans, jointChanSplits, chanValues, Ls_orig)

def solve_skeleton_from_2d(x2ds, splits, labels, effectorLabels, Ps, skelDict, effectorData, rootMat, outerIts=5):
	"""
	Given a posed skeleton and some labelled 2d points, solve the skeleton to better fit the points.
	
	Args:
		x2ds (float[][2]): 2d Detections from all cameras
		splits (int[]): list of camera indices
		labels (int[]): Assigned labels of the x2ds
		effectorLabels (?): For each effector, which label it depends on.
			Joints may be effected by a number of labellings.
		Ps (float[][3][4]): Projection matrices of the cameras.
		skelDict (GskelDict): The Skeleton to process
		effectorData (?): What's this?
		rootMat (float[3][4]): reference frame of the Skeleton.
		outerIts (int): IK Iterations to solve the skeleton. Default = 5.
		
	Returns:
		float[][3]: (x3ds) - the resulting 3D reconstructions.
		int[]: (x3d_labels) - the labels for the 3D points.
		??: (E[singles]) - Equations describing 2D detections not born of the 3D yet.
		int[] (x2d_labels) - labels for the 2D contributions.
		
	Requires:
		Recon.solve_x3ds
		
	"""
	x3ds, x3d_labels, E, x2d_labels = Recon.solve_x3ds(x2ds, splits, labels, Ps)

	# effectorLabels tells, for each effector, which label it depends on
	# effectorLabels[ei] = li
	# given a list of labels, collect all the effectors that depend on those labels; and then find the reordering of the
	# original labels (which may include duplicates) that matches the effectors.

	numLabels = np.max(effectorLabels)+1

	lbl3_inv = -np.ones(numLabels+1,dtype=np.int32)
	lbl3_inv[x3d_labels] = range(len(x3d_labels))
	tmp3 = lbl3_inv[effectorLabels]
	ae3 = np.array(np.where(tmp3 != -1)[0],dtype=np.int32)
	tmp3 = tmp3[ae3]

	lbl2_inv = -np.ones(numLabels+1,dtype=np.int32)
	lbl2_inv[x2d_labels] = range(len(x2d_labels))
	tmp2 = lbl2_inv[effectorLabels]
	ae2 = np.array(np.where(tmp2 != -1)[0],dtype=np.int32)
	tmp2 = tmp2[ae2]
	#
	solveIK1Ray(skelDict, effectorData, x3ds.take(tmp3,axis=0), ae3, E.take(tmp2,axis=0), ae2, outerIts=outerIts, rootMat=rootMat)
	return x3ds, x3d_labels, E, x2d_labels

def solve_skeleton_from_2d_bake(x2ds, splits, labels, effectorLabels, Ps, skelDict, effectorData, rootMat, outerIts=5):
	"""
	Given a posed skeleton and some labelled 2d points, solve the skeleton to better fit the points.
	This method Bakes Ball-joints (3 DoF Joints)
		
	Args:
		x2ds (float[][2]): 2d Detections from all cameras
		splits (int[]): list of camera indices
		labels (int[]): Assigned labels of the x2ds
		effectorLabels (?): For each effector, which label it depends on.
			Joints may be effected by a number of labellings.
		Ps (float[][3][4]): Projection matrices of the cameras.
		skelDict (GskelDict): The Skeleton to process
		effectorData (?): What's this?
		rootMat (float[3][4]): reference frame of the Skeleton.
		outerIts (int): IK Iterations to solve the skeleton. Default = 5.
		
	Returns:
		float[][3]: (x3ds) - the resulting 3D reconstructions.
		int[]: (x3d_labels) - the labels for the 3D points.
		??: (E[singles]) - Equations describing 2D detections not born of the 3D yet.
		int[] (x2d_labels) - labels for the 2D contributions.
	"""

	bake_ball_joints(skelDict)
	ret = solve_skeleton_from_2d(x2ds, splits, labels, effectorLabels, Ps, skelDict, effectorData, rootMat, outerIts=outerIts)
	unbake_ball_joints(skelDict)
	return ret

def solve_skeleton_from_3d(x3ds, labels, effectorLabels, skelDict, effectorData, rootMat):
	"""
	Given a posed skeleton and some labelled 3d points, solve the skeleton to better fit the points.
	
	Args:
		x3ds
		labels
		effectorLabels
		skelDict
		effectorData
		rootMat
		
	Returns:
		(float)? "score" - score of IK goodlyness.
		
	"""
	# IK solving
	which = np.where([li != -1 and li in effectorLabels for li in labels])[0]
	effectorIndices = [effectorLabels.index(li) for li in labels[which]]
	effectorTargets = np.zeros_like(effectorData[1])
	effectorTargets[effectorIndices,:,3] = x3ds[which]
	effectorWeights = effectorData[2]
	effectorWeights[:] = 0
	effectorWeights[effectorIndices,:,3] = 1

	# 'solve_skeleton_from_3d_bake' may be needed here, see 'solve_skeleton_from_2d_bake'
	bake_ball_joints(skelDict)
	# solveIK1Ray(skelDict, effectorData, x3ds, effectorIndices_3d, E, effectorIndices_2d, outerIts=10, rootMat=None):
	# solveIK(skelDict, chanValues, effectorData, effectorTargets, outerIts=10, rootMat=None)
	solveIK(skelDict, skelDict['chanValues'], effectorData, effectorTargets, outerIts=10, rootMat=rootMat)
	unbake_ball_joints(skelDict)

	score = scoreIK(skelDict, skelDict['chanValues'], effectorData, effectorTargets, rootMat=rootMat)
	return score
