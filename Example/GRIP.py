import os
import traceback
from uuid import uuid4
import copy
import ISCV
import operator
import logging
from OpenGL import GLUT

from PySide import QtCore
from PySide import QtGui
import numpy as np

from GCore import Character, State, Retarget, SolveIK
from GCore.base import atdict
from UI import T_SKELETON, T_MOTION, K_NAME, K_FILENAME, K_DRAW
from UI import COLOURS, GLSkel
from UI.GLMeshes import GLMeshes
from UI import errorDialog
from IO import MAReader
import Base_Deformer
import IO

G_Moving_Things = []
G_All_Things = {}
G_Selection = []
G_RTG_Dicts = {}
G_RTG_Keys = []
G_Characters = []
G_Cube_Deformer = None
G_Target_Point = None
G_Temp_Chans = None

target_anim = {'chan_order':[],'anim_dict':np.zeros((10270,157))}
looped = 0

logging.basicConfig()
LOG = logging.getLogger(__name__)

RETARGET_TYPE = 'retarget_type'
FILE_TYPE = 'FILE'
COLOUR_RTG = (1,.6,.3,1)
JOINT_NO = 'Not Set'
SWIZZLE_NO = 'No Swizzle'
SWIZZLE_CUSTOM = 'Custom'
DEFAULT_SWIZZLES = {SWIZZLE_NO: np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
                    '+z+x+y': np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
                    '-y-x-z': np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], dtype=np.float32),
                    '+x-y-z': np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32),
                    '-y-z+x': np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float32),
                    '+z-x-y': np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float32)}
assert(all([s is None or np.linalg.det(s)==1 for s in DEFAULT_SWIZZLES.values()]))

templateObjectData = {'name':'object',
		'colour': (0, 0.8, 1, 1),
		'selected': False,
		'draw':True,
		'visible':True}

def updateHackyDeformer(skel_dict):
	global G_Hack_lattice
	if G_Hack_lattice is None:
		filename = 'D:\Documents\lattice_Hack.hack'
		mappings = IO.load(filename)[1]['mappings']
		G_Hack_lattice = Base_Deformer.CubeDeformer('Hack', skel_dict, mappings)
	else:
		G_Hack_lattice.skel_dict = skel_dict

def addRTGKey(outliner, rtg_key, name, affected_objs):
	global G_All_Things, G_RTG_Keys
	G_RTG_Keys.append(rtg_key)
	print "Adding Key: {}\nRTG_Keys: {}".format(rtg_key, G_RTG_Keys)
	G_All_Things[rtg_key] = {'objecttype':RETARGET_TYPE,
							 'data':{'name':name,
									 'rtg_key':rtg_key,
									 'children':[]
									 }
							 }
	outliner.model.add(G_All_Things[rtg_key])
	for obj in affected_objs:
		if 'rtg' not in obj['data']:
			obj['data']['rtg'] = [rtg_key]
		else:
			obj['data']['rtg'].append(rtg_key)
	# outliner.model.setScene(G_All_Things.values())

def loadSkel(filepath):
	skelDict = IO.load(filepath)[1]
	return skelDict

def loadAsf(filepath):
	asfDict = IO.ASFReader.read_ASF(filepath)
	skelDict = IO.ASFReader.asfDict_to_skelDict(asfDict)
	return skelDict

def getFromState(key, path, *args):
	return State.getKey("/".join([key,path]), *args)

def setToState(key, path, value):
	return State.setKey("/".join([key,path]),value)

def addKeyToState(key, value):
	return State.addKey(key,value)

def deleteKeyFromState(key):
	return State.delKey(key)

def getUndoFromState():
	return State.getUndoCmd()

def getAppNameFromState():
	return State.appName()

#-- State Dict --

def setRetargetSource(rtg_dict, obj, key):
	rtg_dict['attrs']['sourceObject'] = key
	rtg_dict['attrs']['sourceSkeleton'] = copy.deepcopy(obj['skelDict'])
	rtg_dict['attrs']['srcAnim'] = obj['animDict']['dofData']
	return rtg_dict

def setRetargetTarget(rtg_dict, obj, key):
	rtg_dict['attrs']['targetObject'] = key
	rtg_dict['attrs']['targetSkeleton'] = copy.deepcopy(obj['skelDict'])
	## TODO - This currently sets the objects colour but it never gets saved to state
	obj['data']['boneColour'] = COLOUR_RTG
	if hasattr(obj['primitive'],'colour'):
		obj['primitive'].colour = COLOUR_RTG
	return rtg_dict

def newCopyPassDict():
	return 	{
				'enabled':False,
				'name':'Copy Pass',
				'jointPairs':[],
				'copyOffsets':{},
				'copySwizzles':{},
				'copyData':[],
				'defaultOffset':None,
				'defaultSwizzle':None
			}

def newIKPassDict():
	return 	{
				'enabled':False,
				'name':'IK Pass',
				'effectorTargetName':JOINT_NO,
				'effectorJointName':JOINT_NO,
				'jointCutoffName':JOINT_NO,
				'weightPosition':1,
				'weightOrientation':30,
				'jointStiffness':[],
				'effectorTargetOffset':np.eye(3,4)
			}

def newRetargetDict():
	rtg_dict = 	{'type':RETARGET_TYPE,'attrs':
					{
						'enabled':True,
						'sourceFile':None,
						'targetFile':None,
						'sourceObject':None,
						'targetObject':None,
						'targetOffsetY':0,
						'sourceSkeleton':None,
						'targetSkeleton':None,
						'srcAnim':None,
						'X':{
								's':1.0,
								't':np.zeros(3, dtype=np.float32),
								'R':np.eye(3, dtype=np.float32),
								'mat':np.zeros((3, 4), dtype=np.float32)
							},
						'passes':{
									'cp':None,
									'ik':{
											'passes':{},
											'order':{}
										}
								}
					}
				}
	return rebuildRetargetX(rtg_dict)

def addIKPassToRetarget(rtg_dict,IKDict):
	name = IKDict['name']
	rtg_dict['attrs']['passes']['ik']['order'][len(rtg_dict['attrs']['passes']['ik']['passes'])] = name
	rtg_dict['attrs']['passes']['ik']['passes'][name] = IKDict
	return rtg_dict

def addIKPassToRetargetState(rtg_key,IKDict):
	name = IKDict['name']
	ik_passes = getFromState(rtg_key,'attrs/passes/ik/passes')
	ik_order = getFromState(rtg_key,'attrs/passes/ik/order')
	ik_order[len(ik_passes)] = name
	ik_passes[name] = IKDict
	setToState(rtg_key,'attrs/passes/ik/passes',ik_passes)
	setToState(rtg_key,'attrs/passes/ik/order',ik_order)

def addCopyPassToRetarget(rtg_dict, CPDict):
	rtg_dict['attrs']['passes']['cp'] = CPDict
	return rtg_dict

def addCopyPassToRetargetState(rtg_key, CPDict):
	setToState(rtg_key,'attrs/passes/cp',CPDict)

def rebuildRetargetX(rtg_dict):
		rtg_dict['attrs']['X']['mat'][:, :3] = rtg_dict['attrs']['X']['R'] * rtg_dict['attrs']['X']['s']
		rtg_dict['attrs']['X']['mat'][:, 3] = rtg_dict['attrs']['X']['t']
		return rtg_dict

def getNumberOfIKPasses(rtg_key):
	return len(getFromState(rtg_key, 'attrs/passes/ik/order'))

def getIKPassesAndOrder(rtg_key):
	rtg_IK_Passes = getFromState(rtg_key, 'attrs/passes/ik/passes')
	rtg_IK_Order = getFromState(rtg_key, 'attrs/passes/ik/order')
	return rtg_IK_Passes, rtg_IK_Order

def getCopyPass(rtg_key):
	return getFromState(rtg_key,'attrs/passes/cp')

def hasCopyPass(rtg_key):
	try:
		ret = getFromState(rtg_key,'attrs/passes/cp')
		return True if ret is not None else False
	except:
		return False

def getIKPasses(ikPasses,ikOrder):
	return [ikPasses[ikOrder[x]] for x in sorted(ikOrder.keys())]

def getOrderedIKPasses(ikPasses,ikOrder):
	return [(ikPasses[ikOrder[x]], x) for x in sorted(ikOrder.keys())]

def getIKOrder(rtg_key):
	return getFromState(rtg_key, 'attrs/passes/ik/order')

def setIKOrder(rtg_key,ik_order):
	return setToState(rtg_key,'attrs/passes/ik/order',ik_order)

def getIKEnabled(ikPasses,ikOrder):
	ordered_keys = sorted(ikOrder.items(), key=operator.itemgetter(0))
	return [ikPasses[item[1]]['enabled'] for item in ordered_keys]

def getIKLatEnabled(ikPasses,ikOrder):
	ordered_keys = sorted(ikOrder.items(), key=operator.itemgetter(0))
	return [ikPasses[item[1]].get('lat_enabled', False) for item in ordered_keys]

def getIKDeformerMappings(ikPasses,ikOrder):
	ordered_keys = sorted(ikOrder.items(), key=operator.itemgetter(0))
	return [ikPasses[item[1]].get('mappings', None) for item in ordered_keys]

def getIKPertData(ikPasses,ikOrder):
	ordered_keys = sorted(ikOrder.items(), key=operator.itemgetter(0))
	return [ikPasses[item[1]].get('perturbations', {'Amplitude':np.zeros(3,dtype=np.float32), 'Frequency':np.zeros(3,dtype=np.float32)}) for item in ordered_keys]

def getEffectorTargetNames(ikPasses,ikOrder):
	ordered_keys = sorted(ikOrder.items(), key=operator.itemgetter(0))
	return [ikPasses[item[1]]['effectorTargetName'] for item in ordered_keys]

def getEffectorJointNames(ikPasses,ikOrder):
	ordered_keys = sorted(ikOrder.items(), key=operator.itemgetter(0))
	return [ikPasses[item[1]]['effectorJointName'] for item in ordered_keys]

def getJointCutoffNames(ikPasses,ikOrder):
	ordered_keys = sorted(ikOrder.items(), key=operator.itemgetter(0))
	return [ikPasses[item[1]]['jointCutoffName'] for item in ordered_keys]

def getEffectorTargetOffsets(ikPasses,ikOrder):
	ordered_keys = sorted(ikOrder.items(), key=operator.itemgetter(0))
	return [ikPasses[item[1]]['effectorTargetOffset'] for item in ordered_keys]

def getIKWeightsPosition(ikPasses,ikOrder):
	ordered_keys = sorted(ikOrder.items(), key=operator.itemgetter(0))
	return [ikPasses[item[1]]['weightPosition'] for item in ordered_keys]

def getIKWeightsOrientation(ikPasses,ikOrder):
	ordered_keys = sorted(ikOrder.items(), key=operator.itemgetter(0))
	return [ikPasses[item[1]]['weightOrientation'] for item in ordered_keys]

def getJointStiffness(ikPasses,ikOrder):
	ordered_keys = sorted(ikOrder.items(), key=operator.itemgetter(0))
	return [ikPasses[item[1]]['jointStiffness'] for item in ordered_keys]

def getSourceObject(rtg_key):
	global G_All_Things
	rtg_sourceObject = getFromState(rtg_key, 'attrs/sourceObject')
	return G_All_Things[rtg_sourceObject]

def getTargetObject(rtg_key):
	global G_All_Things
	rtg_targetObject = getFromState(rtg_key, 'attrs/targetObject')
	return G_All_Things[rtg_targetObject]

def getSourceObjectName(rtg_key):
	global G_All_Things
	rtg_sourceObject = getFromState(rtg_key, 'attrs/sourceObject')
	return G_All_Things[rtg_sourceObject]['data']['name']

def getTargetObjectName(rtg_key):
	global G_All_Things
	rtg_targetObject = getFromState(rtg_key, 'attrs/targetObject')
	return G_All_Things[rtg_targetObject]['data']['name']

def setEnabled(caller, rtg_key, enabled):
		setToState(rtg_key, 'attrs/enabled', enabled)
		rtg_targetObject = getTargetObject(caller,rtg_key)
		if enabled:
			rtg_targetObject['data']['boneColour'] = COLOUR_RTG
		else:
			rtg_targetObject['data']['boneColour'] = COLOURS['Bone']

def getPass(rtg_key, passName):
	rtg_copyPass = getCopyPass(rtg_key)
	if rtg_copyPass and rtg_copyPass['name'] == passName:
		return rtg_copyPass, 'cp'
	else:
		rtg_ik_passes = getFromState(rtg_key, 'attrs/passes/ik/passes')
		# print rtg_ik_passes.keys()
		# try:
		return rtg_ik_passes[passName], 'ik'
		# except:
		# 	raise ValueError("Pass with name %s not found" % passName)

def setPass(rtg_key, pass_type, pass_name, mpass):
	# try:
	if pass_type == 'cp':
		setToState(rtg_key, 'attrs/passes/cp', mpass)
	else:
		rtg_ik_passes = getFromState(rtg_key, 'attrs/passes/ik/passes')
		if pass_name <> mpass['name']:
			rtg_ik_passes[mpass['name']] = mpass
			del rtg_ik_passes[pass_name]
		else:
			rtg_ik_passes[pass_name] = mpass
		setToState(rtg_key, 'attrs/passes/ik/passes', rtg_ik_passes)
	# except:
	# 	raise ValueError("Pass with name %s not found" % pass_name)

def setPassField(rtg_key,pass_name,field,value):
	mpass, pass_type = getPass(rtg_key,pass_name)
	if isinstance(value,unicode):
		value = str(value)
	mpass[field] = value
	setPass(rtg_key,pass_type,pass_name,mpass)
	return mpass

def swapIKPasses(rtg_key, first_index, second_index):
	rtg_IK_Order = getFromState(rtg_key, 'attrs/passes/ik/order')
	rtg_IK_Order[first_index], rtg_IK_Order[second_index] = rtg_IK_Order[second_index], rtg_IK_Order[first_index]
	setToState(rtg_key, 'attrs/passes/ik/order', rtg_IK_Order)

def getAutoSwizzle(rtg_key, jointPair):
	global G_All_Things
	rtg_source_object = getSourceObject(rtg_key)
	rtg_target_object = getTargetObject(rtg_key)
	srp = rtg_source_object['baseGs']
	trp = rtg_target_object['baseGs']
	scv = rtg_source_object['baseChanValues']
	tcv = rtg_target_object['baseChanValues']
	if srp is None or trp is None:return
	rtg_S = getFromState(rtg_key,'attrs/sourceSkeleton')
	rtg_T = getFromState(rtg_key,'attrs/targetSkeleton')
	s,t=jointPair
	Character.pose_skeleton(srp, rtg_S, scv)
	Character.pose_skeleton(trp, rtg_T, tcv)
	smat = np.eye(3,3,dtype=np.float32) if rtg_S['jointParents'][rtg_S['jointIndex'][s]] == -1 else srp[rtg_S['jointParents'][rtg_S['jointIndex'][s]]][:, :3].T
	tmat = np.eye(3,3,dtype=np.float32) if rtg_T['jointParents'][rtg_T['jointIndex'][t]] == -1 else trp[rtg_T['jointParents'][rtg_T['jointIndex'][t]]][:, :3]
	swiz = np.around(np.dot(smat, tmat), 4).astype(np.float32)
	return swiz

def getAutoCopyOffset(rtg_key, jointPair):
	s, t = jointPair
	rtg_copyPass = getCopyPass(rtg_key)
	rtg_S = atdict(getFromState(rtg_key,'attrs/sourceSkeleton'))
	rtg_T = atdict(getFromState(rtg_key,'attrs/targetSkeleton'))
	Character.pose_skeleton(rtg_S.Gs, rtg_S, np.zeros(rtg_S.numChans, dtype=np.float32))
	Character.pose_skeleton(rtg_T.Gs, rtg_T, np.zeros(rtg_T.numChans, dtype=np.float32))
	rtg_copyPass['copyOffsets'][s] = calculateAutoCopyOffset(rtg_copyPass, rtg_S, rtg_T, s, t)
	jS = rtg_S['jointIndex'][s]
	jT = rtg_T['jointIndex'][t]
	rtg_copyPass, rtg_targetSkeleton = fixJoint(s,jS,jT,rtg_copyPass,rtg_S,rtg_T)
	return rtg_copyPass['copyOffsets'][s]

def calculateAutoCopyOffset(rtg_copyPass, rtg_S, rtg_T, s, t):
	jS = rtg_S['jointIndex'][s]
	jT = rtg_T['jointIndex'][t]
	Ls = rtg_S.Ls[jS]
	Lt = rtg_T.Ls[jT]
	# Gs = np.eye(3,3,dtype=np.float32) if rtg_S['jointParents'][rtg_S['jointIndex'][s]] == -1 else rtg_S.Gs[rtg_S['jointParents'][rtg_S['jointIndex'][s]],:, :3]
	# Gt = np.eye(3,3,dtype=np.float32) if rtg_T['jointParents'][rtg_T['jointIndex'][t]] == -1 else rtg_T.Gs[rtg_T['jointParents'][rtg_T['jointIndex'][t]],:, :3]
	C_O = np.eye(3,4, dtype=np.float32)
	# C_O[:,:3] = np.dot(Gs.T,Gt)
	C_O[:,:3] = np.around(np.dot(Ls[:, :3].T, Lt[:, :3]), 4)
	bS = Retarget.findChildBone(rtg_S, jS)
	bT = Retarget.findChildBone(rtg_T, jT)
	if 0 and bS is not None and bT is not None:
		CbS = np.dot(C_O[:,:3].T, bS)
		CbT = np.dot(C_O[:,:3].T, bT)
		R = Retarget.rotateBetweenVectors(CbS, bT)
		print 'From {} to {}\n'.format(s,t)
		print 'bS: {}\n\nbT: {}\n'.format(bS, bT)
		print 'CbS: {}\n\nCbT: {}\n'.format(CbS, CbT)
		print 'R:\n{}'.format(R)
		C_O[:,:3] = np.dot(C_O[:,:3], R.T)
	rtg_copyPass['copyOffsets'][s] = C_O
	return rtg_copyPass['copyOffsets'][s]

def autoCopyOffsets(rtg_key):
	rtg_copy_pass = getCopyPass(rtg_key)
	rtg_S = atdict(getFromState(rtg_key,'attrs/sourceSkeleton'))
	rtg_T = atdict(getFromState(rtg_key,'attrs/targetSkeleton'))
	Character.pose_skeleton(rtg_S.Gs, rtg_S, np.zeros(rtg_S.numChans, dtype=np.float32))
	Character.pose_skeleton(rtg_T.Gs, rtg_T, np.zeros(rtg_T.numChans, dtype=np.float32))
	for s, t in rtg_copy_pass['jointPairs']: # TODO this is a rubbish way of doing things, says colin
		rtg_copy_pass['copyOffsets'][s] = calculateAutoCopyOffset(rtg_copy_pass, rtg_S, rtg_T, s, t)
	# rtg_copy_pass = fixScalingIssues(rtg_copy_pass, rtg_S, rtg_T)
	setToState(rtg_key,'attrs/passes/cp/copyOffsets',rtg_copy_pass['copyOffsets'])


def copyAutoMatch(rtg_key):
	sourceNames = copy.copy(getFromState(rtg_key,'attrs/sourceSkeleton/jointNames'))
	targetNames = copy.copy(getFromState(rtg_key,'attrs/targetSkeleton/jointNames'))

	jointPairsA = getMatchedPairs(sourceNames,targetNames)
	print "matched", len(jointPairsA), "out of a maximum", min(len(sourceNames), len(targetNames))
	setToState(rtg_key,'attrs/passes/cp/jointPairs',jointPairsA)
	'''
	sourceNames = [y.replace("left","l_").replace("right","r_") for y in [x.lower() for x in sourceNames]]
	targetNames = [y.replace("left","l_").replace("right","r_") for y in [x.lower() for x in targetNames]]
	jointPairsA = self.getMatchedPairs(sourceNames,targetNames)
	'''

def getMatchedPairs(sourceNames, targetNames):
		import difflib
		jointPairs = []
		used = set()
		for sourceName in sourceNames:
			try:
				matchedTarget = difflib.get_close_matches(sourceName, targetNames,2,0)[0]
			except IndexError:
				continue

			# some previous source joint may think it matched this target, but it might not be as
			# good a match as this target. if the best target match for sourceName has already been
			# used then I should check if this is a better match than the one previously chosen
			if matchedTarget in used:

				# find which existing pair used that target. eww
				for p in jointPairs:
					if p[1] != matchedTarget:
						continue

					# which is the best source for the existing target and the current target
					bestSource = difflib.get_close_matches(matchedTarget, [sourceName, p[0]],1,0)[0]
					if p[0] != bestSource:

						# remove the old, and add the new match
						jointPairs.append((bestSource, matchedTarget))
						jointPairs.remove(p)

						# add the removed source back in just in case it can find a better match later.
						sourceNames.append(p[0])
					break
			else:
				used.add(matchedTarget)
				jointPairs.append((sourceName, matchedTarget))
		return jointPairs

def autoSwizzles(rtg_key):
	rtg_source_object = getSourceObject(rtg_key)
	rtg_target_object = getTargetObject(rtg_key)
	srp = rtg_source_object['baseGs']
	trp = rtg_target_object['baseGs']
	scv = rtg_source_object['baseChanValues']
	tcv = rtg_target_object['baseChanValues']
	if srp is None or trp is None:return
	rtg_S = getFromState(rtg_key,'attrs/sourceSkeleton')
	rtg_T = getFromState(rtg_key,'attrs/targetSkeleton')
	Character.pose_skeleton(srp, rtg_S, scv)
	Character.pose_skeleton(trp, rtg_T, tcv)
	rtg_copy_pass = getCopyPass(rtg_key)
	for s, t in rtg_copy_pass['jointPairs']:
		smat = np.eye(3,3,dtype=np.float32) if rtg_S['jointParents'][rtg_S['jointIndex'][s]] == -1 else srp[rtg_S['jointParents'][rtg_S['jointIndex'][s]]][:, :3].T
		tmat = np.eye(3,3,dtype=np.float32) if rtg_T['jointParents'][rtg_T['jointIndex'][t]] == -1 else trp[rtg_T['jointParents'][rtg_T['jointIndex'][t]]][:, :3]
		rtg_copy_pass['copySwizzles'][s] = np.around(np.dot(smat, tmat), 4)
	setToState(rtg_key,'attrs/passes/cp',rtg_copy_pass)

def clearSwizzles(rtg_key):
	setToState(rtg_key,'attrs/passes/cp/copySwizzles',{})

def clearCopyOffsets(rtg_key):
	setToState(rtg_key,'attrs/passes/cp/copyOffsets',{})

def clearPositionOffsets(rtg_key):
	rtg_T = getFromState(rtg_key, 'attrs/targetSkeleton')
	setToState(rtg_key,'attrs/passes/cp/positionOffsets',np.zeros(rtg_T['numChans'], dtype=np.float32))

def printPasses(rtg_key):
	import pprint
	rtg_passes = getFromState(rtg_key,'attrs/passes')
	pprint.pprint(rtg_passes)

def setupMapping(rtg_key):
	retargetData = []

	# IK Pass
	rtg_IK_Passes = getFromState(rtg_key, 'attrs/passes/ik/passes')
	rtg_IK_Order = getFromState(rtg_key, 'attrs/passes/ik/order')
	rtg_sourceSkeleton = atdict(getFromState(rtg_key, 'attrs/sourceSkeleton'))
	rtg_targetSkeleton = atdict(getFromState(rtg_key, 'attrs/targetSkeleton'))
	rtg_X = atdict(getFromState(rtg_key, 'attrs/X'))
	for en, lat_en, lattice, perts, tn, jn, jo, cn, wp, wo, js in zip(getIKEnabled(rtg_IK_Passes,rtg_IK_Order),getIKLatEnabled(rtg_IK_Passes,rtg_IK_Order),
															getIKDeformerMappings(rtg_IK_Passes,rtg_IK_Order),
														  getIKPertData(rtg_IK_Passes,rtg_IK_Order),getEffectorTargetNames(rtg_IK_Passes,rtg_IK_Order),
												getEffectorJointNames(rtg_IK_Passes,rtg_IK_Order),getEffectorTargetOffsets(rtg_IK_Passes,rtg_IK_Order),
												getJointCutoffNames(rtg_IK_Passes,rtg_IK_Order),getIKWeightsPosition(rtg_IK_Passes,rtg_IK_Order),
												getIKWeightsOrientation(rtg_IK_Passes,rtg_IK_Order),getJointStiffness(rtg_IK_Passes,rtg_IK_Order)):
		if not en: continue
		effectorJoints = np.array([rtg_targetSkeleton.jointIndex[tn]], dtype=np.int32)  # T root
		numEffectors = len(effectorJoints)
		effectorOffsets = np.zeros((numEffectors, 3, 4), dtype=np.float32)
		effectorWeights = np.zeros((numEffectors, 3, 4), dtype=np.float32)
		for eo in effectorOffsets[:, :, :3]: eo[:] = jo[:, :3].T * rtg_X.s  # take into account the target scale!
		effectorOffsets[:, :, 3] = 0
		if jo.shape[1] == 4: effectorOffsets[:, :, 3] = jo[:, 3]
		effectorWeights[:, :, :3] = wo  # orientation
		effectorWeights[:, :, 3] = wp  # position
		usedChannels, (usedCAEs, usedCAEsSplits) = SolveIK.computeChannelAffectedEffectors(rtg_targetSkeleton.jointIndex[cn],
																							rtg_targetSkeleton.jointParents,
																							rtg_targetSkeleton.jointChanSplits,
																							effectorJoints)
		usedChannelWeights = np.ones(len(usedChannels), dtype=np.float32)
		for cn, cw in js:
			ci = rtg_targetSkeleton.chanNames.index(cn)
			if ci in usedChannels: usedChannelWeights[list(usedChannels).index(ci)] = cw
		current_lattice = lattice if lat_en else None
		effectorData = (effectorJoints, effectorOffsets, effectorWeights, usedChannels, usedChannelWeights, usedCAEs,
						usedCAEsSplits), [rtg_sourceSkeleton.jointIndex[jn]], current_lattice, perts
		retargetData.append(effectorData)
	rtg_targetSkeleton['retargetData'] = retargetData
	setToState(rtg_key,'attrs/targetSkeleton', rtg_targetSkeleton)


	rtg_copyPass = getFromState(rtg_key, 'attrs/passes/cp')

	if rtg_copyPass and rtg_copyPass['enabled']:
		setToState(rtg_key, 'attrs/passes/cp/copyData', setupCopyData(rtg_copyPass, rtg_sourceSkeleton, rtg_targetSkeleton))


def setupCopyData(rtg_copyPass, rtg_sourceSkeleton, rtg_targetSkeleton):
	jointMapping = np.ones((len(rtg_copyPass['jointPairs']), 4), dtype=np.int32)
	jointSwizzles = []
	jointOffsets = []
	jointPositions = []
	for jmi, (s, t) in enumerate(rtg_copyPass['jointPairs']):
		si, ti = rtg_sourceSkeleton.jointIndex[s], rtg_targetSkeleton.jointIndex[t]
		offset = rtg_copyPass['copyOffsets'].get(s, rtg_copyPass['defaultOffset'])
		swizzle = rtg_copyPass['copySwizzles'].get(s, rtg_copyPass['defaultSwizzle'])
		if swizzle is None: swizzleIndex = -1
		else:
			swizzleIndex = -1
			for mat in enumerate(jointSwizzles):
				if np.all(mat == swizzle): swizzleIndex = si
			if swizzleIndex == -1:
				swizzleIndex = len(jointSwizzles)
				jointSwizzles.append(swizzle)
		if offset is None: offsetIndex = -1
		else:
			offsetIndex = -1
			for mat in enumerate(jointOffsets):
				if np.all(mat == offset): offsetIndex = si
			if offsetIndex == -1:
				offsetIndex = len(jointOffsets)
				jointOffsets.append(offset)
		jointMapping[jmi, :] = np.array([si, ti, swizzleIndex, offsetIndex], dtype=np.int32)
	return jointMapping, np.array(jointSwizzles, dtype=np.float32).reshape(-1, 3, 3), np.array(jointOffsets, dtype=np.float32).reshape(-1, 3, 4)

#-- End State Dict --

def sure(message = 'Your changes will be lost. Are you sure?'):
	if getUndoFromState() not in [None, 'save']:
		ok = QtGui.QMessageBox.critical(None, getAppNameFromState(), message,
									QtGui.QMessageBox.Ok | QtGui.QMessageBox.Default, QtGui.QMessageBox.Cancel)
		return ok != QtGui.QMessageBox.StandardButton.Cancel
	return True

def scaleSelected(caller):
	global G_All_Things, G_Selection
	if len(G_Selection) <> 1:
		errorDialog("Select a single Character", "Select a single Character to be scaled")
		return
	a = G_Selection[0]
	if 'character' in a:
		from PySide.QtGui import QInputDialog
		val, ok = QInputDialog.getDouble(None, "Set Scale", "Scale: ", a['character'].scale, 0.1, 100.0, 2)
		if ok:
			a['character'].scaleCharacter(val)
			caller.qtimeline.refresh()
	else:
		errorDialog('Select a Character', 'Selection must be a target Character')

def retargetSelected(outliner, retarget_widget):
	global G_All_Things, G_Selection, G_RTG_Keys
	if len(G_Selection) != 2:
		errorDialog("Select a source and target", "Select a source with no motion and a target with motion")
		return
	if all([x['animDict'] for x in G_Selection]):
		errorDialog("Can't overwrite motion","Both selected items have motion.  Please select a motion and a skeleton")
		return
	# rtg_key = retarget_widget.rtg_key
	# if rtg_key is not None:
	# 	if not sure("A Retarget operation already exists, do you wish to overwrite it?"):
	# 		return
	# 	deleteKeyFromState(rtg_key)
	# so far, selection is a set - there's no order, so for now, just look for the selected
	# thing with an animDict - that must be the source.
	a,b = G_Selection
	a_key, b_key = None, None
	for thing_key in G_All_Things:
		if a is G_All_Things[thing_key]:
			a_key = thing_key
		elif b is G_All_Things[thing_key]:
			b_key = thing_key
		if a_key is not None and b_key is not None: break
	rtg_dict = newRetargetDict()
	if a['animDict']:
		source_name = a['data']['name']
		target_name = b['data']['name']
		rtg_dict = setRetargetSource(rtg_dict,a,a_key)
		rtg_dict = setRetargetTarget(rtg_dict,b,b_key)
	else:
		source_name = b['data']['name']
		target_name = a['data']['name']
		rtg_dict = setRetargetSource(rtg_dict,b,b_key)
		rtg_dict = setRetargetTarget(rtg_dict,a,a_key)
	rtg_key = uuid4().hex
	rtg_key = addKeyToState(rtg_key,rtg_dict)
	addRTGKey(outliner, rtg_key," to ".join([source_name,target_name]),[a,b])

	# update the rtg_key widget
	retarget_widget.setRtg(rtg_key)
	# self.rtgWidget.setCopyPass(rtg_key.copyPass)
	updateMapping(retarget_widget.parent,rtg_key)
	State.push('Retarget selected')

# @profile
def frameCallback(win, frame):
	global  G_Moving_Things, G_RTG_Dicts, G_RTG_Keys
	for m in G_Moving_Things:
		# what happens if there are gaps in the frameNumbers? too much of an edge case?
		# i've cast the animDict['frameNumbers'] to a set so i can do a faster 'in'
		animDict, skelDict, glSkel = m['animDict'], m['skelDict'], m['primitive']
		if 'maxFrame' not in animDict:
			animDict['maxFrame'] = max(animDict['frameNumbers'])
		if frame % animDict['maxFrame'] not in animDict['frameNumbers']:
			m['data'][K_DRAW] = False # don't draw, there's no motion frame this frame
			continue
		draw_frame = (frame % animDict['maxFrame']) - m['startIndex']
		dofs = animDict['dofData'][draw_frame]
		m['data'][K_DRAW] = True  # ensure that it draws
		Character.pose_skeleton(skelDict['Gs'], skelDict, dofs, m['rootMat'])
		glSkel.setPose(skelDict['Gs'])
	# TODO get parallel working. Currently too much reference to external data which could cause issues
	sorted = np.argsort([x.get('isRoot', False) for x in G_RTG_Dicts.values()])[::-1].tolist()
	char_set = set()
	for ind in sorted:
		rtg_dict = G_RTG_Dicts.values()[ind]
		apply_retarget(frame, rtg_dict, char_set)
	for character in char_set:
		character.updatePose(geo=True, skel=False)
	win.updateGL()

# @profile
def apply_retarget(frame, rtg_dict, char_set):
	global G_All_Things, target_anim, looped, G_Cube_Deformer
	if rtg_dict is None: return
	rtg_enabled = rtg_dict['enabled']
	rtg_sourceSkeleton = atdict(rtg_dict['sourceSkeleton'])
	rtg_targetSkeleton = atdict(rtg_dict['targetSkeleton'])
	rtg_data = rtg_targetSkeleton.get('retargetData', False)
	rtg_X = atdict(rtg_dict['X'])
	rtg_srcAnim = rtg_dict['srcAnim']
	rtg_copyPass = rtg_dict['passes']['cp']
	rtg_targetObject = rtg_dict['targetObject']
	rtg_sourceObject = rtg_dict['sourceObject']
	rtg_targetOffsetY = rtg_dict['targetOffsetY']

	if not rtg_enabled: return
	targetHasCharacter = 'character' in G_All_Things[rtg_targetObject]
	if targetHasCharacter:
		rtg_targetSkeleton = atdict(G_All_Things[rtg_targetObject]['character'].skel_dict)
	if len(target_anim['chan_order']) == 0:
		target_anim['chan_order'] = copy.deepcopy(rtg_targetSkeleton.chanNames)
	rtg_sourceSkeleton.chanValues[:] = rtg_srcAnim[frame % len(rtg_srcAnim)]
	Character.pose_skeleton(rtg_sourceSkeleton['Gs'], rtg_sourceSkeleton, rtg_sourceSkeleton.chanValues)

	# SPINE DATA
	# reg_ex = re.compile('VSS_Spine[0-9]*$')
	# spine_joints = [(x,i) for i, x in enumerate(rtg_sourceSkeleton['jointNames']) if reg_ex.match(x)]
	# spine_splits = [(rtg_sourceSkeleton['jointChanSplits'][2*i+1],rtg_sourceSkeleton['jointChanSplits'][2*(i+1)]) for (_,i) in spine_joints]
	# spine_chan_values = [[rtg_sourceSkeleton['chanValues'][z] for z in xrange(x,y)] for (x,y) in spine_splits]
	# print spine_joints
	# print spine_splits
	# print spine_chan_values
	# f = open('C:/Python/Spine_data.txt','a')
	# f.write(str(spine_chan_values)+'\n')
	# f.close()
	# END SPINE DATA

	rtg_targetSkeleton.chanValues[:] = 0
	if rtg_copyPass and rtg_copyPass['enabled'] and rtg_copyPass['copyData'] <> []:
		Retarget.copy_joints(rtg_sourceSkeleton, rtg_targetSkeleton, rtg_copyPass['copyData']) #, positionOffsets=rtg_copyPass.get('positionOffsets',np.zeros(rtg_targetSkeleton.numChans, dtype=np.float32)))
	if False: rtg_targetSkeleton.chanValues[1] += rtg_targetOffsetY  # make the target character's root be 150mm lower than the source
	if True and rtg_data:
		Ls_old = SolveIK.bake_ball_joints(rtg_targetSkeleton)
		for effectorData, effectorTargetJoints, lattice, perturbations in rtg_data:
			effectorTargets = G_All_Things[rtg_sourceObject]['skelDict']['Gs'][effectorTargetJoints]
			if lattice is not None:
				if G_Cube_Deformer is None:
					G_Cube_Deformer = Base_Deformer.CubeDeformer('Deformer', rtg_sourceSkeleton, [])
				G_Cube_Deformer.mappings = lattice
				G_Cube_Deformer.skel_dict = rtg_sourceSkeleton
				effectorTargets[:,:,3] = G_Cube_Deformer.deformPoints(effectorTargets[:,:,3].reshape(-1,3))
			pert_update = np.zeros(3, dtype=np.float32)
			where = np.where(perturbations['Frequency'] <> 0)[0]
			if where.shape[0] > 0:
				pert_amp = perturbations['Amplitude'][:,where]
				pert_freq = perturbations['Frequency'][:,where]
				pert_update[where] = pert_amp * np.sin((frame * np.pi) / pert_freq)
				effectorTargets[:,:,3] += np.dot(effectorTargets[:,:,:3], pert_update.astype(np.float32).T) #TODO rotate into local frame
			# testDerivatives(skelDict, skelDict.chanValues, effectorData, effectorTargets)
			SolveIK.solveIK(rtg_targetSkeleton, rtg_targetSkeleton.chanValues, effectorData, effectorTargets, outerIts=2, rootMat=rtg_X.mat)
		SolveIK.unbake_ball_joints(rtg_targetSkeleton)
	Xm = [rtg_X.mat, np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)][rtg_targetSkeleton is rtg_sourceSkeleton]
	if targetHasCharacter:
		G_All_Things[rtg_targetObject]['character'].setJointChanValues(rtg_targetSkeleton.chanValues)
		G_All_Things[rtg_targetObject]['character'].updatePose(geo=False)
		char_set.add(G_All_Things[rtg_targetObject]['character'])
	else:
		Character.pose_skeleton(rtg_targetSkeleton.Gs, rtg_targetSkeleton, rtg_targetSkeleton.chanValues, Xm)
		G_All_Things[rtg_targetObject]['primitive'].setPose(rtg_targetSkeleton['Gs'])
		if 'geom' in G_All_Things[rtg_targetObject]:
			poseGeom(G_All_Things[rtg_targetObject]['geom'],rtg_targetSkeleton,
					 G_All_Things[rtg_targetObject]['shape_weights'])

def applyLattice(points, lattice):
	return lattice.deformPoints(points)

def poseGeom(geom_mesh, skel_dict, shape_weights):
	Vs = np.zeros((geom_mesh.numVerts,3), dtype=np.float32)
	for si, sname in enumerate(geom_mesh.names):
		geo = Vs[geom_mesh.vsplits[si]:] # a view on the vertices
		smats,joint_list = shape_weights[sname]
		for joint_name, mi in joint_list.iteritems():
			ji = skel_dict['jointIndex'][joint_name]
			mat = skel_dict['Gs'][ji]
			vis,vs = smats[mi]
			geo[vis] += np.dot(vs, (mat*[10,10,10,1]).T) # TODO fix the scaling in MAReader.evaluate_skinClusters
	geom_mesh.setVs(np.array(Vs,dtype=np.float32))

def getAutoPositionOffset(rtg_key, joint_pair):
	s, t = joint_pair
	rtg_S = atdict(getFromState(rtg_key,'attrs/sourceSkeleton'))
	rtg_T = atdict(getFromState(rtg_key,'attrs/targetSkeleton'))
	rtg_copyPass = getCopyPass(rtg_key)
	if 'positionOffsets' not in rtg_copyPass:
			rtg_copyPass['positionOffsets'] = np.zeros(rtg_T.numChans, dtype=np.float32)
	rtg_copyPass['positionOffsets'] = Retarget.copyJointPose(rtg_S, rtg_T, s, t, offsets=rtg_copyPass['positionOffsets'])
	setToState(rtg_key,'attrs/passes/cp/positionOffsets',rtg_copyPass['positionOffsets'])


def autoPositionOffsets(rtg_key):
	rtg_copyPass = getCopyPass(rtg_key)
	rtg_S = atdict(getFromState(rtg_key,'attrs/sourceSkeleton'))
	rtg_T = atdict(getFromState(rtg_key,'attrs/targetSkeleton'))
	pairs = copy.copy(rtg_copyPass['jointPairs'])
	rtg_copyPass['positionOffsets'] = np.zeros(rtg_T.numChans, dtype=np.float32)
	for s,t in pairs:
		rtg_copyPass['positionOffsets'] = Retarget.copyJointPose(rtg_S, rtg_T, s, t, offsets=rtg_copyPass['positionOffsets'])
	setToState(rtg_key,'attrs/passes/cp/positionOffsets',rtg_copyPass['positionOffsets'])

def fixScalingIssues(rtg_copyPass, rtg_sourceSkeleton, rtg_targetSkeleton):
	# rtg_sourceSkeleton = atdict(rtg_dict['sourceSkeleton'])
	# rtg_targetSkeleton = atdict(rtg_dict['targetSkeleton'])
	# rtg_copyPass = rtg_dict['passes']['cp']
	copy_data = setupCopyData(rtg_copyPass, rtg_sourceSkeleton, rtg_targetSkeleton)
	Retarget.copyJoints2(rtg_sourceSkeleton, rtg_targetSkeleton, copy_data)
	Character.pose_skeleton(rtg_targetSkeleton['Gs'], rtg_targetSkeleton)
	pairs = copy.copy(rtg_copyPass['jointPairs'])
	for s,t in pairs:
		jS = rtg_sourceSkeleton['jointIndex'][s]
		bS = Retarget.findChildBone(rtg_sourceSkeleton, jS)
		jT = rtg_targetSkeleton['jointIndex'][t]
		bT = Retarget.findChildBone(rtg_targetSkeleton, jT)
		if bS is None or bT is None: continue
		bS = np.dot(rtg_sourceSkeleton['Gs'][jS,:,3], bS)
		bT = np.dot(rtg_targetSkeleton['Gs'][jT,:,3], bT)
		if np.dot(bS.T, bT) < 0:
			rtg_copyPass, rtg_targetSkeleton = fixJoint(s,jS,jT,rtg_copyPass,rtg_sourceSkeleton,rtg_targetSkeleton)
			# Flip Joint
	return rtg_copyPass

def fixJoint(s, jS, jT, rtg_copyPass, rtg_sourceSkeleton, rtg_targetSkeleton):
	from itertools import product
	vals = np.zeros(8)
	flips = [np.array(x, dtype=np.float32) for x in product([-1,1],[-1,1],[-1,1])]
	bS = Retarget.findChildBone(rtg_sourceSkeleton, jS)
	for fi, flip in enumerate(flips):
		fix = np.diag(flip)
		rtg_copyPass['copyOffsets'][s] = np.dot(fix, rtg_copyPass['copyOffsets'][s])
		copy_data = setupCopyData(rtg_copyPass,rtg_sourceSkeleton,rtg_targetSkeleton)
		Retarget.copyJoints2(rtg_sourceSkeleton, rtg_targetSkeleton, copy_data)
		Character.pose_skeleton(rtg_targetSkeleton['Gs'], rtg_targetSkeleton)
		bT = Retarget.findChildBone(rtg_targetSkeleton, jT)
		try:
			vals[fi] = np.dot(bS.T, bT)
		except:
			vals[fi] = 0
	fi = np.argmax(vals)
	fix = np.diag(flips[fi])
	rtg_copyPass['copyOffsets'][s] = np.dot(fix, rtg_copyPass['copyOffsets'][s])
	copy_data = setupCopyData(rtg_copyPass,rtg_sourceSkeleton,rtg_targetSkeleton)
	Retarget.copyJoints2(rtg_sourceSkeleton, rtg_targetSkeleton, copy_data)
	Character.pose_skeleton(rtg_targetSkeleton['Gs'], rtg_targetSkeleton)
	return rtg_copyPass, rtg_targetSkeleton


def tempCallback(*x):
	pass

def fileImportDialog(caller, timeline, outliner):
	filename, _ = QtGui.QFileDialog.getOpenFileName(caller, "Import (asf, xcp)..", os.environ.get("HOME"), "Files (*.asf *.xcp *.x2d *.skel *.anim *.amc *.ma)")
	if not filename:
		return
	if not os.path.exists(filename):
		errorDialog("File not found", "Could not find file %s" % filename, '')
		return
	QtGui.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
	fileImport(caller, outliner, timeline, filename)
	QtGui.QApplication.restoreOverrideCursor()
	State.push('File Import')

def fileImport(caller, outliner, timeline, filename, newKey = None, block_update = False):
	''' TODO: get most of this out of here into an import/load/io module that returns data from
	from filepaths (that goes for self.loadSkel and self.loadAsf) '''
	newKey = uuid4().hex if newKey is None else newKey
	new = None
	filename = str(filename) # fix unicode issues
	try:
		if filename.endswith('xcp'):
			from IO import ViconReader
			mats, xcp_data = ViconReader.loadXCP(filename)
			camera_ids = ['%s:%s'%(x['LABEL'],x['DEVICEID']) for x in xcp_data]
			caller.addCameras(mats, camera_ids)
		elif filename.endswith('asf'):
			new = addSkel(loadAsf(filename), filename, newKey, caller)
		elif filename.endswith('skel'):
			new = addSkel(loadSkel(filename), filename, newKey, caller)
		# TODO: fix assumption that there's a skel or asf with matching name to the anim/amc
		elif filename.endswith('anim'):
			animDict = IO.load(filename)[1]  #filename.replace('.anim', '.skel'))[1]
			skelDict = loadSkel(filename.replace('.anim', '.skel'))
			new = addMotion(skelDict, animDict, filename, newKey, caller, timeline)
		elif filename.endswith('amc'):
			asfDict = IO.ASFReader.read_ASF(filename.replace('.amc', '.asf'))
			animDict = IO.ASFReader.read_AMC(filename, asfDict)
			skelDict = IO.ASFReader.asfDict_to_skelDict(asfDict)
			new = addMotion(skelDict, animDict, filename, newKey, caller, timeline)
		elif filename.endswith('vss'):
			pass
		elif filename.endswith('ma'):
			ma_character, _ = MAReader.loadMayaCharacter(filename)
			new = addCharacter(ma_character, filename, newKey, caller)
		else:
			errorDialog("Filetype not recognised", "Grip can't read %s files" % filename.split('.')[-1], '')
		if new is not None:
			outliner.model.add(new)
		if not block_update: caller.refresh()
		try:
			if 'state_data' in new:
				addKeyToState(newKey, new['state_data'])
			else:
				State.addKey(newKey,{'type':FILE_TYPE,'attrs':{'path':filename, 'key':newKey}})
			State.push('File import')
			return newKey
		except Exception, e:
			print str(e)
			return None
	except Exception, e:
		errorDialog("Error Importing...", str(e), traceback.format_exc())
		LOG.error("Error importing", exc_info=True)

def parseMayaFile(filename):
	objs,nodeLists = MAReader.read_MA(filename)
	primitives,primitives2D,mats,camera_ids,movies, skels, \
        shape_weights, fields, dobjs = MAReader.maya_to_state(objs,nodeLists, use_State=False)
	assert len(primitives) == 1, "Error: Currently only supports one mesh per file."
	maPrimitive = primitives[0]
	skelDict = skels[0]
	geomInd = [i for i in xrange(len(maPrimitive['names'])) if maPrimitive['names'][i] in shape_weights]
	Vs, Bs, Ts, Names, Faces = [], [], [], [], []
	for gi in geomInd:
		Vs.append(np.array(maPrimitive['verts'][gi],dtype=np.float32))
		Bs.append(maPrimitive['bones'][gi])
		Ts.append(maPrimitive['transforms'][gi])
		Names.append(maPrimitive['names'][gi])
		Faces.append(maPrimitive['faces'][gi])
	geom = {'Names':Names,'Vs':Vs,'Faces':Faces,'Bs':Bs,'Ts':Ts,'shape_weights':shape_weights}
	return skelDict, geom

def newSceneDict():
	return {'data':{},
			'objecttype':None,
			'children':[],
			'parent':None,
			'primitive':None,
			'skelDict':None,
			'rootMat':None,
			'baseGs':None,
			'baseChanValues':None,
			'animDict':None,
			'startIndex':None}

def addMotion(skelDict, animDict, filename, newKey, caller, timeline):
	global G_Moving_Things, G_All_Things
	glSkel = GLSkel(skelDict['Bs'], skelDict['Gs'])
	new = newSceneDict()
	data = dict(templateObjectData) # start with default
	data.update(glSkel.d) # update with the defaults for a skel
	new['data'] = data
	new['objecttype'] = T_MOTION
	new['data'][K_NAME] = os.path.basename(filename).split('.')[0]
	new['data'][K_FILENAME] = filename
	new['animDict'], new['skelDict'] = animDict, skelDict
	new['baseGs'] = np.copy(skelDict['Gs'])
	new['baseChanValues'] = np.copy(skelDict['chanValues'])
	new['rootMat'] = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
	glSkel.d = new['data'] # the glskel keeps a pointer to the user data.
	new['primitive'] = glSkel
	caller.view().primitives.append(glSkel)
	# hack (because you can't index a set)
	new['startIndex'] = animDict['frameNumbers'][0]
	min_frame, max_frame = animDict['frameNumbers'][0], animDict['frameNumbers'][-1]
	# for speed (need to do 'in' a lot)
	animDict['frameNumbers'] = set(animDict['frameNumbers'])
	G_Moving_Things.append(new)
	G_All_Things[newKey]=new
	# extend the timeline to include the start and end frame numbers from the anim
	timeline.setRange(max(1,min(timeline.lo, min_frame)), max(timeline.hi, max_frame))
	return new

def addCharacter(character, filename, newKey, caller):
	global G_All_Things, templateObjectData, G_Characters
	new = newSceneDict()
	new['data'] = dict(templateObjectData) # start with default
	# new['data'].update(glSkel.d) # update with the defaults for a skel
	new['objecttype'] = T_SKELETON
	new['data'][K_NAME] = character.skel_dict['name'] if 'DISABLE_name' in character.skel_dict else os.path.basename(filename).split('.')[0]
	new['data'][K_FILENAME] = filename
	new['skelDict'] = character.skel_dict
	new['baseGs'] = np.copy(character.skel_dict['Gs'])
	new['baseChanValues'] = np.copy(character.skel_dict['chanValues'])
	# glSkel.d = new['data'] # the glskel keeps a pointer to the user data.
	new['character'] = character
	character.state_data['attrs']['key'] = newKey
	character.state_data['attrs']['filename'] = filename
	new['state_data'] = character.state_data
	for primitive in character.primitives:
		caller.view().primitives.append(primitive)
	new['primitive'] = character.skel_primitive
	G_All_Things[newKey]=new
	G_Characters.append(character)
	return new

def addSkel(skelDict, filename, newKey, caller, geom = None):
	global G_All_Things
	try:
		Vs, Bones, Ts, Faces, Names = [], [], [], [], []
		for i in xrange(skelDict['numJoints']):
			Names.append(skelDict['jointNames'][i])
			Ts.append(skelDict['Gs'][i])
			Bs = [[0,0,0]]
			for j in [x for x in xrange(skelDict['numJoints']) if skelDict['jointParents'][x] == i]:
				Bs.append(skelDict['Ls'][j][:,3])
			Vs.append(Bs)
			Bones.append([(0,i) for i in range(1,len(Bs))])
			Faces.append([])
		glSkel = GLMeshes(names=Names,verts=Vs,faces=Faces,bones=Bones,transforms=Ts)
		# glSkel = GLSkel(skelDict['Bs'], skelDict['Gs'])
	except Exception:
		import pprint;pprint.pprint(skelDict)  #.keys()
		LOG.error("Failed making GLSkel from skeldict.  Bad skel file %s?" % filename, exc_info=True)
		errorDialog("Error Importing...", 'Failed to read skeleton data',
				traceback.format_exc())
		return
	new = newSceneDict()
	new['data'] = dict(templateObjectData) # start with default
	# new['data'].update(glSkel.d) # update with the defaults for a skel
	new['objecttype'] = T_SKELETON
	new['data'][K_NAME] = skelDict['name'] if 'DISABLE_name' in skelDict else os.path.basename(filename).split('.')[0]
	new['data'][K_FILENAME] = filename
	new['skelDict'] = skelDict
	new['baseGs'] = np.copy(skelDict['Gs'])
	new['baseChanValues'] = np.copy(skelDict['chanValues'])
	# glSkel.d = new['data'] # the glskel keeps a pointer to the user data.
	new['primitive'] = glSkel
	if geom is not None:
		geom_mesh = GLMeshes(names=geom['Names'],verts=geom['Vs'],
							 faces=geom['Faces'],bones=geom['Bs'],
							 transforms=geom['Ts'])
		new['geom'] = geom_mesh
		new['shape_weights'] = geom['shape_weights']
		caller.view().primitives.append(geom_mesh)
	caller.view().primitives.append(glSkel)
	G_All_Things[newKey]=new
	return new

def deleteSelected(caller, outliner, retargeter):
	global G_Selection
	for s in copy.copy(G_Selection): # must use a copy because selection changes as things are deleted
		delete(caller, outliner, retargeter, s)
	caller.updateGL()
	State.push('Delete selected')

def deleteRetarget(rtg_key, outliner, retargeter, refresh=True):
	global G_All_Things
	for key, obj in G_All_Things.items():
		if 'rtg' in obj['data'] and rtg_key in obj['data']['rtg']:
			obj['data']['rtg'].remove(rtg_key)
			if len(obj['data']['rtg']) == 0 and obj.get('animDict',None) is None:
				rtg_targetObject = G_RTG_Dicts[rtg_key]['targetObject']
				rtg_targetSkeleton = atdict(G_RTG_Dicts[rtg_key]['targetSkeleton'])
				rtg_targetSkeleton['chanValues'][:] = 0
				Character.pose_skeleton(rtg_targetSkeleton.Gs, rtg_targetSkeleton)
				if 'character' in obj:
					obj['character'].setJointChanValues(rtg_targetSkeleton.chanValues)
					obj['character'].updatePose()
				else:
					obj['primitive'].setPose(rtg_targetSkeleton['Gs'])
					try:
						poseGeom(G_All_Things[rtg_targetObject]['geom'],rtg_targetSkeleton,
						 G_All_Things[rtg_targetObject]['shape_weights'])
					except KeyError:
						pass # No Geometry associated
		elif obj['objecttype'] == RETARGET_TYPE and rtg_key == obj['data']['rtg_key']:
			G_All_Things.pop(key,None)
			outliner.model.remove(obj)
	deleteKeyFromState(rtg_key)
	G_RTG_Keys.remove(rtg_key)
	updateRTGDict()
	retargeter.clear()
	if refresh: retargeter.parent.qtimeline.refresh()

def clearAll(caller,outliner,retargeter,silent=False):
	global G_All_Things, G_Moving_Things, G_RTG_Keys
	if not silent:
		if not sure(): return
	view = caller.view()
	for key, obj in copy.copy(G_All_Things.items()):
		if key not in G_All_Things: continue # Retarget has already been deleted
		try:
			G_Moving_Things.remove(obj)
		except ValueError:
			pass
		if 'rtg' in obj['data'] and len(obj['data']['rtg'])>0:
			rtg_keys = obj['data']['rtg']
			for rtg_key in rtg_keys:
				deleteRetarget(rtg_key,outliner,retargeter, refresh=False)
			caller.qtimeline.refresh()
		try:
			if 'character' in obj:
				for primitive in obj['character'].primitives:
					view.primitives.remove(primitive)
			else:
				if 'primitive' in obj: view.primitives.remove(obj['primitive'])
				if 'geom' in obj:
					view.primitives.remove(obj['geom'])
			outliner.model.remove(obj)
		except ValueError:
			pass
		del G_All_Things[key]
	caller.updateGL()

def delete(caller, outliner, retargeter, obj):
	global G_All_Things, G_Moving_Things, G_Selection, G_RTG_Keys
	if 'rtg_key' in obj['data']:
		rtg_key = obj['data']['rtg_key']
		deleteRetarget(rtg_key,outliner,retargeter)
	else:
		if 'rtg' in obj['data'] and len(obj['data']['rtg'])>0:
			if sure('One or more retargets are associated with this object. Remove Retarget(s)?'):
				rtg_keys = copy.copy(obj['data']['rtg'])
				for rtg_key in rtg_keys:
					deleteRetarget(rtg_key,outliner,retargeter, refresh=False)
				caller.qtimeline.refresh()
			else:
				return
		try:
			if 'character' in obj:
				for primitive in obj['character'].primitives:
					caller.view().primitives.remove(primitive)
			else:
				caller.view().primitives.remove(obj['primitive'])
			outliner.model.remove(obj)
		except KeyError:
			pass
		except ValueError:
			pass
		try:
			G_Moving_Things.remove(obj)
		except ValueError:
			pass
	# try:
	# 	G_Selection.remove(obj)
	# except ValueError:
	# 	pass # wasn't selected
	for key in G_All_Things:
		try:
			if G_All_Things[key] == obj:
				del G_All_Things[key]
				break
		except ValueError:
			continue
	outliner.model.remove(obj)

def hilightPrimitive(p,indexes):
	if hasattr(p['primitive'], 'hilight'):
		p['primitive'].hilight(None)
	if indexes is None:
		return
	for x in indexes:
		if hasattr(p['primitive'], 'hilight'): p['primitive'].hilight(x) #(indexes)

def updateSelection(retarget_panel, selected, deselected):
	''' modify the current selection. typically called via signal from the outliner when the
	selection changes'''
	global G_Selection, G_All_Things
	rtg_key = None
	if len(selected) == 1 and selected[0]['objecttype'] == RETARGET_TYPE:
		rtg_key = selected[0]['data']['rtg_key']
		if rtg_key is not None and retarget_panel.rtg_key <> rtg_key:
			retarget_panel.setRtg(rtg_key)
	print "Selected Key: {}".format(rtg_key or None)
	G_Selection += [obj for obj in selected if obj not in G_Selection]
	for obj in deselected:
		G_Selection.remove(obj)
	for obj in deselected: obj['data'][K_SELECTED] = False
	if rtg_key is None:
		for obj in selected: obj['data'][K_SELECTED] = True
	print "Selected: {}".format([obj['data']['name'] for obj in G_Selection])

def load100Motions(caller,outliner, timeline):
	QtGui.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
	path = os.path.join(os.environ['GRIP_DATA'],'SkeletonTool')
	files = ['man_run.anim']
	for _ in range(100):
		for f in files:
			fileImport(caller,outliner,timeline,os.path.join(path, f))
	import random
	for p in caller.view().primitives:
		p.offset = (random.uniform(-10000,10000),0,random.uniform(-10000,10000))
	# outliner = caller.set_widget['Outliner']
	# outliner.model.setScene(G_All_Things.values())
	# outliner.model.refresh()
	caller.refresh()
	QtGui.QApplication.restoreOverrideCursor()
	State.push('Load 100 motions')

def updateRTGDict(rtg_key = None):
	global G_RTG_Dicts, G_RTG_Keys
	if rtg_key is None:
		G_RTG_Dicts = {}
		for rtg_key in G_RTG_Keys:
			G_RTG_Dicts[rtg_key] = getFromState(rtg_key,'attrs')
	else:
		G_RTG_Dicts[rtg_key] = getFromState(rtg_key,'attrs')

def updateMapping(caller, rtg_key=None):
	if rtg_key is not None: setupMapping(rtg_key)
	updateRTGDict()
	caller.refresh()

def setOutlinerItemRTGKey(outliner, obj, rtg_key):
	index = outliner.model.indexOf(obj)
	index.internalPointer().rtg_key = rtg_key

def setLoadingText(caller,status):
	if status:
		x, y = caller.view().width, caller.view().height
		caller.view().displayText = [{'x':x/2,'y':y/2,'s':'Loading...',
									 'font':GLUT.GLUT_BITMAP_TIMES_ROMAN_24,
									  'color':(1.,0.,0.)}]
	else:
		caller.view().displayText = []
	caller.updateGL()

def reconstructCharacter(attrs, caller):
	global G_All_Things, templateObjectData, G_Characters
	character = Character(attrs['skeleton'], attrs['geometry'])
	new = newSceneDict()
	filename = attrs['filename']
	newKey = attrs['key']
	new['data'] = dict(templateObjectData) # start with default
	# new['data'].update(glSkel.d) # update with the defaults for a skel
	new['objecttype'] = T_SKELETON
	new['data'][K_NAME] = character.skel_dict['name'] if 'DISABLE_name' in character.skel_dict else os.path.basename(filename).split('.')[0]
	new['data'][K_FILENAME] = filename
	new['skelDict'] = character.skel_dict
	new['baseGs'] = np.copy(character.skel_dict['Gs'])
	new['baseChanValues'] = np.copy(character.skel_dict['chanValues'])
	# glSkel.d = new['data'] # the glskel keeps a pointer to the user data.
	new['character'] = character
	character.state_data['key'] = newKey
	character.state_data['filename'] = filename
	new['state_data'] = character.state_data
	for primitive in character.primitives:
		caller.view().primitives.append(primitive)
	G_Characters.append(character)
	G_All_Things[newKey]=new
	return new

def retargeterLoad(caller, outliner, retargeter, timeline):
	if not sure(): return
	filename, filtr = QtGui.QFileDialog.getOpenFileName(caller, 'Choose a file to open', '.', 'RTG (*.rtg)')
	if filename == '': return # did not load
	caller.new()
	clearAll(caller,outliner,retargeter,silent=True)
	setLoadingText(caller,True)
	State.load(filename)
	state_keys = State.uniqueKeys()
	retarget_keys = []
	for key in state_keys:
		obj = getFromState(key,'')
		obj_type = obj.get('type', '')
		if obj_type not in [Character.Type, FILE_TYPE, RETARGET_TYPE]:
			print obj.keys()
		# obj_type = getFromState(key,'type', "")
		if obj_type == Character.Type:
			new = reconstructCharacter(obj['attrs'], caller)
			if new is not None:
				outliner.model.add(new)
		if obj_type == FILE_TYPE:
			filename = obj['attrs']['path']
			if not os.path.exists(filename):
				start_bit = filename.find('ExampleData')
				filename = 'D:/' + filename[start_bit:]
			fileImport(caller,outliner,timeline,filename,newKey=key)
		if obj_type == RETARGET_TYPE:
			retarget_keys.append(key)
	for key in retarget_keys: # Ensure all files are loaded before retargets are added
		print "Retarget Key Found!"
		source_object = getSourceObject(key)
		target_object = getTargetObject(key)
		target_object['data']['boneColour'] = COLOUR_RTG
		retarget_name = 'From {} to {}'.format(source_object['data']['name'],target_object['data']['name'])
		setupMapping(key)
		addRTGKey(outliner,key,retarget_name,[source_object,target_object])
	caller.setFilename(filename)
	updateMapping(caller)
	setLoadingText(caller, False)
	State.push('Load Retargeter')

def retargeterSave(caller):
	retargeterSaveAs(caller, caller.filename)

def retargeterSaveAs(caller, filename = None):
	if filename is None or filename == '':
		filename, filtr = QtGui.QFileDialog.getSaveFileName(caller, 'Choose a file name to save as', '.', 'RTG (*.rtg)')
	if filename == '': return # did not save
	State.save(filename)
	retargeterSetFilename(caller, filename)

def retargeterSetFilename(caller, filename):
	caller.filename = filename
	caller.setWindowTitle(State.appName() + ' - ' + caller.filename.replace('\\','/').rpartition('/')[-1])

def checkItems(caller, outliner, timeline):
	global G_All_Things, G_Moving_Things
	changed = False
	state_keys = State.uniqueKeys()
	state_files = []
	for key in state_keys.values():
		try:
			obj_type = getFromState(key,'type')
		except KeyError:
			continue
		if obj_type == FILE_TYPE:
			state_files.append(getFromState(key,'attrs'))
	for state_file in state_files:
		if state_file['key'] not in G_All_Things.keys():
			changed = True
			fileImport(caller,outliner,timeline,state_file['path'],state_file['key'])
	state_file_keys = set([obj['key'] for obj in state_files])
	for global_key, global_thing in G_All_Things.items():
		if global_key not in state_file_keys and 'primitive' in global_thing:
			changed = True
			G_All_Things.pop(global_key,None)
	return changed

def undo(caller, retarget_panel, outliner, timeline):
	undo_cmd = State.getUndoCmd()
	if undo_cmd in ['LOAD','DELETE']: return
	State.undo()
	# if checkItems(caller, outliner, timeline):
	# 	outliner.refresh()
	retarget_panel.refresh()

def redo(caller, retarget_panel, outliner, timeline):
	State.redo()
	retarget_panel.refresh()
	# if checkItems(caller, outliner, timeline):
	# 	outliner.refresh()


if __name__ == "__main__":
	from UI import GRetargetUI
	GRetargetUI.main()
