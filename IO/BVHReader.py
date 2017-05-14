#!/usr/bin/env python

import numpy as np
import sys
from GCore import Character

def read_BVH(filename, lengthScale = 1.0, angleScale = np.radians(1)):
	'''Read an BVH file from disk. Returns a bone dict.'''
	return decode_BVH(parse_BVH(open(filename,'r').readlines()), lengthScale=lengthScale, angleScale=angleScale)

def parse_BVH(bvhLines):
	'''Parse an BVH file into a dict of str:one of (str,list of dict(str:str),dict(str:str)).'''
	bvhDict = {':comment':''}
	bvhLines = map(str.strip, bvhLines)
	chans = ['Xposition', 'Yposition', 'Zposition', 'Xrotation', 'Yrotation', 'Zrotation']
	while len(bvhLines):
		line = bvhLines.pop(0)
		if line == '': continue # ignore blank lines
		if line.startswith('#'): bvhDict[':comment']+=line+'\n'; continue # store comments
		if line.startswith('HIERARCHY'):
			bvhDict['HIERARCHY'] = []
			depth = 0
			parents = []
			parent = None
			tail = bvhDict['HIERARCHY']
			while len(bvhLines):
				line = bvhLines.pop(0)
				if line.startswith('End'):
					tmp = bvhLines.pop(0)
					assert(tmp == '{')
					endOffset = bvhLines.pop(0).split()
					tail.append([None,parent,map(float,endOffset[1:]),[]])
					tmp = bvhLines.pop(0)
					assert(tmp == '}')
				if line.startswith('ROOT') or line.startswith('JOINT'):
					_,name = line.split(None,1)
					brace = bvhLines.pop(0)
					assert(brace == '{')
					depth += 1
					parents.append(parent)
					offset = bvhLines.pop(0).split()
					assert(offset[0] == 'OFFSET')
					channels = bvhLines.pop(0).split()
					assert(channels[0] == 'CHANNELS')
					assert(int(channels[1]) == len(channels)-2)
					item = [name,parent,map(float,offset[1:]),[chans.index(c) for c in channels[2:]]]
					tail.append(item)
					parent = name
					if len(bvhLines) and bvhLines[0].startswith('JOINT'): continue
				if line == '}':
					depth -= 1
					parent = parents.pop()
					if depth == 0: break
			continue
		if line.startswith('MOTION'):
			frameCount = bvhLines.pop(0).split()
			frameTime = bvhLines.pop(0).split()
			bvhDict['MOTION'] = {'frameCount':int(frameCount[1]),'fps':1.0/float(frameTime[2]), 'data': np.array([map(float,line.split()) for line in bvhLines if len(line)])}
	return bvhDict

def decode_BVH(bvhDict, lengthScale = 1.0, angleScale = np.radians(1)): # mm/rad output, assuming mm/deg input
	'''Decode a parsed BVH dict into a dictionary of sensible arrays with known units.'''
	angleScale = np.radians(1)
	name = 'charname'
	numJoints = len(bvhDict['HIERARCHY'])
	jointNames = []
	jointIndex = {}
	jointParents = []
	jointChans = []
	jointTs = []
	jointChanSplits = [0]
	chanNames = []
	chanData = bvhDict['MOTION']['data']
	numChans = chanData.shape[1]
	chanExts = [':tx',':ty',':tz',':rx',':ry',':rz']

	aa = 0
	for jn,parent,offset,chans in bvhDict['HIERARCHY']:
		print "aa= ", aa
		aa = aa + 1
		ji = len(jointNames)
		if jn == None: jn = parent+'_dummy'
		jointNames.append(jn)
		jointIndex[jn] = ji
		jointTs.append(offset)
		pi = -1
		if parent != None: pi = jointNames.index(parent)
		jointParents.append(pi)
		jc = len(jointChans) + np.sum(np.array(chans)<=2)
		jointChans.extend(chans)
		for ci in chans: chanNames.append(jn+chanExts[ci])
		jointChanSplits.append(jc)
		jointChanSplits.append(len(jointChans))

	# TODO FIXME stupid reversed rotation channel order from ASF legacy
	chanOrder = [s for ss in [list(range(a,b))+list(range(c-1,b-1,-1)) for a,b,c in zip(jointChanSplits[0:-1:2],jointChanSplits[1::2],jointChanSplits[2::2])] for s in ss]
	assert(len(jointChans) == len(chanOrder))
	jointChans = [jointChans[c] for c in chanOrder]
	chanNames = [chanNames[c] for c in chanOrder]
	chanData = chanData[:,chanOrder].copy()
	numChans = len(chanNames)
	
	Gs = np.zeros((numJoints,3,4),dtype=np.float32) # GLOBAL mats
	Ls = np.zeros((numJoints,3,4),dtype=np.float32) # LOCAL mats
	Bs = np.zeros((numJoints,3),dtype=np.float32) # BONES

	Ls[:,:,3] = jointTs
	aa = 0
	for ji,pi in enumerate(jointParents):
		print "aa= ", aa
		print "ji=", ji
		print "pi= ", pi
		aa = aa+1
		Ls[ji,:3,:3] = np.eye(3)
		Gs[ji,:3,:3] = np.eye(3)
		if pi != -1:
			Gs[ji,:,3] = Gs[pi,:,3] + Ls[ji,:,3]
			Bs[pi] = jointTs[ji] # TODO we only support one bone per joint; so we ought to introduce dummy joints?


	dofScales    = [s for ss in [[lengthScale]*(b-a)+[angleScale]*(c-b) for a,b,c in zip(jointChanSplits[0:-1:2],jointChanSplits[1::2],jointChanSplits[2::2])] for s in ss]
	chanData *= dofScales

	return { 'name'           : str(name),
			 'numJoints'      : int(numJoints),
			 'jointNames'     : jointNames,  # list of strings
			 'jointIndex'     : jointIndex, # dict of string:int
			 'jointParents'   : np.array(jointParents,dtype=np.int32),
			 'jointChans'     : np.array(jointChans,dtype=np.int32), # 0 to 5 : tx,ty,tz,rx,ry,rz
			 'jointChanSplits': np.array(jointChanSplits,dtype=np.int32),
			 'chanNames'      : chanNames,   # list of strings
			 'chanValues'     : np.zeros(numChans,dtype=np.float32),
			 'numChans'       : int(numChans),
			 'Bs'             : np.array(Bs, dtype=np.float32),
			 'Ls'             : np.array(Ls, dtype=np.float32),
			 'Gs'             : np.array(Gs, dtype=np.float32),
			 'dofData'        : np.array(chanData,dtype=np.float32),
			 'frameNumbers'   : np.array(range(len(chanData)),dtype=np.int32),
			 'fps'            : bvhDict['MOTION']['fps'],
			 'markerParents'  : np.array([],dtype=np.int32),
			 'markerOffsets'  : np.zeros((0,3),dtype=np.float32),
			 'markerNames'    : [],
 			 'markerWeights'  : np.array([],dtype=np.float32),
			 'numMarkers'     : int(0),
			 'sticks'         : [],
			}

def convertBVH_to_SKELANIM(bvh, skelFilename, animFilename):
	import IO
	bvhDict = read_BVH(bvh)
	skelDict = bvhDict_to_skelDict(bvhDict)
	animDict = {'dofData':skelDict.pop('dofData'),'frameNumbers':skelDict.pop('frameNumbers'),'fps':skelDict.pop('fps'),'frameCount':skelDict.pop('frameCount')}
	IO.save(skelFilename, skelDict)
	IO.save(animFilename, animDict)

def convertBVH_to_SKEL(bvh, skelFilename):
	import IO
	bvhDict = read_BVH(bvh)
	skelDict = bvhDict_to_skelDict(bvhDict)
	IO.save(skelFilename, skelDict)

if __name__ == '__main__':

	import IO
	import os; atHome=os.path.isdir('/dev')
	homeDir = '../../data/'
	#first = 'c3po'
	#second = 'c3po2'
	#skelDict = IO.load(homeDir+'%s.skel' % first)[1]
	#animDict = IO.load(homeDir+'%s.anim' % first)[1]
	#animDict2 = IO.load(homeDir+'%s.anim' % second)[1]

	scene = 'trot'
	scene = 'gallop'
	dr = 'C:/Users/ColinD/Desktop/'
	if atHome: dr = '../../../../Captury/'
	filename = '%s%s.bvh' % (dr,scene)
	skelDict = read_BVH(filename)
	animDict = {'dofData':skelDict.pop('dofData'),'frameNumbers':skelDict.pop('frameNumbers')}

	# recompute the channels; change the root channel order; remove no-effect channels
	dofData = animDict['dofData']
	jointChans = skelDict['jointChans']
	jointChanSplits = skelDict['jointChanSplits']
	numFrames = len(dofData)
	numJoints = skelDict['numJoints']
	import ASFReader
	for ji in range(numJoints):
		c0,c1,c2=jointChanSplits[2*ji:2*ji+3]
		if c2 == c1 + 3: # 3-rot
			axes = ''.join([[None,None,None,'x','y','z'][ct] for ct in jointChans[c1:c2]])
			targetAxes = ['zxy','zyx'][ji!=0]
			jointChans[c1:c2] = [ord(c)-ord('x')+3 for c in targetAxes]
			for dofs in dofData:
				R = ASFReader.composeR(dofs[c1:c2],axes)
				dofs[c1:c2] = ASFReader.decomposeR(R,targetAxes)
	dofData[np.where(dofData*dofData < 1e-12)] = 0.0

	# delete no-effect channels
	data = np.zeros((numFrames,numJoints,3),dtype=np.float32)
	nonzeros = np.where(np.mean(dofData,axis=0) != 0)[0]
	jointChans = jointChans[nonzeros].copy()
	dofData = dofData[:,nonzeros].copy()
	jointChanSplits = np.array([np.sum(nonzeros < si) for si in jointChanSplits],dtype=np.int32)
	skelDict['jointChans'] = jointChans
	skelDict['jointChanSplits'] = jointChanSplits
	chanNames = []
	chanExts = [':tx',':ty',':tz',':rx',':ry',':rz']
	for jn,c0,c1 in zip(skelDict['jointNames'], jointChanSplits[:-1:2],jointChanSplits[2::2]):
		for ci in jointChans[c0:c1]: chanNames.append(jn+chanExts[ci])
	print chanNames
	skelDict['chanNames'] = chanNames
	skelDict['numChans'] = len(nonzeros)
	skelDict['chanValues'] = np.zeros(len(nonzeros),dtype=np.float32)
	animDict['dofData'] = dofData

	skelDict2 = read_BVH(filename)

	d2c = ASFReader.decomposeDofs(dofData[:-1])
	D2C = np.zeros((45,33),dtype=np.float32)
	dofNames = skelDict['chanNames']
	for di,ci,wt in d2c:
		D2C[ci,di] = wt * 0.01
		
	print [(di,skelDict['chanNames'][ci], wt) for di,ci,wt in d2c]
	#exit()
	[(0, 'root:tx', 100), (1, 'root:ty', 100), (2, 'root:tz', 100), (3, 'root:rz', 100), (4, 'root:ry', 100), (5, 'root:rx', 100), (6, 'spine_2:rx', -100), (6, 'spine_3:rx', -100), (6, 'spine_1:rx', 100), (7, 'left_hip:rz', 100), (8, 'left_hip:rx', 100), (9, 'left_stifle:rx', 100), (10, 'left_hock:rx', 100), (11, 'right_hip:rz', 100), (12, 'right_hip:rx', 100), (13, 'right_stifle:rx', 100), (14, 'right_hock:rx', 100), (15, 'tail_1:rz', 100), (15, 'tail_2:rz', 75), (15, 'tail_3:rz', 50), (15, 'tail_4:rz', 25), (16, 'tail_1:rx', 100), (16, 'tail_2:rx', 75), (16, 'tail_3:rx', 50), (16, 'tail_4:rx', 25), (17, 'tail_1:rz', -33), (17, 'tail_3:rz', 33), (17, 'tail_4:rz', 67), (17, 'tail_5:rz', 100), (18, 'tail_1:rx', -33), (18, 'tail_3:rx', 33), (18, 'tail_4:rx', 67), (18, 'tail_5:rx', 100), (19, 'left_clavicle:rx', 100), (20, 'left_shoulder:rz', 100), (21, 'left_shoulder:rx', 100), (22, 'left_elbow:rx', 100), (23, 'left_knee:rx', 100), (24, 'right_clavicle:rx', 100), (25, 'right_shoulder:rz', 100), (26, 'right_shoulder:rx', 100), (27, 'right_elbow:rx', 100), (28, 'right_knee:rx', 100), (29, 'neck_1:ry', 100), (29, 'neck_2:ry', 20), (29, 'neck_3:ry', 10), (30, 'neck_1:rx', 100), (30, 'neck_2:rx', 20), (30, 'neck_3:rx', 10), (31, 'head:rz', 100), (32, 'head:rx', 100)]

	homeDir = '../../data/'
	import IO
	IO.save(homeDir+'%s.skel' % scene, skelDict)
	IO.save(homeDir+'%s.anim' % scene, animDict)

	
	#u,s,vt = np.linalg.svd(dofData[:,6:])[1]
	#print np.linalg.pca(dofData[:,6:])
	
	#print nonzeros
	#print np.sum(dofData==0,axis=0)
	##print list(dofData[:,zeros])
	#print len(jointChans),len(nonzeros)
	
	#for fi,dofs in enumerate(dofData):
		#dofs[:6] = 0.0
		#Character.pose_skeleton(skelDict['Gs'], skelDict, dofs)
		#data[fi] = skelDict['Gs'][:,:,3]
		
	
	def setFrame(fi):
		QGLViewer.timeline.frameStep = 1
		drawing_skel2 = True
		global animDict
		dofs = animDict['dofData'][(fi-animDict['frameNumbers'][0])%len(animDict['frameNumbers'])].copy()
		#dofs[[2,5]] = dofData[0,[2,5]]
		Character.pose_skeleton(skelDict['Gs'], skelDict, dofs)
		QGLViewer.skel.setPose(skelDict['Gs'])
		if drawing_skel2:
			dofs = skelDict2['dofData'][(fi-skelDict2['frameNumbers'][0])%len(skelDict2['frameNumbers'])].copy()
			Character.pose_skeleton(skelDict2['Gs'], skelDict2, dofs)
			QGLViewer.skel2.setPose(skelDict['Gs'])
		QGLViewer.view.updateGL()

	timeRange = (animDict['frameNumbers'][0],animDict['frameNumbers'][-1])
	from UI import QGLViewer
	QGLViewer.makeViewer([], timeRange=timeRange, callback=setFrame, skelDict=skelDict, altSkelDict=skelDict)

