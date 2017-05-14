#!/usr/bin/env python

import numpy as np
import math

def read_ASF(filename):
	'''Read an ASF file from disk. Returns a bone dict.'''
	return decode_ASF(parse_ASF(open(filename,'r').readlines()))

def parse_ASF(asfLines):
	'''Parse an ASF file into a dict of str:one of (str,list of dict(str:str),dict(str:str)).'''
	asfDict = {':comment':''}
	asfLines = map(str.strip, asfLines)
	while len(asfLines):
		line = asfLines.pop(0)
		if line == '': continue # ignore blank lines
		if line.startswith('#'): asfDict[':comment']+=line+'\n'; continue # store comments
		assert(line.startswith(':'))
		line.replace('\t',' ')
		fw,_,rw = line.partition(' ')
		if rw != '': asfDict[fw] = rw; continue # inline string
		content = []
		while len(asfLines) and not asfLines[0].startswith(':'):
			line = asfLines.pop(0)
			if line != '' and not line.startswith('#'): # ignore comments and blank lines (shouldn't really be here)
				content.append(line.partition(' '))
		if len(content) and content[0][0] == 'begin': # a list
			data,datum = [],{}
			while len(content):
				if content[0][0].startswith('(') and k == 'limits': # hack to support limits which are a list of (min,max) pairs which can span multiple lines
					v1,_,v2 = content.pop(0)
					datum[k] = datum[k]+v1+' '+v2
					continue
				k,_,v = content.pop(0)
				if k == 'limits(': k,v = 'limits','('+v
				if   k == 'end'  : data.append(datum)
				elif k == 'begin': datum = {}
				else:              assert(not datum.has_key(k)); datum[k] = v
			asfDict[fw] = data
		else: # a dict
			datum = {}
			while len(content):
				k,_,v = content.pop(0)
				assert(not datum.has_key(k))
				datum[k] = v
			asfDict[fw] = datum
	return asfDict

def asfR(a, angleScale):
	x,y,z,order = a['axis'].lower().split()
	R = np.eye(3,dtype=np.float32)
	for v,c in [(z,order[2]),(y,order[1]),(x,order[0])]: # rotations multiply on the right in reverse order
		cv,sv=np.cos(float(v) * angleScale),np.sin(float(v) * angleScale)
		if   c == 'x': R[:,1],R[:,2] = R[:,1]*cv+R[:,2]*sv, R[:,2]*cv-R[:,1]*sv
		elif c == 'y': R[:,2],R[:,0] = R[:,2]*cv+R[:,0]*sv, R[:,0]*cv-R[:,2]*sv
		elif c == 'z': R[:,0],R[:,1] = R[:,0]*cv+R[:,1]*sv, R[:,1]*cv-R[:,0]*sv
	return R

def asfDofs(a):
	if not a.has_key('dof'): return ['','']
	dofs = a['dof'].lower().split()
	return [''.join([t[1] for t in dofs if t.startswith('t')]),''.join([r[1] for r in dofs if r.startswith('r')])]

def asfDofNames(a):
	return ['t'+x for x in a[0]]+['r'+x for x in a[1]]

def asfT(a, lengthScale):
	x,y,z = map(float,a['direction'].split())
	l = float(a['length'])*lengthScale
	return [x*l,y*l,z*l]

def decode_ASF(asfDict, lengthScale = 25.4): # mm output, assuming inches input
	'''Decode a parsed ASF dict into a dictionary of sensible arrays with known units.'''
	# TODO, bones may have (min,max) limits; probably should decode this
	# TODO, other versions of asf?
	assert(asfDict[':version']=='1.10')
	angleScale = {'deg':np.radians(1),'rad':1.0}[asfDict[':units']['angle']]
	lengthScale /= float(asfDict[':units']['length'])
	assert(asfDict[':units']['angle'] == 'deg')
	name = asfDict[':name']
	asf_boneData   = list(asfDict[':bonedata']) # make a copy because we will insert a root node for simplicity
	asf_root       = asfDict[':root']
	ori = asf_root['orientation'].split()
	axi = asf_root['axis'].split()
	if len(axi) == 4 and len(ori) == 3: assert(ori == ['0','0','0']) # weird case; if the orientation is put in the axis then ignore the actual orientation (presumably it's zero...)
	else: axi = ori + axi
	asf_boneData.insert(0,{'id':'0','name':'root','direction':asf_root['position'],'length':'1',\
						   'axis':' '.join(axi),'dof':asf_root['order']})
	asf_hierarchy  = asfDict[':hierarchy']
	numBones       = len(asf_boneData)
	boneNames      = [a['name'] for a in asf_boneData]
	# is 'id' really a compulsory field? not used for anything, not even in amc where it might have reduced the file size
	#boneIds        = [a['id'] for a in asf_boneData]
	boneTs         = [asfT(a, lengthScale) for a in asf_boneData]
	boneRs         = [asfR(a, angleScale) for a in asf_boneData]
	boneDofs       = [asfDofs(a) for a in asf_boneData]
	dofNames       = [n+':'+x for n,a in zip(boneNames,boneDofs) for x in asfDofNames(a)]
	boneParents    = [-1]*numBones
	boneMap        = dict([(n, ni) for ni,n in enumerate(boneNames)])
	for parent,v in asf_hierarchy[0].iteritems():
		pid = boneMap[parent]
		for child in v.split():
			cid = boneMap[child]
			boneParents[cid] = pid
			assert(pid < cid) # we can't use this representation unless parents come before children
	dofCounts    = [len(d[0])+len(d[1]) for d in boneDofs]
	asfDofSplits    = [sum(dofCounts[:i]) for i in xrange(len(dofCounts)+1)]
	dofScales    = [s for ss in [[lengthScale]*len(d[0])+[angleScale]*len(d[1]) for d in boneDofs] for s in ss]
	numDofs      = asfDofSplits[-1]
	return { 'name'           : str(name),
			 'numBones'       : int(numBones),
			 'boneNames'      : boneNames,  # list of strings
			 #'boneIds'        : boneIds,    # list of strings
			 'boneTs'         : np.array(boneTs,dtype=np.float32), # bone translation (global axes, but relative to parent)
			 'boneRs'         : np.array(boneRs,dtype=np.float32), # bone orientation (global axes) encoded as 3x3 matrix
			 'boneDofs'       : boneDofs,   # list of [string,string]s
			 'asfDofSplits'   : np.array(asfDofSplits,dtype=np.int32),
			 'boneParents'    : np.array(boneParents,dtype=np.int32),
			 'dofNames'       : dofNames,   # list of strings
			 'dofScales'      : np.array(dofScales,dtype=np.float32),
			 'numDofs'        : int(numDofs),
			 'lengthScale'    : float(lengthScale),
			 'angleScale'     : float(angleScale) }

def asfDict_to_skelDict(asfDict):
	name = asfDict['name']
	numBones = asfDict['numBones']
	boneNames = asfDict['boneNames']
	boneDofs = asfDict['boneDofs']
	boneParents = asfDict['boneParents']
	dofNames = asfDict['dofNames']
	#dofScales = asfDict['dofScales']
	numDofs = asfDict['numDofs']
	#lengthScale = asfDict['lengthScale']
	#angleScale = asfDict['angleScale']
	# convert the ASF data into our skeleton data format
	jointChans = [[ord(tc)-ord('x') for tc in t]+[ord(rc)-ord('x')+3 for rc in r] for t,r in boneDofs]
	jointChans = np.array([x for y in jointChans for x in y],dtype=np.int32) # flatten
	jointChanSplits = [x for y in [[len(t),len(r)] for t,r in boneDofs] for x in y]
	jointChanSplits = np.array([sum(jointChanSplits[:ji]) for ji in xrange(len(jointChanSplits)+1)],dtype=np.int32)
	Gs, Ls, Bs = boneMatrices(asfDict)
	jointIndex = {}
	for ji,jn in enumerate(boneNames): jointIndex[jn] = ji

	return { 'name'           : str(name),
			 'numJoints'      : int(numBones),
			 'jointNames'     : boneNames,  # list of strings
			 'jointIndex'     : jointIndex, # dict of string:int
			 'jointParents'   : np.array(boneParents,dtype=np.int32),
			 'jointChans'     : np.array(jointChans,dtype=np.int32), # 0 to 5 : tx,ty,tz,rx,ry,rz
			 'jointChanSplits': np.array(jointChanSplits,dtype=np.int32),
			 'chanNames'      : dofNames,   # list of strings
			 'chanValues'     : np.zeros(numDofs,dtype=np.float32),
			 'numChans'       : int(numDofs),
			 'Bs'             : np.array(Bs, dtype=np.float32),
			 'Ls'             : np.array(Ls, dtype=np.float32),
			 'Gs'             : np.array(Gs, dtype=np.float32)
			}

def addTrunnions(asfDict):
	'''Add some strategic no-dof bones that will reveal the orientation of joints.
	These have the additional benefit of making it possible to solve dofs from only joint positions.'''
	# TODO make this work on skelDict
	numBones       = asfDict['numBones']
	boneParents    = asfDict['boneParents']
	boneTs         = asfDict['boneTs']
	boneRs         = asfDict['boneRs']
	boneDofs       = asfDict['boneDofs']
	boneNames      = asfDict['boneNames']
	addBoneParents, addBoneTs, addBoneRs = [],[],[]
	# immutable data, should be passed in
	boneChildren = [np.where(boneParents == bi)[0] for bi in xrange(numBones)]
	numAdded = 0
	# add some end-of-bones where they would be lost.
	boneEnd = [len(bcs)==0 and len(dofs[0])+len(dofs[1]) != 0 for bcs,dofs in zip(boneChildren,boneDofs)]
	# add end-of-bones where the child allows translation
	for pi,dofs in zip(boneParents, boneDofs):
		if pi != -1 and len(dofs[0]) != 0: boneEnd[pi] = True
	for bi in range(numBones):
		if boneEnd[bi]:
			addBoneParents.append(bi)
			addBoneTs.append([0,0,0])
			addBoneRs.append(boneRs[bi])
			boneDofs.append(['',''])
			boneNames.append(boneNames[bi]+'_bone_end')
			numAdded += 1
	for bi,(dofs,bcs,bn,bT,bR) in enumerate(zip(boneDofs, boneChildren, boneNames, boneTs, boneRs)):
		numRotDofs = len(dofs[1])
		numChildren = len(bcs)
		bRT = np.abs(np.dot(bR.T, bT)) # closer to zero is more perpendicular; closer to norm(bT) is more parallel
		if numRotDofs == 0: continue # zero-rotation dofs can be resolved
		if numChildren > 1: continue # if more than 1 child then assume all angles can be resolved
		if numChildren == 1 and numRotDofs == 1: # 1-rot with child can be resolved, unless the rotation is on-axis
			if bRT[int(ord(dofs[1])-ord('x'))] <= 1.0: continue
		#if (numChildren == 1 and numRotDofs <= 2): continue # of no-children joints, only 3+rots earn a trunnion
		axis     = np.argmin(bRT)
		boneT    = bR[:,axis]*30. - bT # 30mm is good; shorter trunnions have larger errors
		boneName = bn+'_trunnion_'+'xyz'[axis]
		addBoneParents.append(bi)
		addBoneTs.append(boneT)
		addBoneRs.append(bR)
		boneDofs.append(['',''])
		boneNames.append(boneName)
		numAdded += 1
	asfDict['boneParents']  = np.concatenate((boneParents,np.array(addBoneParents,dtype=np.int32)))
	asfDict['boneTs']       = np.concatenate((asfDict['boneTs'], np.array(addBoneTs,dtype=np.float32).reshape(-1,3)))
	asfDict['boneRs']       = np.concatenate((asfDict['boneRs'], np.array(addBoneRs,dtype=np.float32).reshape(-1,3,3)))
	asfDict['asfDofSplits'] = np.concatenate((asfDict['asfDofSplits'],np.array([asfDict['asfDofSplits'][-1]]*numAdded,dtype=np.int32)))
	asfDict['numBones']     = numBones + numAdded
	print 'numAdded',numAdded
	return asfDict

def boneMatrices(asfDict):
	'''Generate Global and Local matrices from a bone_dict. The Local matrices, together with the
	dofs and parents, functionally describe the skeleton.'''
	boneParents     = asfDict['boneParents']
	numBones = len(boneParents)

	Gs = np.zeros((numBones,3,4),dtype=np.float32) # GLOBAL mats
	Ls = np.zeros((numBones,3,4),dtype=np.float32) # LOCAL mats
	Bs = np.zeros((numBones,3),dtype=np.float32) # BONES
	# boneRs are in global coordinate system, so they form the rotations of the bone.
	Gs[:,:,:3] = asfDict['boneRs']
	# boneTs are in global coordinate system, being measured relative to their parent
	Bs[:] = asfDict['boneTs']
	Gs[:,:,3] = Bs[:]
	# turn the delta translations into global translations by accumulating the parent offset from the root
	for bi,pi in enumerate(boneParents):
		if pi != -1: Gs[bi,:,3] += Gs[pi,:,3]
	# in our joint representation, the transform of the bone translation happens on the child
	for bi in xrange(numBones-1,-1,-1):
		pi = boneParents[bi]
		if pi != -1: Gs[bi,:,3] = Gs[pi,:,3]
	# now solve for the local matrices using the equation Gs_bi = Gs_pi * Ls_bi
	for bi,pi in enumerate(boneParents):
		if pi == -1:
			Bs[bi] = 0
			Ls[bi,:,:] = Gs[bi,:,:]
		else:
			Gs_pi_T = Gs[pi,:,:3].T
			Bs[bi] = np.dot(Gs[bi,:,:3].T, Bs[bi])
			Ls[bi,:,:3] = np.dot(Gs_pi_T, Gs[bi,:,:3])
			Ls[bi,:,3]  = np.dot(Gs_pi_T, Gs[bi,:,3] - Gs[pi,:,3])
	Bs[np.where(Bs*Bs<0.01)] = 0 # just tidy up a bit by zeroing any tiny offsets < 0.1mm
	return Gs, Ls, Bs

def read_AMC(filename, asfDict):
	'''Read an AMC file from disk. Returns a dictionary of extracted data from the file.'''
	return parse_AMC(open(filename,'r').readlines(), asfDict)

def parse_AMC(amc, asfDict):
	if asfDict.has_key('asfDofSplits'): # this is an asfDict
		boneNames    = asfDict['boneNames']
		asfDofSplits = asfDict['asfDofSplits']
		dofScales    = asfDict['dofScales']
		numDofs      = asfDict['numDofs']
	else: # this is a skelDict (our internal format)
		boneNames    = asfDict['jointNames']
		asfDofSplits = asfDict['jointChanSplits']
		lengthScale,angleScale = 25.4,np.radians(1)
		dofScales    = [s for ss in [[lengthScale]*(b-a)+[angleScale]*(c-b) for a,b,c in zip(asfDofSplits[0:-1:2],asfDofSplits[1::2],asfDofSplits[2::2])] for s in ss]
		asfDofSplits = asfDofSplits[::2]
		numDofs      = asfDict['numChans']
	amc.append('eof') # append a dummy single-word string to mark the end
	amc = [a.split() for a in amc if len(a) and a[0] not in '#:']
	frameStarts  = np.where(np.array(map(len, amc)) == 1)[0] # single-word lines
	numFrames    = len(frameStarts)-1
	frameNumbers = np.zeros(numFrames,dtype=np.int32)
	dofData      = np.zeros((numFrames,numDofs), dtype=np.float32)
	dofMap       = dict([(name, dofData[:,asfDofSplits[ni]:asfDofSplits[ni+1]]) for ni,name in enumerate(boneNames)])
	for fi,(f0,f1) in enumerate(zip(frameStarts[:-1],frameStarts[1:])):
		frameNumbers[fi] = int(amc[f0][0]) # preserve the original numbering of the frameNumbers
		for line in amc[f0+1:f1]:
			tmp = map(float,line[1:])
			dofMap[line[0]][fi,:len(tmp)] = tmp
	dofData *= dofScales
	return { 'dofData'      : dofData,
			 'frameNumbers' : frameNumbers }

def pose_skeleton(Gs, Ls, boneParents, boneDofs, dofSplits, dofValues):
	'''Fill in the Gs from the dofValues and skeleton definition.'''
	assert(dofValues.shape == (dofSplits[-1],))
	for Gs_bi,Ls_bi,pi,(tchans,rchans),di in zip(Gs, Ls, boneParents, boneDofs, dofSplits):
		nt,nr = len(tchans),len(rchans)
		if pi == -1: Gs_pi = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]],dtype=np.float32)
		else       : Gs_pi = Gs[pi]
		#Gs_bi = Gs_pi * Ls[bi] * Dof0 * Dof1 * ...
		np.dot(Gs_pi[:,:3], Ls_bi, out=Gs_bi)
		Gs_bi[:,3] += Gs_pi[:,3]
		if nt: # translation DOFs
			for c,v in zip(tchans, dofValues[di:di+nt]):
				Gs_bi[:,3] += Gs_pi[:,ord(c)-ord('x')] * v
			di += nt
		if nr: # rotation DOFs
			Gs_bi[:,:3] = np.dot(Gs_bi[:,:3], composeR(dofValues[di:di+nr],axes=rchans))
	return Gs

def extractSkeletonDofs(Gs, Ls, boneParents, boneDofs, dofSplits, dofValues):
	'''Fill in the dofValues and global rotations, given target joint positions.'''
	targets = Gs[:,:,3].copy()
	# debugging ... let's make sure we don't use uninitialised data!
	assert(dofValues.shape == (dofSplits[-1],))
	Gs[:,:,:] = float('inf')
	dofValues[:] = float('inf')
	numBones = len(boneParents)
	boneDofCounts = dofSplits[1:] - dofSplits[:-1] # number of dofs per bone
	# this "list of bones that are directly driven only by this bone" could be precomputed
	# here we remove the influence of child bones with translation dofs, since we added end-of-bones points to deal with that
	# we also remove the influence of child bones at the same position, since that could cause numerical problems
	boneZeroChildren = [[bi for bi in np.where(boneParents == bi)[0] if boneDofs[bi][0] == '' and not np.all(Ls[bi,:,3]==0)] for bi in xrange(numBones)]
	# for our skeletons, it is good enough to consider only grandchildren; but here all zero-dof descendents are added
	for bi in xrange(numBones-1,-1,-1):
		pi,bdc = boneParents[bi],boneDofCounts[bi]
		if pi != -1 and bdc == 0: boneZeroChildren[pi].extend(boneZeroChildren[bi])
	for bi,(tgt_bi,Gs_bi,Ls_bi,pi,(tchans,rchans),di,bzcs) in enumerate(zip(targets,Gs, Ls, boneParents, boneDofs, dofSplits, boneZeroChildren)):
		nt,nr = len(tchans),len(rchans)
		if pi == -1: Gs_pi = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]],dtype=np.float32)
		else       : Gs_pi = Gs[pi]
		#Gs[bi] = Gs[pi] * Ls[bi] * Dof0 * Dof1 * ...
		# assume that Gs_pi is complete for now; TODO it might have an unsolved DOF, which is the rotation around the bone axis.
		# we added trunnions to fix this, but it might be possible to generate the trunnions on the fly using cross products of other joints
		np.dot(Gs_pi[:,:3], Ls_bi, out=Gs_bi)
		Gs_bi[:,3] += Gs_pi[:,3]
		if nt: # translation DOFs
			for ddi,c in enumerate(tchans,start=di):
				dofValues[ddi] = v = np.dot(Gs_pi[:,ord(c)-ord('x')], tgt_bi - Gs_bi[:,3])
				Gs_bi[:,3] += Gs_pi[:,ord(c)-ord('x')] * v
			di += nt
		if nr: # rotation DOFs
			numChildren = len(bzcs)
			if numChildren == 0: # no way to solve the dofs, so just zero them
				dofValues[di:di+nr] = 0
			else:
				Lt = np.zeros((numChildren, 3),dtype=np.float32)
				Rt = np.zeros((numChildren, 3),dtype=np.float32)
				for ci,cbi in enumerate(bzcs):
					# we need to solve the R matrix from equations like:  R * Ls[ck] *...* Ls[cj] * Ls[ci,:,3] = Gs[bi,:,:3].T * (Gs[ci,:,3] - Gs[bi,:,3])
					# these equations are the columns of: R Lt.T = Rt.T
					Lt[ci,:] = Ls[cbi,:,3]
					Rt[ci,:] = np.dot(Gs_bi[:,:3].T, targets[cbi] - Gs_bi[:,3])
					pbi = boneParents[cbi]
					while pbi != bi:
						Lt[ci,:] = np.dot(Ls[pbi,:,:3], Lt[ci,:]) + Ls[pbi,:,3]
						pbi = boneParents[pbi]
				rv = fitPointsAndDecomposeR(Lt, Rt, axes=rchans)
				dofValues[di:di+nr] = rv[:nr]
				#if nr == 2: print 'hopefully all zero',nr,rv,rv[nr:]
			Gs_bi[:,:3] = np.dot(Gs_bi[:,:3], composeR(dofValues[di:di+nr],axes=rchans))
	return Gs, dofValues

def composeR(rs, axes='xyz'):
	'''Compose a vector of 3 radians into a 3x3 rotation matrix.
	The rotation order is traditional right-to-left 'xyz'=R(z)*R(y)*R(x).
	The values should be given in the same order (ie in this example: x,y,z).'''
	i = ord(axes[0])-ord('x')
	if len(axes) == 1: parity = 1 # single channel
	else             : parity = (ord(axes[1])-ord(axes[0])+3)
	j,k = (i+parity)%3,(i+2*parity)%3
	if ((parity%3) == 2): rs = -rs
	R = np.zeros((3,3),dtype=np.float32)
	if len(rs) == 1:
		ci,si = math.cos(rs[0]),math.sin(rs[0])
		R[i,i],R[j,i],R[k,i],R[i,j],R[j,j],R[k,j],R[i,k],R[j,k],R[k,k] = 1,0,0,0,ci,si,0,-si,ci
	elif len(rs) == 2:
		ci,cj,si,sj = math.cos(rs[0]),math.cos(rs[1]),math.sin(rs[0]),math.sin(rs[1])
		R[i,i],R[j,i],R[k,i],R[i,j],R[j,j],R[k,j],R[i,k],R[j,k],R[k,k] = cj,0,-sj,si*sj,ci,cj*si,ci*sj,-si,cj*ci
	else:
		ci,cj,ck = np.cos(rs, dtype=np.float32); si,sj,sk = np.sin(rs, dtype=np.float32)
		cc,cs,sc,ss = ci*ck,ci*sk,si*ck,si*sk
		R[i,i],R[j,i],R[k,i],R[i,j],R[j,j],R[k,j],R[i,k],R[j,k],R[k,k] = ck*cj,sk*cj,-sj,sc*sj-cs,ss*sj+cc,cj*si,cc*sj+ss,cs*sj-sc,cj*ci
	return R

def decomposeR(R, axes='xyz'):
	'''Decompose a 3x3 rotation matrix into a vector of 3 radians.
	The rotation order is traditional right-to-left 'xyz'=R(z)*R(y)*R(x).
	The returned values will be in the order specified.'''
	i = ord(axes[0])-ord('x')
	if len(axes) == 1: parity = 1 # single channel
	else:              parity = (ord(axes[1])-ord(axes[0])+3)
	j,k = (i+parity)%3,(i+2*parity)%3
	cj = math.sqrt(R[i,i]*R[i,i] + R[j,i]*R[j,i])
	if cj > 1e-30: ret = np.array([math.atan2(R[k,j],R[k,k]),math.atan2(-R[k,i],cj),math.atan2(R[j,i],R[i,i])],dtype=np.float32)
	else:          ret = np.array([math.atan2(-R[j,k],R[j,j]),math.atan2(-R[k,i],cj),0.0],dtype=np.float32)
	if ((parity%3) == 2): ret = -ret
	return ret #[:len(axes)]

def fitPointsAndDecomposeR(A, B, axes='xyz'):
	'''Given Nx3 matrices A and B with coordinates of N corresponding points.
	Solve R A.T = B.T for rotation matrix R having rotation order and degrees of freedom specified by axes.'''
	R = np.dot(B.T, A) # NOT np.dot(B.T, np.linalg.pinv(A.T,rcond=0.0001))
	if len(axes) == 1: # special case: minimise in 1D (otherwise the solve is unstable)
		i = ord(axes[0])-ord('x')
		R[i,:] = R[:,i] = 0
		R[i,i] = 1
	T = np.linalg.svd(R) # U,S,VT
	R = np.dot(T[0],T[2])
	if np.linalg.det(R) < 0: T[0][:,2] *= -1; R = np.dot(T[0],T[2])
	if len(axes) == 2: # force a 2-parameter estimation of joint angles (potentially better than 3-parameter estimation & zeroing the third value)
		# rewrite the matrix as the outer product of (1,sin,cos) vectors, and compose only the first singular value
		i,j = ord(axes[0])-ord('x'),ord(axes[1])-ord('x')
		k = (2*j+3-i)%3
		svd = np.linalg.svd([[1.0,-R[k,i],R[i,i]],[-R[j,k],R[i,j],R[k,j]],[R[j,j],R[i,k],R[k,k]]])
		[[_,R[k,i],R[i,i]],[R[j,k],R[i,j],R[k,j]],[R[j,j],R[i,k],R[k,k]]] = np.outer(svd[0][:,0],svd[2][0,:])*svd[1][0]
		R[j,i] = 0 # this forces the third value to be 0
		R[k,i]*=-1
		R[j,k]*=-1
	return decomposeR(R, axes)

def fitPoints(A,B, out=None):
	'''Given Nx3 matrices A and B with coordinates of N corresponding points.
	Solve RT A.T = B.T for rotation-translation matrix [R;T].
	R (A - mean(A)).T = (B - mean(B)).T for rotation matrix R.'''
	RT = out
	if RT is None: RT = np.zeros((3,4),dtype = np.float32)
	Bmean,Amean = np.mean(B,axis=0),np.mean(A,axis=0)
	R = np.dot((B - Bmean).T, (A - Amean))
	S0,S1,S2 = np.linalg.svd(R) # U,S,VT
	np.dot(S0,S2,out=R)
	if np.linalg.det(R) < 0: S0[:,2] *= -1; np.dot(S0,S2,out=R)
	RT[:,:3] = R
	RT[:,3] = (Bmean-np.dot(R,Amean))
	return RT

def makeTriangles(graph):
	'''Given a graph of edges (lo,hi), find all the ordered triangles.'''
	gdict = {}
	for lo,hi in graph: gdict[lo] = []; gdict[hi] = []
	for lo,hi in graph: gdict[lo].append(hi)
	tris = [[lo,mid,hi] for lo,mids in gdict.iteritems() for mid in mids for hi in gdict[mid]]
	return tris

def rigidTriangles(data, threshold = 100.):
	'''Given data = numFrames x numVerts x 3 animation data, compute rigid triangles.'''
	print data.shape
	dm, dd = makeVertsDistanceMatrix(data)
	print 'dmdd',dm.shape
	graph = makeGraph(dm,dd, threshold)
	print 'graph',len(graph), graph[:10]
	tris = makeTriangles(graph)
	print 'tris',len(tris), tris[:10]
	filtTris = []
	for t in tris:
		D = data[:,t,:] # numFrames x triVerts x 3
		D0 = D[0]
		dx,dy = D0[1]-D0[0],D0[2]-D0[0]
		if np.dot(dx,dy)**2/(np.dot(dx,dx)*np.dot(dy,dy)) > 0.9: continue # weed out too-straight triangles
		filtTris.append(t)
	print 'filtTris',len(filtTris)
	RTs = stabilizeGroups(data, filtTris)
	return filtTris, RTs

def stabilizeAssignment(data, assignment):
	'''Given data = numFrames x numVerts x 3 animation data and group label per vertex,
	compute stabilizing RTs = numGroups x numFrames x 3 x 4 (to the first frame).'''
	groups = [np.where(assignment == gi)[0] for gi in xrange(max(assignment)+1)]
	return stabilizeGroups(data, groups)

def stabilizeGroups(data, groups):
	'''Given data = numFrames x numVerts x 3 animation data and list of groups of vertices,
	compute stabilizing RTs = numGroups x numFrames x 3 x 4 (to the first frame).'''
	numGroups = len(groups)
	numFrames = data.shape[0]
	RTs = np.zeros((numGroups,numFrames,3,4), dtype=np.float32)
	for group,RT in zip(groups,RTs):
		D = data[:,group,:]
		for r,d in zip(RT,D): fitPoints(d, D[0], out=r)
	return RTs

def assignmentResidual(data, RTs, thresholdDistance):
	'''Given data = numFrames x numVerts x 3 animation data and stabilizing RTs = numTris x numFrames x 3 x 4
	compute the reconstruction residual for assigning each vertex to each of the triangles.'''
	numTris = RTs.shape[0]
	numVerts = data.shape[1]
	res = np.zeros((numTris,numVerts),dtype=np.float32)
	for ti,RT in enumerate(RTs):
		alignData = applyRT_list(RT, data)
		# calculate the variance of each point
		res2 = np.mean(np.sum((alignData[0] - alignData)**2,axis=2,dtype=np.float32),axis=0,dtype=np.float32)
		np.clip(res2,0,thresholdDistance,out=res[ti])
	return res

def bestTriangle(res, resids):
	bestImprovement,bestIndex = 0,-1
	for index,res2 in enumerate(resids):
		replace = np.where(res > res2)[0]
		improvement = np.sum(res[replace] - res2[replace])
		if improvement > bestImprovement: bestImprovement,bestIndex = improvement,index
	return bestImprovement/len(res),bestIndex

def assignAndStabilize(data, RTs, thresholdDistance):
	'''Given data = numFrames x numVerts x 3 animation data and stabilizing RTs = numGroups x numFrames x 3 x 4
	assign each data point to one of the triangles and compute the minimum reconstruction residual.
	Returns the assignment, the residuals, and the stabilized data points (to the first frame).'''
	numVerts = data.shape[1]
	res = np.ones(numVerts,dtype=np.float32)*thresholdDistance
	assignment = -np.ones(numVerts,dtype=np.int32)
	stableData = np.zeros_like(data)
	for gi,RT in enumerate(RTs):
		alignData = applyRT_list(RT, data)
		# calculate the variance of each point
		res2 = np.mean(np.sum((alignData[0] - alignData)**2,axis=2,dtype=np.float32),axis=0,dtype=np.float32)
		replace = np.where(res2 < res)[0]
		res[replace] = res2[replace]
		stableData[:,replace,:] = alignData[:,replace,:]
		assignment[replace] = gi
	return assignment, res, stableData

def unstabilize(stableData, RTs):
	'''Given stableData = numGroups x 3 animation data and stabilizing RTs = numGroups x numFrames x 3 x 4
	Returns the data = numFrames x numGroups x 3, animated (undoing the stabilizing transform).'''
	numGroups = stableData.shape[0]
	numFrames = RTs.shape[1]
	data = np.zeros((numFrames,numGroups,3),dtype=np.float32)
	for gi,(RT,sd) in enumerate(zip(RTs,stableData)):
		data[:,gi,:] = unapplyRT_list(RT,sd.reshape(1,-1)).reshape(-1,3)
	return data

def invert_matrix_array(RTs):
	ret = np.zeros_like(RTs)
	ret[:,:3,:3] = np.transpose(RTs[:,:3,:3],(0,2,1))
	for rti,rto in zip(RTs,ret):
		rto[:,3] = -np.dot(rto[:3,:3],rti[:,3])
	return ret
	
def transform_pair_residual(RT1, RT2):
	numFrames = RT1.shape[0]
	R1T = np.transpose(RT1[:,:,:3],axes=(0,2,1))
	R2T = np.transpose(RT2[:,:,:3],axes=(0,2,1))
	T1 = RT1[:,:,3]
	T2 = RT2[:,:,3]
	A = np.zeros((numFrames*3,3),dtype=np.float32)
	B = np.zeros((numFrames*3),dtype=np.float32)
	A[:] = (R1T - R2T).reshape(-1,3)
	for b,r1,t1,r2,t2 in zip(B.reshape(-1,3),R1T,T1,R2T,T2): b[:] = np.dot(r1,t1)-np.dot(r2,t2)
	O,res,_,_ = np.linalg.lstsq(A, B, rcond=0.0001)
	res = np.mean((B-np.dot(A,O))**2) # why isn't res this?
	O = np.dot(RT1[0,:,:3],O)+RT1[0,:,3]
	return res,O
	
def sharedStablePoints(RTs, threshold=float('inf')):
	'''Given stabilizing RTs = numGroups x numFrames x 3 x 4, look for pairs of groups (g1,g2) that have a common stable point.
	RTs[g1,fi,:,:3] * xi + RTs[g1,fi,:,3] = O
	RTs[g2,fi,:,:3] * xi + RTs[g2,fi,:,3] = O
	(RTs[g2,fi,:,:3].T - RTs[g1,fi,:,:3].T) . O =  RTs[g2,fi,:,:3].T . RTs[g2,fi,:,3] - RTs[g1,fi,:,:3].T . RTs[g1,fi,:,3]
	Return a list of group pairs and stable points.'''
	numGroups = RTs.shape[0]
	ret = []
	for (g1,g2) in ((g1,g2) for g1 in xrange(numGroups) for g2 in xrange(numGroups)):
		if g1 == g2: continue
		res,O = transform_pair_residual(RTs[g1],RTs[g2])
		if res < threshold:
			ret.append((g1,g2,O,res))
	return ret

def groupRepresentatives(data, stabilizedPointToGroup):
	'''Given data = numFrames x numPoints x 3 and stabilizedPointToGroup = numPoints (groupIndex),
	Choose a representative (central) point in each group.'''
	numGroups = max(stabilizedPointToGroup)+1
	ret = -np.ones(numGroups,dtype=np.int32)
	for gi in xrange(numGroups):
		points = np.where(stabilizedPointToGroup == gi)[0]
		D = data[0,points,:]
		res = np.sum((D - np.mean(D,axis=0))**2,axis=1)
		try:
			ret[gi] = points[np.argmin(res)]
		except:
			continue
	return ret

def applyRT(RT, data):
	'''RT is 3x4; data is a numVerts x 3'''
	return np.dot(data, RT[:,:3].T) + RT[:,3] # numVerts x 3

def applyRT_list(RT, data):
	'''RT is numFrames x 3 x 4; data is a numFrames x numVerts x 3'''
	ret = np.zeros_like(data)
	for o,d,r,t in zip(ret, data, RT[:,:,:3], RT[:,:,3]):
		o[:] = np.dot(d,r.T) + t
	return ret

def unapplyRT_list(RT, data):
	'''RT is numFrames x 3 x 4; data is a numVerts x 3'''
	numFrames = RT.shape[0]
	numVerts = data.shape[0]
	ret = np.zeros((numFrames,numVerts,3),dtype=np.float32)
	for o,r,t in zip(ret, RT[:,:,:3], RT[:,:,3]):
		o[:] = np.dot(data - t,r)
	return ret
	
def greedyTriangles(data, maxNum = None, triangleThreshold = 500., thresholdDistance = 100.):
	'''Given data = numFrames x numVerts x 3 animation data
	1) set model M = {}
	2) form T = rigidTriangles(data)
	3) find the t in T - M that minimises residual(data,M + {t})
	4) if the residual reduced by enough: M += {t}, goto 3

	where residual(data, model) is a (robust) measure of the total variance of the data, given that each data point must be assigned
	to move rigidly with one of the triangles in the model.'''
	if maxNum is None: maxNum = data.shape[0]
	M = []
	numVerts = data.shape[1]
	res = np.ones(numVerts,dtype=np.float32)*thresholdDistance
	tris,RTs = rigidTriangles(data, threshold=triangleThreshold)
	numTris = len(tris)
	print 'numTriangles',numTris
	resBest = thresholdDistance
	resids = assignmentResidual(data,RTs,thresholdDistance)
	triIndices = []
	remainIndices = list(range(numTris))
	for it in range(maxNum):
		improvement,bestIndex = bestTriangle(res, resids[remainIndices])
		if improvement == 0.0: print '0.0 improvement'; break
		ti = remainIndices.pop(bestIndex)
		triIndices.append(ti)
		res = np.min((res,resids[ti]),axis=0)
		print 'it %d/%d minres %2.2fmm' % (it+1,maxNum,np.sqrt(np.mean(res)))
	return {'tris':tris,'RTs':RTs,'triIndices':triIndices}

def makeDistanceMatrix(boneDict, animDict):
	'''Make a distance matrix for the joints. Probably we want to do this for the markers instead!'''
	boneParents = boneDict['boneParents']
	boneDofs = boneDict['boneDofs']
	dofData = animDict['dofData']
	dofSplits = animDict['dofSplits']
	Gs, Ls, Bs = boneMatrices(boneDict)
	numFrames = dofData.shape[0]
	numJoints = len(boneParents)
	vertices = np.zeros((numFrames,numJoints,3), dtype=np.float32)
	for fi,dofValues in enumerate(dofData):
		Gs = pose_skeleton(Gs, Ls, boneParents, boneDofs, dofSplits, dofValues)
		vertices[fi,:,:] = Gs[:,:,3]
	return makeVertsDistanceMatrix(vertices)

def makeVertsDistanceMatrix(vertices):
	'''Given a numFrames x numVerts x 3 data matrix of animating vertices, compute the distance matrix.'''
	numVerts = vertices.shape[1]
	print 'numVerts',numVerts
	dm = np.zeros((numVerts,numVerts),dtype=np.float32)
	dd = np.zeros((numVerts,numVerts),dtype=np.float32)
	for bi in xrange(numVerts):
		for bj in xrange(bi):
			d = vertices[:,bi] - vertices[:,bj]
			d2 = np.sum(d*d,axis=-1)
			d2_mean = np.mean(d2)
			d2 -= d2_mean
			d2_dev = math.sqrt(np.mean(d2*d2))
			dm[bi,bj] = dm[bj,bi] = d2_mean
			dd[bi,bj] = dd[bj,bi] = d2_dev
	return dm, dd

def makeGraph(dm, dd, threshold = 4000):
	'''Given a matrix of mean square-distance and deviation of that, choose the stiff edges.'''
	graph = []
	numJoints = dm.shape[0]
	for bi in xrange(numJoints):
		for bj in xrange(bi):
			if dd[bi,bj] < threshold: graph.append([bj,bi]) # roughly s.d. of 1cm at 20cm (2x10x200)
	return graph

def retargetJoints(joints, targetJoints_tm1, graph, targetInvLengths, targetDirs, positionConstraints = [], oits=7, iits = 7):
	'''Implement velocity constraints by predicting the position of the joint from targetJoints_tm1.'''
	numJoints = joints.shape[0]
	targetJoints = np.zeros((numJoints,3), dtype=np.float32)
	if targetJoints_tm1 is None: targetJoints[:] = joints
	else:                        targetJoints[:] = targetJoints_tm1
	def err(tjs, graph, tils, tds, pcs):
		E = 0
		for (bi,pi),til, td in zip(graph, tils, tds):
			d       = (tjs[bi]-tjs[pi])*til
			dg      = (d[0]*d[0]+d[1]*d[1]+d[2]*d[2]) - 1.0
			if til == 1.0: dg += 1.0
			d_td    = d - td
			do      = (d_td[0]*d_td[0]+d_td[1]*d_td[1]+d_td[2]*d_td[2]) # orientation
			E      += dg*dg + do
		for ti,tp,tw in pcs:
			dp      = (tjs[ti]-tp)*tw
			E      += np.dot(dp,dp)
		return E
	def derr2(tjs, graph, tils, tds, pcs):
		size = (len(graph)*4+len(pcs)*3)
		njs = len(tjs)
		E = np.zeros((size),dtype=np.float32)
		dE = np.zeros((size,njs,3),dtype=np.float32)
		ni = 0
		for ((bi,pi),til,td) in zip(graph, tils, tds):
			d       = (tjs[bi]-tjs[pi])*til
			dg      = (d[0]*d[0]+d[1]*d[1]+d[2]*d[2]) - 1.0
			if til == 1.0: dg += 1.0
			d_td    = d - td
			E[ni]   = dg
			dE[ni,bi] += d*(2*til)
			dE[ni,pi] -= d*(2*til)
			R = range(ni+1,ni+4)
			E[R]   = d_td
			dE[R,bi,[0,1,2]] += til
			dE[R,pi,[0,1,2]] -= til
			ni += 4
		for ti,tp,tw in pcs:
			dp         = (tjs[ti]-tp)*tw
			E[ni:ni+3] = dp
			dE[[ni,ni+1,ni+2],ti,[0,1,2]] = tw
			ni += 3
		assert(ni == size)
		return E,dE

	for oit in xrange(oits):
		E,dE = derr2(targetJoints, graph, targetInvLengths, targetDirs, positionConstraints)
		A = dE.reshape(-1,numJoints*3)
		delta = np.linalg.lstsq(A, -E, rcond=0.0001)
		delta,alpha,bestAlpha,bestE = delta[0].reshape(-1,3),0.1,0.0,err(targetJoints, graph, targetInvLengths, targetDirs, positionConstraints)
		for iit in xrange(iits):
			testJoints = targetJoints + (bestAlpha+alpha)*delta
			testE = err(testJoints, graph, targetInvLengths, targetDirs, positionConstraints)
			if testE < bestE: bestAlpha,bestE = bestAlpha+alpha,testE
			else: alpha *= -0.707 # toggle around the best
			if bestAlpha+alpha < 0: alpha=-alpha
		targetJoints += bestAlpha*delta
		print oit, bestE
	E,targetJoints = bestE,targetJoints+bestAlpha*delta
	return targetJoints

def decorrelate(M):
	# renormalise
	mx,mn = np.max(M,axis=1),np.min(M,axis=1)
	scl = np.array([[mv,Mv][Mv > -mv] for Mv,mv in zip(mx,mn)]).reshape(-1,1)
	M /= scl
	for it in xrange(2):
		last_di = -1
		for ci in xrange(M.shape[1]):
			di = np.argmax(np.abs(M[:,ci]))
			if di > last_di and abs(M[di,ci]) > 0.1:
				last_di += 1
				# swap it in position
				M[[di,last_di],:] = M[[last_di,di],:]
				di = last_di
			# now zero out other rows
			if abs(M[di,ci]) > 0.5: # actually it should be really close to 1.0
				for dj in xrange(M.shape[0]):
					if dj != di: # anything above 0.4 gives the same result
						M[dj,:] -= M[di,:] * M[dj,ci]/M[di,ci]
		# renormalise
		mx,mn = np.max(M,axis=1),np.min(M,axis=1)
		scl = np.array([[mv,Mv][Mv > -mv] for Mv,mv in zip(mx,mn)]).reshape(-1,1)
		M /= scl
	return M
		
def decomposeDofs(animData):
	'''Given an animation, find a mostly diagonal, sparse, low-dimensional, linear space that best approximates/compresses it.'''
	import pylab as pl
	# animData_frames,channels = animDofs_frames,dofs * d2c_dofs,channels. we want the sparse d2c coding.
	# assume the first 6 dofs are root
	tmp = animData[:-1,6:] - animData[1:,6:]
	u,s,vt = np.linalg.svd(tmp, full_matrices=False)
	#pl.plot(np.log(s/s[0]))
	#pl.hold()
	#pl.show()
	rank = np.where(s>=s[0]*1e-4)[0][-1]+1
	print 'rank',rank
	vt = vt[:rank,:]
	nch = vt.shape[1]
	scale = np.exp(-(1.0/(nch*nch))*np.array(range(nch))).reshape(1,nch) # gently weight the channels to encourage a diagonal
	vt = vt * scale
	u,s,vt = np.linalg.svd(np.dot(vt.T,vt))
	vt = decorrelate(vt[:rank,:] / scale)
	vt[:] = np.around(vt * 100.0)
	sel = np.where(np.abs(vt) >= 3)
	print 'num shared dofs',len(zip(*sel))
	pl.imshow((vt[:,:]))
	pl.hold()
	pl.show()
	return [(i,i,100) for i in range(6)]+[(x+6,y+6,int(vt[x,y])) for x,y in zip(*sel)]

def convertASFAMC_to_SKELANIM(asf, amc, skelFilename, animFilename):
	import IO
	asfDict = read_ASF(asf)
	#skelDict = addTrunnions(skelDict)
	animDict = read_AMC(amc, asfDict)
	skelDict = asfDict_to_skelDict(asfDict)
	IO.save(skelFilename, skelDict)
	IO.save(animFilename, animDict)

def convertASF_to_SKEL(asf, skelFilename):
	import IO
	asfDict = read_ASF(asf)
	#asfDict = addTrunnions(asfDict)
	skelDict = asfDict_to_skelDict(asfDict)
	IO.save(skelFilename, skelDict)
