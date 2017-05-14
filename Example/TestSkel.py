import numpy as np
import sys, os

from PySide import QtCore, QtGui, QtOpenGL

import ISCV
import IO
from IO import MovieReader, ASFReader
import UI
from Example import GViewer
from UI import GLSkeleton, GLGrid, GLPoints3D, GLBones


def calibrateCamera(objpoints,imgpoints,width,height):
	#print np.min(imgpoints,axis=0),np.max(imgpoints,axis=0)
	import cv2
	print objpoints.shape, imgpoints.shape,width,height
	camMat = np.eye(3,dtype=np.float32)
	camMat[0,0],camMat[1,1] = 2000,2000
	camMat[0,2],camMat[1,2] = width*0.5,height*0.5
	distCoeffs = np.zeros(5,dtype=np.float32)
	ret, K_3x3, dist, rv, tv = \
		cv2.calibrateCamera([np.array(objpoints,dtype=np.float32)], [np.array(imgpoints,dtype=np.float32)], \
							(width,height),camMat,distCoeffs,\
							flags=cv2.CALIB_USE_INTRINSIC_GUESS|cv2.CALIB_FIX_ASPECT_RATIO|cv2.CALIB_FIX_K2|cv2.CALIB_FIX_K3|cv2.CALIB_ZERO_TANGENT_DIST)
	P = np.eye(4,dtype=np.float64)
	P[:3,:3] = np.dot(K_3x3,cv2.Rodrigues(np.array(rv[0]))[0])
	P[:3,3] = np.dot(K_3x3, np.array(tv[0])).T
	return ret, P, dist[0]

def setFrame(newFrame):
	global frame, view, allSkels,points,joints,bones
	frame = newFrame
	for Gs3, Ls3, skelDict3, animData3,skel3 in allSkels:
		dofs3 = animData3[frame%len(animData3)]
		Gs3 = ASFReader.pose_skeleton(Gs3, Ls3, skelDict3['jointParents'], skelDict3['jointDofs'], skelDict3['dofSplits'], dofs3)
		skel3.vertices[:] = Gs3[:,:,3]

	global md,img,g_detectingDots,g_readingMovie
	if g_readingMovie and md is not None:
		try:
			MovieReader.readFrame(md, seekFrame=(frame-videoFrameOffset)/4)
		except:
			frame = videoFrameOffset
			MovieReader.readFrame(md, seekFrame=(frame-videoFrameOffset)/4)
		if g_detectingDots:
			ret = ISCV.detect_bright_dots(img, 254, 200, 190)
			good = [r for r in ret if min(r.sxx,r.syy) > 0.1 and min(r.sxx,r.syy) < 100.0] # and r.sxy*r.sxy<=0.01*r.sxx*r.syy]
			print len(good),'good points'
			for r in good:
				#print r.sx,r.sy,r.sxx,r.sxy,r.syy
				img[int(r.sy-5):int(r.sy+5),int(r.sx-5):int(r.sx+5),:] = [0,255,0]
		view.refreshImageData()
	global animJoints,stablePointsGroups,displayFrames,groupRepresentatives
	pfr = np.searchsorted(goodFrames, frame)
	points.vertices = displayFrames[pfr % len(displayFrames)]
	if animJoints is not None: joints.vertices[:] = animJoints[pfr%len(animJoints)]
	bones.vertices[::2] = joints.vertices
	bones.vertices[1::2] = points.vertices[groupRepresentatives[stablePointsGroups]]

	view.updateGL()

def nextFrame():
	global frame
	setFrame(frame + 4)

def generateSkeleton(cacheId=622, x3d_filename='', perc=0.9, triangleThreshold=1000, thresholdDistance=25., useFrames=range(0, 514),
                     numComps=30, labelGraphThreshold=4, stepSize=1):

	directory = os.path.join(os.environ['GRIP_DATA'], '140113_A2_GRIP_GenPeople')
	c3d_filename = 'ROM.c3d'
	cameraB = 7292,34,(47,2.3,-0.2),(67,788,79)
	cameraA = 5384,49.8,(5.4,1.1,-0.7),(4,1135,0) #5045,52,(5.6,0.9,-0.7),(0,1130,0)
	camera = cameraA
	startFrame = 0

	tempDir = os.environ['GRIP_TEMP']

	#import logging
	#logging.basicConfig(level=logging.DEBUG)
	from IO import C3D

	graph_out_fn = None

	# labelGraphThreshold = 4
	# stepSize = 1

	if True: # bo data
		c = C3D.read(os.path.join(directory, c3d_filename))
		c3d_frames, c3d_fps = c['frames'],c['fps']
		pointLabels = c['labels']
		print 'c3d fps = ', c3d_fps
		numFramesVisiblePerPoint = np.sum(c3d_frames[:,:,3]==0,axis=0)
		numPointsVisiblePerFrame = np.sum(c3d_frames[:,:,3]==0,axis=1)
		print 'threshold',0.40 * len(c3d_frames)
		goodPoints = np.where(numFramesVisiblePerPoint > 0.90 * len(c3d_frames))[0]
		goodFrames = np.where(np.sum(c3d_frames[:,goodPoints,3]==0,axis=1) == len(goodPoints))[0]
		print len(goodPoints),len(goodFrames) # 290 x 6162 (80%), 283 x 8729 (90%), 275x10054 (96%)
		frames = c3d_frames[goodFrames,:,:][:,goodPoints,:][:,:,:3]
		pointLabels = [pointLabels[g] for g in goodPoints]
		#badPoint = pointLabels.index('BoDense:A_Neck_1')

		data_fn = 'W90-28-10.IO'
		skel_out_fn = None
		triangleThreshold = 1000.
	else: # orn data
		# cacheId = 622
		# perc = 0.9
		# triangleThreshold = 1000.      # Bone threshold
		# thresholdDistance = 25.        # Joint threshold
		# useFrames = range(2370, 3500)
		# useFrames = range(2650, 3500)
		# useFrames = range(2600, 3480)
		# useFrames = range(2650, 3480)
		# useFrames = range(0, 2000)
		# useFrames = range(0, 514)
		#useFrames = [] #range(0, 1000)
		#useFrames.extend(range(2650, 3480))
		#useFrames.extend(range(4824, 5253))
		# numComps = 30

		# useFrames = range(0, 333)
		# useFrames = range(4824, 5253)

		print 'CacheId:', cacheId
		print 'Good point percentage:', perc
		print 'Triangle threshold:', triangleThreshold
		print 'Distance threshold:', thresholdDistance
		print 'Frames:', useFrames[0], '-', useFrames[-1]

		_, x3d_data = IO.load(x3d_filename)
		data_fn = 'W90-28-8.romtracks_T%d.IO' % cacheId
		location = '/root/tracks'
		# location = '/root/skeleton/reconstruction/collection/c3ds'
		c3d_frames = x3d_data[location]['x3ds']
		print c3d_frames.shape
		c3d_frames = np.transpose(c3d_frames, axes=(1, 0, 2))
		#frames = frames[:, blueIds, :]
		print c3d_frames.shape
		pointLabels = x3d_data[location]['x3ds_labels']

		if False:
			goodPoints = np.arange(c3d_frames.shape[1])
			goodFrames = np.arange(len(c3d_frames))
		else:
			numFramesVisiblePerPoint = np.sum(c3d_frames[useFrames, :, 3] == 0, axis=0)
			numPointsVisiblePerFrame = np.sum(c3d_frames[useFrames, :, 3] == 0, axis=1)
			goodPoints = np.where(numFramesVisiblePerPoint > perc * len(useFrames))[0]
			goodFrames = np.where(np.sum(c3d_frames[:, goodPoints, 3] == 0, axis=1) == len(goodPoints))[0]

		print '# Good points: %d | # Good frames: %d' % (len(goodPoints), len(goodFrames))
		print goodFrames[:4]
		frames = c3d_frames[goodFrames, :, :][:, goodPoints, :][:, :, :3]
		pointLabels = [int(pointLabels[g]) for g in goodPoints]

		skel_out_fn = None
		graph_out_fn = None

	data = frames[::stepSize, :, :].copy()
	first_time_only = not os.path.exists(os.path.join(tempDir, data_fn))
	if first_time_only: # generate the file
		M = ASFReader.greedyTriangles(data, numComps, triangleThreshold=triangleThreshold, thresholdDistance=thresholdDistance**2) # only every Nth frame
		IO.save(os.path.join(tempDir, 'M90_T%d.IO' % cacheId), M)
		_, M = IO.load(os.path.join(tempDir, 'M90_T%d.IO' % cacheId))
		stabilizedPointToGroup,stabilizedPointResiduals,stabilizedFrames = ASFReader.assignAndStabilize(data, M['RTs'][M['triIndices'][:28]], thresholdDistance=thresholdDistance**2)
		W = {'stabilizedPointToGroup':stabilizedPointToGroup,'stabilizedPointResiduals':stabilizedPointResiduals,'stabilizedFrames':stabilizedFrames}
		IO.save(os.path.join(tempDir, data_fn), W)
	else:
		_data = IO.load(os.path.join(tempDir, data_fn))[1]
		stabilizedPointToGroup = _data['stabilizedPointToGroup']
		stabilizedPointResiduals = _data['stabilizedPointResiduals']
		stabilizedFrames = _data['stabilizedFrames']

	print 'numFrames = %d' % len(stabilizedFrames)
	print 'number of labelled points %d' % np.sum(stabilizedPointToGroup != -1)
	print 'RMS of labelled points %fmm' %np.sqrt(np.mean(stabilizedPointResiduals[np.where(stabilizedPointToGroup!=-1)]))
	first_time_only = True
	print stabilizedPointToGroup
	num_groups = max(stabilizedPointToGroup)+1
	stabilized_groups = [np.where(stabilizedPointToGroup==gi)[0] for gi in range(num_groups)]
	if first_time_only:
		if True: # tighten the fit
			# thresh = [10,10,9,9] #,10,10,9,7,9,9,6,9,9,9,]
			thresh = [thresholdDistance, thresholdDistance, thresholdDistance - 1, thresholdDistance - 1, thresholdDistance - 2, thresholdDistance - 2]
			# thresh = [20, 20, 19, 19, 10, 10, 9, 9]
			for t in thresh:
				#stabilizedPointToGroup[badPoint] = -1 # unlabel
				RTs = ASFReader.stabilizeAssignment(data, stabilizedPointToGroup)
				stabilizedPointToGroup,stabilizedPointResiduals,stabilizedFrames = ASFReader.assignAndStabilize(data, RTs, thresholdDistance = float(t)**2)
				print 'number of labelled points %d' % np.sum(stabilizedPointToGroup != -1)
				print 'RMS of labelled points %fmm' %np.sqrt(np.mean(stabilizedPointResiduals[np.where(stabilizedPointToGroup!=-1)]))
		else:
			RTs = ASFReader.stabilizeAssignment(data, stabilizedPointToGroup)
			stabilizedPointToGroup,stabilizedPointResiduals,stabilizedFrames = ASFReader.assignAndStabilize(data, RTs, thresholdDistance = 10.**2)

		global animJoints,stablePointsGroups,displayFrames,groupRepresentatives
		stablePointsData = ASFReader.sharedStablePoints(RTs, threshold=3.**2)
		stablePointsGroups = [sp[0] for sp in stablePointsData]
		stablePoints = np.array([sp[2] for sp in stablePointsData],dtype=np.float32)
		print 'num stable points',len(stablePoints)

	def residual(gi, leaf_indices, RTs):
		'''given a group and a list of attachment points, choose the best attachment point and return the residual.'''
		tmp = [(ASFReader.transform_pair_residual(RTs[gi], RTs[gj]),gj) for gj in leaf_indices]
		return min(tmp)
		
	# make a skeleton from stabilizedPointToGroup
	root_group = 0
	leaf_nodes = set([root_group])
	skel_group_indices = [root_group]
	skel_joint_parents = [-1]
	groups = set(range(stabilizedPointToGroup.max()+1))
	groups.remove(root_group)
	RTs = ASFReader.stabilizeAssignment(data, stabilizedPointToGroup)	
	joints = []
	joints.append(np.mean(data[0,stabilized_groups[root_group]],axis=0))
	bones = []
	bones.append([])
	G = np.eye(3,4,dtype=np.float32)
	G[:,3] = np.mean(data[0,stabilized_groups[root_group]],axis=0)
	Gs = [G]
	while groups:
		residuals = [(residual(gi,leaf_nodes,RTs),gi) for gi in groups]
		(((res,O),parent),group) = min(residuals)
		groups.remove(group)
		leaf_nodes.add(group)
		skel_group_indices.append(group)
		pi = skel_group_indices.index(parent)
		skel_joint_parents.append(pi)
		joint_world = np.float32(O)
		joints.append(joint_world)
		bones.append([np.mean(data[0,stabilized_groups[group]],axis=0) - O])
		bones[pi].append(joint_world - joints[pi])
		print group,parent
		G = np.eye(3,4,dtype=np.float32)
		G[:,3] = O
		Gs.append(G)
	print skel_group_indices
	print skel_joint_parents

	numJoints = len(skel_joint_parents)
	jointNames = map(str, skel_group_indices)
	jointIndex = dict(zip(jointNames,range(len(jointNames))))
	jointParents = skel_joint_parents
	jointChans = [0,1,2]+[3,4,5]*numJoints
	jointChanSplits = [0,3,6]
	for x in range(numJoints-1):
		jointChanSplits.append(jointChanSplits[-1])
		jointChanSplits.append(jointChanSplits[-1]+3)
	dofNames = [jointNames[ji]+[':tx',':ty',':tz',':rx',':ry',':rz'][jointChans[di]] for ji in range(numJoints) for di in range(jointChanSplits[2*ji],jointChanSplits[2*ji+2])]
	numDofs = len(dofNames)

	def mult_inv(Gs_pi, Gs_gi):
		# Gs_pi^-1 Gs_gi = Ls_gi
		R = np.linalg.inv(Gs_pi[:3,:3])
		ret = np.dot(R, Gs_gi)
		ret[:,3] -= np.dot(R,Gs_pi[:,3])
		return ret

	Ls = np.float32([mult_inv(Gs[pi],Gs[gi]) if pi != -1 else Gs[gi] for gi,pi in enumerate(skel_joint_parents)])
	Bs = bones
	
	print map(len,Bs)
	
	markerParents = [skel_group_indices.index(gi) for gi in stabilizedPointToGroup if gi != -1]
	markerNames = [('%d'%pi) for pi,gi in enumerate(stabilizedPointToGroup) if gi != -1]
	labelNames = [('%d' % pointLabels[pi]) for pi, gi in enumerate(stabilizedPointToGroup) if gi != -1]
	markerOffsets = [np.dot(Gs[skel_group_indices.index(gi)][:3,:3].T, data[0][pi]-Gs[skel_group_indices.index(gi)][:3,3]) for pi,gi in enumerate(stabilizedPointToGroup) if gi != -1]

	skel_dict = {
		'name': 'skeleton',
		'numJoints': int(numJoints),
		'jointNames': jointNames,  # list of strings
		'jointIndex': jointIndex,  # dict of string:int
		'jointParents': np.int32(jointParents),
		'jointChans': np.int32(jointChans),  # 0 to 5 : tx,ty,tz,rx,ry,rz
		'jointChanSplits': np.int32(jointChanSplits),
		'chanNames': dofNames,  # list of strings
		'chanValues': np.zeros(numDofs, dtype=np.float32),
		'numChans': int(numDofs),
		'Bs': Bs,
		'Ls': np.float32(Ls),
		'Gs': np.float32(Gs),
		'markerParents': np.int32(markerParents),
		'markerNames': markerNames,
		'markerOffsets': np.float32(markerOffsets),
		'markerWeights': np.ones(len(markerNames), dtype=np.float32),
		'rootMat': np.eye(3, 4, dtype=np.float32),
		'labelNames': labelNames
	}

	if graph_out_fn is not None and labelGraphThreshold != -1:
		print 'Generating labelling graph...'
		from GCore import Label as GLabel
		c3d_data = c3d_frames[goodFrames, :, :][:, goodPoints, :][:, :, :]
		c3d_data = c3d_data[::stepSize, :, :]
		# graph = GLabel.graph_from_c3ds(skel_dict, markerNames, c3d_data, threshold=3)
		graph = GLabel.graph_from_c3ds(skel_dict, markerNames, c3d_data, threshold=labelGraphThreshold)
		IO.save(graph_out_fn, {'/root/graph': {'label_graph': graph}})
		print 'Labelling graph saved to:', graph_out_fn

	if skel_out_fn is not None: IO.save(skel_out_fn, skel_dict)


	def test_skeleton(sd):
		'''TODO, write some code to verify that a dict actually is a skeleton.'''
		assert isinstance(sd['name'],str), 'name key should be a string'
		numJoints = sd['numJoints']
		assert isinstance(numJoints,int), 'numJoints key should be an int'
		
		
	
	
	animJoints = None
	showStabilized = False
	if showStabilized:
		displayFrames = stabilizedFrames
		pointToGroup = stabilizedPointToGroup
	else: # show animated
		displayFrames = frames #c3d_frames[:,:,:3]
		displayLabels = pointLabels
		if first_time_only: # generate the file
			framesRTs = ASFReader.stabilizeAssignment(displayFrames, stabilizedPointToGroup)
			IO.save(os.path.join(tempDir, 'tmp90-28.IO'), {'framesRTs':framesRTs, 'stabilizedPointToGroup':stabilizedPointToGroup, 'stablePoints':stablePoints, 'stablePointsGroups':stablePointsGroups})
		for k,v in IO.load(os.path.join(tempDir, 'tmp90-28.IO'))[1].iteritems():locals()[k] = v
		animJoints = ASFReader.unstabilize(stablePoints, framesRTs[stablePointsGroups])
		print 'animJoints shape', animJoints.shape
		pointToGroup = -np.ones(displayFrames.shape[1],dtype=np.int32)
		print goodPoints.shape, pointToGroup.shape, stabilizedPointToGroup.shape
		pointToGroup = stabilizedPointToGroup
		#pointToGroup[goodPoints] = stabilizedPointToGroup # for displayFrames = c3d_frames[:,:,:3]
	groupRepresentatives = ASFReader.groupRepresentatives(data, stabilizedPointToGroup)
	numJoints = len(stablePoints)
	boneEdges = np.array(range(2*numJoints),dtype=np.int32)
	boneVertices = np.zeros((numJoints*2,3),dtype=np.float32)
	boneVertices[::2] = stablePoints
	boneVertices[1::2] = displayFrames[0,groupRepresentatives[stablePointsGroups]]

	#import cv2
	#movie = cv2.VideoCapture(directory+movieFilename)
	#frameOk, frameData = movie.read()
	#global md
	#md = {'buffer':frameData, 'height':frameData.shape[0], 'width':frameData.shape[1]}


	global app, win, view, frame, points,joints,bones
	app = QtGui.QApplication(sys.argv)
	app.setStyle('plastique')
	win = QtGui.QMainWindow()
	win.setFocusPolicy(QtCore.Qt.StrongFocus) # get keyboard events
	win.setWindowTitle('Imaginarium Skeleton Reconstruction Test %d' % cacheId)
	panel = GViewer.QGLPanel()
	view = panel.view
	view.setMinimumWidth(640)
	view.setMinimumHeight(480)
	win.setCentralWidget(panel)
	timelineDock = QtGui.QDockWidget('Timeline')
	timeline = UI.QTimeline(win)
	timeline.cb = setFrame
	timeline.setRange(0,goodFrames[-1])
	timelineDock.setWidget(timeline)
	timelineDock.setFeatures(QtGui.QDockWidget.DockWidgetMovable|QtGui.QDockWidget.DockWidgetFloatable)

	frame = startFrame
	view.addCamera(UI.QGLViewer.Camera('default'))
	grid = GLGrid()
	view.primitives.append(grid)

	points = GLPoints3D(displayFrames[frame])
	from colorsys import hsv_to_rgb
	colorTable = np.array([hsv_to_rgb((h * 0.618033988749895)%1, 0.5, 0.95) for h in xrange(max(pointToGroup)+2)],dtype=np.float32)
	colorTable[-1] = 0
	points.colours = colorTable[pointToGroup]
	#points.names = displayLabels
	#points.pointSize = 3
	view.primitives.append(points)
	joints = GLPoints3D(stablePoints)
	joints.names = map(str, xrange(len(stablePoints)))
	view.primitives.append(joints)
	bones = GLBones(boneVertices,boneEdges)
	view.primitives.append(bones)
	win.addDockWidget(QtCore.Qt.BottomDockWidgetArea, timelineDock)
	win.show()
	
	global md, img, g_detectingDots, g_readingMovie
	md, img, g_detectingDots = None, None, False
	g_readingMovie = False
	if g_readingMovie:
		md = MovieReader.open_file(os.path.join(directory, movieFilename))
		img = np.frombuffer(md['vbuffer'],dtype=np.uint8).reshape(md['vheight'],md['vwidth'],3)
		view.setImageData(md['vbuffer'],md['vheight'],md['vwidth'],3)

	global allSkels
	allSkels = []

	app.connect(app, QtCore.SIGNAL('lastWindowClosed()') , app.quit)
	sys.exit(app.exec_())

if __name__ == '__main__':
	generateSkeleton(cacheId=804, x3d_filename=r'HarlequinTrackTestFull.x3d', perc=0.95, triangleThreshold=300,
	                 thresholdDistance=20., useFrames=range(4820, 5255), numComps=30, labelGraphThreshold=1.5, stepSize=2)
