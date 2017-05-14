#!/usr/bin/env python

import ISCV
import numpy as np
from IO import ViconReader, MovieReader, MAReader, IO, C3D, ASFReader
from UI import QGLViewer, QApp  # gui
from GCore import Label, Recon, Retarget, SolveIK, Calibrate, Character, Opengl
from UI.GLMeshes import GLMeshes
#from UI.GLSkel import GLSkel
#from UI.GLSkeleton import GLSkeleton

def intersectRaysCB(fi):
	global x2d_frames,mats,Ps,c3d_frames,view,primitives,primitives2D,track3d,prev_frame,track_orn,orn_graph,boot,g_all_skels,md,orn_mapper,mar_mapper
	skipping = prev_frame is None or np.abs(fi - prev_frame) > 10
	prev_frame = fi
	view = QApp.view()
	points,altpoints = primitives
	g2d = primitives2D[0]
	frame = x2d_frames[fi]
	x2ds_data,x2ds_splits = ViconReader.frameCentroidsToDets(frame,mats)
	g2d.setData(x2ds_data,x2ds_splits)
	if skipping: 
		x3ds,x3ds_labels = track3d.boot(x2ds_data, x2ds_splits)
		#trackGraph = Label.TrackGraph()
		boot = -10
	else:
		x3ds,x3ds_labels = track3d.push(x2ds_data, x2ds_splits)
	if False:
		boot = boot+1
		if boot == 0:
			x2d_threshold_hash = 0.01
			penalty = 10.0 # the penalty for unlabelled points. this number should be about 10. to force more complete labellings, set it higher.
			maxHyps = 500 # the number of hypotheses to maintain.
			print "booting:"
			numLabels = len(orn_graph[0])
			l2x = -np.ones(numLabels, dtype=np.int32)
			label_score = ISCV.label_from_graph(x3ds, orn_graph[0], orn_graph[1], orn_graph[2], orn_graph[3], maxHyps, penalty, l2x)
			clouds = ISCV.HashCloud2DList(x2ds_data, x2ds_splits, x2d_threshold_hash)
			which = np.array(np.where(l2x != -1)[0],dtype=np.int32)
			pras_score, x2d_labels, vels = Label.project_assign(clouds, x3ds[l2x[which]], which, Ps, x2d_threshold=x2d_threshold_hash)
			print fi, label_score, pras_score
			labelled_x3ds = x3ds[l2x[which]]
			print track_orn.bootPose(x2ds_data, x2ds_splits, x2d_labels)
		if boot > 0:
			track_orn.push(x2ds_data, x2ds_splits, its=4)
	#x3ds,x2ds_labels = Recon.intersect_rays(x2ds_data, x2ds_splits, Ps, mats, seed_x3ds = None)
	points.setData(x3ds)
	if c3d_frames != None:
		c3ds = c3d_frames[(fi-832)/2]
		true_labels = np.array(np.where(c3ds[:,3]==0)[0],dtype=np.int32)
		x3ds_true = c3ds[true_labels,:3]
		altpoints.setData(x3ds_true)

	ci = view.cameraIndex()-1
	if True: #ci == -1:
		MovieReader.readFrame(md, seekFrame=max((fi - 14)/4,0))
		QApp.app.refreshImageData()
	(orn_skel_dict, orn_t) = g_all_skels['orn']
	orn_mesh_dict, orn_skel_mesh, orn_geom_mesh = orn_t
	orn_anim_dict = orn_skel_dict['anim_dict']
	orn_skel_dict['chanValues'][:] = orn_anim_dict['dofData'][fi]
	Character.updatePoseAndMeshes(orn_skel_dict, orn_skel_mesh, orn_geom_mesh)
	(mar_skel_dict, mar_t) = g_all_skels['mar']
	mar_anim_dict = mar_skel_dict['anim_dict']
	mar_mesh_dict, mar_skel_mesh, mar_geom_mesh = mar_t
	Character.updatePoseAndMeshes(mar_skel_dict, mar_skel_mesh, mar_geom_mesh, mar_anim_dict['dofData'][fi])
	
	from PIL import Image
	#orn_geom_mesh.setImage((md['vbuffer'],(md['vheight'],md['vwidth'],3)))
	#orn_geom_mesh.refreshImage()

	w,h = 1024,1024
	cam = view.cameras[0]
	cam.refreshImageData(view)
	aspect = float(max(1,cam.bindImage.width()))/float(cam.bindImage.height()) if cam.bindImage is not None else 1.0
	orn_mapper.project(orn_skel_dict['geom_Vs'], aspect)
	data = Opengl.renderGL(w, h, orn_mapper.render, cam.bindId)
	orn_geom_mesh.setImage(data)
	mar_mapper.project(mar_skel_dict['geom_Vs'], aspect)
	data = Opengl.renderGL(w, h, mar_mapper.render, cam.bindId)
	mar_geom_mesh.setImage(data)
	#image = Image.fromstring(mode='RGB', size=(w, h), data=data)
	#image = image.transpose(Image.FLIP_TOP_BOTTOM)
	#image.save('screenshot.png')

	if 0:
		global g_screen
		image = Opengl.renderGL(1920, 1080, Opengl.quad_render, (cam.bindId, g_screen))
		import pylab as pl
		pl.imshow(image)
		pl.show()
	view.updateGL()

def load_skels(directory):
	ted_dir = os.path.join(os.environ['GRIP_DATA'],'ted')
	_,ted_skel_dict = IO.load(os.path.join(ted_dir,'ted_body6.skel'))

	asf_filename = os.path.join(directory,'Orn.asf')
	amc_filename = os.path.join(directory,'Orn.amc')
	asf_dict = ASFReader.read_ASF(asf_filename)
	orn_anim_dict = ASFReader.read_AMC(amc_filename, asf_dict)
	orn_skel_dict = ASFReader.asfDict_to_skelDict(asf_dict)
	orn_skel_dict['anim_dict'] = orn_anim_dict

	# we are going to try to transfer the geometry from ted to orn
	# the challenge here is that the joints/bones are not the same
	# because the rigging is rather different, our best chance is to pose the characters and transfer using joint names
	
	# orn's joint names begin 'VSS_'
	orn = [t[4:] for t in orn_skel_dict['jointNames']]; orn[0] = 'root'
	# ted's joint names begin 'GenTed:'. the first joint is 'Reference' (on the floor) and the second 'Hips'.
	ted = [t[7:] for t in ted_skel_dict['jointNames']]; ted[0] = 'root'; ted[1] = 'root'
	#ted_skel_dict['Ls'][0][:3,:3] = 1.03 * np.eye(3,3) # apparently, ted is 3% smaller than orn?

	# ted has extra joints compared to orn
	ted_extras = ['RightUpLeg_Roll','LeftUpLeg_Roll','RightArm_Roll','LeftArm_Roll','Neck1','Neck2',\
				'LeftHand1','LeftInHandIndex','LeftInHandMiddle','LeftInHandRing','LeftInHandPinky',\
				'RightHand1','RightInHandIndex','RightInHandMiddle','RightInHandRing','RightInHandPinky',\
				'HeadEnd','LeftToeBaseEnd','RightToeBaseEnd',\
				'LeftHandThumb4','LeftHandIndex4','LeftHandMiddle4','LeftHandRing4','LeftHandPinky4',\
				'RightHandThumb4','RightHandIndex4','RightHandMiddle4','RightHandRing4','RightHandPinky4'
		]
	ted_extras = sorted([ted.index(n) for n in ted_extras])
	# map ted's extra joints to their parent
	for ji in ted_extras:
		jpi = ted_skel_dict['jointParents'][ji]
		ted[ji] = ted[jpi]
	# some of ted's names differ
	name_mapping = dict([(ot,orn_skel_dict['jointNames'][orn.index(t.replace('Spine3','Chest').replace('_Roll','Roll'))]) for t,ot in zip(ted,ted_skel_dict['jointNames'])])
	print zip(ted_skel_dict['jointNames'],[name_mapping[t] for t in ted_skel_dict['jointNames']])
	print list(enumerate(ted_skel_dict['jointNames']))
	print list(enumerate(orn_skel_dict['jointNames']))
	
	orn_indices = np.array([orn_skel_dict['jointIndex'][name_mapping[t]] for t in ted_skel_dict['jointNames']],dtype=np.int32)
	
	# solve ted into orn's position.
	# we generate one constraint per joint and zero the weights of those that aren't constrained
	numJoints = ted_skel_dict['numJoints']
	markerParents = np.arange(numJoints,dtype=np.int32)
	markerOffsets = np.zeros((numJoints,3),dtype=np.float32)
	markerWeights = np.ones(numJoints,dtype=np.float32)
	once = set(orn_indices)
	for mi,oi in enumerate(orn_indices):
		if oi in once: once.remove(oi)
		else: markerWeights[mi] = 0; print 'weighting zero',mi,ted_skel_dict['jointNames'][mi]
	# don't fit the shoulders, to avoid ted's head leaning back
	markerWeights[ted_skel_dict['jointIndex']['GenTed:LeftShoulder']] = 0
	markerWeights[ted_skel_dict['jointIndex']['GenTed:RightShoulder']] = 0
	markerWeights[0] = 0
	markerWeights[1] = 1
	p_o_w = markerParents, markerOffsets, markerWeights
	effectorData = SolveIK.make_effectorData(ted_skel_dict, p_o_w = p_o_w)
	effectorTargets = np.zeros_like(ted_skel_dict['Gs'])
	effectorTargets[:,:,3] = orn_skel_dict['Gs'][:,:,3][orn_indices]

	jps = ted_skel_dict['jointParents']
	cvs,jcss = ted_skel_dict['chanValues'],ted_skel_dict['jointChanSplits']
	def kill_joint(ji):
		cvs[jcss[2*ji]:jcss[2*ji+2]] = 0
		#if jcss[2*ji] == jcss[2*ji+2]: kill_joint(jps[ji])
	for it in range(20):
		SolveIK.solveIK(ted_skel_dict, ted_skel_dict['chanValues'], effectorData, effectorTargets, outerIts = 4)
		print it,SolveIK.scoreIK(ted_skel_dict, ted_skel_dict['chanValues'], effectorData, effectorTargets)
		# zero all joints that are only in ted
		for ji in ted_extras: kill_joint(ji)
		# for some reason, the Head and Hands wander off: keep them straight
		nji = ted_skel_dict['jointIndex']['GenTed:Neck']
		cvs[jcss[2*nji]:jcss[2*nji+2]] *= [0,0,1] # zero first two channels only...
		#kill_joint(nji)
		hji = ted_skel_dict['jointIndex']['GenTed:Head']
		cvs[jcss[2*hji]:jcss[2*hji+2]] = -cvs[jcss[2*nji]:jcss[2*nji+2]]
		#kill_joint(hji)
		for jn,ji in ted_skel_dict['jointIndex'].iteritems():
			if 'Hand' in jn: kill_joint(ji)
		print it,SolveIK.scoreIK(ted_skel_dict, ted_skel_dict['chanValues'], effectorData, effectorTargets)
	# kill all end effectors' parents
	#for ji in xrange(len(jps)):
	#	if ji not in list(jps): kill_joint(jps[ji])
	print SolveIK.scoreIK(ted_skel_dict, ted_skel_dict['chanValues'], effectorData, effectorTargets)
	orn_skel_dict['geom_dict'] = ted_skel_dict['geom_dict']
	orn_skel_dict['geom_Vs'] = ted_skel_dict['geom_Vs'].copy()
	orn_skel_dict['geom_vsplits'] = ted_skel_dict['geom_vsplits'].copy()
	orn_skel_dict['geom_Gs'] = ted_skel_dict['geom_Gs'].copy()
	orn_skel_dict['shape_weights'] = Character.shape_weights_mapping(ted_skel_dict, orn_skel_dict, name_mapping)
	return ted_skel_dict, orn_skel_dict

def main(x2d_filename, xcp_filename, c3d_filename = None):
	'''Generate a 3D view of an x2d file, using the calibration.'''
	global x2d_frames,mats,Ps,c3d_frames,primitives,primitives2D,track3d,prev_frame,track_orn,orn_graph,boot,orn_mapper,mar_mapper
	prev_frame = None
	c3d_frames = None
	if c3d_filename != None:
		c3d_dict = C3D.read(c3d_filename)
		c3d_frames, c3d_fps, c3d_labels = c3d_dict['frames'],c3d_dict['fps'],c3d_dict['labels']
	mats,xcp_data = ViconReader.loadXCP(xcp_filename)
	camera_ids = [int(x['DEVICEID']) for x in xcp_data]
	print 'loading 2d'
	x2d_dict = ViconReader.loadX2D(x2d_filename)
	x2d_frames = x2d_dict['frames']
	cameras_info = ViconReader.extractCameraInfo(x2d_dict)
	print 'num frames', len(x2d_frames)
	Ps = [m[2]/(m[0][0,0]) for m in mats]
	track3d = Label.Track3D(mats)
	
	primitives = QGLViewer.makePrimitives(vertices = [], altVertices=[])
	primitives2D = QGLViewer.makePrimitives2D(([],[0]))
	
	global g_all_skels,md
	directory = os.path.join(os.environ['GRIP_DATA'],'151110')
	_,orn_skel_dict = IO.load(os.path.join(directory,'orn.skel'))
	movie_fn = os.path.join(directory,'50_Grip_RoomCont_AA_02.v2.mov')
	md = MovieReader.open_file(movie_fn, audio=True, frame_offset=0, volume_ups=10)

	asf_filename = os.path.join(directory,'Martha.asf')
	amc_filename = os.path.join(directory,'Martha.amc')
	asf_dict = ASFReader.read_ASF(asf_filename)
	mar_skel_dict = ASFReader.asfDict_to_skelDict(asf_dict)
	mar_skel_dict['anim_dict'] = ASFReader.read_AMC(amc_filename, asf_dict)
	for k in ('geom_Vs','geom_vsplits','geom_Gs'): mar_skel_dict[k] = orn_skel_dict[k].copy()
	mar_skel_dict['shape_weights'] = orn_skel_dict['shape_weights']
	mar_skel_dict['geom_dict'] = orn_skel_dict['geom_dict']

	orn_vss = ViconReader.loadVSS(os.path.join(directory,'Orn.vss'))
	orn_vss_chan_mapping = [orn_vss['chanNames'].index(n) for n in orn_skel_dict['chanNames']]
	orn_anim_dict = orn_skel_dict['anim_dict']
	orn_vss_anim = np.zeros((orn_anim_dict['dofData'].shape[0],orn_vss['numChans']),dtype=np.float32)
	orn_vss_anim[:,orn_vss_chan_mapping] = orn_anim_dict['dofData']
	orn_anim_dict['dofData'] = orn_vss_anim
	orn_vss['anim_dict'] = orn_anim_dict
	for x in ['geom_dict','geom_Vs','geom_vsplits','geom_Gs','shape_weights']: orn_vss[x] = orn_skel_dict[x]
	orn_skel_dict = orn_vss

	g_all_skels = {}
	orn_mesh_dict, orn_skel_mesh, orn_geom_mesh = orn_t = Character.make_geos(orn_skel_dict)
	g_all_skels['orn'] = (orn_skel_dict, orn_t)
	orn_skel_dict['chanValues'][:] = 0
	Character.updatePoseAndMeshes(orn_skel_dict, orn_skel_mesh, orn_geom_mesh)

	mar_mesh_dict, mar_skel_mesh, mar_geom_mesh = mar_t = Character.make_geos(mar_skel_dict)
	g_all_skels['mar'] = (mar_skel_dict, mar_t)

	#ted_mesh_dict, ted_skel_mesh, ted_geom_mesh = ted_t = Character.make_geos(ted_skel_dict)
	#g_all_skels['ted'] = (ted_skel_dict, ted_t)
	#ted_skel_dict['chanValues'][0] += 1000
	#Character.updatePoseAndMeshes(ted_skel_dict, ted_skel_mesh, ted_geom_mesh)

	mnu = orn_skel_dict['markerNamesUnq']
	mns = orn_skel_dict['markerNames']
	effectorLabels = np.array([mnu.index(n) for n in mns], dtype=np.int32)
	orn_graph = Label.graph_from_skel(orn_skel_dict, mnu)
	boot = -10

	track_orn = Label.TrackModel(orn_skel_dict, effectorLabels, mats)

	#ted = GLSkel(ted_skel_dict['Bs'], ted_skel_dict['Gs']) #, mvs=ted_skel_dict['markerOffsets'], mvis=ted_skel_dict['markerParents'])
	#ted = GLSkeleton(ted_skel_dict['jointNames'],ted_skel_dict['jointParents'], ted_skel_dict['Gs'][:,:,3])
	#ted.setName('ted')
	#ted.color = (1,1,0)
	#orn = GLSkeleton(orn_skel_dict['jointNames'],orn_skel_dict['jointParents'], orn_skel_dict['Gs'][:,:,3])
	#orn.setName('orn')
	#orn.color = (0,1,1)
	
	#square = GLMeshes(names=['square'],verts=[[[0,0,0],[1000,0,0],[1000,1000,0],[0,1000,0]]],vts=[[[0,0],[1,0],[1,1],[0,1]]],faces=[[[0,1,2,3]]],fts=[[[0,1,2,3]]])
	#square.setImageData(np.array([[[0,0,0],[255,255,255]],[[255,255,255],[0,0,0]]],dtype=np.uint8))
	#orn_geom_mesh.setImageData(np.array([[[0,0,0],[255,255,255]],[[255,255,255],[0,0,0]]],dtype=np.uint8))
	
	P = Calibrate.composeP_fromData((60.8,),(-51.4,14.7,3.2),(6880,2860,5000),0) # roughed in camera for 151110
	ks = (0.06,0.0)
	mat = Calibrate.makeMat(P, ks, (1080, 1920))
	orn_mapper = Opengl.ProjectionMapper(mat)
	orn_mapper.setGLMeshes(orn_geom_mesh)
	orn_geom_mesh.setImage((md['vbuffer'],(md['vheight'],md['vwidth'],3)))

	mar_mapper = Opengl.ProjectionMapper(mat)
	mar_mapper.setGLMeshes(mar_geom_mesh)
	mar_geom_mesh.setImage((md['vbuffer'],(md['vheight'],md['vwidth'],3)))

	global g_screen
	g_screen = Opengl.make_quad_distortion_mesh()

	QGLViewer.makeViewer(mat=mat,md=md,layers = {\
		#'ted':ted, 'orn':orn,
		#'ted_skel':ted_skel_mesh,'ted_geom':ted_geom_mesh,\
		#'square':square,
		'orn_skel':orn_skel_mesh,'orn_geom':orn_geom_mesh,\
		'mar_skel':mar_skel_mesh,'mar_geom':mar_geom_mesh,\
			},
	primitives=primitives, primitives2D=primitives2D, timeRange=(0, len(x2d_frames) - 1, 4, 25.0), callback=intersectRaysCB, mats=mats,camera_ids=camera_ids)


if __name__=='__main__':
	import sys, os
	print sys.argv
	if len(sys.argv) == 1:
		directory = os.path.join(os.environ['GRIP_DATA'],'151110')
		x2d_filename = os.path.join(directory,'50_Grip_RoomCont_AA_02.x2d')
		xcp_filename = os.path.join(directory,'50_Grip_RoomCont_AA_02.xcp')
		main(x2d_filename, xcp_filename)
	elif len(sys.argv) == 3:
		x2d_filename, xcp_filename = sys.argv[1:]
		main(x2d_filename, xcp_filename)
	elif len(sys.argv) == 4:
		x2d_filename, xcp_filename, c3d_filename = sys.argv[1:]
		main(x2d_filename, xcp_filename, c3d_filename)


