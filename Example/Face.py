#!/usr/bin/env python

import os, sys
import numpy as np
import ISCV
from IO import ASFReader, OBJReader, ViconReader, MovieReader, C3D, IO
from UI import QApp,QGLViewer
from UI import GLSkel, GLCameras, GLMeshes, GLGeometry, GLPoints3D
from GCore import Calibrate, Character

def animateHead(newFrame):
	global ted_geom,ted_geom2,ted_shape,tony_geom,tony_shape,tony_geom2,tony_obj,ted_obj,diff_geom,c3d_frames,extract
	global tony_shape_vector,tony_shape_mat,ted_lo_rest,ted_lo_mat,c3d_points
	global md,movies
	tony_geom.image,tony_geom.bindImage,tony_geom.bindId = ted_geom.image,ted_geom.bindImage,ted_geom.bindId # reuse the texture!
	fo = 55
	MovieReader.readFrame(md, seekFrame=((newFrame+fo)/2))
	view = QApp.view()
	for ci in range(0,4): view.cameras[ci+1].invalidateImageData()
	ci = view.cameras.index(view.camera)-1
	if ci >= 0: MovieReader.readFrame(movies[ci], seekFrame=(newFrame+fo)) # only update the visible camera
	frac = (newFrame % 200) / 100.
	if (frac > 1.0): frac = 2.0 - frac
	fi = newFrame%len(c3d_frames)

	if ted_skel: # move the skeleton
		dofs = ted_anim['dofData'][fi*2 - 120]
		Character.pose_skeleton(ted_skel['Gs'], ted_skel, dofs)
		ted_glskel.setPose(ted_skel['Gs'])
		offset = ted_skel['Gs'][13] # ted_skel['jointNames'].index('VSS_Head')

		cams = QApp.app.getLayers()['cameras']
		tmp = np.eye(4,4,dtype=np.float32); tmp[:3,:] = offset
		cams.setTransform(tmp)
	
		if ci >= 0: # move the camera view to be correct
			camRT = mats[ci][1]
			RT = np.dot(camRT, np.linalg.inv(tmp))
			view.cameras[ci+1].setRT(RT)

		# update the face geometries to fit the skeleton
		ted_geom.setPose(offset.reshape(1,3,4))
		tony_geom.setPose(offset.reshape(1,3,4))
		#TODO head_points,c3d_points,surface_points,ted_geom2

	frame = c3d_frames[fi][extract]
	which = np.where(frame[:,3] == 0)[0]
	x3ds = frame[which,:3]
	#print which,x3ds.shape,ted_lo_rest.shape,ted_lo_mat.shape
	bnds = np.array([[0,1]]*ted_lo_mat.shape[0],dtype=np.float32)
	tony_shape_vector[:] = OBJReader.fitLoResShapeMat(ted_lo_rest, ted_lo_mat, x3ds, Aoffset=10.0, Boffset=3.0, x_0=tony_shape_vector, indices=which, bounds = bnds)
	#global tony_shape_vectors; tony_shape_vector[:] = tony_shape_vectors[newFrame%len(tony_shape_vectors)]

	#tony_shape_vector *= 0.
	#tony_shape_vector += (np.random.random(len(tony_shape_vector)) - 0.5)*0.2
	if 1:
		ted_shape_v = np.dot(ted_shape_mat_T, tony_shape_vector).reshape(-1,3)
	else:
		ted_shape_v = np.zeros_like(ted_obj['v'])
		ISCV.dot(ted_shape_mat_T, tony_shape_vector, ted_shape_v.reshape(-1))
	tony_shape_v = ted_shape_v
	#tony_shape_v = tony_shape['v']*frac
	ted_geom.setVs(ted_obj['v'] + ted_shape_v) #ted_shape['v'] * frac)
	tony_geom.setVs(tony_obj['v'] + tony_shape_v - np.array([200,0,0],dtype=np.float32))
	ted_geom2.setVs(ted_obj['v'] * (1.0 - frac) + tony_tedtopo_obj['v'] * frac + np.array([200,0,0],dtype=np.float32))
	#if len(ted_shape_v) == len(tony_shape_v):
	#	tony_geom2.setVs(tony_obj['v'] + ted_shape_v - [400,0,0])
	#	diff_geom.setVs(ted_obj['v'] + tony_shape_v - ted_shape_v - [600,0,0])

	#print [c3d_labels[i] for i in which]
	surface_points.vertices = np.dot(ted_lo_mat.T, tony_shape_vector).T + ted_lo_rest
	surface_points.colour = [0,1,0,1] # green
	c3d_points.vertices = x3ds
	c3d_points.colour = [1,0,0,1] # red

	QApp.app.refreshImageData()
	QApp.app.updateGL()


if __name__ == '__main__':

	global ted_skel,ted_anim,ted_xcp
	ted_skel,ted_anim, ted_xcp = None,None,None
	if True: # body animation
		ted_dir = os.path.join(os.environ['GRIP_DATA'],'ted')
		asf_filename = os.path.join(ted_dir,'32_Quad_Dialogue_01.asf')
		amc_filename = os.path.join(ted_dir,'32_Quad_Dialogue_01.amc')
		xcp_filename = os.path.join(ted_dir,'32_Quad_Dialogue_01.xcp')
		skelFilename = os.path.join(ted_dir,'ted.skel')
		animFilename = os.path.join(ted_dir,'ted.anim')
		try:
			ted_skel = IO.load(skelFilename)[1]
			ted_anim = IO.load(animFilename)[1]
		except:
			print 'generating ted skel and anim'
			ASFReader.convertASFAMC_to_SKELANIM(asf_filename, amc_filename, skelFilename, animFilename)
			ted_skel = IO.load(skelFilename)[1]
			ted_anim = IO.load(animFilename)[1]
		ted_xcp_mats,ted_xcp_data = ViconReader.loadXCP(xcp_filename)

	if True: # facial animation

		global ted_geom,ted_geom2,ted_shape,tony_geom,tony_shape,tony_geom2,tony_obj,ted_obj,diff_geom,c3d_frames
		global tony_shape_vector,tony_shape_mat,ted_lo_rest,ted_lo_mat,c3d_points
		global md,movies

		ted_dir = os.path.join(os.environ['GRIP_DATA'],'ted')
		wavFilename = os.path.join(ted_dir,'32T01.WAV')
		md = MovieReader.open_file(wavFilename)
		c3d_filename = os.path.join(ted_dir,'201401211653-4Pico-32_Quad_Dialogue_01_Col_wip_02.c3d')
		c3d_dict = C3D.read(c3d_filename)
		c3d_frames, c3d_fps, c3d_labels = c3d_dict['frames'],c3d_dict['fps'],c3d_dict['labels']
		if False: # only for cleaned-up data
			c3d_subject = 'TedFace'
			which = np.where([s.startswith(c3d_subject) for s in c3d_labels])[0]
			c3d_frames = c3d_frames[:,which,:]
			c3d_labels = [c3d_labels[i] for i in which]
			print c3d_labels
		if False: # this is for the cleaned-up data (don't apply the other offset...)
			offset = Calibrate.composeRT(Calibrate.composeR( (0.0,0.0, 0)),(0,0,-8),0) # 0.902
			c3d_frames[:,:,:3] = np.dot(c3d_frames[:,:,:3] - offset[:3,3],offset[:3,:3])[:,:,:3]
		offset =  Calibrate.composeRT(Calibrate.composeR( (3.9,-38.7, 0)),(-159.6,188.8,123-12),0) # 0.902
		c3d_frames[:,:,:3] = np.dot(c3d_frames[:,:,:3] - offset[:3,3],offset[:3,:3])[:,:,:3]

		geos = []
		dat_directory = os.path.join(os.environ['GRIP_DATA'],'dat')

		if False: # experiments involving deformation transfer
			geos_filename = 'geos'
			if not os.path.exists(geos_filename):
				ted_dir = os.path.join(os.environ['GRIP_DATA'],'ted')
				ted_obj = OBJReader.readFlatObjFlipMouth(os.path.join(ted_dir,'ted.obj'))
				ted_shape = OBJReader.readFlatObjFlipMouth(os.path.join(ted_dir,'tedopen.obj'))
				ted_shape['v'] -= ted_obj['v']
				tony_obj = OBJReader.readFlatObjFlipMouth(os.path.join(ted_dir,'tony.obj'))
				nearVerts = trianglesToNearVerts(ted_obj['tris'], steps = 15)
				IO.save(geos_filename,(ted_obj,ted_shape,tony_obj,nearVerts))
			else:
				_,(ted_obj,ted_shape,tony_obj,nearVerts) = IO.load('geos')

			for target in ['ape']: #['andy','avatar','baboon','bigdog','evilelf','fatbat','feline','fishman','kadel','lizardman','mannequin','shaman','ted','tony','troll','wolf']:
				if True:
					#target = 'baboon'
					target_filename = dat_directory+target+'.dat'
					if True: #not os.path.exists(target_filename):
						ted_dir = os.path.join(os.environ['GRIP_DATA'],'ted')
						tony_obj = OBJReader.readFlatObjFlipMouth(ted_dir + target+'.obj')
						if target == 'ape' or target == 'apenew': flipMouth(tony_obj) # the ape's mouth is already flipped!
						print tony_obj['v'].shape, ted_obj['v'].shape

						print np.mean(map(len,nearVerts))
						vts = ted_obj['vt']

						tony_shape = {'v':0*tony_obj['v']}


						if True:
							print 'computing x-to-x scheme'

							ted_ccs = connectedComponents(ted_obj['tris'])
							print len(ted_ccs),map(len,ted_ccs)
							tony_ccs = connectedComponents(tony_obj['tris'])
							print len(tony_ccs),map(len,tony_ccs)
							for gp in range(7):
								print gp,np.mean(ted_obj['vt'][ted_ccs[gp]],axis=0) - np.mean(tony_obj['vt'][tony_ccs[gp]],axis=0)
							ted_vts_copy = ted_obj['vt'].copy()
							tony_vts_copy = tony_obj['vt'].copy()
							tony_vts_copy[tony_ccs[0]] += np.array([-0.0029, 0],dtype=np.float32)
							tony_vts_copy[tony_ccs[3],0] = 0.715+ tony_vts_copy[tony_ccs[3],0]
							tony_vts_copy[tony_ccs[4],0] = 0.715+ tony_vts_copy[tony_ccs[4],0]

							(mw,mws,mis),(mw2,mw2s,mi2s),x2s,D = OBJReader.computeTopoMap(ted_obj, tony_obj, ted_vts_copy, tony_vts_copy)
							print len(np.where(mws > 0.98)[0])
							tony_tedtopo_obj = { 'v':x2s,'vt':ted_obj['vt'], 'tris':ted_obj['tris'] }
							tony_shape = {'v':OBJReader.renderGeo(OBJReader.renderMotion(D, ted_shape['v']), mws,mis)} # reuse everything
						elif True:
							Dsrc = OBJReader.computeLocalCoordinates(ted_obj['v'], vts, nearVerts)
							Dtgt = OBJReader.computeLocalCoordinates(tony_obj['v'], vts, nearVerts)
							localMotion = OBJReader.computeMotion(ted_shape['v'], Dsrc)
							tony_shape['v']= OBJReader.renderMotion(Dtgt, localMotion)
						else:
							steps = 3
							tony_incr = tony_obj['v'].copy()
							ted_incr = ted_obj['v'].copy()
							ted_step = ted_shape['v'] * (1.0/steps)
							for it in xrange(steps):
								Dtgt = OBJReader.computeLocalCoordinates(tony_incr, vts, nearVerts)
								Dsrc = OBJReader.computeLocalCoordinates(ted_incr, vts, nearVerts)
								localMotion = OBJReader.computeMotion(ted_step, Dsrc)
								tony_incr += OBJReader.renderMotion(Dtgt, localMotion)
								ted_incr += ted_step
							tony_shape['v'][:] = tony_incr - tony_obj['v']
						IO.save(target_filename,(tony_obj,tony_shape))
					else:
						_,(tony_obj,tony_shape) = IO.load(target_filename)
				else: #except Exception, e:
					print 'oops',target,e

		if True:
			geos_filename = os.path.join(dat_directory,'ted_new.dat')
			if not os.path.exists(geos_filename):
				ted_obj = OBJReader.readFlatObjFlipMouth(os.path.join(ted_dir,'Ted_NEWRIG_Neutral_Moved.obj'))
			else:
				_,(ted_obj,nearVerts) = IO.load(geos_filename)
			target = 'andy'
			tony_obj = OBJReader.readFlatObjFlipMouth(ted_dir + target + '.obj')
			_,tony_shapes = IO.load(dat_directory+target+'_shapes.dat')
			num_shapes = len(tony_shapes)
			print num_shapes
			tony_shape_mat = np.zeros((num_shapes,tony_shapes[0]['v'].shape[0],3),dtype=np.float32)
			for t,ts in zip(tony_shape_mat, tony_shapes): t[:] = ts['v']
			tony_shape_vector = 0.2*np.ones(num_shapes,dtype=np.float32)
			tony_shape_v = np.dot(tony_shape_mat.T, tony_shape_vector).T
			tony_tedtopo_obj = {'v': tony_obj['v'].copy() }
			ted_shape = {'v':tony_shape_v.copy()}
			tony_shape = {'v':tony_shape_v.copy()}


		if True: # ted_shape_mat
			try:
				ted_shape_mat = IO.load('ted_shape_mat')[1]
			except:
				geos_filename = dat_directory+'ted_new.dat'
				if not os.path.exists(geos_filename):
					ted_obj = OBJReader.readFlatObjFlipMouth(os.path.join(ted_dir,'Ted_NEWRIG_Neutral_Moved.obj'))
				else:
					_,(ted_obj,nearVerts) = IO.load(geos_filename)
				_,ted_shapes = IO.load(dat_directory+'ted_shapes.dat')
				print ted_shapes
				num_shapes = len(ted_shapes)
				ted_shape_mat = np.zeros((num_shapes,ted_shapes[0]['v'].shape[0],3),dtype=np.float32)
				for t,ts in zip(ted_shape_mat, ted_shapes): t[:] = ts['v']
				IO.save('ted_shape_mat',ted_shape_mat)
				# HACK scale ted... it looks like the correct value is 0.90197, which Shridhar introduced
			ted_obj['v'] *= 0.902
			ted_shape_mat *= 0.902
			lo_geo = c3d_frames[0,:,:3]
			extract,weights,indices = OBJReader.getMapping(ted_obj['v'], ted_obj['tris'], lo_geo, threshold = 20.0)
			for it,v in enumerate(zip(weights,indices)):
				print it,v
			ted_lo_rest, ted_lo_mat = OBJReader.makeLoResShapeMat(ted_obj['v'], ted_shape_mat, weights, indices)
			print np.sum(ted_shape_mat),np.sum(ted_lo_mat)
			ted_shape_mat_T = ted_shape_mat.reshape(ted_shape_mat.shape[0],-1).T.copy()

		tmp = np.sort(np.sum((ted_shape_mat_T!=0), axis=1))
		dtmp = tmp[1:]-tmp[:-1]
		diff = np.where(dtmp)[0]
		print dtmp[diff]
		print 'sort',diff[1:]-diff[:-1]
		u,s,vt = np.linalg.svd(ted_shape_mat_T, full_matrices=False)
		print s/s[0]

		#ted_obj['v'] -= np.mean(ted_obj['v'],axis=0)
		#tony_obj['v'] -= np.mean(tony_obj['v'],axis=0)
		#tony_tedtopo_obj['v'] -= np.mean(tony_tedtopo_obj['v'],axis=0)
		#rotate90(tony_obj,10)
		tony_obj['v'] -= np.array([0, 1750, 0],dtype=np.float32)
		tony_tedtopo_obj['v'] -= np.array([0, 1750, 0],dtype=np.float32)
		display_offset = np.array([0,0,0],dtype=np.float32) # show above the ground plane
		tony_obj['v'] += display_offset
		tony_tedtopo_obj['v'] += display_offset
		ted_obj['v'] += display_offset
		ted_lo_rest += display_offset
		c3d_frames[:,:,:3] += display_offset
		offset[:3,3] -= np.dot(offset[:3,:3],display_offset)

		draw_normals = False
		if draw_normals:
			geos.append(GLGeometry(vs=zip(tony_obj['v'],tony_obj['v']+Dtgt[:,:,0]*0.005), tris=range(Dtgt.shape[0]*2), drawStyle = 'wire',colour=[1,0,0,1]))
			geos.append(GLGeometry(vs=zip(tony_obj['v'],tony_obj['v']+Dtgt[:,:,1]*0.005), tris=range(Dtgt.shape[0]*2), drawStyle = 'wire',colour=[0,1,0,1]))
			geos.append(GLGeometry(vs=zip(tony_obj['v'],tony_obj['v']+Dtgt[:,:,2]*0.005), tris=range(Dtgt.shape[0]*2), drawStyle = 'wire',colour=[0,0,1,1]))

		#Dsrc = OBJReader.computeLocalCoordinates(ted_obj['v'], vts, nearVerts)
		#Dtgt = OBJReader.computeLocalCoordinates(ted_obj['v'] + ted_shape['v'], vts, nearVerts)
		#localMotion = OBJReader.computeMotion((tony_obj['v'] + [200,0,0])-ted_obj['v'], Dsrc)
		#tony_shape['v']= OBJReader.renderMotion(Dtgt, localMotion)+ (ted_obj['v'] + ted_shape['v']) - (tony_obj['v'] + [200,0,0])

		drawStyle='smooth'#'wire_over_smooth'
		ted_geom = GLMeshes(['ted'],[ted_obj['v']], [ted_obj['tris']], vts = [ted_obj['vt']], transforms = [np.eye(3,4)])
		#ted_geom = GLGeometry(vs = ted_obj['v'], vts = ted_obj['vt'], tris = ted_obj['tris'], transformData=None, drawStyle=drawStyle)
		xspacer = np.array([200,0,0],dtype=np.float32)
		ted_geom2 = GLGeometry(vs = ted_obj['v'] + xspacer, vts = ted_obj['vt'], tris = ted_obj['tris'], transformData=None, drawStyle=drawStyle)
		tony_geom = GLMeshes(['tony'], [tony_obj['v'] - xspacer], [tony_obj['tris']], vts=[tony_obj['vt']], transforms=[np.eye(3,4)]) #GLGeometry(vs = tony_obj['v'] - xspacer, vts = tony_obj['vt'], tris = tony_obj['tris'], transformData=None, drawStyle=drawStyle)
		#tony_geom2 = GLGeometry(vs = tony_obj['v'] + [-400,0,0], vts = tony_obj['vt'], tris = tony_obj['tris'], transformData=None, drawStyle=drawStyle)
		#diff_geom = GLGeometry(vs = ted_obj['v'] + [-600,0,0], vts = ted_obj['vt'], tris = ted_obj['tris'], transformData=None, drawStyle=drawStyle)

		xcp_filename = '201401211653-4Pico-32_Quad_Dialogue_01.xcp'
		xcp_filename = '201401211653-4Pico-32_Quad_Dialogue_01_Col_wip_01.xcp'
		xcp,xcp_data = ViconReader.loadXCP(os.path.join(ted_dir, xcp_filename))

		movie_filename0 = '201401211653-4Pico-32_Quad_Dialogue_01_0.mp4'
		movie_filename1 = '201401211653-4Pico-32_Quad_Dialogue_01_1.mp4'
		movie_filename2 = '201401211653-4Pico-32_Quad_Dialogue_01_2.mp4'
		movie_filename3 = '201401211653-4Pico-32_Quad_Dialogue_01_3.mp4'

		interest,fovX,(ox,oy),pan_tilt_roll,tx_ty_tz,distortion = 452,76.3,(-0.017,0.042),(-19,-19.7,83),(60,418,171),(0.291979,0.228389)
		interest,fovX,(ox,oy),pan_tilt_roll,tx_ty_tz,distortion = 440,76.3,(-0.017,0.042),(-17.6,-18.7,84.0),(59.,418.,176.),(0.291979,0.228389)
		cams = [
			#(-7.14,-9.68,74.7),(91,-114,112)
			#(-14.5,-12.6,86.5),(56,-73,195)
			#(5.8,-35.6,-16),(-158,95,-12)
			#(-14.8,-15.4,83.5),(61,-85,191)
			#cam1 tgt (-13.8,-13.8,86.4),(54,-73,183)
			#cam1 src (-13.8,-13.3,87.4),(56,-72,184)
			#cam2 tgt (49,-16,199)
			#cam2 src (50,-17,199)
			#cam3 tgt (29.4,2.8,-88.1) (-99,-5,178) src (29.6,3.0,-88.1) (-100,-5,178)
			#cam4 tgt (22,-8,-91.2),(-87,-50,184) src (22.5,-9.2,-91.2),(-88,-53,180)
			#cam2 tgt (-10.0,2.2,90.0),(44,-16,193) src (-12.5,1.3,90.0),(49,-19,193)
			#cam2 tgt (-13.2,2.4,89.4),(52,-16,190) src (-11.1,2.3,89.7),(44,-15,193)
		(341,76.34,(-0.017246,0.0424024),(-20.5+0.1,20.2-0.5,99.5-1.0),(219.5-2.,-180.-2.,202.-1.),(0.291979,0.228389)),
		(307,76.41,(0.036673,0.081622),(-20.4-1.2+2.-2.1-0.6,37.0+1.1+1.3+0.1-0.3-0.2,101.5+0.3-0.3+1.0+0.7),(215.-1-5+8-1,-126.+1+4+3-3,182.-2.5-3),(0.292402446,0.214726448)),
		(215,76.4,(0.019527,0.037984),(29.3-0.2,35.0-0.2,-104.6+0.1),(63+2.,-119-1,170+1),(0.29172,0.22112)),
		(261,76.7,(-0.050844,0.078558),(18.1-0.3,24.8+1.2,-101.7),(81.+2,-158+4,199),(0.29115,0.228548)),
		]
		for ci,(c,x) in enumerate(zip(cams,xcp)):
			interest,fovX,(ox,oy),pan_tilt_roll,tx_ty_tz,distortion = c
			K,RT = Calibrate.composeK(fovX,ox=ox,oy=oy),Calibrate.composeRT(Calibrate.composeR(pan_tilt_roll),tx_ty_tz,0)
			K,RT = Calibrate.decomposeKRT(xcp[ci][2]) # !!!!use the original values!!!!
			P = np.dot(K,RT)[:3,:]
			P2 = x[2]
			print 'diff',np.dot(P2[:3,:3],np.linalg.pinv(P[:3,:3])),'dt',P[:,3]-P2[:,3]
			RT = np.dot(RT,offset)
			xcp[ci] = [K[:3,:3],RT[:3,:4],np.dot(K,RT)[:3,:],distortion,-np.dot(RT[:3,:3].T,RT[:3,3])]

		distortion = (0.291979,0.228389)
		K,RT = Calibrate.composeK(fovX,ox=ox,oy=oy),Calibrate.composeRT(Calibrate.composeR(pan_tilt_roll),tx_ty_tz,0)
		mat0 = [K[:3,:3],RT[:3,:3],np.dot(K,RT)[:3,:],distortion,-np.dot(RT[:3,:3].T,RT[:3,3])]
		md0 = MovieReader.open_file(os.path.join(ted_dir, movie_filename0))
		md1 = MovieReader.open_file(os.path.join(ted_dir, movie_filename1))
		md2 = MovieReader.open_file(os.path.join(ted_dir, movie_filename2))
		md3 = MovieReader.open_file(os.path.join(ted_dir, movie_filename3))
		mats = xcp
		camera_ids = [d['DEVICEID'] for d in xcp_data]
		movies = [md0,md1,md2,md3]

		# the tiara points are defined in an svg file, in units of bogons
		# in the file there is a matrix scale of 0.95723882 (dots per bogon) and a diameter of 14.06605 bogons = 3.8mm
		# 25.4 mmpi / 90 dpi * 0.95723882 dpb_from_svg = 3.8mm / 14.06605 bogon_diameter_from_svg = 0.270 mm_per_bogon
		mm_per_bogon = 0.270154067
		head_pts_bogons = np.array([
			[  23.24843216, -273.46289062],
			[  53.5888443 , -290.25338745],
			[  65.81463623, -341.46832275],
			[ 101.07259369, -361.53491211],
			[ 122.78975677, -391.83300781],
			[ 136.22935486, -352.81604004],
			[ 114.80623627, -318.71374512],
			[ 167.69758606, -335.17553711],
			[ 214.97885132, -337.76928711],
			[ 268.80731201, -338.53485107],
			[ 316.49282837, -331.34683228],
			[ 350.15072632, -349.76928711],
			[ 363.43197632, -315.17553711],
			[ 170.24447632, -407.59741211],
			[ 221.22885132, -405.47241211],
			[ 270.54135132, -409.98803711],
			[ 325.93197632, -398.12866211],
			[ 362.99447632, -379.26928711],
			[ 395.51010132, -350.64428711],
			[ 426.2131958 , -314.92553711],
			[ 417.29135132, -288.36303711],
			[ 447.74447632, -274.65991211]], dtype=np.float32)
		head_pts = np.zeros((head_pts_bogons.shape[0],3),dtype=np.float32)
		head_pts[:,0] = mm_per_bogon*head_pts_bogons[:,0]
		head_pts[:,1] = -mm_per_bogon*head_pts_bogons[:,1] # our y-axis is up
		print head_pts
		head_pts += [150,-100,0] - np.mean(head_pts,axis=0)
		#head_pts += [85,-193,0]
		head_pts = np.dot(head_pts - offset[:3,3],offset[:3,:3])

		c3d_points = GLPoints3D([])
		surface_points = GLPoints3D([])
		head_points = GLPoints3D(head_pts); head_points.colour = (0,1,1,1.0)

		# generate the animation
		if False:
			tsv_filename = 'tony_shape_vectors6'
			try:
				tony_shape_vectors = IO.load(tsv_filename)[1]
			except:
				tony_shape_vectors = np.zeros((len(c3d_frames), ted_lo_mat.shape[0]),dtype=np.float32)
				bnds = np.array([[0,1]]*ted_lo_mat.shape[0],dtype=np.float32)
				x_0 = np.zeros(ted_lo_mat.shape[0],dtype=np.float32)
				for fi, frame in enumerate(c3d_frames):
					which = np.where(frame[:,3] == 0)[0]
					x3ds = frame[which,:3]
					#print which,x3ds.shape,ted_lo_rest.shape,ted_lo_mat.shape
					x_0[:] = tony_shape_vectors[fi] = fitLoResShapeMat(ted_lo_rest, ted_lo_mat, x3ds, indices=which, bounds=bnds, x_0=x_0)
					print '\rfitting',fi,; sys.stdout.flush()
				IO.save(tsv_filename,tony_shape_vectors)

		primitives = [head_points,c3d_points,surface_points,ted_geom,ted_geom2,tony_geom]
		primitives.extend(geos)
		ted_glskel = GLSkel(ted_skel['Bs'],ted_skel['Gs']) if ted_skel else None
		if ted_glskel: primitives.append(ted_glskel)
		ted_cameras = GLCameras([x['USERID'] for x in ted_xcp_data],ted_xcp_mats) if ted_xcp_mats else None
		if ted_cameras: primitives.append(ted_cameras)

		QGLViewer.makeViewer(timeRange = (0,len(c3d_frames)-1), mats = mats, camera_ids = camera_ids, movies = movies, callback = animateHead, primitives = primitives)
		exit()
