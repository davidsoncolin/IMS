#!/usr/bin/env python
'''
nothing to do with dots tool. detecting dots in video. 
'''
import numpy as np
from IO import MovieReader
from IO import ViconReader
import ISCV
from GCore import Label, Recon, Calibrate, Detect
from UI import QApp, QGLViewer
from UI import GLPoints3D
import sys

VICON_tiara_x3ds = np.array([[-59.253918, -19.042404, 0.0], [-51.057335, -14.506386, 0.0], [-47.754486, -0.67046356, 0.0], [-38.229404, 4.7506027, 0.0], [-32.362427, 12.93576, 0.0], [-28.731663, 2.3951645, 0.0], [-34.519203, -6.8177032, 0.0], [-20.230392, -2.3704834, 0.0], [-7.4571686, -1.6697769, 0.0], [7.0848083, -1.4629517, 0.0], [19.967247, -3.4048233, 0.0], [29.060066, 1.5720749, 0.0], [32.648048, -7.7735672, 0.0], [-19.542339, 17.19458, 0.0], [-5.7687035, 16.620499, 0.0], [7.5532684, 17.840416, 0.0], [22.517273, 14.636559, 0.0], [32.529854, 9.5416183, 0.0], [41.314079, 1.8084564, 0.0], [49.60865, -7.8411026, 0.0], [47.19838, -15.017075, 0.0], [55.425415, -18.719025, 0.0]],dtype=np.float32)


def fit_points(X,C):
	'''Solve X A + B = C.
	X and C should be Nx2 matrices.'''
	x = X[1:]-X[0]
	c = C[1:]-C[0]
	A = np.linalg.lstsq(np.dot(x.T,x),np.dot(x.T,c))[0] # 2x2
	#B = C[0] - np.dot(X[0],A)
	B = np.mean(C - np.dot(X,A),axis=0)
	print B
	return A,B

def find_labels(x2ds, model, x2ds_indices, model_indices, threshold, labels_out):
	A,B = fit_points(x2ds[x2ds_indices], model[model_indices])
	print 'cf',np.dot(x2ds[x2ds_indices],A)+B - model[model_indices]
	cloud = ISCV.HashCloud2D(model, threshold)
	L = np.dot(x2ds,A)+B
	scores,matches,matches_splits = cloud.score(L)
	sc = ISCV.min_assignment_sparse(scores, matches, matches_splits, threshold**2, labels_out)
	ms = np.sum(labels_out != -1)
	return ((sc-(len(x2ds)-len(model))*(threshold**2))/len(model))**0.5,ms

def get_movie_frame(md, frame, deinterlacing):
	'''Read a MovieReader frame and return it together with a filtered version.'''
	if deinterlacing: field = frame & 1; frame /= 2
	try:
		MovieReader.readFrame(md, seekFrame=frame)
	except:
		print 'oops',frame; return None,None
	img = np.frombuffer(md['vbuffer'],dtype=np.uint8).reshape(md['vheight'],md['vwidth'],3)

	if deinterlacing: # TODO check even/odd
		y = np.arange(0,md['vheight'],2)
		if field: img[y,:] = img[y+1,:] # odd
		else:     img[y+1,:] = img[y,:] # even
	return img



def filter_movie_frame(img, small_blur, large_blur):
	mx = min(img.shape[0],img.shape[1])/2
	large_blur = min(max(large_blur,1),mx-1)
	small_blur = min(max(small_blur,0),large_blur-1)
	filtered_img = ISCV.filter_image(img, small_blur, large_blur)
	return filtered_img


def get_processed_movie_frame(md, frame, small_blur, large_blur, deinterlacing):
	'''Read a MovieReader frame and return it together with a filtered version.'''
	img = get_movie_frame(md, frame, deinterlacing)
	filtered_img = filter_movie_frame(img, small_blur, large_blur)
	return img, filtered_img

def movies_to_detections(movies, frames, deinterlacing, attrs):
	'''Run the dot detection on all the movies over all the frames.'''
	ret = {}
	for fi in frames:
		p0,p1 = [],[]
		for ci,md in enumerate(movies):
			img,data = get_processed_movie_frame(md, fi, attrs['small_blur'], attrs['large_blur'], deinterlacing)
			if img is None: print 'done';return ret
			if True: # run dot detectors
				good_darks, pts0 = Detect.detect_dots(255-data, attrs['threshold_dark_inv'], attrs)
				good_lights,pts1 = Detect.detect_dots(data, attrs['threshold_bright'], attrs)
				p0.append(pts0)
				p1.append(pts1)
		def make_bounds(lens): return np.array([sum(lens[:x]) for x in xrange(len(lens)+1)],dtype=np.int32)
		data0 = np.array(np.concatenate(p0),dtype=np.float32).reshape(-1,2),make_bounds(map(len,p0))
		data1 = np.array(np.concatenate(p1),dtype=np.float32).reshape(-1,2),make_bounds(map(len,p1))
		ret[fi] = (data0,data1)
		print fi,len(data0[0]),len(data1[0]),'good points (darks,lights)'
	return ret

def get_labels(frames, x3ds_seq, detections_seq, mats, x2d_threshold = 0.01):
	'''Project all the 3d points in all the views and label the detections.'''
	num_cameras = len(mats)
	ret = {}
	Ps = np.array([m[2]/(m[0][0,0]) for m in mats],dtype=np.float32)
	for fi in frames:
		print fi,'\r',
		x3ds,x3ds_labels = x3ds_seq[fi]
		x2ds_raw_data,splits = detections_seq[fi][0]
		assert(num_cameras+1 == len(splits))
		x2ds_labels = -np.ones(len(x2ds_raw_data),dtype=np.int32)
		x2ds_data,_ = Calibrate.undistort_dets(x2ds_raw_data, splits, mats)
		if len(x2ds_data):
			clouds = ISCV.HashCloud2DList(x2ds_data, splits, x2d_threshold)
			sc,x2ds_labels,x2ds_vels = Label.project_assign(clouds, x3ds, x3ds_labels, Ps, x2d_threshold)
			zeros = np.where(x2ds_labels == -1)[0]
			# these lines remove all the data for the unlabelled points
			x2ds_data[zeros] = -1
			x2ds_raw_data[zeros] = -1
		ret[fi] = x2ds_raw_data,splits,x2ds_labels
	return ret

def picked(view,data,clearSelection=True):
	if data is None or clearSelection: primitives[2].setData(np.zeros((0,3),dtype=np.float32))
	if data is None: view.updateGL(); return
	print data
	(type,pi,index,depth) = data
	if type == '3d' and pi == 1:
		primitives[2].setData(np.concatenate((primitives[2].vertices,primitives[0].vertices[index:index+1])))
	view.updateGL()


def get_dark_and_light_points(colour_image, frame_index, camera_index, opts):
	data = filter_movie_frame(colour_image, opts['small_blur'], opts['large_blur'])
	good_darks, pts0 = Detect.detect_dots(255-data, opts['threshold_dark_inv'], opts)
	good_lights,pts1 = Detect.detect_dots(data, opts['threshold_bright'], opts)
	return good_darks, pts0, good_lights, pts1, data

#@profile
def setFrame(frame):
	global State, mats, movieFilenames, primitives, detectingTiara
	global movies, primitives2D, deinterlacing, detectingWands, dot_detections, track3d, prev_frame, booting, trackGraph
	key = State.getKey('dotParams/attrs')
	
	skipping,prev_frame = (frame != prev_frame and frame-1 != prev_frame),frame
	booting = 10 if skipping else booting-1

	p0,p1 = [],[]

	if True: #dot_detections is None:

		for pair in enumerate(movies):
			pts = process_frame(deinterlacing, detectingTiara, detectingWands, frame, key, pair)
			p0.append(pts[0])
			p1.append(pts[1])
		def make_bounds(lens): return np.array([sum(lens[:x]) for x in xrange(len(lens)+1)],dtype=np.int32)
		data0 = np.array(np.concatenate(p0),dtype=np.float32).reshape(-1,2),make_bounds(map(len,p0))
		data1 = np.array(np.concatenate(p1),dtype=np.float32).reshape(-1,2),make_bounds(map(len,p1))
	else:
		#dot_detections = movies_to_detections(movies, [frame], deinterlacing, key)
		data0,data1 = dot_detections[frame] if dot_detections.has_key(frame) else dot_detections.values()[0]
		for ci,md in enumerate(movies):
			try:
				MovieReader.readFrame(md, seekFrame=frame)
			except:
				print 'oops',frame; return None,None
			#img = np.frombuffer(md['vbuffer'],dtype=np.uint8).reshape(md['vheight'],md['vwidth'],3)
			QApp.view().cameras[ci+1].invalidateImageData()
			data0 = data0[0].copy(),data0[1] # so that undistort doesn't modify the raw detections
			data1 = data1[0].copy(),data1[1]
	# TODO, move this to the viewer...
	data0 = ViconReader.frameCentroidsToDets( data0, mats )
	data1 = ViconReader.frameCentroidsToDets( data1, mats )

	primitives2D[0].setData(data0[0],data0[1])
	primitives2D[1].setData(data1[0],data1[1])

	#print x2ds_labels
	if len(movieFilenames) is not 1:
		if 1:
			#x2ds_data, x2ds_splits = data0 # dark points only
			x2ds_data, x2ds_splits = data1 # light points only
			if skipping:
				x3ds,x3ds_labels = track3d.boot(x2ds_data, x2ds_splits)
				#trackGraph = Label.TrackGraph()
			else:
				x3ds,x3ds_labels = track3d.push(x2ds_data, x2ds_splits)
				# coarse bounding box
				if False:
					for xi,x in zip(x3ds_labels,x3ds):
						if x[0] < -200 or x[0] > 200 or x[1] < 800 or x[1] > 1200 or x[2] < -50 or x[2] > 300:
							track3d.x2ds_labels[np.where(track3d.x2ds_labels == xi)[0]] = -1
							x[:] = 0
			primitives[0].setData(x3ds)
			#trackGraph.push(x3ds,x3ds_labels)
			#primitives[0].graph = trackGraph.drawing_graph()
		elif False:
			Ps = np.array([m[2]/(m[0][0,0]) for m in mats],dtype=np.float32)
			data = data0 # dark points
			#data = data1 # light points
			x3ds,x2ds_labels = Recon.intersect_rays(data[0], data[1], Ps, mats, tilt_threshold = 0.003, x2d_threshold = 0.02, x3d_threshold = 5.0, min_rays=2)
			primitives[0].setData(x3ds)
		if detectingTiara:
			global c3d_frames
			frame = c3d_frames[(frame-55) % len(c3d_frames)]
			which = np.where(frame[:,3] == 0)[0]
			x3ds = frame[which,:3]
			#print frame,'len',len(x3ds)
			primitives[1].setData(x3ds)
	QApp.app.refreshImageData()
	QApp.app.updateGL()

#@profile
def process_frame(deinterlacing, detectingTiara, detectingWands, frame, opts, pair):
	ci, md = pair
	img = get_movie_frame(md, frame, deinterlacing)
	#data = filter_movie_frame(img, small_blur, large_blur)
	#img, data = get_processed_movie_frame(md, frame, small_blur, large_blur, deinterlacing)
	QApp.view().cameras[ci + 1].invalidateImageData()
	"""
	if 1:  # show the filtered image
		img[:] = data
		pass
	if 0:  # crush the image to see the blobs
		lookup = np.zeros(256, dtype=np.uint8)
		lookup[threshold_bright:] = 255
		lookup[255 - threshold_dark_inv:threshold_bright] = 128
		img[:] = lookup[img]
	"""
	if 1:
		good_darks, pts0, good_lights, pts1, data = get_dark_and_light_points(img, frame, ci, opts)
		if 1:  # show the filtered image
			#print "data before insertion", type(data), data.shape
			#sys.exit(0)
			img[:] = data
		if 0:  # crush the image to see the blobs
			lookup = np.zeros(256, dtype=np.uint8)
			lookup[threshold_bright:] = 255
			lookup[255 - threshold_dark_inv:threshold_bright] = 128
			img[:] = lookup[img]
		# good_darks, pts0 = Detect.detect_dots(255-data, opts['threshold_dark_inv'], opts)
		# good_lights,pts1 = Detect.detect_dots(data, opts['threshold_bright'], opts)
		print ci, frame, len(pts0), len(pts1), 'good points (darks,lights)'

		if detectingWands:
			ratio = 2.0;
			x2d_threshold = 0.5;
			straightness_threshold = 0.01 * 2;
			match_threshold = 0.07 * 2
			x2ds_labels = -np.ones(pts1.shape[0], dtype=np.int32)
			x2ds_splits = np.array([0, pts1.shape[0]], dtype=np.int32)
			ISCV.label_T_wand(pts1, x2ds_splits, x2ds_labels, ratio, x2d_threshold, straightness_threshold,
							  match_threshold)
			print x2ds_labels

			for r, li in zip(good_lights, x2ds_labels):
				if li != -1:  # make some red boxes
					dx, dy = 10, 10
					img[int(r.sy - dy):int(r.sy + dy), int(r.sx - dx):int(r.sx + dx), 0] = 128
	else:
		pts0 = pts1 = []
	return (pts0, pts1)


def tighten_calibration((x3s,x3s_labels), (x2s,x2s_splits,x2s_labels), mats):
	x3s_original = x3s.copy()
	x2s_labels_original = x2s_labels.copy()
	for it in range(10):
		x2d_threshold = 0.08 # - it * 0.04/50.
		Ps = np.array([m[2]/(m[0][0,0]) for m in mats],dtype=np.float32)
		u2s,_ = Calibrate.undistort_dets(x2s, x2s_splits, mats)
		x3s, x3s_labels, E, x2d_labels = Recon.solve_x3ds(u2s, x2s_splits, x2s_labels_original, Ps, True)
		clouds = ISCV.HashCloud2DList(u2s, x2s_splits, x2d_threshold)
		sc,x2s_labels,_ = Label.project_assign(clouds, x3s, x3s_labels, Ps, x2d_threshold)
		print 'it',it,sc
		tiara_xis = np.where(x3s_labels < len(VICON_tiara_x3ds))[0]
		tiara_lis = x3s_labels[tiara_xis]
		tiara_true = VICON_tiara_x3ds[tiara_lis] + [0,1000,0]
		tiara_xs = x3s[tiara_xis]
		# now solve the tiara into place by finding a rigid transform
		RT,inliers = Calibrate.rigid_align_points_inliers(tiara_xs, tiara_true, scale=True)
		x3s = np.dot(x3s,RT[:3,:3].T) + RT[:,3]
		x3s[tiara_xis] = tiara_true
		singles = np.where([x in list(x2d_labels) for x in x2s_labels])[0]
		x2s_labels[singles] = -1
		for ci,P in enumerate(Ps): # first pass: solve cameras from 2d-3d correspondences
			x2s_which = np.where(map(lambda x:x!=-1,x2s_labels[x2s_splits[ci]:x2s_splits[ci+1]]))[0]+x2s_splits[ci]
			xls = x2s_labels[x2s_which]
			x3s_which = [list(x3s_labels).index(xi) for xi in xls]
			cv2_mat = Calibrate.cv2_solve_camera_from_3d(x3s[x3s_which], x2s[x2s_which], None, solve_distortion = True, solve_principal_point = False, solve_focal_length = True)
			rms = cv2_mat[2]
			print 'camera rms',ci,rms,'distortion cf',cv2_mat[1], mats[ci][3]
			mats[ci] = Calibrate.makeMat(cv2_mat[0],cv2_mat[1],mats[ci][5])
		if True: # try to make the fovs and distortions be shared
			f = np.mean([m[0][0,0] for m in mats])
			k = np.mean([m[3] for m in mats],axis=0)
			for ci,m in enumerate(mats):
				m[0][0,0] = m[0][1,1] = f
				np.dot(m[0],m[1],m[2])
				m[3][:] = k
			ISCV.undistort_points(x2s, 0, 0, float(k[0]), float(k[1]), u2s)
			for ci,P in enumerate(Ps): # second pass: enforce shared focal and distortion
				x2s_which = np.where(map(lambda x:x!=-1,x2s_labels[x2s_splits[ci]:x2s_splits[ci+1]]))[0]+x2s_splits[ci]
				xls = x2s_labels[x2s_which]
				x3s_which = [list(x3s_labels).index(xi) for xi in xls]
				cv2_mat = Calibrate.cv2_solve_camera_from_3d(x3s[x3s_which], u2s[x2s_which], Kin=mats[ci][0], solve_distortion = False, solve_principal_point = False, solve_focal_length = False)
				rms = cv2_mat[2]
				print 'camera rms',ci,rms,'distortion cf',cv2_mat[1], mats[ci][3]
				mats[ci] = Calibrate.makeMat(cv2_mat[0],mats[ci][3],mats[ci][5])
	nontiara_xis = np.where(x3s_labels >= len(VICON_tiara_x3ds))[0]
	nontiara_lis = x3s_labels[nontiara_xis]
	nontiara_true = x3s_original[nontiara_xis]
	nontiara_xs = x3s[nontiara_xis]
	RT,inliers = Calibrate.rigid_align_points_inliers(nontiara_true, nontiara_xs, scale=True)
	return RT

def test_2D(frames, x3ds, detections, mats, x2d_threshold = 0.025):
	'''Test the labelling of a 2d point sequence by propagating the labels to the next frame.'''
	import IO
	print 'loading 2d'
	print 'num frames', len(frames)
	prev_x2ds, prev_splits = detections[frames[0]]
	prev_vels = np.zeros_like(prev_x2ds)
	clouds = ISCV.HashCloud2DList(prev_x2ds, prev_splits, 6./2000.)
	x3ds_labels = np.arange(len(x3ds),dtype=np.int32)
	Ps = np.array([m[2]/(m[0][0,0]) for m in mats],dtype=np.float32)
	sc,prev_labels,_ = Label.project_assign(clouds, x3ds, x3ds_labels, Ps, 6./2000.)

	ret = []
	for fi in frames:
		x2ds, splits = detections[fi]
		clouds = ISCV.HashCloud2DList(x2ds, splits, x2d_threshold)
		sc,labels,vels = clouds.assign_with_vel(prev_x2ds, prev_vels, prev_splits, prev_labels, x2d_threshold)
		prev_x2ds,prev_splits,prev_labels,prev_vels = x2ds,splits,labels,vels
		ret.append(labels)
	return ret

def main():
	global State, mats, movieFilenames, primitives
	global movies, primitives2D, deinterlacing, detectingWands
	import IO
	import sys,os
	deinterlacing = False
	detectingWands = False
	detectingTiara = False
	dot_detections = None
	detections_filename = None
	frame_offsets = None
	firstFrame,lastFrame = 0,5000
	drawDotSize = 4.0
	fovX,(ox,oy),pan_tilt_roll,tx_ty_tz,distortion = 50.,(0,0),(0,0,0),(0,1250,0),(0,0)
	mats = []
	grip_directory = os.environ['GRIP_DATA']
	
	if 0:
		fovX,(ox,oy),pan_tilt_roll,tx_ty_tz,distortion = 37.9,(0,0),(-66.0,3.5,-0.2),(4850,1330,3280),(0,0) # roughed in
		K,RT = Calibrate.composeK(fovX,ox,oy),Calibrate.composeRT(Calibrate.composeR(pan_tilt_roll),tx_ty_tz,0)
		mat0 = [K[:3,:3],RT[:3,:4],np.dot(K,RT)[:3,:],distortion,-np.dot(RT[:3,:3].T,RT[:3,3]),[1920,1080]]
		fovX,(ox,oy),pan_tilt_roll,tx_ty_tz,distortion = 55.8,(0,0),(-103.6,3.5,-0.3),(2980,1380,-2180),(0,0) # roughed in
		K,RT = Calibrate.composeK(fovX,ox,oy),Calibrate.composeRT(Calibrate.composeR(pan_tilt_roll),tx_ty_tz,0)
		mat1 = [K[:3,:3],RT[:3,:4],np.dot(K,RT)[:3,:],distortion,-np.dot(RT[:3,:3].T,RT[:3,3]),[1920,1080]]
		fovX,(ox,oy),pan_tilt_roll,tx_ty_tz,distortion = 49.3,(0,0),(27.9,4.0,-0.2),(-5340,1150,5030),(0,0) # roughed in
		K,RT = Calibrate.composeK(fovX,ox,oy),Calibrate.composeRT(Calibrate.composeR(pan_tilt_roll),tx_ty_tz,0)
		mat2 = [K[:3,:3],RT[:3,:4],np.dot(K,RT)[:3,:],distortion,-np.dot(RT[:3,:3].T,RT[:3,3]),[1920,1080]]
		fovX,(ox,oy),pan_tilt_roll,tx_ty_tz,distortion = 50.6,(0,0),(-156.6,4.9,0.2),(-105,1400,-4430),(0,0) # roughed in
		K,RT = Calibrate.composeK(fovX,ox,oy),Calibrate.composeRT(Calibrate.composeR(pan_tilt_roll),tx_ty_tz,0)
		mat3 = [K[:3,:3],RT[:3,:4],np.dot(K,RT)[:3,:],distortion,-np.dot(RT[:3,:3].T,RT[:3,3]),[1920,1080]]
		mats = [mat0, mat1, mat2, mat3]
		xcp_filename = '154535_Cal168_Floor_Final.xcp'
		directory = os.path.join(grip_directory,'REFRAME')
		movieFilenames = ['001E0827_01.MP4', '001F0813_01.MP4', '001G0922_01.MP4', '001H0191_01.MP4']
		#mats,movieFilenames = mats[:1],movieFilenames[:1] # restrict to single-view
		frame_offsets = [119+160,260,339,161]
		small_blur,large_blur = 1,25
		min_dot_size = 1.0
		max_dot_size = 20.0
		circularity_threshold = 3.0
		threshold_bright,threshold_dark_inv = 250,250 #135,135
	elif 0:
		xcp_filename = '201401211653-4Pico-32_Quad_Dialogue_01_Col_wip_01.xcp'
		detections_filename = 'detections.dat'
		detectingTiara = True
		pan_tilt_roll = (0,0,90)
		distortion = (0.291979,0.228389)
		directory = os.path.join(os.environ['GRIP_DATA'],'ted')
		movieFilenames = ['201401211653-4Pico-32_Quad_Dialogue_01_%d.mpg' % xi for xi in range(1)]
		firstFrame = 511
		small_blur,large_blur = 1,20
		min_dot_size = 1.0
		max_dot_size = 16.0
		circularity_threshold = 3.0
		threshold_bright,threshold_dark_inv = 0,170
	elif 1:
		xcp_filename = '50_Grip_RoomCont_AA_02.xcp'
		detections_filename = 'detections.dat'
		pan_tilt_roll = (0,0,0)
		distortion = (0.291979,0.228389)
		directory = os.path.join(os.environ['GRIP_DATA'],'151110')
		movieFilenames = ['50_Grip_RoomCont_AA_02.v2.mov']
		firstFrame = 0
		small_blur,large_blur = 1,20
		min_dot_size = 1.0
		max_dot_size = 16.0
		circularity_threshold = 3.0
		threshold_bright,threshold_dark_inv = 170,170

	attrs = dict([(v,eval(v)) for v in ['small_blur','large_blur','threshold_bright','threshold_dark_inv','circularity_threshold','min_dot_size','max_dot_size']])

	primitives2D = QGLViewer.makePrimitives2D(([],[]),([],[]))
	primitives = []
	if len(movieFilenames) is 1:
		# TODO: time_base, timecode
		K,RT = Calibrate.composeK(fovX,ox,oy),Calibrate.composeRT(Calibrate.composeR(pan_tilt_roll),tx_ty_tz,0)
		mats = [[K[:3,:3],RT[:3,:4],np.dot(K,RT)[:3,:],distortion,-np.dot(RT[:3,:3].T,RT[:3,3]),[1920,1080]]]
		camera_ids=['video']
		movies=[MovieReader.open_file(os.path.join(directory,movieFilenames[0]), audio=False)]
	else: # hard coded cameras
		if xcp_filename.endswith('.xcp'):
			if detectingTiara: # gruffalo
				c3d_filename = os.path.join(directory, '201401211653-4Pico-32_Quad_Dialogue_01_Col_wip_02.c3d')
				from IO import C3D
				c3d_dict = C3D.read(c3d_filename)
				global c3d_frames
				c3d_frames, c3d_fps, c3d_labels = c3d_dict['frames'],c3d_dict['fps'],c3d_dict['labels']
				c3d_subject = ''#'TedFace'
				which = np.where([s.startswith(c3d_subject) for s in c3d_labels])[0]
				c3d_frames = c3d_frames[:,which,:]
				c3d_labels = [c3d_labels[i] for i in which]
				print len(c3d_frames)
			xcp,xcp_data = ViconReader.loadXCP(os.path.join(directory, xcp_filename))
			mats.extend(xcp)
		elif xcp_filename.endswith('.cal'):
			from IO import OptitrackReader
			xcp,xcp_data = OptitrackReader.load_CAL(os.path.join(directory, xcp_filename))
			mats = xcp
			print 'mats',len(mats),len(movieFilenames)
			assert(len(mats) == len(movieFilenames))
		camera_ids = []
		movies = []
		for ci,mf in enumerate(movieFilenames):
			fo = 0 if frame_offsets is None else frame_offsets[ci]
			movies.append(MovieReader.open_file(os.path.join(directory,mf), audio=False, frame_offset=fo))
		camera_ids = ['cam_%d'%ci for ci in xrange(len(mats))]
		print len(mats),len(movies),len(camera_ids)
	primitives.append(GLPoints3D([]))
	primitives.append(GLPoints3D([]))
	primitives.append(GLPoints3D([]))
	primitives[0].colour = (0,1,1,0.5)   # back-projected "cyan" points
	primitives[1].colour = (0,0,1,0.5)
	primitives[1].pointSize = 5
	primitives[2].colour = (1,0,0,0.99)

	if len(movieFilenames) != 1 and detections_filename != None:
		try:
			dot_detections = IO.load(detections_filename)[1]
		except:
			numFrames = len(c3d_frames) # TODO HACK HACK
			dot_detections = movies_to_detections(movies, range(numFrames), deinterlacing, attrs)
			IO.save(detections_filename,dot_detections)
			
		if detectingTiara:
			x3ds_seq = {}
			for fi in dot_detections.keys():
				frame = c3d_frames[(fi-55) % len(c3d_frames)]
				which = np.array(np.where(frame[:,3] == 0)[0],dtype=np.int32)
				x3ds_seq[fi] = np.concatenate((VICON_tiara_x3ds + np.array([150,-100,0],dtype=np.float32),frame[which,:3])), \
							   np.concatenate((np.arange(len(VICON_tiara_x3ds),dtype=np.int32),which+len(VICON_tiara_x3ds)))

			dot_labels = get_labels(dot_detections.keys(), x3ds_seq, dot_detections, mats, x2d_threshold = 0.05)

			calibration_fi = 546-2-6

			RT = tighten_calibration(x3ds_seq[calibration_fi], dot_labels[calibration_fi], mats)
			for v in c3d_frames:
				v[:,:3] = np.dot(v[:,:3],RT[:3,:3].T) + RT[:,3]
			
			if True:
				dot_detections = IO.load(detections_filename)[1]
				x3ds_seq = {}
				for fi in dot_detections.keys():
					frame = c3d_frames[(fi-55) % len(c3d_frames)]
					which = np.array(np.where(frame[:,3] == 0)[0],dtype=np.int32)
					x3ds_seq[fi] = np.concatenate((VICON_tiara_x3ds + np.array([0,1000,0],dtype=np.float32),frame[which,:3])), \
								   np.concatenate((np.arange(len(VICON_tiara_x3ds),dtype=np.int32),which+len(VICON_tiara_x3ds)))

				#dot_labels = get_labels(dot_detections.keys(), x3ds_seq, dot_detections, mats, x2d_threshold = 0.05)

	if detectingTiara:
		primitives.append(GLPoints3D(VICON_tiara_x3ds + [0,1000,0]))
		primitives[-1].pointSize = 5

	global track3d, prev_frame, booting, trackGraph
	track3d = Label.Track3D(mats[:len(movies)], x2d_threshold = 0.03, x3d_threshold = 5.0, min_rays=3, boot_interval = 2) #tilt_threshold = 0.01, gruffalo
	trackGraph = Label.TrackGraph()
	prev_frame = 0
	booting=1

	from UI import QApp
	from PySide import QtGui
	from GCore import State
	# Modified the options parameter for fields to be the range of acceptable values for the box
	# Previously would crash if small_blur got too low
	QApp.fields = { 'image filter':[
		('small_blur', 'Small blur radius', 'This is part of the image filter which controls the size of smallest detected features.', 'int', small_blur, {"min":0,"max":None}),
		('large_blur', 'Large blur radius', 'This is part of the image filter which controls the size of largest detected features.', 'int', large_blur, {"min":0,"max":None}),
		('threshold_bright', 'threshold_bright', 'This is part of the image filter which controls the size of smallest detected features.', 'int', threshold_bright, {"min":0,"max":255}),
		('threshold_dark_inv', 'threshold_dark_inv', 'This is part of the image filter which controls the size of largest detected features.', 'int', threshold_dark_inv, {"min":0,"max":255}),
		('circularity_threshold', 'circularity_threshold', 'How circular?.', 'float', circularity_threshold, {"min":0,"max":100}),
		('min_dot_size', 'min_dot_size', 'min_dot_size smallest detected features.', 'float', min_dot_size, {"min":0,"max":100}),
		('max_dot_size', 'max_dot_size', 'max_dot_size largest detected features.', 'float', max_dot_size, {"min":0,"max":100}),
	]}
	State.addKey('dotParams', {'type':'image filter', 'attrs':attrs})
	State.setSel('dotParams')
	appIn = QtGui.QApplication(sys.argv)
	appIn.setStyle('plastique')
	win = QApp.QApp()
	win.setWindowTitle('Imaginarium Dots Viewer')
	QGLViewer.makeViewer(primitives=primitives,primitives2D=primitives2D, timeRange = (firstFrame,lastFrame), callback=setFrame, mats=mats,camera_ids=camera_ids,movies=movies,pickCallback=picked,appIn=appIn,win=win)

if __name__ == '__main__':
	main()

