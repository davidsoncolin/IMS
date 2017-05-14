import os, sys
import numpy as np
from GCore import Calibrate, Retarget, Face, State
from IO import IO, MovieReader, PyTISStream, SocketServer, JPEG
from UI import QApp

class WebCam(object):
	cv2Properties = {
		"POS_MSEC": cv2.cv.CV_CAP_PROP_POS_MSEC,
	# Current position of the video file in milliseconds or video capture timestamp.
		"POS_FRAMES": cv2.cv.CV_CAP_PROP_POS_FRAMES,  # 0-based index of the frame to be decoded/captured next.
		"POS_AVI_RATIO": cv2.cv.CV_CAP_PROP_POS_AVI_RATIO,
	# Relative position of the video file: 0 - start of the film, 1 - end of the film.
		"FRAME_WIDTH": cv2.cv.CV_CAP_PROP_FRAME_WIDTH,  # Width of the frames in the video stream.
		"FRAME_HEIGHT": cv2.cv.CV_CAP_PROP_FRAME_HEIGHT,  # Height of the frames in the video stream.
		"FPS": cv2.cv.CV_CAP_PROP_FPS,  # Frame rate.
		"FOURCC": cv2.cv.CV_CAP_PROP_FOURCC,  # 4-character code of codec.
		"FRAME_COUNT": cv2.cv.CV_CAP_PROP_FRAME_COUNT,  # Number of frames in the video file.
		"FORMAT": cv2.cv.CV_CAP_PROP_FORMAT,  # Format of the Mat objects returned by retrieve() .
		"MODE": cv2.cv.CV_CAP_PROP_MODE,  # Backend-specific value indicating the current capture mode.
		"BRIGHTNESS": cv2.cv.CV_CAP_PROP_BRIGHTNESS,  # Brightness of the image (only for cameras).
		"CONTRAST": cv2.cv.CV_CAP_PROP_CONTRAST,  # Contrast of the image (only for cameras).
		"SATURATION": cv2.cv.CV_CAP_PROP_SATURATION,  # Saturation of the image (only for cameras).
		"HUE": cv2.cv.CV_CAP_PROP_HUE,  # Hue of the image (only for cameras).
		"GAIN": cv2.cv.CV_CAP_PROP_GAIN,  # Gain of the image (only for cameras).
		"EXPOSURE": cv2.cv.CV_CAP_PROP_EXPOSURE,  # Exposure (only for cameras).
		"CONVERT_RGB": cv2.cv.CV_CAP_PROP_CONVERT_RGB,
	# Boolean flags indicating whether images should be converted to RGB.
		"RECTIFICATION": cv2.cv.CV_CAP_PROP_RECTIFICATION,
	# Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently)
	}

	def __init__(self):
		self.isOpened = False
		self.cap = None
		self.properties = None
		self.currentFrame = None

	def Open(self, movieFilename):
		import cv2 # required for webcam
		self.properties = {}
		self.cap = cv2.VideoCapture()
		if not self.cap.open(movieFilename): return False
		self.isOpened = True
		self.populateProperties()
		for property in self.properties:
			print "{}: {}".format(property, self.properties[property])
		return True

	def populateProperties(self):
		self.properties = {}
		if not self.isOpened: return False
		for propertyName, propertyCode in self.cv2Properties.iteritems():
			self.properties[propertyName] = self.cap.get(propertyCode)
		return True

	def GetProperty(self, propertyName):
		return self.properties.get(propertyName, None)

	def SetProperty(self, propertyName, value):
		self.cap.set(self.cv2Properties[propertyName], value)
		self.properties[propertyName] = value

	def GetFrame(self):
		ret, img = self.cap.read()
		# print ret,
		return cv2.cvtColor(img, cv2.cv.CV_BGR2RGB) if ret else None

def update_rbfn(md, short_name='Take', mapping_file=None):
	global g_rbfn, g_predictor
	# TODO these groups must have weights, this can't initialise weights
	groups, slider_splits, slider_names, marker_names = extract_groups(g_rbfn)

	# update the neutral
	if mapping_file:
		fi = mapping_file[mapping_file.keys()[0]]['Neutral']
	else:
		g = groups[0][1]
		print g.keys()
		active_poses = [pn for pn in g['marker_data'].keys() if pn not in g.get('disabled', [])]
		ni = [ap.rsplit('_',2)[1]=='Neutral' for ap in active_poses].index(True)
		fi = int(active_poses[ni].rsplit('_',2)[2])
	print 'neutral on frame',fi
	MovieReader.readFrame(md, fi)
	img = np.frombuffer(md['vbuffer'],dtype=np.uint8).reshape(md['vheight'],md['vwidth'],3).copy()
	vs = Face.detect_face(img, g_predictor)
	vs = Face.track_face(img, g_predictor, vs)
	clear_neutral()
	g_rbfn['neutral'] = stabilize_shape(vs)[0]
	for (gn,group) in groups:
		gmd,gsd,gis = {},{},{}
		for pose_key,pose_data in group['marker_data'].iteritems():
			sd = group['slider_data'][pose_key]
			test_short_name,pose_name,frame_number = pose_key.rsplit('_',2)
			assert(test_short_name == short_name)
			fi = int(frame_number)
			print fi
			if mapping_file:
				if pose_name not in mapping_file[gn]:
					print 'WARNING: pose %s missing; removing from rbfn' % pose_name
					continue
				fi = mapping_file[gn].pop(pose_name)
				print 'remapping to',fi
			MovieReader.readFrame(md, fi)
			img = np.frombuffer(md['vbuffer'],dtype=np.uint8).reshape(md['vheight'],md['vwidth'],3).copy()
			vs = Face.detect_face(img, g_predictor)
			if vs is None:
				print 'failed to boot'
				for vi in range(max(fi-300,0),fi):
					MovieReader.readFrame(md, vi)
					img2 = np.frombuffer(md['vbuffer'],dtype=np.uint8).reshape(md['vheight'],md['vwidth'],3).copy()
					vs = Face.detect_face(img2, g_predictor)
					if vs is not None:
						print 'booted on frame',vi
						for vi2 in range(vi+1,fi):
							MovieReader.readFrame(md, vi2)
							img2 = np.frombuffer(md['vbuffer'],dtype=np.uint8).reshape(md['vheight'],md['vwidth'],3).copy()
							vs = Face.track_face(img2, g_predictor, vs)
						break
					if vi == fi-1: print 'don\'t know what to do'
			vs = Face.track_face(img, g_predictor, vs)
			#Face.show_image(img,vs)
			#vs, head_pan, head_tilt, A = stabilize_shape(vs)
			print pose_name
			#tmp = pose_data.reshape(-1,3)[:,:2]
			#Face.show_image(None,tmp-np.mean(tmp,axis=0),(vs-np.mean(vs,axis=0))*5)
			pose_data = np.hstack((vs,np.zeros((vs.shape[0],1),dtype=np.float32)))
			pose_key = '_'.join((short_name,pose_name,str(fi)))
			gmd[pose_key] = pose_data
			gsd[pose_key] = sd
			gis[pose_key] = JPEG.compress(img)
		group['marker_data'] = gmd
		group['slider_data'] = gsd
		group['images'] = gis
	if mapping_file: print 'left overs:',mapping_file

def retrack_refresh_rbfn():
	grip_dir = os.environ['GRIP_DATA']
	movie_fn,_ = QApp.app.loadFilename('Choose a movie to open', grip_dir, 'Movie Files (*.mp4 *.mov *.avi *.flv *.mpg)')
	md = MovieReader.open_file(movie_fn, audio=False)
	update_rbfn(md)

def retrack_remap_rbfn():
	grip_dir = os.environ['GRIP_DATA']
	movie_fn,_ = QApp.app.loadFilename('Choose a movie to open', grip_dir, 'Movie Files (*.mp4 *.mov *.avi *.flv *.mpg)')
	txt_fn,_ = QApp.app.loadFilename('Choose a text file of frame indices to open', grip_dir, 'Text Files (*.txt)')
	md = MovieReader.open_file(movie_fn, audio=False)
	lines = map(str.strip,(open(txt_fn,'r').readlines()))
	mapping_file = {}
	for l in lines:
		pose_name,frame_number,group_names = l.split(':')
		for gn in group_names.split(','):
			mapping_file.setdefault(gn,{})[pose_name] = int(frame_number)
	print mapping_file.keys()
	print mapping_file
	update_rbfn(md, mapping_file=mapping_file)

def extract_x2ds(group, pn, marker_names):
	new_shape = group['marker_data'][pn].reshape(-1,3)[:,:2]
	marker_indices = [marker_names.index(mn) for mn in group['marker_names']]
	out_shape = np.zeros((len(marker_names),2),dtype=np.float32)
	#out_shape[:len(marker_indices)] = new_shape[marker_indices]
	out_shape[:len(new_shape)] = new_shape # TODO this is a hack; markers were added since
	return out_shape

def extract_slider_marker_data_from_group(group, slider_names, marker_names):
	global g_predictor
	gmd,gsd = group['marker_data'],group['slider_data']
	e0,e1 = group['edge_list'].T
	active_poses = [pn for pn in gmd.keys() if pn not in group.get('disabled', [])]
	slider_indices = [slider_names.index(sn) for sn in group['slider_names']]
	marker_indices = [marker_names.index(mn) for mn in group['marker_names']]
	slider_data = np.zeros((len(active_poses), len(slider_indices)), dtype=np.float32)
	marker_data = np.zeros((len(active_poses), len(e0)), dtype=np.float32)
	# put Neutral in the first spot
	ni = [ap.rsplit('_',2)[1]=='Neutral' for ap in active_poses].index(True)
	active_poses = [active_poses.pop(ni)]+active_poses
	for pi, pose_name in enumerate(active_poses):
		x2ds = extract_x2ds(group, pose_name, marker_names)
		x2ds = stabilize_shape(x2ds)[0][marker_indices]
		marker_data[pi,:] = np.linalg.norm(x2ds[e0]-x2ds[e1], axis=1)
		slider_data[pi,:] = gsd[pose_name][slider_indices]
	return slider_data, marker_data

def retrain_RBFN_no_linear():
	retrain_RBFN(use_linear=False)

def retrain_RBFN(use_linear=True):
	'''rebuild the RBFN. testing the logic'''
	global g_rbfn
	clear_neutral()
	g_rbfn['neutral'] = stabilize_shape(g_rbfn['neutral'])[0]
	groups, slider_splits, slider_names, marker_names = extract_groups(g_rbfn)
	slider_names,marker_names = g_rbfn['slider_names'],g_rbfn['marker_names']
	for gn,group in groups:
		slider_data,marker_data = extract_slider_marker_data_from_group(group, slider_names, marker_names)
		if use_linear:
			group['md0'] = marker_data[0]
			m0 = marker_data-group['md0'].reshape(1,-1)
			linear = np.linalg.lstsq(m0, np.arcsin(np.clip(slider_data,0,1)))[0] * 0.2
			group['linear'] = linear
			slider_data -= np.sin(np.dot(m0, linear))
		else:
			group.pop('linear',None)
			group.pop('md0',None)
		w,c,b = Retarget.trainRBFN(marker_data, slider_data)
		#print w-group['weights'],c-group['centres'],b-group['betas']
		#assert np.allclose(w, group['weights'], 1e-4,1e-4)
		#assert np.allclose(c, group['centres'])
		#assert np.allclose(b, group['betas'])
		group['weights'], group['centres'], group['betas'] = w,c,b

def export_rbfn():
	global g_rbfn
	IO.save('out.rbfn',g_rbfn)
		
def extract_groups(rbfn):
	'''extract some useful structures from the rbfn'''
	groups = [(gn,g) for gn,g in rbfn['rbfn'].iteritems() if (isinstance(g, dict) and g.get('enabled', True) and g['weights'] is not None)]
	slider_splits = np.cumsum([0]+[len(g['slider_names']) for gn,g in groups])
	if 'slider_names' not in rbfn:
		rbfn['slider_names'] = sorted(set([sn for gn,g in groups for sn in g['slider_names']]))
	if 'marker_names' not in rbfn:
		rbfn['marker_names'] = sorted(set([mn for gn,g in groups for mn in g['marker_names']]))
	return groups, slider_splits, rbfn['slider_names'], rbfn['marker_names']
			
def applyRetarget(rbfn, x2ds):
	'''run the retargetting function on the tracking data.
	returns slider names and values.'''
	groups, slider_splits, slider_names, marker_names = extract_groups(rbfn)
	slider_values = np.zeros(slider_splits[-1], dtype=np.float32)
	for (gn,group),s0,s1 in zip(groups, slider_splits[:-1], slider_splits[1:]):
		e0,e1 = np.int32([marker_names.index(n) for n in group['marker_names']])[group['edge_list'].T]
		data = np.linalg.norm(x2ds[e0]-x2ds[e1], axis=1)
		slider_values[s0:s1] = Retarget.evaluateRBFN(group['weights'], group['centres'], group['betas'], data.reshape(1,-1))[0]
		linear = group.get('linear',None)
		if linear is not None: slider_values[s0:s1] += np.sin(np.dot(data-group['md0'], linear))
	return slider_names, slider_values

def strip_slider_names(slider_names, strip):
	return [x[len(strip):] if x.startswith(strip) else x for x in slider_names]

def dirty_cb(dirty):
	if '/root/ui/attrs/movie_filename' in dirty:
		fn = State.getKey('/root/ui/attrs/movie_filename')
		global g_md
		g_md = MovieReader.open_file(fn)
		QApp.app.qtimeline.setRange(0,g_md['vmaxframe'])
	for dk in dirty:
		if dk.startswith('/root/ui/attrs/'):
			QApp.app.refresh()
	global g_mode, g_frame, g_rbfn
	if g_mode == 1 and not '/root/sliders/attrs' in dirty: # RBFN view; changing frame sets all the sliders; we avoid that case
		for key in dirty:
			if key.startswith('/root/sliders/attrs'):
				si = g_rbfn['slider_names'].index(key[len('/root/sliders/attrs/'):])
				group,gn,pn,slider_indices,slider_names,pose_splits = rbfn_info_from_frame(g_frame[g_mode])
				print 'rbfn slider value changed:',key,si,'from',group['slider_data'][pn][si],'to',State.getKey(key)
				group['slider_data'][pn][si] = State.getKey(key)
				rbfn_view_cb(g_frame[g_mode]) # TODO, force an update of the geo
				
	#QApp.app.updateMenus()

def import_movie():
	global g_directory
	vid_filename,_ = QApp.app.loadFilename('Choose a movie to open',directory=g_directory,filtr='Movie Files (*.mp4 *.mov *.avi *.flv)')
	if vid_filename: State.setKey('/root/ui/attrs/movie_filename',vid_filename)

def stabilize_shape(vs, setting_neutral=False, neutral=None, head_pan_shape=None, head_tilt_shape=None):
	global g_neutral_corrective_shape
	if neutral is None: global g_rbfn; neutral = g_rbfn['neutral']
	if head_pan_shape is None: global g_head_pan_shape; head_pan_shape = g_head_pan_shape
	if head_tilt_shape is None: global g_head_tilt_shape; head_tilt_shape = g_head_tilt_shape
	norm_shape = vs - np.mean(vs, axis=0)
	dx = norm_shape[16] - norm_shape[0]
	c,s = dx / np.sum(dx**2)
	A0 = np.float32([[c,-s],[s,c]])
	norm_shape = np.dot(norm_shape, A0)
	# measure head_pan and head_tilt
	head_pan = np.sum((norm_shape - neutral) * head_pan_shape)
	head_tilt = np.sum((norm_shape - neutral) * head_tilt_shape)
	# straighten head
	norm_shape -= head_pan * head_pan_shape + head_tilt * head_tilt_shape
	
	rigid_indices = np.int32([0,1,15,16,39,42,45,27,30,31,33,35,36]) # most-rigid points
	global g_predictor
	rigid_ref_shape = g_predictor['ref_shape'][rigid_indices]
	rigid_t0 = np.mean(rigid_ref_shape, axis=0)
	rigid_ref_pinv = np.linalg.pinv(rigid_ref_shape - rigid_t0)

	n0 = np.mean(norm_shape[rigid_indices], axis=0)
	A = Face.normalizing_A(norm_shape[rigid_indices] - n0, rigid_ref_pinv)
	norm_shape = np.dot(norm_shape - n0, A) + rigid_t0
	# set the neutral from the current values
	if setting_neutral and g_neutral_corrective_shape is 0:
		g_neutral_corrective_shape = neutral - norm_shape
		#IO.save('neutral.out',g_neutral_corrective_shape)
	# compensate for neutral
	norm_shape += g_neutral_corrective_shape
	return norm_shape, head_pan, head_tilt, np.dot(A0,A)

def clear_neutral():
	global g_neutral_corrective_shape
	State.setKey('/root/ui/attrs/setting_neutral',False)
	g_neutral_corrective_shape = 0

def rbfn_pose_splits():
	global g_rbfn
	groups = extract_groups(g_rbfn)[0]
	pose_names = [g['marker_data'].keys() for gn,g in groups]
	pose_splits = np.int32([0]+list(np.cumsum(map(len,pose_names))))
	return pose_splits

def rbfn_info_from_frame(fi):
	global g_rbfn
	groups, slider_splits, slider_names, marker_names = extract_groups(g_rbfn)
	pose_names = [g['marker_data'].keys() for gn,g in groups]
	pose_splits = np.int32([0]+list(np.cumsum(map(len,pose_names))))
	fi = np.clip(fi,0,pose_splits[-1]-1)
	gi = np.where(pose_splits <= fi)[0][-1]
	gn,group = groups[gi]
	pi = fi-pose_splits[gi]
	pn = pose_names[gi][pi]
	slider_indices = xrange(len(slider_names))
	return group,gn,pn,slider_indices,slider_names,pose_splits

def filter_data(markers, last_smooth, delta = 0.025):
	if last_smooth is None: return markers.copy()
	diff = markers - last_smooth
	oned = len(diff.shape) == 1
	if oned: diff = diff.reshape(-1,1)
	alpha = np.linalg.norm(diff, axis=1) / delta
	alpha = np.clip(alpha, 0, 1)
	diff = alpha.reshape(-1,1) * diff
	if oned: diff = diff.reshape(-1)
	return last_smooth + diff

def setFrame_cb(fi):
	attrs = State.getKey('/root/ui/attrs/')
	global g_setting_frame
	if g_setting_frame: return
	g_setting_frame = True
	try: # within this loop we handle the timeline, which could trigger calling this function recursively
		global g_mode, g_frame, g_TIS_server, g_neutral_corrective_shape
		global g_smooth_pose
		view = QApp.view()
		cid = view.cameraIndex()
		if cid != g_mode: # deal with changing modes
			g_mode = cid
			if g_mode == 0:
				if g_md is not None: QApp.app.qtimeline.setRange(0, g_md['vmaxframe'])
			elif g_mode == 1:
				pose_splits = rbfn_pose_splits()
				QApp.app.qtimeline.setRange(0, pose_splits[-1]-1)
			new_frame = g_frame.get(g_mode,fi)
			if new_frame != fi:
				QApp.app.qtimeline.frame = new_frame
				fi = new_frame
	except Exception as e:
		print 'exc setFrame',e
	g_setting_frame = False
	g_frame[g_mode] = fi
	
	if not attrs['setting_neutral']: g_neutral_corrective_shape = 0
	
	new_pose,new_shape,norm_shape,img,slider_names,slider_values,A = [track_view_cb,rbfn_view_cb][g_mode](fi,attrs)

	
	mirror_scale = -1 if attrs['mirroring'] else 1
	h,wm = img.shape[0]*0.5,img.shape[1]*0.5*mirror_scale

	geo_vs = np.zeros((new_shape.shape[0],3), dtype=np.float32)	
	if attrs['debugging']: # display the stabilised data
		geo_vs[:,:2] = norm_shape
		geo_vs *= 200
		geo_vs[:,:2] += np.int32(np.mean(new_shape, axis=0)/200)*200
	else: # display the tracking data
		geo_vs[:,:2] = new_shape

	geo_mesh,image_mesh,bs_mesh = QApp.app.getLayers(['geo_mesh', 'image_mesh', 'bs_mesh'])
	
	bs_mesh.visible = attrs['show_harpy']
	if bs_mesh.visible:
		global g_bs_vs, g_bs_shape_mat_T
		bs_mesh.setVs(g_bs_vs + np.dot(g_bs_shape_mat_T, np.clip(slider_values[:-3],0,1)))
		# compute the Harpy position
		R = Calibrate.composeR(new_pose*[1,-1,-1])
		if g_mode == 1: R = np.eye(3) # TODO
		bs_ts = Calibrate.composeRT(R,[0,1720,0],0) # compensate for the offset of the Harpy (temples ~1720mm above origin)
		scale = 1.0/np.linalg.norm(160.*A) # IPD (64mm) / 0.4 (ref_shape) = 160.
		off = np.mean(new_shape[[0,16]],axis=0) # get the position of the temples (pixels)
		g_smooth_pose[g_mode] = filter_data(np.float32([scale,off[0],off[1]]), g_smooth_pose.setdefault(g_mode,None), 10.0)
		pose = g_smooth_pose[g_mode]
		bs_ts[:3] *= pose[0]
		bs_ts[:3,3] += [pose[1]-abs(wm),1000+pose[2]-h,0]
		# offset screen-right 300mm
		bs_ts[:3,3] += (pose[0]*attrs['harpy_xoffset'])*np.float32([np.cos(np.radians(view.camera.cameraRoll)),-np.sin(np.radians(view.camera.cameraRoll)),0.0])
		bs_mesh.transforms[0] = bs_ts.T
	
	geo_mesh.setVs(geo_vs)
	geo_mesh.colour=[0 if attrs['streaming_TIS'] else 1,1 if attrs['streaming_TIS'] else 0,0,1]
	geo_mesh.transforms[0][:,:3] = [[mirror_scale,0,0],[0,1,0],[0,0,1],[-wm,1000-h,0.1]]
	image_mesh.setVs(np.float32([[-wm,-h,0],[wm,-h,0],[wm,h,0],[-wm,h,0]]))
	image_mesh.setImage(img)
	if attrs['unreal']:
		if not attrs['streaming_TIS']: toggle_unreal()
		ret, activeConnections = g_TIS_server.WriteAll(PyTISStream.getBlendshapeData(slider_names, slider_values))
		if not ret:
			print "Server is not Initialised"
			State._setKey('/root/ui/attrs/streaming_TIS', False)
	else:
		# Turn off streaming
		if attrs['streaming_TIS']: toggle_unreal()
	QApp.app.updateGL()
	
def rbfn_view_cb(fi, attrs):
	# g_mode = 1
	global g_rbfn
	group,gn,pn,slider_indices,slider_names,pose_splits = rbfn_info_from_frame(fi)

	QApp.view().displayText = [(10, 100, gn), (10,125, pn)]
	img = group['images'][pn]
	img = JPEG.decompress(img)
	h,wm = img.shape[0]*0.5,img.shape[1]*0.5

	out_shape = extract_x2ds(group, pn, g_rbfn['marker_names'])
	
	svs = group['slider_data'][pn][slider_indices]
	State._setKey('/root/sliders/attrs', dict(zip(slider_names, svs))) # NO UNDO

	# compensate for roll, translation and scale
	norm_shape, head_pan, head_tilt, A = stabilize_shape(out_shape)

	# extract angles from the measured values
	mirror_scale = -1 if attrs['mirroring'] else 1
	new_pose = np.degrees(np.arctan2([head_pan*mirror_scale, head_tilt, -mirror_scale*A[1][0]],[2,2,A[1][1]]))
	
	head_roll = -np.arctan2(A[1][0],A[1][1])
	head_pan = np.arctan2(head_pan, 2.0)
	head_tilt = np.arctan2(head_tilt, 2.0)
	#print head_roll, head_pan, head_tilt

	slider_names, slider_values = applyRetarget(g_rbfn, norm_shape)
	svs[np.where(svs < 1e-4)] = 0
	slider_values[np.where(slider_values < 1e-4)] = 0
	#print zip(slider_values,svs)
	slider_names.extend(['NeckRoll','NeckPan','NeckTilt'])
	svs = np.clip(svs,0,1)
	slider_values = np.float32(list(svs)+list(np.degrees([head_roll,head_pan,head_tilt])))

	return new_pose,out_shape,norm_shape,img,slider_names,slider_values,A

def track_view_cb(fi, attrs):
	# g_mode = 0
	global g_webcam, g_md, g_rbfn, g_predictor
	# runtime options and state
	global g_prev_smooth_shape, g_prev_vs, g_hmc_boot, g_settle, g_head_pan_tilt_roll

	if attrs['using_webcam']:
		if g_webcam is None:
			g_webcam = WebCam()
			g_webcam.Open(State.getKey('/root/ui/attrs/cam_offset') + State.getKey('/root/ui/attrs/webcam_index'))
			g_webcam.SetProperty('FPS', State.getKey('/root/ui/attrs/cam_fps'))
			g_webcam.SetProperty('FRAME_WIDTH', State.getKey('/root/ui/attrs/cam_width'))
			g_webcam.SetProperty('FRAME_HEIGHT', State.getKey('/root/ui/attrs/cam_height'))
		if g_webcam is None:
			img = np.zeros((16,16,3),dtype=np.uint8)
		else:
			img = g_webcam.GetFrame()
			if img is None:
				img = np.zeros((16,16,3),dtype=np.uint8)
	elif g_md is not None:
		MovieReader.readFrame(g_md, seekFrame=fi) # only update the visible camera
		img = np.frombuffer(g_md['vbuffer'], dtype=np.uint8).reshape(g_md['vheight'],g_md['vwidth'],3)
		#QApp.app.qtimeline.setRange(0, g_md['vmaxframe'])
	else:
		img = np.zeros((16,16,3),dtype=np.uint8)
	
	mirror_scale = -1 if attrs['mirroring'] else 1
	rotate = attrs['rotate']

	if g_settle >= 0:
		if g_settle == 0 and g_prev_vs is not None:
			g_hmc_boot = g_prev_vs.copy()
		g_settle = g_settle - 1
	else:
		if attrs['HMC_mode'] and g_hmc_boot is not None: g_prev_vs = g_hmc_boot.copy()
		if attrs['booting'] or Face.test_reboot(img, g_prev_vs):
			g_prev_vs = Face.detect_face(img, g_predictor, 2, rotate)
			g_hmc_boot = None # in case we didn't detect a face
			g_settle = 10 # go into settle mode (10 frames)
			if g_prev_vs is not None:
				State.setKey('/root/ui/attrs/booting',False)
				if attrs['HMC_mode']: g_hmc_boot = g_prev_vs.copy()
	g_prev_vs = Face.track_face(img, g_predictor, g_prev_vs, rotate=rotate)

	# compensate for roll, translation and scale
	norm_shape, head_pan, head_tilt, A = stabilize_shape(g_prev_vs, setting_neutral=attrs['setting_neutral'])
	# dejitter
	if attrs['filtering']:
		g_prev_smooth_shape = filter_data(norm_shape, g_prev_smooth_shape)
	else:
		g_prev_smooth_shape = norm_shape.copy()
	# extract angles from the measured values
	head_pan_tilt_roll = np.degrees(np.arctan2([head_pan*mirror_scale, head_tilt, -mirror_scale*A[1][0]],[2,2,A[1][1]]))
	g_head_pan_tilt_roll = filter_data(head_pan_tilt_roll, g_head_pan_tilt_roll, 3.0)

	camera = QApp.view().camera
	camera.lockedUpright = False
	camera.cameraRoll = (-90*rotate if rotate != -1 else g_head_pan_tilt_roll[2])

	ret = g_prev_smooth_shape.copy()
	if attrs['mirroring']:
		flip_order = [16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0, 26,25,24,23,22,21,20,19,18,17, 27,28,29,30, 35,34,33,32,31, \
			  45,44,43,42, 47,46, 39,38,37,36, 41,40, 54,53,52,51,50,49,48, 59,58,57,56,55, 64,63,62,61,60, 67,66,65, 69,68]
		ret = ret[flip_order]
	slider_names, slider_values = applyRetarget(g_rbfn, ret)
	#State._setKey('/root/sliders/attrs', dict(zip(slider_names, slider_values))) # NO UNDO
	slider_names.extend(['NeckPan','NeckTilt','NeckRoll'])
	slider_values = np.float32(list(slider_values)+list(g_head_pan_tilt_roll))

	return g_head_pan_tilt_roll.copy(),g_prev_vs.copy(),norm_shape,img,slider_names,slider_values,A

def toggle_unreal():
	global g_TIS_server
	streaming_TIS = State.getKey('/root/ui/attrs/streaming_TIS')
	if streaming_TIS:
		g_TIS_server.Stop()
		State._setKey('/root/ui/attrs/streaming_TIS', False)
		print "Stopping Server"
	else:
		print 'Trying to start Server'
		if g_TIS_server.Start('',6500):
			State._setKey('/root/ui/attrs/streaming_TIS', True)
		else:
			print 'ARGH!!! Failed to start server'
			

def keypress_cb(view, key):
	if key == ord('R'):
		State.setKey('/root/ui/attrs/booting',True)
		State.push('boot face')
	if key == ord('N'):
		State.setKey('/root/ui/attrs/setting_neutral',True)
		State.push('set neutral')
	if key == ord('S'):
		toggle_unreal()
	if key == ord('Z'):
		State.setKey('/root/ui/attrs/debugging',not State.getKey('/root/ui/attrs/debugging'))
		State.push('debugging')

def convert_rbfn(rbfn_filename):
	print 'using pickle'
	import pickle
	RBFN = pickle.load(open(rbfn_filename,'rb'))
	groups, slider_splits, slider_names, marker_names = extract_groups({'rbfn':RBFN})
	print RBFN.keys()
	print RBFN['Eyes']['marker_data'].keys()
	pn = ''.join([ap if ap.rsplit('_',2)[1]=='Neutral' else '' for ap in RBFN['Eyes']['marker_data'].keys()])
	print pn
	tmp = extract_x2ds(RBFN['Eyes'], pn, marker_names)
	RBFN_neutral_shape = stabilize_shape(tmp, neutral=0)[0]
	return {'rbfn':RBFN,'neutral':RBFN_neutral_shape,'marker_names':marker_names,'slider_names':slider_names}

def main():
	from UI import QGLViewer
	from UI import GLMeshes, GLPoints3D

	global g_setting_frame
	g_setting_frame = False
	# static data
	global g_webcam, g_md, g_rbfn, g_predictor, g_head_pan_shape, g_head_tilt_shape
	# runtime options and state
	global g_prev_smooth_shape, g_prev_vs, g_hmc_boot, g_neutral_corrective_shape, g_settle, g_head_pan_tilt_roll, g_smooth_pose
	global g_directory, g_TIS_server, g_mode, g_frame

	g_TIS_server = SocketServer.SocketServer()
	g_mode, g_frame = 0,{}

	grip_dir = os.environ['GRIP_DATA']
	g_directory = grip_dir

	g_webcam,g_md = None,None

	g_prev_vs, g_prev_smooth_shape = None,None
	g_hmc_boot = None
	#clear_neutral()
	g_neutral_corrective_shape = IO.load(os.path.join(g_directory,'neutral.out'))[1]
	g_settle = -1
	g_head_pan_tilt_roll = None
	g_smooth_pose = {}

	aam = IO.load(os.path.join(g_directory,'aam.out'))[1]
	if 0:
		svt = np.float32(aam['shapes']).reshape(-1,140)
		svt = np.dot(aam['shapes_u'],aam['shapes_s'].reshape(-1,1)*aam['shapes_vt'])
		svt = aam['shapes_s'].reshape(-1,1)*aam['shapes_vt']
		tmp = svt.reshape(svt.shape[0],-1,2)
		Sx,Sy = tmp[:,:,0],tmp[:,:,1]
		tmp = np.dot(np.dot(Sy.T,np.dot(Sx,Sx.T)),Sy)
		u,s,vt = np.linalg.svd(tmp, full_matrices=False)
		print s
		g_head_pan_shape = np.zeros((svt.shape[1]/2,2),dtype=np.float32)
		g_head_tilt_shape = np.zeros((svt.shape[1]/2,2),dtype=np.float32)
		g_head_pan_shape[:,0] = g_head_tilt_shape[:,1] = vt[0]
		print np.sum(g_head_pan_shape * aam['shapes_vt'][0].reshape(-1,2))
		print np.sum(g_head_tilt_shape * aam['shapes_vt'][1].reshape(-1,2))
	g_head_pan_shape = aam['shapes_vt'][0].reshape(-1,2)
	g_head_tilt_shape = aam['shapes_vt'][1].reshape(-1,2)
	g_head_tilt_shape = g_head_pan_shape[:,::-1]*np.float32([1,-1])
	print np.sum(g_head_pan_shape*g_head_tilt_shape)
	g_head_pan_shape *= np.linalg.norm(g_head_pan_shape)**-0.5
	g_head_tilt_shape *= np.linalg.norm(g_head_tilt_shape)**-0.5
	if np.sum(g_head_pan_shape[:,0] < 1): g_head_pan_shape = -g_head_pan_shape
	if np.sum(g_head_tilt_shape[:,1] > 1): g_head_tilt_shape = -g_head_tilt_shape
	#print np.sum(g_head_pan_shape * g_head_tilt_shape)
	#print np.dot(g_head_pan_shape[:,0],g_head_tilt_shape[:,1])

	g_predictor = Face.load_predictor(os.path.join(g_directory,'train.out'))
	rbfn_filename = os.path.join(g_directory,'rbfn.out')
	g_rbfn = IO.load(rbfn_filename)[1]
	#g_rbfn = convert_rbfn(rbfn_in_filename)
	#IO.save(rbfn_filename, g_rbfn)

	
	ref_shape = g_predictor['ref_shape']
	cx,cy = np.mean(ref_shape,axis=0)
	vx,vy = (np.var(ref_shape,axis=0)**0.5) * 2.5
	geo_bs = []
	ref_fs = Face.triangulate_2D(ref_shape)
	for p0,p1,p2 in ref_fs:
		geo_bs.append((p0,p1))
		geo_bs.append((p1,p2))
		geo_bs.append((p2,p0))
	geo_vs = np.zeros((len(ref_shape),3), dtype=np.float32)
	geo_fs = []
	geo_ts = np.float32([[1,0,0,0],[0,1,0,1000],[0,0,1,0]])
	geo_vts = np.zeros_like(ref_shape)
	
	img_vs = np.float32([[-1000,-1000,0],[1000,-1000,0],[1000,1000,0],[-1000,1000,0]])
	img_fs = np.int32([[0,1,2,3]])
	img_ts = np.float32([[1,0,0,0],[0,1,0,1000],[0,0,1,0]])
	img_vts = np.float32([[0,1],[1,1],[1,0],[0,0]])
	markup_mesh = GLPoints3D(vertices=geo_vs, edges=np.int32(geo_bs), names=[], colour=[0,1,0,1],edgeColour=[1,1,1,1])
	geo_mesh = GLMeshes(names=['geo_mesh'],verts=[geo_vs],faces=[geo_fs],transforms=[geo_ts],bones=[geo_bs], vts=[geo_vts], colour=[1,0,0,1])
	image_mesh = GLMeshes(names=['image_mesh'],verts=[img_vs],faces=[img_fs],transforms=[img_ts],vts=[img_vts])

	global g_bs_vs, g_bs_shape_mat, g_bs_fs, g_bs_vts, g_bs_shape_mat_T
	bs_dict = IO.load(os.path.join(g_directory,'harpy_ma.out'))[1]['blendShapes']['Harpy_cFace_GEOShape']
	obj_scale = 10.0
	g_bs_vs = np.float32(bs_dict['vs']*obj_scale)
	bs_dict['pts'] = [b*obj_scale for b in bs_dict['pts']]
	g_bs_fs = bs_dict['fs'] # warning: mix of quads and triangles :-(
	assert bs_dict['vts'].keys() == range(len(bs_dict['vts'].keys()))
	g_bs_vts = bs_dict['vts'].values()
	g_bs_ts = np.float32([[1,0,0,800],[0,1,0,-600],[0,0,1,300]])
	bs_mesh = GLMeshes(names=['bs_mesh'],verts=[g_bs_vs],faces=[g_bs_fs],transforms=[g_bs_ts],vts=[g_bs_vts],visible=False)

	rbfn_groups, rbfn_slider_splits, rbfn_slider_names, rbfn_marker_names = extract_groups(g_rbfn)
	slider_names = [(x[8:-2]+'.translateY' if x.startswith('get_ty') else x) for x in bs_dict['wt_names']]
	try:
		slider_order = [slider_names.index(x) for x in rbfn_slider_names]
	except Exception as e:
		print 'error',e
		slider_order = []
	g_bs_shape_mat = bs_dict['matrix'] = np.zeros((len(bs_dict['pts']), len(bs_dict['vs']), 3),dtype=np.float32)
	for m,ct,pt in zip(g_bs_shape_mat,bs_dict['cts'],bs_dict['pts']): m[ct] = pt
	g_bs_shape_mat = g_bs_shape_mat[slider_order]
	g_bs_shape_mat_T = g_bs_shape_mat.transpose(1,2,0).copy()

	layers = {'image_mesh':image_mesh,'geo_mesh':geo_mesh,'bs_mesh':bs_mesh,'markup_mesh':markup_mesh}
	app,win = QGLViewer.makeApp()
	outliner = win.qoutliner
	#for gi,geo in enumerate(layers.keys()): outliner.addItem(geo, data='_OBJ_'+geo, index=gi)

	State.setKey('ui',{'type':'ui','attrs':{\
		'harpy_xoffset':300.0,'show_harpy':True,'rotate':0,'mirroring':False,'unreal':True,'streaming_TIS':False,\
		'using_webcam':False,'HMC_mode':True,'booting':True,'filtering':True,'setting_neutral':True,'debugging':False, \
		'webcam_index':0,'cam_offset':700,'cam_fps':50,'cam_width':1280,'cam_height':720, 'movie_filename':''}})
	if True: # running on deployed face machine at 720p50
		State.setKey('/root/ui',{'type':'ui','attrs':{\
			'harpy_xoffset':300.0,'show_harpy':False,'rotate':1,'mirroring':False,'unreal':True,'streaming_TIS':False,\
			'using_webcam':True,'HMC_mode':True,'booting':True,'filtering':True,'setting_neutral':True,'debugging':False, \
			'webcam_index':0,'cam_offset':700,'cam_fps':50,'cam_width':1280,'cam_height':720, 'movie_filename':''}})
	win.setFields('ui',     [
		('show_harpy',      'show_harpy','Whether to display the Harpy','bool', False),
		('harpy_xoffset',   'xoffset', 'Pixels to offset Harpy to right', 'float', 300.0),
		('rotate',          'rotation','Rotate image 0=up,1=left,2=down,3=right,-1=any angle','int', 0),
		('mirroring',       'mirror',  'Show reversed',                 'bool', False),
		('unreal',          'unreal',  'Whether to connect to unreal',  'bool', True),
		#('streaming_TIS',   'streaming_TIS',  'Whether currently streaming',   'bool', False),
		('using_webcam',    'webcam',  'Whether using the webcam',      'bool', False),
		('HMC_mode',        'HMC_mode','Boot every frame',              'bool', True),
		('booting',         'boot',    'Boot at next chance',           'bool', True),
		('filtering',       'filter',  'Whether to filter noise',       'bool', True),
		('setting_neutral', 'neutral', 'Set neutral at next chance',    'bool', False),
		('debugging',       'debug',   'Show rbfn input for debugging', 'bool', False),
		('webcam_index',    'camindex', 'The index of the webcam',      'int',  0),
		('cam_offset',      'camoffset','The offset of the webcam',     'int',  700),
		('cam_fps',         'fps',      'The frame rate of the webcam', 'int',  50),
		('cam_width',       'width',    'The width of the webcam image', 'int',  1280),
		('cam_height',      'height',   'The height of the webcam image', 'int',  720),
		('movie_filename',  'movie',   'The filename of the movie', 'string',  ''),
		])
	slider_names = sorted(g_rbfn['slider_names'])
	win.setFields('sliders', [(sn,sn,'Slider %d'%si,'float',0.0) for si,sn in enumerate(slider_names)])
	State.setKey('/root/sliders', {'type':'sliders','attrs':{sn:0.0 for sn in slider_names}})
	outliner.set_root('/root')
	#outliner.addItem('sliders', data='sliders', index=1)
	win.outliner.raise_()
	#win.select('ui')
	QApp.app.dirtyCB = dirty_cb
	QApp.app.addMenuItem({'menu':'&File','item':'Import &movie','tip':'Import a movie file','cmd':import_movie})
	QApp.app.addMenuItem({'menu':'&Edit','item':'Retrain rbfn','tip':'Train the rbfn','cmd':retrain_RBFN})
	QApp.app.addMenuItem({'menu':'&Edit','item':'Retrain rbfn (no linear)','tip':'Train the rbfn with no linear part','cmd':retrain_RBFN_no_linear})
	QApp.app.addMenuItem({'menu':'&Edit','item':'Retrack refresh rbfn','tip':'Refresh the rbfn','cmd':retrack_refresh_rbfn})
	QApp.app.addMenuItem({'menu':'&Edit','item':'Retrack remap rbfn','tip':'Rebuild the rbfn','cmd':retrack_remap_rbfn})
	QApp.app.addMenuItem({'menu':'&File','item':'Export rbfn','tip':'Export the rbfn','cmd':export_rbfn})
	State.clearUndoStack()
	QGLViewer.makeViewer(appName='StreamVideoTrack',timeRange=(0,100), callback=setFrame_cb, keyCallback=keypress_cb, layers=layers, mats=[Calibrate.makeMat(Calibrate.composeRT(np.eye(3)*[10,10,1],[0,1000,6000],1000),[0,0],[1920,1080])], camera_ids=['RBFN'])

	# Ensure the server has stopped when program terminates
	g_TIS_server.Stop()



if __name__ == '__main__':
	main()
