import numpy as np
import time, os
import FaceTrack
from GCore import Calibrate
try:
	import freenect
except:
	def surface_to_array(surface):
		import ctypes
		buffer_interface = surface.get_buffer()
		address = ctypes.c_void_p()
		size = Py_ssize_t()
		_PyObject_AsWriteBuffer(buffer_interface, ctypes.byref(address), ctypes.byref(size))
		bytes = (ctypes.c_byte * size.value).from_address(address.value)
		bytes.object = buffer_interface
		return bytes

	if 0:
		import thread
		import pykinect
		from pykinect import nui
		
		class freenect:
			# a dummy class with static methods to emulate a module
			def __init__(self):
				pass
		
			@staticmethod
			def init():
				global g_kinect, g_screen_lock, g_depth_data, g_video_data
				g_screen_lock = thread.allocate()
				g_depth_data, g_video_data = None,None
				g_kinect = nui.Runtime()
				g_kinect.depth_frame_ready += freenect.depth_frame_ready    
				g_kinect.video_frame_ready += freenect.video_frame_ready    
				#g_kinect.video_stream.open(nui.ImageStreamType.Video, 2, nui.ImageResolution.Resolution1280x1024, nui.ImageType.Color)
				g_kinect.video_stream.open(nui.ImageStreamType.Video, 2, nui.ImageResolution.Resolution640x480, nui.ImageType.Color)
				#g_kinect.depth_stream.open(nui.ImageStreamType.Depth, 2, nui.ImageResolution.Resolution640x480, nui.ImageType.Depth)
				#print 'init'

			@staticmethod
			def depth_frame_ready(frame):
				#print 'depth_frame_ready'
				global g_kinect, g_screen_lock, g_depth_data, g_video_data
				with g_screen_lock: g_depth_data = frame.image.bits

			@staticmethod
			def video_frame_ready(frame):
				#print 'video_frame_ready'
				global g_kinect, g_screen_lock, g_depth_data, g_video_data
				with g_screen_lock: g_video_data = frame.image.bits
				
			@staticmethod
			def sync_get_video():
				#print 'sync_get_video'
				global g_kinect, g_screen_lock, g_depth_data, g_video_data
				#print g_kinect.video_stream.height,g_kinect.video_stream.width,len(g_video_data)
				video = np.fromstring(g_video_data,dtype=np.uint8).reshape(-1,g_kinect.video_stream.width,4)
				video = video[:,:,[2,1,0]].copy()
				#print 'video',np.median(video)
				return [video]

			@staticmethod
			def DEPTH_REGISTERED(): pass # a dummy value
			
			@staticmethod
			def sync_get_depth(format=None):
				#print 'sync_get_depth'
				global g_kinect, g_screen_lock, g_depth_data, g_video_data
				depth = np.zeros((480,640),dtype=np.float32)
				#depth = np.fromstring(g_depth_data,dtype=np.uint16).reshape(g_kinect.depth_stream.height,g_kinect.depth_stream.width).astype(np.float32)
				depth[:] = 1800.0
				return [depth]
				
			@staticmethod
			def open_device(kinect,num): return freenect()
			
			def set_led(self, val): pass
			
			def set_tilt_degs(self, degs):
				global g_kinect, g_screen_lock, g_depth_data, g_video_data
				g_kinect.camera.elevation_angle = degs
			
			def update_tilt_state(self): pass

			def get_tilt_state(self): 
				global g_kinect, g_screen_lock, g_depth_data, g_video_data
				print dir(g_kinect.camera)
				from GCore import atdict
				self.tilt_state = atdict({'tilt_angle':52,'tilt_status':0,'accelerometer_x':85,'accelerometer_y':743,'accelerometer_z':369})
				return self.tilt_state

			def close_device(self): pass
	else:
		class freenect:
			# a dummy class with static methods to emulate a module
			def __init__(self):
				pass
		
			@staticmethod
			def init():
				global g_record
				_,g_record = IO.load('dump')

			@staticmethod
			def sync_get_video():
				global g_frame, g_record
				if g_frame not in g_record: g_frame = 1
				#print g_frame
				video = g_record[g_frame]['video']
				return [video]

			@staticmethod
			def DEPTH_REGISTERED(): pass # a dummy value
			
			@staticmethod
			def sync_get_depth(format=None):
				global g_frame, g_record
				if g_frame not in g_record: g_frame = 1
				#print g_frame
				depths = g_record[g_frame]['depths']
				return [depths]
				
			@staticmethod
			def open_device(kinect,num): return freenect()
			
			def set_led(self, val): pass
			
			def set_tilt_degs(self, degs): pass
			
			def update_tilt_state(self): pass

			def get_tilt_state(self): 
				from GCore import atdict
				self.tilt_state = atdict({'tilt_angle':85,'tilt_status':743,'accelerometer_x':369,'accelerometer_y':52,'accelerometer_z':0.0})
				return self.tilt_state

			def close_device(self): pass
	
import ISCV
import IO
import sys
from PySide import QtGui, QtCore
from UI import QGLViewer, QApp
from GCore import Calibrate, State
from UI import GLMeshes

def make_coords(height,width):
	'''Generate a uniform grid on the pixels, scaled so that the x-axis runs from -1 to 1.'''
	pix_coord = np.zeros((height,width,2),dtype=np.float32)
	pix_coord[:,:,1] = np.arange(height).reshape(-1,1)
	pix_coord[:,:,0] = np.arange(width)
	coord = (pix_coord - np.array([0.5*(width-1),0.5*(height-1)],dtype=np.float32)) * np.array([2.0/width,-2.0/width],dtype=np.float32)
	return coord,pix_coord

def clean_depth_buffer(coord, mat, depths, kernel_size = 2):
	'''Given camera calibration data, clean the geometry using a median filter.'''
	# step 1: depth per pixel
	K,RT,P,ks,T,wh = mat
	h,w = coord.shape[:2]
	coord = coord.copy()
	Calibrate.undistort_points_mat(coord.reshape(-1,2), mat, coord.reshape(_1,2))
	shape = [h,w,3]
	which = np.where(depths.reshape(-1) == 0)[0]
	depths.reshape(-1)[which] = 0
	# step 2: median filter
	filtered_depths = 1e20*np.ones([h,w,1],dtype=np.float32)
	filtered_depths.reshape(-1)[:] = depths.reshape(-1)
	filtered_depths.reshape(-1)[which] = 1e20
	if 0:
		for y in range(kernel_size,h-kernel_size):
			for x in range(kernel_size,w-kernel_size):
				d = depths[y-kernel_size:y+kernel_size+1,x-kernel_size:x+kernel_size+1].reshape(-1)
				which = np.where(d != 0)[0]
				if len(which): filtered_depths[y,x] = np.median(d[which])
	#filtered_depths[:,:,0] = depths # HACK, now the results should look the same
	# step 3: ray per pixel
	rays = np.dot(coord,RT[:2,:3]) # ray directions (unnormalized)
	rays -= np.dot([-K[0,2],-K[1,2],K[0,0]],RT[:3,:3])
	rays /= (np.sum(rays**2,axis=-1)**0.5).reshape(h,w,1) # normalized ray directions
	# step 4: compose
	return (rays * filtered_depths).reshape(depths.shape[0],depths.shape[1],3)+T

def depths_to_points(rays, T, depths):
	h,w = depths.shape[:2]
	which = np.where(depths.reshape(-1) == 0)[0]
	ret = (rays * depths.reshape(h,w,1)).reshape(h,w,3)+T
	ret.reshape(-1,3)[which] = 1e20
	return ret

def make_faces(height,width):
	return np.array([(vi+1,vi,vi+width,vi+width+1) for vi in xrange((height-1)*width) if (vi%width)!=width-1],dtype=np.int32)

def eval_shape_predictor(predictor, img, rect):
	rect = np.array(rect,dtype=np.int32)
	ref_pinv,ref_shape,forest_splits,forest_leaves,anchor_idx,deltas = predictor
	if 1: # all-in-one C 1.8ms
		current_shape = ref_shape.copy()
		forest_leaves2 = forest_leaves.reshape(forest_leaves.shape[0],forest_leaves.shape[1],forest_leaves.shape[2],-1)
		ISCV.eval_shape_predictor(ref_pinv, ref_shape, forest_splits, forest_leaves2, anchor_idx, deltas, img, rect, current_shape)
		return current_shape

global g_record
g_record = {}

def extract_depths(img, vts):
	clip_min,clip_max = np.array([[0,0],[img.shape[1]-1,img.shape[0]-1]],dtype=np.int32)
	p = np.int32(vts) # as pixels
	np.clip(p,clip_min,clip_max,out=p) # clip to image
	return img[p[:,1],p[:,0]] # and sample

#@profile
def cb(frame):
	global g_record, g_frame
	g_frame = frame
	global g_camera_rays, g_camera_mat
	#print 'in cb'
	img = freenect.sync_get_video()[0]
	geom_mesh = QApp.app.getLayer('geom_mesh')
	geom_mesh.setImage(img)
	
	if 0:
		depths = freenect.sync_get_depth(format=freenect.DEPTH_REGISTERED)[0]
		#print 'depths',np.median(depths)
		
		#if frame not in g_record: return
		#img,depths = g_record[frame]['video'],g_record[frame]['depths']
		#g_record[frame] = {'video':img.copy(),'depths':depths.copy()}
		#if frame == 99: IO.save('dump',g_record)
		
		depths_sum = np.array(depths != 0,dtype=np.int32)
		depths_lo = np.array(depths[::2,::2]+depths[1::2,::2]+depths[::2,1::2]+depths[1::2,1::2],dtype=np.float32)
		lookup = np.array([0,1,0.5,1.0/3,0.25],dtype=np.float32)
		depths_lo = depths_lo * lookup[(depths_sum[::2,::2]+depths_sum[1::2,::2]+depths_sum[::2,1::2]+depths_sum[1::2,1::2]).reshape(-1)].reshape(depths_lo.shape)
		K,RT,P,ks,T,wh = g_camera_mat
		vs = depths_to_points(g_camera_rays, T, depths_lo)
		geom_mesh.setVs(vs.reshape(-1,3))

	#QApp.view().setImage(img, img.shape[0], img.shape[1], img.shape[2])
	#camera = QApp.view().camera
	#geom_mesh.image = camera.image
	#geom_mesh.bindImage = camera.bindImage
	#geom_mesh.bindId = camera.bindId
	global g_predictor, g_detector, reference_3d, geo_vs, geo_vts
	h,w,_3 = img.shape
	
	global g_prev_vs
	try: g_prev_vs
	except: g_prev_vs = None
	use_prev_vs = True
	
	tmp = FaceTrack.detect_face(img, g_detector, g_predictor) if g_prev_vs is None else g_prev_vs
	tmp = FaceTrack.track_face(img, g_predictor, tmp, numIts=5)
	if use_prev_vs: g_prev_vs = tmp
	if frame == 0 or FaceTrack.test_reboot(img, g_prev_vs): g_prev_vs = None
	geo_vts[:len(tmp)] = tmp
	if g_predictor[0]: geo_vts[:,1] = img.shape[0]-geo_vts[:,1]
	
	current_shape = geo_vts[:len(tmp)].copy()

	if 0:
		ds = extract_depths(vs, current_shape*0.5)
		M,inliers = Calibrate.rigid_align_points_inliers(ds, reference_3d, scale=True, threshold_ratio=200.0)
		ds = np.dot(ds,M[:3,:3].T)+M[:,3]
		reference_3d[inliers] = ds[inliers]
		ds[:] = reference_3d[:]
		M[1,3] += 1000
	else:
		M = np.eye(3,4,dtype=np.float32)
		M[1,3] += 1000
	geom_mesh.setPose(M.reshape(1,3,4))
	
	ref_pinv = g_predictor[1]
	xform = np.dot(ref_pinv,current_shape)
	ut,s,v = np.linalg.svd(xform)
	s = (s[0]*s[1])**-0.5
	xform_inv = np.dot(v.T,ut.T)*s
	current_shape = np.dot(current_shape - np.mean(current_shape,axis=0), xform_inv) * 100.
	geo_vs[:] = 0
	geo_vs[:len(current_shape),:2] = current_shape * [1,1 if g_predictor[0] else -1] # convert from y-down to y-up
	#geo_vs[:68] = ds
	#geo_vs[:68,:] += [0,100,5500]
	#print geo_vts[:4],w,h
	geo_mesh = QApp.app.getLayer('geo_mesh')
	geo_mesh.setVs(geo_vs,vts=geo_vts*np.array([1.0/w,1.0/h],dtype=np.float32))
	geo_mesh.setImage(img)
	#geo_mesh.transforms[0][:,:3] = [[1,0,0],[0,1,0],[0,0,1],[0,1000,0.1]]

	if 0:
		global g_model
		w,h = 160,160
		shp = geo_vs[:68,:2]
		shape_u, tex_u, A_inv, mn = FaceTrack.fit_aam(g_model, tmp, img)
		FaceTrack.render_aam(g_model, A_inv*0.5, mn*0.5, shape_u, tex_u, img)

	img_mesh = QApp.app.getLayer('img_mesh')
	img_mesh.setImage(img)

	QApp.view().updateGL()


#@profile
def main():
	global g_model
	g_model = IO.load('model.train')[1]
	
	global g_predictor, g_detector, reference_3d, geo_vs, geo_vts, rect
	import dlib
	g_detector = dlib.get_frontal_face_detector()
	rect = None
	grip_dir = os.environ['GRIP_DATA']
	pred_fn = os.path.join(grip_dir,'out.train')
	g_predictor = FaceTrack.load_predictor(pred_fn, cutOff=15)
	yup,ref_pinv,reference_shape,splits,leaves,anchor_idx,deltas = g_predictor
	size = reference_shape.shape[0]
	geo_vs = np.zeros((size+4,3),dtype=np.float32)
	geo_vs[:size,:2] = reference_shape
	geo_vts = np.zeros((size+4,2),dtype=np.float32)
	geo_vts[:size] = reference_shape + 0.5
	geo_ts = np.array([[1,0,0,0],[0,1,0,1000],[0,0,1,0]],dtype=np.float32)
	geo_fs = FaceTrack.triangulate_2D(reference_shape)
	geo_bs = []
	for p0,p1,p2 in geo_fs:
		geo_bs.append((p0,p1))
		geo_bs.append((p1,p2))
		geo_bs.append((p2,p0))
	reference_3d = np.zeros((reference_shape.shape[0],3),dtype=np.float32)
	reference_3d[:,:2] = reference_shape*[100,-100]
	
	img_vs = np.array([[0,0,0],[640,0,0],[640,480,0],[0,480,0]],dtype=np.float32)
	img_vts = np.array([[0,1],[1,1],[1,0],[0,0]],dtype=np.float32)
	img_fs = np.array([[0,1,2,3]],dtype=np.int32)
	img_ts = np.array([[1,0,0,0],[0,1,0,1000],[0,0,1,0]],dtype=np.float32)
	
	geo_mesh = GLMeshes(names=['geo_mesh'],verts=[geo_vs],faces=[geo_fs],transforms=[geo_ts],bones=[geo_bs],vts=[geo_vts])
	img_mesh = GLMeshes(names=['img_mesh'],verts=[img_vs],faces=[img_fs],transforms=[img_ts],bones=[None],vts=[img_vts])
	kinect = freenect.init()
	tilt,roll = 0,0

	if 1:
		kdev = freenect.open_device(kinect,0)
		freenect.set_led(kdev,0) # turn off LED
		freenect.set_tilt_degs(kdev,30)
		kstate = freenect.get_tilt_state(kdev)
		freenect.update_tilt_state(kdev)
		tilt_angle,tilt_status = kstate.tilt_angle,kstate.tilt_status
		ax,ay,az = kstate.accelerometer_x,kstate.accelerometer_y,kstate.accelerometer_z
		#bottom facing down: (85, 743, 369, 52, 0)
		#right side down: (916, 71, 96, 112, 0)
		#front side down: (52, 63, -863, -128, 0)
		freenect.close_device(kdev)
		y_axis = np.array((ax,ay,az),dtype=np.float32)
		y_axis = y_axis/np.linalg.norm(y_axis)
		roll = np.degrees(np.arctan2(ax,ay))
		tilt = -np.degrees(np.arctan2(az,(ax**2+ay**2)**0.5))

	fovX = 45.
	pan_tilt_roll = (0,tilt,roll)
	tx_ty_tz = (0,1000,6000)
	P = Calibrate.composeP_fromData((fovX,),(pan_tilt_roll),(tx_ty_tz),0)

	
	global g_camera_rays, g_camera_mat
	h,w = 480/2,640/2
	coord,pix_coord = make_coords(h,w)
	#P = np.eye(3,4,dtype=np.float32)
	#P[0,0] = P[1,1] = 2.0
	k1,k2 = 0,0
	g_camera_mat = Calibrate.makeMat(P,(k1,k2),[w,h])
	K,RT,P,ks,T,wh = g_camera_mat
	coord_undist = coord.copy()
	Calibrate.undistort_points_mat(coord.reshape(-1,2), g_camera_mat, coord_undist.reshape(-1,2))
	g_camera_rays = np.dot(coord_undist,RT[:2,:3]) # ray directions (unnormalized)
	g_camera_rays -= np.dot([-K[0,2],-K[1,2],K[0,0]],RT[:3,:3])
	g_camera_rays /= (np.sum(g_camera_rays**2,axis=-1)**0.5).reshape(h,w,1) # normalized ray directions	
	names = ['kinect']
	vs = [np.zeros((h*w,3),dtype=np.float32)]
	ts = [np.eye(3,4,dtype=np.float32)]
	vts = [pix_coord*(1.0/w,1.0/h)]
	faces = [make_faces(h,w)]
	mats = None
	geom_mesh = GLMeshes(names=names,verts=vs,faces=faces,transforms=ts,vts=vts)
	layers = {'geom_mesh':geom_mesh, 'geo_mesh':geo_mesh, 'img_mesh':img_mesh}
	QGLViewer.makeViewer(layers=layers, mats = mats, callback=cb, timeRange=(0,10000))

if __name__ == '__main__':
	main()
