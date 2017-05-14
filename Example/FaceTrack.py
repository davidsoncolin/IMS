#!/usr/bin/env python

import os
import numpy as np
import ISCV
from UI import QApp,QGLViewer
from UI import GLMeshes
from GCore import Face,Opengl
from IO import MovieReader, IO

def import_movie_cb():
	grip_dir = os.environ['GRIP_DATA']
	movie_fn, _ = QApp.app.loadFilename('Choose a movie to open', grip_dir, 'Movie Files (*.mp4 *.mov *.avi *.flv *.mpg)')
	global md
	md = MovieReader.open_file(movie_fn)

def set_frame_cb(frame):
	global g_aam_model, g_images, g_shapes, g_predictor, template_vs
	indx = frame/4
	which = frame%4
	img = np.zeros((160,160,3),dtype=np.uint8)
	if which == 0:
		img = g_images[indx].copy()
		shp = g_shapes[indx].copy()
		tex = textures[indx]
	elif which == 1:
		img = g_images[indx].copy()
		shp = g_shapes[indx].copy()
		tex = textures[indx]
		shp = Face.track_face(img, g_predictor, shp)
	elif which == 2:
		shp = g_aam_model['shapes'][indx]
		tex = g_aam_model['textures'][indx]
	else:
		shp = g_aam_model['ref_shape'] + np.dot(g_aam_model['shapes_u'][indx]*g_aam_model['shapes_s'],\
			g_aam_model['shapes_vt']).reshape(-1,2)
		tex = g_aam_model['texture_mean'] + np.dot(g_aam_model['texture_u'][indx]*g_aam_model['texture_s'],\
			g_aam_model['texture_vt']).reshape(-1,3)
	width,height = img.shape[1],img.shape[0]
	w,h = width*0.5,height*0.5
	if which >= 2:
		shp = Face.normalize_shape(shp, model['ref_pinv'])
		shp = (shp + 1) * [w,h]
		np.clip(tex,0,255,out=tex)
		Face.render_texture(tex, img, shp, model['model_indices'], model['model_weights'])
	geo_mesh = QApp.app.getLayer('geo_mesh')
	size = 68+4
	vs = np.zeros((size,3),dtype=np.float32)
	vs[:size-4,:2] = shp
	vs[size-4:size,:2] = Face.get_boundary(shp, template_vs)
	geo_mesh.setVs(vs)
	geo_mesh.transforms[0][:,:3] = [[1,0,0],[0,1,0],[0,0,1],[-w,1000-h,0.1]]
	image_mesh = QApp.app.getLayer('image_mesh')
	image_mesh.setVs(np.array([[-w,-h,0],[w,-h,0],[w,h,0],[-w,h,0]], dtype=np.float32))
	image_mesh.setImage(img)
	QApp.view().updateGL()

#@profile
def set_frame_cb2(frame):
	global g_predictor, g_predictor_dlib, g_detector
	size = (len(g_predictor['ref_shape'])+4)
	geo_vs = np.zeros((size,3), dtype=np.float32)
	ref_vs = np.zeros((size,3), dtype=np.float32)

	global g_prev_vs
	try: g_prev_vs
	except: g_prev_vs = None
	if 0: # show_images
		global g_jpgs; fn = g_jpgs[frame%len(g_jpgs)]
		img = Face.load_image(fn)
		img = Face.fix_image(img, max_size=640)
		use_prev_vs = False # images need booting every frame
	else: # show_movies
		global md; MovieReader.readFrame(md, seekFrame=frame) # only update the visible camera
		img = np.frombuffer(md['vbuffer'], dtype=np.uint8).reshape(md['vheight'],md['vwidth'],3)
		use_prev_vs = True
		
	if 0: # undistort_stuff
		global g_screen
		global g_tid, g_bid
		g_tid,g_bid = Opengl.bind_streaming_image(img, g_tid, g_bid)
		img = Opengl.renderGL(img.shape[1], img.shape[0], Opengl.quad_render, (g_tid, g_screen, 0.85))
		#Opengl.unbind_image(bid)

	if 0: # rotated_image
		img = img.transpose((1,0,2)).copy()
	if 0: # gamma_image
		lookup = np.array([int(((x/255.0)**0.4545)*255.0) for x in range(256)], dtype=np.uint8)
		img = lookup[img]
	#img[:,600:1000] = 0 #img[:,200:600].copy()
	if 0: # test_rotate
		import scipy; img = scipy.misc.imrotate(img, frame, interp='bilinear')
	if 0: # test_rotate_right
		import scipy; img[:,-img.shape[0]:] = scipy.misc.imrotate(img[:,-img.shape[0]:], frame, interp='bilinear')
	if 0: # test_filter_image
		img = ISCV.filter_image(img,4,16)

	w,h = img.shape[1]*0.5,img.shape[0]*0.5

	boot = g_prev_vs
	if boot is None: boot = Face.detect_face(img, g_predictor, 2) # ,-1) # put -1 at end to boot at any angle
	tmp = Face.track_face(img, g_predictor, boot)
	if use_prev_vs and boot is not None: g_prev_vs = tmp
	if frame == 0 or Face.test_reboot(img, g_prev_vs): g_prev_vs = None
	global template_vs
	geo_vs[:size-4,:2] = tmp
	geo_vs[size-4:size,:2] = Face.get_boundary(geo_vs[:size-4,:2], template_vs)

	if 0: # show_aam
		global g_aam_model
		shape_u, tex_u, A_inv, mn  = Face.fit_aam(g_aam_model, tmp, img)
		Face.render_aam(g_aam_model, A_inv*0.1, mn*0.1, shape_u, tex_u, img)
		su,tu = Face.normalized_aam_coords(g_aam_model, shape_u, tex_u)
		res = Face.aam_residual(g_aam_model, tmp, img)
		QApp.view().displayText = [(10,100,'%f' % np.linalg.norm(tu)),(10,125,'%f' % np.linalg.norm(su)),(10,150,'%f'%res)]

	if 0: # show_extracted_texture
		global g_aam_model_indices,g_aam_model_weights
		pixels = Face.extract_texture(img, geo_vs[:size,:2], g_aam_model_indices, g_aam_model_weights)
		global template_vs
		Face.render_texture(pixels, img, template_vs, g_aam_model_indices, g_aam_model_weights)

	geo_mesh = QApp.app.getLayer('geo_mesh')
	geo_mesh.setVs(geo_vs)
	geo_mesh.transforms[0][:,:3] = [[1,0,0],[0,1,0],[0,0,1],[-w,1000-h,0.1]]
	image_mesh = QApp.app.getLayer('image_mesh')
	image_mesh.setVs(np.array([[-w,-h,0],[w,-h,0],[w,h,0],[-w,h,0]], dtype=np.float32))
	image_mesh.setImage(img)
	QApp.view().updateGL()

def keyCB(view, key):
	#print 'keyCB',repr(key),type(key)
	#if key == ord('S'): save_db()
	if key == ord('R'):
		global g_prev_vs
		g_prev_vs = None

if __name__ == '__main__':
	global g_screen  # undistort stuff
	g_screen = Opengl.make_quad_distortion_mesh(dist=(0.2,0))

	global g_aam_model, g_predictor
	grip_dir = os.environ['GRIP_DATA']
	g_aam_model = Face.load_aam(os.path.join(grip_dir,'aam.io'))

	if 0:
		global g_jpgs
		tmp_dir = os.path.join(grip_dir,'ims_faces')
		g_jpgs = sorted([os.path.join(tmp_dir,x) for x in os.listdir(tmp_dir) if x.endswith('.jpg')])

	rotated_video = False
	md = None #MovieReader.open_file(movie_fn)

	global g_predictor
	g_predictor = Face.load_predictor(os.path.join(grip_dir,'train.out'))

	global template_vs
	template_vs = g_predictor['ref_shape'] * 100
	cx,cy = np.mean(template_vs[:,:2],axis=0)
	vx,vy = (np.var(template_vs[:,:2],axis=0)**0.5) * 2.5
	template_vs = np.vstack((template_vs,np.array([[cx-vx,cy-vy],[cx+vx,cy-vy],[cx+vx,cy+vy],[cx-vx,cy+vy]], dtype=np.float32)))
	template_vs -= np.int32(np.min(template_vs, axis=0))
	#show_image(shape=template_vs)

	geo_bs = []
	ref_fs = Face.triangulate_2D(template_vs)
	for p0,p1,p2 in ref_fs:
		geo_bs.append((p0,p1))
		geo_bs.append((p1,p2))
		geo_bs.append((p2,p0))

	size = len(template_vs)

	geo_vs = np.zeros((size,3), dtype=np.float32)
	geo_fs = []
	geo_ts = np.array([[1,0,0,0],[0,1,0,1000],[0,0,1,0]], dtype=np.float32)
	geo_vts = [[0,0]]*size
	
	img_vs = [[-1000,-1000,0],[1000,-1000,0],[1000,1000,0],[-1000,1000,0]]
	img_fs = [[0,1,2,3]]
	img_ts = np.array([[1,0,0,0],[0,1,0,1000],[0,0,1,0]], dtype=np.float32)
	img_vts = [[0,1],[1,1],[1,0],[0,0]]
	geo_mesh = GLMeshes(names=['geo_mesh'],verts=[geo_vs],faces=[geo_fs],transforms=[geo_ts],bones=[geo_bs], vts=[geo_vts], colour=[1,0,0,1])
	image_mesh = GLMeshes(names=['image_mesh'],verts=[img_vs],faces=[img_fs],transforms=[img_ts],bones = [[]], vts=[img_vts])
	layers = {'image_mesh':image_mesh,'geo_mesh':geo_mesh}
	if 0:
		ref_vs = np.zeros((size,3), dtype=np.float32)
		ref_vs[:,:2] = template_vs
		ref_mesh = GLMeshes(names=['ref_mesh'],verts=[ref_vs],faces=[geo_fs],transforms=[geo_ts],bones=[geo_bs], vts=[geo_vts], colour=[0,1,0,1])
		layers['ref_mesh'] = ref_mesh

	app,win = QGLViewer.makeApp(appName='Imaginarium FaceTrack')
	QApp.app.addMenuItem({'menu':'&File','item':'Import &movie','tip':'Import a movie file','cmd':import_movie_cb})
	QApp.app.qoutliner.set_root('')
	#for gi,geo in enumerate(layers.keys()): outliner.addItem(geo, data='_OBJ_'+geo, index=gi)

	global g_aam_model_indices,g_aam_model_weights
	g_aam_model_indices,g_aam_model_weights = Face.make_sample_model(template_vs, ref_fs, grid_size=1.0)
	
	QGLViewer.makeViewer(timeRange=(0,8000), callback=set_frame_cb2, keyCallback=keyCB, layers=layers)
