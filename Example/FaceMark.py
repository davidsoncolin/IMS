#!/usr/bin/env python

import os
import numpy as np
from GCore import State, Face
from IO import IO, JPEG, MovieReader
from UI import QApp, QGLViewer, GLMeshes, GLPoints3D

'''
this is a database tool. it enables the creation and editing of a database of points.
in this case, the points are facial landmark annotations.
state keys:
/images          : a list of the images in the db (h,w,3 uint8)
/shapes          : a list of the vertices in the db (num_verts,2 float32)
/labels          : a list of the labels for each vertex (num_verts int32)
/predictor       : a dict of the tracking model trained from the db
/aam             : a dict of the appearance model trained from the db
/edges           : array, the edges from triangulation of the vertices (num_edges,2 int32)
/tris            : array, the triangles from triangulation of the vertices (num_tris,3 int32)
/vnames          : a list of names of the vertices (num_verts str)
/markup_mesh_sel : the currently selected vertex (int)
/flip            : array, for each vertex the mirror-image (num_verts int32)
/order           : array, the original ordering of the images (num_images int32)
/frame_number    : the index of the currently-editing frame (int)
'''

global g_predictor, g_aam_model
g_predictor, g_aam_model = None, None

def get_predictor():
	global g_predictor
	if g_predictor is None: g_predictor = State.getKey('/predictor', None)
	return g_predictor

def get_aam():
	global g_aam_model
	if g_aam_model is None: g_aam_model = State.getKey('/aam', None)
	return g_aam_model

def images_shapes_not_all_labels_unmarked():
	images = State.getKey('/images')
	shapes = State.getKey('/shapes')
	labels = State.getKey('/labels')
	which = np.where(np.sum(np.int32(labels)==0, axis=1) == 0)[0]
	return [JPEG.decompress(images[x]) for x in which], [shapes[x] for x in which], [labels[x] for x in which]

def retrain():
	predictor = get_predictor()
	images,shapes,labels = images_shapes_not_all_labels_unmarked()
	flip = State.getKey('/flip', None)
	Face.double_data(images, shapes, flip_order=flip)
	predictor = Face.retrain_shape_predictor(predictor, images, shapes)
	State.setKey('/predictor', predictor)
	State.push('retrain predictor')

def train():
	images,shapes,labels = images_shapes_not_all_labels_unmarked()
	flip = State.getKey('/flip', None)
	Face.double_data(images, shapes, flip_order=flip)
	predictor = Face.train_shape_predictor(images, shapes)
	State.setKey('/predictor', predictor)
	State.push('train predictor')

def train_aam():
	images,shapes,labels = images_shapes_not_all_labels_unmarked()
	flip = State.getKey('/flip', None)
	aam_model = Face.train_aam_model(images,shapes, flip_order=flip, texture_rank=20)
	State.setKey('/aam',aam_model)
	State.push('train aam')

def boot_face():
	fi = frame_number()
	img = get_frame_image(fi)
	predictor = get_predictor()
	shp = Face.detect_face(img, predictor)
	if shp is not None:
		set_frame_markup(fi, shp)
		State.push('boot face markup')

def reorder_images():
	images = State.getKey('/images')
	shapes = State.getKey('/shapes')
	labels = State.getKey('/labels')
	old_order = State.getKey('/order', np.arange(len(images), dtype=np.int32))
	norm_shapes, ref_shape, ref_pinv = Face.normalized_shapes(shapes)
	norm_shapes -= ref_shape
	shapes_u,shapes_s,shapes_vt = np.linalg.svd(norm_shapes.reshape(norm_shapes.shape[0],-1), full_matrices=0)
	wts = np.sum(shapes_u**2,axis=1)
	order = np.argsort(wts)[::-1]
	images = [images[o] for o in order]
	shapes = [shapes[o] for o in order]
	labels = [labels[o] for o in order]
	State.setKey('/images', images)
	State.setKey('/shapes', shapes)
	State.setKey('/labels', labels)
	State.setKey('/order', old_order[order])
	State.push('reorder images')

def test():
	images,shapes,labels = images_shapes_not_all_labels_unmarked()
	predictor = State.getKey('/predictor')
	Face.test_shape_predictor(predictor, images, shapes)

def export_predictor(): save_object('predictor', 'Predictor Files (*.io)', '/predictor')
def import_predictor(fn = None):
	if fn is None: fn,_ = QApp.app.loadFilename('Choose predictor to open', cwd(), 'Predictor Files (*.io)')
	State.setKey('/predictor', Face.load_predictor(fn))
	set_cwd(fn)

def export_aam(): save_object('aam', 'AAM Files (*.io)', '/aam')
def import_aam(): load_object('aam', 'AAM Files (*.io)', '/aam')

def save_object(desc, filetype, key):
	print 'exporting',desc
	aam_fn,_ = QApp.app.loadFilename('Choose a file to write '+desc, cwd(), desc)
	IO.save(aam_fn,State.getKey(key))

def load_object(desc, filetype, key):
	print 'imorting',desc
	fn,_ = QApp.app.loadFilename('Choose '+desc+' to open', cwd(), filetype)
	State.setKey(key,IO.load(fn)[1])
	set_cwd(fn)

def get_frame_markup(fi = None):
	if fi is None: fi = frame_number()
	shp = State.getKey('/shapes/%d'%fi, None)
	labels = State.getKey('/labels/%d'%fi, None)
	return shp, labels

def set_frame_image(fi, img):
	State.setKey('/images/%d'%fi, JPEG.compress(img))

def get_frame_image(fi):
	img = State.getKey('/images/%d'%fi, None)
	if isinstance(img, np.ndarray):
		print 'compressing image',fi
		set_frame_image(fi, img)
	img = JPEG.decompress(img)
	if img is not None and (len(img.shape) != 3 or img.shape[2] != 3 or img.dtype != np.uint8):
		print 'repairing img',fi,img.shape,img.dtype
		ret = np.zeros((img.shape[0],img.shape[1],3), dtype=np.uint8)
		ret[:,:,:3] = img.reshape(img.shape[0], img.shape[1], -1)[:,:,:3]
		img = ret
		print 'repaired img',fi,img.shape,img.dtype
		set_frame_image(fi, img)
	return img

def set_frame_markup(fi, shp):
	State.setKey('/shapes/%d'%fi, np.float32(shp))

def set_frame_labels(fi, lbl):
	State.setKey('/labels/%d'%fi,np.int32(lbl))

def delete_image():
	fi = frame_number
	images = State.getKey('/images', [])
	shapes = State.getKey('/shapes', [])
	labels = State.getKey('/labels', [])
	images.pop(fi)
	shapes.pop(fi)
	labels.pop(fi)
	State.setKey('/images', images)
	State.setKey('/shapes', shapes)
	State.setKey('/labels', labels)
	State.push('delete image')

def add_image(img, shp=None):
	if shp is None: shp = get_predictor()['ref_shape']
	if not State.hasKey('/images') or not State.hasKey('/shapes') or not State.hasKey('/labels'):
		State.setKey('/images', [])
		State.setKey('/shapes', [])
		State.setKey('/labels', [])
	fi = len(State.getKey('/images'))
	lbl = np.zeros(len(shp), dtype=np.int32) # unlabelled
	set_frame_image(fi, img)
	set_frame_markup(fi, shp)
	set_frame_labels(fi, lbl)
	State.push('add image')

def add_vertex(vname = None):
	# TODO later, we want to initialise every vertex that has a zero label
	shapes = State.getKey('/shapes')
	labels = State.getKey('/labels')
	vnames = State.getKey('/vnames')
	if vname is None: vname = 'pt_%d' % len(vnames)
	shapes = np.float32(shapes)
	labels = np.int32(labels)
	num_images, num_vertices = labels.shape
	shapes = list(np.hstack((shapes,np.zeros((num_images,1,2), dtype=np.float32))))
	labels = list(np.hstack((labels,np.zeros((num_images,1), dtype=np.int32))))
	vnames.append(vname)
	State.setKey('/shapes', shapes)
	State.setKey('/labels', labels)
	State.setKey('/vnames', vnames)
	State.push('add vertex')

def get_selected_vertex():
	return State.getKey('/markup_mesh_sel',-1)
	#markup_mesh = QApp.view().getLayer('markup_mesh')
	#return markup_mesh.selectedIndex

def set_selected_vertex(vi):
	State.setKey('/markup_mesh_sel',vi)
	markup_mesh = QApp.view().getLayer('markup_mesh')
	markup_mesh.selectedIndex = vi
	
def delete_selected_vertex():
	vi = get_selected_vertex()
	shapes = State.getKey('/shapes')
	labels = State.getKey('/labels')
	vnames = State.getKey('/vnames')
	num_images, num_vertices = labels.shape
	which = range(0, vi)+range(vi+1, num_vertices())
	shapes = list(np.float32(shapes)[:,which])
	labels = list(np.int32(labels)[:,which])
	vnames = [vnames[x] for x in which]
	State.setKey('/shapes', shapes)
	State.setKey('/labels', labels)
	State.setKey('/vnames', vnames)
	State.push('delete vertex')

def retriangulate():
	shp = get_predictor()['ref_shape']
	edges = []
	tris = Face.triangulate_2D(shp)
	for p0,p1,p2 in tris:
		edges.append((p0,p1))
		edges.append((p1,p2))
		edges.append((p2,p0))
	edges = np.int32(edges)
	tris = np.int32(tris)
	State.setKey('/edges', edges)
	State.setKey('/tris', tris)
	State.push('triangulate')

def update_flip_order_last1():
	'''A hacky method that makes the last added point be its own flip.'''
	flip = State.getKey('/flip', [16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,\
		26,25,24,23,22,21,20,19,18,17,27,28,29,30,35,34,33,32,31,45,44,43,42,47,46,\
		39,38,37,36,41,40,54,53,52,51,50,49,48,59,58,57,56,55,64,63,62,61,60,67,66,65])
	flip = list(flip)
	flip.extend([len(flip)])
	print flip
	flip = np.int32(flip)
	State.setKey('/flip',flip)
	State.push('update flip last 1')

def update_flip_order_last2():
	'''A hacky method that makes the last two added points be each others flip.'''
	flip = State.getKey('/flip', [16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,\
		26,25,24,23,22,21,20,19,18,17,27,28,29,30,35,34,33,32,31,45,44,43,42,47,46,\
		39,38,37,36,41,40,54,53,52,51,50,49,48,59,58,57,56,55,64,63,62,61,60,67,66,65])
	flip = list(flip)
	flip.extend([len(flip)+1,len(flip)])
	print flip
	flip = np.int32(flip)
	State.setKey('/flip', flip)
	State.push('update flip last 2')

def set_frame_labels(fi, lbl):
	State.setKey('/labels/%d'%fi,np.int32(lbl))

def num_vertices():
	# this is the number of vertices in the DB; it could be different from the number of vertices in the predictor or aam...
	return len(State.getKey('/labels/0', []))

def num_aam_vertices():
	return len(State.getKey('/aam/ref_shape', []))

def num_predictor_vertices():
	return len(State.getKey('/predictor/ref_shape', []))

def shape_to_aam(shp):
	size = num_vertices()
	assert(shp.shape == (size,2))
	asize = num_aam_vertices()
	ashp = np.zeros((asize,2), dtype=np.float32)
	ashp[:min(size,asize)] = shp[:min(size,asize)]
	return ashp

def shape_to_predictor(shp):
	size,psize = num_vertices(),num_predictor_vertices()
	assert(shp.shape == (size,2))
	pshp = np.zeros((psize,2), dtype=np.float32)
	pshp[:min(size,psize)] = shp[:min(size,psize)]
	return pshp

def update_markup_mesh(view):
	aam_model = get_aam()
	shp, lbl = get_frame_markup()
	markup_mesh,ref_mesh = view.getLayers(['markup_mesh','ref_mesh'])
	size,asize = num_vertices(),num_aam_vertices()
	vs = np.zeros((size,3), dtype=np.float32)
	avs = np.zeros((asize,3), dtype=np.float32)
	if shp is not None: vs[:,:2] = shp
	markup_mesh.setVs(vs, drawStyles=lbl)
	avs[:min(size,asize)] = vs[:min(size,asize)]
	if shp is not None and aam_model is not None:
		ashp = shape_to_aam(shp)
		shape_u, A_inv, mn = Face.fit_aam_shape(aam_model, ashp)
		aweights = np.ones(asize, dtype=np.float32)
		aweights[np.where(lbl[:min(size,asize)] == 2)] = 1000.0
		ashp[:] = Face.weighted_fit_aam_shape(aam_model, ashp, aweights)
		avs[:asize,:2] = ashp
	ref_mesh.setVs(avs)

def boot_from_ref_shape(view):
	size,psize = num_vertices(),num_predictor_vertices()
	fi = frame_number()
	shp, labels = get_frame_markup(fi)
	if shp is None: return
	ref_mesh = view.getLayer('ref_mesh').vertices[:,:2]
	pref_mesh = shape_to_predictor(ref_mesh)
	img = get_frame_image(fi)
	if img is not None:
		Face.track_face(img, get_predictor(), out=pref_mesh)
	which_verts = np.where(labels[:min(size,psize)] != 2)
	labels[which_verts] = 1
	shp[which_verts] = pref_mesh[which_verts]
	set_frame_markup(fi, shp)
	set_frame_labels(fi, labels)

def select_vertex(view, vi):
	fi = frame_number()
	set_selected_vertex(vi)
	view.setTool('3d drag vertex')
	view.drag_data = vi
	shp, labels = get_frame_markup(fi)
	if labels is None: return
	labels[vi] = 2
	set_frame_labels(fi, labels)

def move_vertex(view, vi, position, delta=False):
	fi = frame_number()
	shp, labels = get_frame_markup(fi)
	if shp is None: return
	if delta: shp[vi] += position
	else: shp[vi] = position
	set_frame_markup(fi, shp)

def unlabel_selected_vertex(view, delta=False):
	fi = frame_number()
	vi = get_selected_vertex()
	if vi == -1: return False
	shp, labels = get_frame_markup(fi)
	if labels is None: return False
	labels[vi] = 0
	set_frame_labels(fi, labels)
	return True

def unlabel_all_vertices(view, delta=False):
	fi = frame_number()
	shp, labels = get_frame_markup(fi)
	if labels is None: return False
	labels[:] = 0
	set_frame_labels(fi, labels)

def frame_number():
	return State.getKey('/frame_number',0)

def update_gui_image(view, img):
	if img is None: img = np.zeros((8,8,3), dtype=np.uint8)
	width,height = img.shape[1],img.shape[0]
	w,h = width*0.5,height*0.5
	markup_mesh,ref_mesh,image_mesh = view.getLayers(['markup_mesh','ref_mesh','image_mesh'])
	markup_mesh.transform[:,:3] = [[1,0,0],[0,1,0],[0,0,1],[-w,1000-h,0.1]]
	ref_mesh.transform[:,:3] = [[1,0,0],[0,1,0],[0,0,1],[-w,1000-h,0.05]] # ref_mesh is behind
	image_mesh.setVs(np.array([[-w,-h,0],[w,-h,0],[w,h,0],[-w,h,0]], dtype=np.float32))
	image_mesh.setImage(img)

def set_frame_cb(fi):
	view = QApp.view() # TODO should be part of the callback?
	if fi != frame_number():
		State.setKey('/frame_number', fi)
		State.push('change frame')
	img = get_frame_image(fi)
	update_gui_image(view, img)
	update_markup_mesh(view)
	QApp.app.updateMenus() # trigger a full refesh here (TODO not needed?)

def drawGL_cb(view):
	if QApp.app.qtimeline.frame != frame_number():
		QApp.app.qtimeline.frame = frame_number()
	update_markup_mesh(view)

def dragCB(view, data):
	vi = view.drag_data
	#print 'dragCB',data,vi
	offset = (view.mouseDx * view.toolRX + view.mouseDy * view.toolRY)
	move_vertex(view, vi, offset[:2], delta=True)
	State.push('move vertex')
	QApp.app.updateMenus()

def pickedCB(view,data,clearSelection=True):
	#print 'pickedCB',view,data,clearSelection
	if data is None:
		QApp.app.select(None) # TODO achievable through the view?
	else:
		primitive_type,pn,pi,distance = data
		if primitive_type is '3d':
			p = view.primitives[pn]
			if isinstance(p,GLPoints3D):
				if pi == -1: # edge
					print 'picked edge'
					return
				#name = p.names[pi]
				print "Picked:", pi
				if p is view.getLayer('ref_mesh'):
					move_vertex(view, pi, p.vertices[pi,:2])
				select_vertex(view,pi)
				State.push('select vertex')
				QApp.app.updateMenus()

def keyCB(view, key):
	#print 'keyCB',repr(key),type(key)
	if key == ord('R'):
		boot_from_ref_shape(view)
		State.push('boot markup')
		QApp.app.updateMenus()
	if key == ord('U'):
		if unlabel_selected_vertex(view):
			State.push('unlabel vertex')
			QApp.app.updateMenus()
	if key == ord('Z'):
		unlabel_all_vertices(view)
		State.push('unlabel all vertices')
		QApp.app.updateMenus()

def dirtyCB(dirty):
	# triggered by a state change, this allows the app to synch with the state
	# TODO this is the correct place to deal with changes due to eg menu items or hot keys
	# should be able to remove eg updateGL from everywhere else, really
	#print 'dirty',dirty
	if dirty:
		outliner = QApp.app.qoutliner
		outliner.set_root(outliner.root) # TODO this causes a total rebuild

	global g_predictor, g_aam_model
	if '/predictor' in dirty: g_predictor=None
	if '/aam' in dirty: g_aam_model = None
	if '/vnames' in dirty:
		#print 'setting names',State.getKey('/vnames')
		QApp.view().getLayer('markup_mesh').names = State.getKey('/vnames', [])
	if '/markup_mesh_sel' in dirty:
		QApp.view().getLayer('markup_mesh').selectedIndex = State.getKey('/markup_mesh_sel', -1)
	if '/edges' in dirty:
		QApp.view().getLayer('markup_mesh').edges = State.getKey('/edges', None)
		QApp.view().getLayer('ref_mesh').edges = State.getKey('/edges', None)

def cwd():
	cwd = State.getKey('/cwd', None)
	if cwd is None: cwd = os.environ['GRIP_DATA']+'/'
	return cwd

def set_cwd(fn):
	State.setKey('/cwd', os.path.dirname(fn))

def import_image():
	image_fn,_ = QApp.app.loadFilename('Choose an image to open', cwd(), 'Image Files (*.jpg *.jpeg *.png *.bmp *.tif)')
	if image_fn == '': return # cancel
	img = Face.load_image(image_fn)
	images,shapes = [],[]
	add_image(img)
	State.push('Import image')

def import_movie_frames():
	movie_fn, _ = QApp.app.loadFilename('Choose a movie to open', cwd(), 'Movie Files (*.mp4 *.mov *.avi *.flv *.mpg)')
	if movie_fn == '': return # cancel
	set_cwd(movie_fn)
	txt_fn, _ = QApp.app.loadFilename('Choose a text file of frame indices to open', cwd(), 'Text Files (*.txt)')
	md = MovieReader.open_file(movie_fn, audio=False)
	images,shapes = [],[]
	if txt_fn == '': frames = range(0, md['vmaxframe'], 100)
	#if txt_fn == '': frames = range(30000, 38300, 100)
	else: frames = [int(l.split(':')[1]) for l in open(txt_fn,'r').readlines()]
	for fi in frames:
		print fi,'/',frames[-1]
		MovieReader.readFrame(md, fi)
		add_image(np.frombuffer(md['vbuffer'], dtype=np.uint8).reshape(md['vheight'],md['vwidth'], 3).copy())
	State.push('Import movie frames')

if __name__ == '__main__':

	if 1:
		grip_dir = os.environ['GRIP_DATA']
		pred_fn = os.path.join(grip_dir,'train.out')
		import_predictor(pred_fn)

	if 0:
		State.load('mu2.scn')
		State.g_dirty.clear()

	if 0:
		vnames = ['aEdge1', 'aEdge2', 'aEdge3', 'aEdge4', 'aEdge5', 'aEdge6', 'aEdge7', 'aEdge8', 'aEdge9', 'aEdge10', 'aEdge11', 'aEdge12', 'aEdge13', 'aEdge14', 'aEdge15', 'aEdge16', 'aEdge17', # 0:17
				'arEyebrow1', 'arEyebrow2', 'arEyebrow3', 'arEyebrow4', 'arEyebrow5', # 17:22
				'alEyebrow5', 'alEyebrow4', 'alEyebrow3', 'alEyebrow2', 'alEyebrow1', # 22:27
				'anoseBridge1', 'anoseBridge2', 'anoseBridge3', 'anoseBridge5', 'anostril1', 'anostril2', 'anostril3', 'anostril4', 'anostril5', # 27:36
				'arEyeOuter', 'arEyeUpper1', 'arEyeUpper2', 'arEyeInner', 'arEyeLower2', 'arEyeLower1', # 36:42
				'alEyeInner', 'alEyeUpper2', 'alEyeUpper1', 'alEyeOuter', 'alEyeLower1', 'alEyeLower2', # 42:48
				'arMouthOuterCorner', 'arMouthOuterUpper1', 'arMouthOuterUpper2', 'acMouthOuterUpper', 'alMouthOuterUpper2', 'alMouthOuterUpper1', 'alMouthOuterCorner',
				'alMouthOuterLower1', 'alMouthOuterLower2', 'acMouthOuterLower', 'arMouthOuterLower2', 'arMouthOuterLower1', 'arMouthInnerCorner', 'arMouthInnerUpper',
				'acMouthInnerUpper', 'alMouthInnerUpper', 'alMouthInnerCorner', 'alMouthInnerLower', 'acMouthInnerLower', 'arMouthInnerLower',  # 48:68
				]
		State.setKey('/vnames', vnames)

	if 0:
		subset = [0,7,8,9,16,27,30,31,33,35,36,39,42,45,48,54]
		flip = [subset.index(x) for x in [16,9,8,7,0,27,30,35,33,31,45,42,39,36,54,48]]
		vnames = [vnames[x] for x in subset]
		State.setKey('/flip', flip)
		State.setKey('/vnames', vnames)
		shapes = State.getKey('/shapes')
		labels = State.getKey('/labels')
		shapes = list(np.float32(shapes)[:,subset])
		labels = list(np.int32(labels)[:,subset])
		State.setKey('/shapes', shapes)
		State.setKey('/labels', labels)

	img_vs = np.float32([[-1000,-1000,0],[1000,-1000,0],[1000,1000,0],[-1000,1000,0]])
	img_fs = np.int32([[0,1,2,3]])
	img_ts = np.float32([[1,0,0,0],[0,1,0,1000],[0,0,1,0]])
	img_vts = np.float32([[0,1],[1,1],[1,0],[0,0]])

	if not State.hasKey('/edges') or not State.hasKey('/tris'):
		retriangulate()

	template_vs = get_predictor()['ref_shape'] * 100
	size = len(template_vs)
	markup_x2ds = np.zeros((size,3), dtype=np.float32)
	ref_vs = np.zeros((size,3), dtype=np.float32)
	ref_vs[:,:2] = template_vs
	
	markup_mesh = GLPoints3D(vertices=markup_x2ds, edges=None, names=[], colour=[0,1,0,1], edgeColour=[1,1,1,1])
	ref_mesh = GLPoints3D(vertices=ref_vs, edges=State.getKey('/edges', None), edgeColour=[0.0,1.0,0.0,0.5], colour=[0.0,1.0,1.0,0.5])
	image_mesh = GLMeshes(names=['image_mesh'], verts=[img_vs], faces=[img_fs], transforms=[img_ts], bones=[[]], vts=[img_vts])
	layers = {'image_mesh':image_mesh,'markup_mesh':markup_mesh,'ref_mesh':ref_mesh}
	app,win = QGLViewer.makeApp(appName='Imaginarium FaceMark')
	outliner = QApp.app.qoutliner
	outliner.set_root('')
	QApp.app.addMenuItem({'menu':'&Edit','item':'&Retrain','tip':'Retrain the face model (1 hour)','cmd':retrain})
	QApp.app.addMenuItem({'menu':'&Edit','item':'&Train','tip':'Train the face model (1 day)','cmd':train})
	QApp.app.addMenuItem({'menu':'&Edit','item':'Train &AAM','tip':'Train the AAM (1 minute)','cmd':train_aam})
	QApp.app.addMenuItem({'menu':'&Edit','item':'Te&st','tip':'Test the face model (1 minute)','cmd':test})
	QApp.app.addMenuItem({'menu':'&File','item':'Export &predictor','tip':'Export the predictor as a file','cmd':export_predictor})
	QApp.app.addMenuItem({'menu':'&File','item':'Import &predictor','tip':'Import the predictor from a file','cmd':import_predictor})
	QApp.app.addMenuItem({'menu':'&File','item':'Export &aam','tip':'Export the aam to a file','cmd':export_aam})
	QApp.app.addMenuItem({'menu':'&File','item':'Import &aam','tip':'Import the aam from a file','cmd':import_aam})
	QApp.app.addMenuItem({'menu':'&Edit','item':'&Reorder images','tip':'Reorder by likelihood','cmd':reorder_images})
	QApp.app.addMenuItem({'menu':'&Test','item':'&Import movie','tip':'Add some frames from a movie','cmd':import_movie_frames})
	QApp.app.addMenuItem({'menu':'&Test','item':'&Import image','tip':'Add an image','cmd':import_image})
	QApp.app.addMenuItem({'menu':'&Test','item':'&Boot Face','tip':'Boot the face','cmd':boot_face,'shortcut':'Ctrl+B'})
	QApp.app.addMenuItem({'menu':'&Test','item':'&Add vertex','tip':'Add a vertex to the DB','cmd':add_vertex})
	QApp.app.addMenuItem({'menu':'&Test','item':'&Delete vertex','tip':'Remove selected vertex from the DB','cmd':delete_selected_vertex})
	QApp.app.addMenuItem({'menu':'&Test','item':'Delete image','tip':'Remove this image from the DB','cmd':delete_image})
	QApp.app.addMenuItem({'menu':'&Test','item':'Retriangu&late','tip':'Re-triangulate the mesh','cmd':retriangulate})
	QApp.app.addMenuItem({'menu':'&Test','item':'Update flip order last 1','tip':'Last vertex is its own flip','cmd':update_flip_order_last1})
	QApp.app.addMenuItem({'menu':'&Test','item':'Update flip order last 2','tip':'Last two vertices are each others flip','cmd':update_flip_order_last2})
	State.clearUndoStack()
	QGLViewer.makeViewer(timeRange=(0,8000), callback=set_frame_cb, pickCallback=pickedCB, dragCallback=dragCB, keyCallback=keyCB, drawCallback=drawGL_cb, dirtyCallback=dirtyCB, layers=layers)
