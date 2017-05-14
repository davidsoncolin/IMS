#!/usr/bin/env python

import numpy as np
import ISCV
from OpenGL import GL,GLU,GLUT
from OpenGL.arrays import vbo

def bind_image(image):
	h,w,_3 = image.shape
	assert _3 == 3,'must be rgb'
	tid = GL.glGenTextures(1)
	GL.glBindTexture(GL.GL_TEXTURE_2D, tid)
	GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
	GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
	GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, w, h, 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, image)
	GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
	return tid

def unbind_image(tid):
	GL.glDeleteTextures([tid])

def bind_streaming_image(image, tid = None, bid = None):
	h,w,_3 = image.shape
	assert _3 == 3,'must be rgb'
	if tid is None: tid = GL.glGenTextures(1)
	if bid is None: bid = GL.glGenBuffers(1) # for streaming video
	GL.glBindTexture(GL.GL_TEXTURE_2D, tid)
	GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, bid)
	GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
	GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
	GL.glBufferData(GL.GL_PIXEL_UNPACK_BUFFER, image.size, image, GL.GL_STREAM_DRAW)
	GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, w, h, 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, None)
	GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)
	GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
	return tid, bid

def renderGL(w, h, func, data):
	# TODO store original GL context and restore it afterwards?
	startGL()
	fbid = GL.glGenFramebuffers(1)
	GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbid)
	dbid = GL.glGenRenderbuffers(1)
	GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, dbid)
	GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT, w, h)
	tid = GL.glGenTextures(1)
	GL.glBindTexture(GL.GL_TEXTURE_2D, tid)
	GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
	GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
	GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, w, h, 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, None)
	GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, tid, 0)
	GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER, dbid)
	assert GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) == GL.GL_FRAMEBUFFER_COMPLETE, 'GL error'
	GL.glViewport(0, 0, w, h)
	if isinstance(data,tuple): func(*data)
	else:                      func(data)
	data = GL.glReadPixels(0, 0, w, h, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
	GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
	GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, 0)
	GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
	GL.glDeleteTextures([tid])
	GL.glDeleteFramebuffers(1,[fbid])
	GL.glDeleteRenderbuffers(1,[dbid])
	return np.fromstring(data, dtype=np.uint8).reshape(h,w,3)

def startGL():
	if GLUT.glutInit() == (): return # once only
	GLUT.glutInitDisplayMode(GLUT.GLUT_RGBA | GLUT.GLUT_DEPTH)
	GLUT.glutInitWindowSize(1,1)
	GLUT.glutCreateWindow('offscreen rendering context window')
	GLUT.glutHideWindow()

def make_quad_distortion_mesh(ooa=1.0,w=64,h=64,Kox=0,Koy=0,dist=(0.29,0.22)):
	startGL()
	xsc,ysc,w4 = 1.0/w,1.0/h,w*4
	vs,vts=[],[]
	#quads2 = list(range(0,w4,4)) + list(range(w4-3,w4*h,w4)) + list(range(w4*h-2,w4*(h-1),-4)) + list(range(w4*(h-1)+3,0,-w4))
	quads = np.arange(w4*h,dtype=np.int32)
	v0 = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]],dtype=np.float32)
	for y in range(h):
		for x in range(w):
			xs = v0+[x,y,0]
			vs.extend(xs)
			vts.extend(xs[:,:2])
	vs = np.float32((np.float32(vs)*[2*xsc,2*ysc,0]-[1,1,1])*[1,ooa,1])
	vs[:,2] *= -1 # TODO
	vts = np.float32(np.float32(vts)*[xsc,ysc])
	ISCV.undistort_points(vs, -float(Kox), -float(Koy), float(dist[0]), float(dist[1]), vs)
	vs = vbo.VBO(vs, usage='GL_STATIC_DRAW_ARB')
	vts = vbo.VBO(vts, usage='GL_STATIC_DRAW_ARB')
	quads = vbo.VBO(quads, target=GL.GL_ELEMENT_ARRAY_BUFFER, usage='GL_STATIC_DRAW_ARB')
	return vs,vts,quads

def quad_render(tid,(vs,vts,quads),scale=1.0):
	# render the movie texture with the x2ds into the texture coords in the buffer
	GL.glClearColor(1,0,0,1)
	GL.glClear(GL.GL_COLOR_BUFFER_BIT|GL.GL_DEPTH_BUFFER_BIT)
	GL.glShadeModel(GL.GL_FLAT)
	GL.glDisable(GL.GL_LIGHTING)
	GL.glEnable(GL.GL_TEXTURE_2D)
	GL.glBindTexture(GL.GL_TEXTURE_2D, tid)
	GL.glMatrixMode(GL.GL_PROJECTION)
	GL.glLoadIdentity()
	GL.glMatrixMode(GL.GL_MODELVIEW)
	GL.glLoadIdentity()
	GLU.gluOrtho2D(-1/scale, 1/scale, -1/scale, 1/scale) # (-1,-1) at bottom-left
	GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
	vs.bind()
	GL.glVertexPointerf(vs)
	GL.glEnableClientState(GL.GL_TEXTURE_COORD_ARRAY)
	vts.bind()
	GL.glTexCoordPointerf(vts)
	quads.bind()
	GL.glShadeModel(GL.GL_FLAT)
	GL.glBindTexture(GL.GL_TEXTURE_2D, tid)
	GL.glEnable(GL.GL_TEXTURE_2D)
	GL.glDrawElementsui(GL.GL_QUADS, quads)
	GL.glDisable(GL.GL_TEXTURE_2D)
	quads.unbind()
	vts.unbind()
	vs.unbind()
	GL.glDisableClientState(GL.GL_TEXTURE_COORD_ARRAY)
	GL.glDisableClientState(GL.GL_VERTEX_ARRAY)

class ProjectionMapper:
	def __init__(self, mat):
		self.mat = mat
	
	def setGLMeshes(self, glmesh):
		self.vs_mapping = glmesh.vs_mapping
		self.vts = glmesh.vts
		self.tris = glmesh.tris
	
	def project(self, x3ds, aspect):
		# set the buffer as the texture
		K,RT,P,ks,T,wh = self.mat
		
		#num_pts = len(x3ds)
		#x2ds, splits, labels = ISCV.project(x3ds, np.arange(num_pts,dtype=np.int32), P[:3,:4].reshape(1,3,4))
		#x2s = 1e10*np.ones((num_pts,2),dtype=np.float32)
		#x2s[labels,:] = x2ds
		
		# project the 3D vertices into the camera using the projection matrix
		proj = np.dot(x3ds,P[:3,:3].T) + P[:3,3]
		ds = -proj[:,2]
		x2s = proj[:,:2]/ds.reshape(-1,1)
		# distort the projections using the camera lens
		ISCV.distort_points(x2s, float(-K[0,2]), float(-K[1,2]), float(ks[0]), float(ks[1]), x2s)
		# convert to texture coordinates
		x2s *= [0.5,-0.5*aspect]
		x2s += 0.5
		self.x2ds = x2s
		if 0: # wip
			self.ds = ds
			GL.glClearColor(0,0,0,1)
			GL.glClear(GL.GL_COLOR_BUFFER_BIT|GL.GL_DEPTH_BUFFER_BIT)
			
			GL.glMatrixMode(GL.GL_PROJECTION)
			GL.glLoadIdentity()
			GL.glMultMatrixf(np.array([[1,0,0,0],[0,aspect,0,0],[0,0,-1,-1],[0,0,cameraInterest*-0.02,0]],dtype=np.float32))

			GL.glMatrixMode(GL.GL_MODELVIEW)
			GL.glLoadIdentity()
			GL.glMultMatrixd(P.T)
			GL.glDisable(GL.GL_TEXTURE_2D)
			GL.glEnable(GL.GL_DEPTH_TEST)
			GL.glShadeModel(GL.GL_FLAT)
			GL.glDisable(GL.GL_LIGHTING)
			GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
			GL.glVertexPointerf(x3ds)
			self.tris.bind()
			GL.glDrawElementsui(GL.GL_TRIANGLES, self.tris)
			self.tris.unbind()

	def render(self, tid):
		# render the movie texture with the x2ds into the texture coords in the buffer
		x2ds = np.array(self.x2ds[self.vs_mapping],dtype=np.float32)
		GL.glClearColor(0,0,0,1)
		GL.glClear(GL.GL_COLOR_BUFFER_BIT|GL.GL_DEPTH_BUFFER_BIT)
		GL.glShadeModel(GL.GL_FLAT)
		GL.glDisable(GL.GL_LIGHTING)
		GL.glEnable(GL.GL_TEXTURE_2D)
		GL.glBindTexture(GL.GL_TEXTURE_2D, tid)
		GL.glMatrixMode(GL.GL_PROJECTION)
		GL.glLoadIdentity()
		GL.glMatrixMode(GL.GL_MODELVIEW)
		GL.glLoadIdentity()
		GLU.gluOrtho2D(0, 1, 0, 1) # (0,0) at bottom-left
		GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
		self.vts.bind()
		GL.glVertexPointerf(self.vts)
		GL.glEnableClientState(GL.GL_TEXTURE_COORD_ARRAY)
		v = vbo.VBO(x2ds, usage='GL_STATIC_DRAW_ARB')
		v.bind()
		GL.glTexCoordPointerf(v)
		self.tris.bind()
		GL.glDrawElementsui(GL.GL_TRIANGLES, self.tris)
		self.tris.unbind()
		v.unbind()
		self.vts.unbind()


if(__name__ == '__main__'):

	def drawStuff(aspect):
		GL.glMatrixMode(GL.GL_PROJECTION)
		GL.glLoadIdentity()
		GLU.gluPerspective(40.,aspect,1.,40.)
		GL.glMatrixMode(GL.GL_MODELVIEW)
		GL.glLoadIdentity()
		GLU.gluLookAt(0,0,10, 0,0,0, 0,1,0)
		GL.glClearColor(0,1,0,1)
		GL.glShadeModel(GL.GL_SMOOTH)
		GL.glEnable(GL.GL_CULL_FACE)
		GL.glEnable(GL.GL_DEPTH_TEST)
		GL.glEnable(GL.GL_LIGHTING)
		lightPosition = [10,4,10,1]
		lightColor = [1, 1, 1, 1]
		GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, lightPosition)
		GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE, lightColor)
		GL.glLightf(GL.GL_LIGHT0, GL.GL_CONSTANT_ATTENUATION, 0.1)
		GL.glLightf(GL.GL_LIGHT0, GL.GL_LINEAR_ATTENUATION, 0.05)
		GL.glEnable(GL.GL_LIGHT0)
		GL.glClear(GL.GL_COLOR_BUFFER_BIT|GL.GL_DEPTH_BUFFER_BIT)
		color = [1,0,0,1]
		GL.glMaterialfv(GL.GL_FRONT,GL.GL_DIFFUSE,color)
		GLUT.glutSolidSphere(2, 40, 40)

	w,h = 512,512
	data = renderGL(w,h, drawStuff, w/float(h))
	from PIL import Image
	image = Image.frombytes(mode='RGB', size=(w, h), data=data)
	image = image.transpose(Image.FLIP_TOP_BOTTOM)
	image.save('screenshot.png')
