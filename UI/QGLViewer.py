#!/usr/bin/env python

from math import sin, cos, tan
from PySide.QtOpenGL import QGLFormat
import numpy as np
from time import time

from PySide import QtCore, QtGui, QtOpenGL
from OpenGL import GL, GLU, GLUT
import QApp

from UI import GLSkeleton
from UI import DRAWOPT_DEFAULT, DRAWOPT_IMAGE, DRAWOPT_HUD, GLGrid, GLMeshes, COLOURS
import ISCV
g_fmt = QGLFormat()
g_fmt.setSwapInterval(0)
from GCore import Calibrate
import IO

class Camera:
	'''Represents a camera using world interest coordinate x,y,z (cameraT), pan,tilt,roll
	(cameraPan,cameratilt,cameraRoll), and interest distance (cameraInterest), and intrinsics horizontal field of view
	(cameraFovX), optical centre (cameraKox,cameraKoy) squareness (cameraKsquare) and skewness (cameraKskew).
	Forms a projection matrix compatible with opengl (2d x-coord in [-1,1], looking in negative z axis).
	Also holds distortion parameters and can undistort the background plate.
	Units are degrees and mm. Negative pan,tilt,roll is left,up,anticlockwise.'''
	def __init__(self, name):
		self.name = str(name)
		self.resetData = IO.encode({'FovX':45.,'Pan':0.,'Tilt':0.,'Roll':0.,'T':np.array([0,1000,0],dtype=np.float32),\
			'Interest':6000.,'Ksquare':1.,'Kskew':0.,'Kox':0.,'Koy':0.,'Distortion':None})
		self.imageInvalidated = False
		self.drawingFrustrum = True
		self.drawingDistortion = True
		self.lockedUpright = True
		self.distortionData = {}
		self.bindImage = None
		self.imageFlipped = False
		self.reset2D()
		self.reset3D()
		self.clearImage()

	def reset2D(self):
		self.cameraOx,self.cameraOy,self.cameraZoom = 0.,0.,1.

	def reset3D(self):
		for k,v in IO.decode(self.resetData)[0].iteritems(): self.__dict__.update((('camera'+k,v),))

	def setResetData(self):
		d = {}
		for k in ['Interest','FovX','Pan','Tilt','Roll','T','Kskew','Kox','Koy','Distortion']: d[k] = self.__dict__['camera'+k]
		self.resetData = IO.encode(d)

	def clearImage(self):
		self.image = None

	def undistort_points(self, x2ds, x2ds_out=None):
		if self.cameraDistortion is None and x2ds_out is not None: x2ds_out[:] = x2ds
		if self.cameraDistortion is None: return
		if x2ds_out is None: x2ds_out = x2ds
		ISCV.undistort_points	(x2ds, -float(self.cameraKox), -float(self.cameraKoy),\
								float(self.cameraDistortion[0]), float(self.cameraDistortion[1]), x2ds_out)
		
	def computeDistortionMap(self):
		# compute the distortion map
		key = (self.bindImage.width() if self.bindImage is not None else None,self.bindImage.height() if self.bindImage is not None else None,self.cameraDistortion,self.imageFlipped) if self.drawingDistortion else None
		if self.distortionData.has_key(key): return self.distortionData[key]
		ooa = float(self.bindImage.height())/float(max(1,self.bindImage.width())) if self.bindImage is not None else 1.0
		if key is None:
			vs = [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1]]
			vts = [[0,0],[1,0],[1,1],[0,1]]
			quads = np.arange(4,dtype=np.int32)
			quads2 = quads
		else:
			w,h = 32,32
			xsc,ysc,w4 = 1.0/w,1.0/h,w*4
			vs,vts=[],[]
			quads2 = list(range(0,w4,4)) + list(range(w4-3,w4*h,w4)) + list(range(w4*h-2,w4*(h-1),-4)) + list(range(w4*(h-1)+3,0,-w4))
			quads = np.arange(w4*h,dtype=np.int32)
			v0 = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]],dtype=np.float32)
			for y in range(h):
				for x in range(w):
					xs = v0+[x,y,0]
					vs.extend(xs)
					vts.extend(xs[:,:2])
			vs = np.array((np.array(vs,dtype=np.float32)*[2*xsc,2*ysc,0]-[1,1,1])*[1,ooa,1],dtype=np.float32)
			vts = np.array(vts,dtype=np.float32)*[xsc,ysc]
			self.undistort_points(vs)
		if self.imageFlipped: vts[:,1] = 1-vts[:,1]
		from OpenGL.arrays import vbo
		vs = vbo.VBO(np.array(vs,dtype=np.float32)*self.cameraInterest, usage='GL_STATIC_DRAW_ARB')
		vts = vbo.VBO(np.array(vts,dtype=np.float32), usage='GL_STATIC_DRAW_ARB')
		quads = vbo.VBO(np.array(quads,dtype=np.int32), target=GL.GL_ELEMENT_ARRAY_BUFFER, usage='GL_STATIC_DRAW_ARB')
		quads2 = vbo.VBO(np.array(quads2,dtype=np.int32), target=GL.GL_ELEMENT_ARRAY_BUFFER, usage='GL_STATIC_DRAW_ARB')
		self.distortionData = {key:(vs,vts,quads,quads2)}
		return vs,vts,quads,quads2


	def paintCamera(self, view, drawOpts=DRAWOPT_DEFAULT):
		if self.imageInvalidated:
			self.refreshImageData(view)
		if self.image is not self.bindImage:
			if self.bindImage is not None:
				view.deleteTexture(self.bindId)
				self.bindId,self.bindImage,self.imageFlipped = long(0),None,False
			if self.image is not None:
				self.bindId = view.bindTexture(self.image)
				self.bindImage = self.image

		GL.glDisable(GL.GL_DEPTH_TEST)
		GL.glColor3f(1,1,1)
		GL.glDisableClientState(GL.GL_COLOR_ARRAY)

		vs,vts,quads,quads2 = self.computeDistortionMap()
		GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
		vs.bind()
		GL.glVertexPointerf(vs)
		GL.glEnableClientState(GL.GL_TEXTURE_COORD_ARRAY)
		vts.bind()
		GL.glTexCoordPointerf(vts)
		if self.drawingFrustrum:
			quads2.bind()
			GL.glShadeModel(GL.GL_FLAT)
			GL.glLineWidth(2)
			GL.glDrawElementsui(GL.GL_LINE_LOOP, quads2)
			quads2.unbind()
		if drawOpts & DRAWOPT_IMAGE and self.bindImage is not None and self.bindId != 0:
			quads.bind()
			GL.glShadeModel(GL.GL_FLAT)
			GL.glBindTexture(GL.GL_TEXTURE_2D, self.bindId)
			GL.glEnable(GL.GL_TEXTURE_2D)
			GL.glDrawElementsui(GL.GL_QUADS, quads)
			GL.glDisable(GL.GL_TEXTURE_2D)
			quads.unbind()
		GL.glShadeModel(GL.GL_SMOOTH)
		vts.unbind()
		vs.unbind()
		GL.glDisableClientState(GL.GL_TEXTURE_COORD_ARRAY)
		GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
		GL.glEnable(GL.GL_DEPTH_TEST)
		GL.glLoadIdentity()

	def setImage(self, image):
		self.image = image
		self.imageFlipped = False

	def setImageData(self, data, height, width, chans):
		#if self.bindImage is not None:
			##self.deleteTexture(self.bindId)
			#self.bindId,self.bindImage,self.imageFlipped = long(0),None,False
		self.setImage(QtGui.QImage(data,width,height,[QtGui.QImage.Format_RGB888,QtGui.QImage.Format_ARGB32][chans-3]))

	def refreshImageData(self, view):
		'''This can be used after setImageData to notify the graphics card that the image contents has changed.'''
		if self.bindImage is not None and self.bindImage is self.image and view.bindBID != 0:
			GL.glBindTexture(GL.GL_TEXTURE_2D, self.bindId)
			GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, view.bindBID)
			self.imageFlipped = True
			GL.glBufferData(GL.GL_PIXEL_UNPACK_BUFFER, self.bindImage.byteCount(),
							np.frombuffer(self.bindImage.bits(),dtype=np.uint8), GL.GL_STREAM_DRAW)
			GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, self.bindImage.width(), self.bindImage.height(), 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, None)
			GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)
			GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
		self.imageInvalidated = False

	def invalidateImageData(self):
		self.imageInvalidated = True

	def R(self, upright=False):
		'''Yields the orientation matrix for the view, R.'''
		return Calibrate.composeR([self.cameraPan,self.cameraTilt,0 if (self.lockedUpright and upright) else self.cameraRoll])

	def RT(self, upright=False):
		'''Yields the extrinsic matrix for the view [R T].'''
		return Calibrate.composeRT(self.R(upright),self.cameraT,self.cameraInterest)

	def K(self):
		'''Yields the intrinsic matrix for the view, K.'''
		return Calibrate.composeK(self.cameraFovX,self.cameraKox,self.cameraKoy,self.cameraKsquare,self.cameraKskew)

	def P(self, upright=True):
		'''Yields the camera projection matrix P = K RT.'''
		return Calibrate.composeKRT(self.K(),self.RT(upright = upright))

	def setP(self, P, distortion = None, setInterest = True, store = False):
		'''Set the view to match the given camera projection matrix. P can be 4x4 or 3x4.'''
		K, RT = Calibrate.decomposeKRT(P)
		self.setK(K)
		self.setRT(RT, setInterest)
		self.cameraDistortion = distortion
		if distortion is not None: self.cameraDistortion = (float(distortion[0]),float(distortion[1])) # make it hashable
		if store: self.setResetData()

	def setK(self, K):
		'''Set the view to match the given intrinsic matrix.'''
		self.cameraFovX,self.cameraKox,self.cameraKoy,self.cameraKsquare,self.cameraKskew = Calibrate.decomposeK(K)

	def setRT(self, RT, setInterest = True):
		'''Set the view to match the given extrinsic matrix.'''
		(self.cameraPan, self.cameraTilt, self.cameraRoll), self.cameraT, self.cameraInterest = Calibrate.decomposeRT(RT, self.cameraInterest, setInterest)
		if self.cameraInterest == 0: self.cameraInterest = 1e-6

	def testP(self):
		K, RT = Calibrate.decomposeKRT(self.P())
		print ('K', K, self.K())
		print ('RT', RT, self.RT())
		print (self.cameraFovX,self.cameraKox,self.cameraKoy,self.cameraKsquare,self.cameraKskew)
		print ((self.cameraPan, self.cameraTilt, self.cameraRoll), self.cameraT, self.cameraInterest)
		print ('cf')
		print (Calibrate.decomposeK(K))
		print (Calibrate.decomposeRT(RT, self.cameraInterest, False))

	def T(self):
		'''Yields the actual position of the camera in 3D.'''
		return self.cameraInterest*self.R()[2,:] + self.cameraT

	def tostring(self):
		T = self.T()
		return self.name+ ' i%f f%f p%f t%f r%f tx%f ty%f tz%f d%s o%f,%f' % (self.cameraInterest,self.cameraFovX,self.cameraPan,self.cameraTilt,self.cameraRoll,T[0],T[1],T[2],self.cameraDistortion,self.cameraKox,self.cameraKoy)


class QGLView(QtOpenGL.QGLWidget):
	# emit on object selection. value is index in self.primitives (or None) 
	pickSignal = QtCore.Signal(object, object, bool)
	# emit on drag
	dragSignal = QtCore.Signal(object, str)
	drawSignal = QtCore.Signal(object)
	keySignal = QtCore.Signal(object, int)

	# keep a reference to the first QGlWidget, so all the glwidgets can share context
	shareWidget = None

	def __init__(self, parent=None):
		QtOpenGL.QGLWidget.__init__(self, g_fmt, parent, QGLView.shareWidget)
		if QGLView.shareWidget is None: QGLView.shareWidget = self
		# print "GL Format:\n{}".format(self.format())
		self.width,self.height,self.aspect = 1,1,1
		self.mouseX, self.mouseY = -1,-1
		self.camera = Camera('Perspective')
		self.cameras = [self.camera]
		self.tool = None
		self.marquee = None
		self.drawingHUD = True
		self.setMouseTracking(True) # try to capture mouseover for HUD; TODO this doesn't seem to work
		self.textColor = (1,1,1)
		self.primitives = []
		self.primitives2D = []
		self.lastTimes = [time()]
		self.displayText = []
		self.layers = {}
		self.selectionLayers = []

		self.setFocusPolicy(QtCore.Qt.WheelFocus) # get all events

		#: OR mask of draw options - passed to primitive paintGL calls to control what kinds of
		#: things are drawn
		self.drawOpts = DRAWOPT_DEFAULT

	def numCameras(self):
		return len(self.cameras)

	def cameraIndex(self):
		if self.camera is None: return -1
		return self.cameras.index(self.camera)

	def addCamera(self, camera, switch=False):
		'''add a camera to the view

		:param Camera camera: the :class:`Camera` to add
		:param bool switch: make this new camera current if True'''
		self.cameras.append(camera)
		if switch:
			self.camera = camera

	def setImage(self, imageFilename):
		self.camera.setImage(QtGui.QPixmap(imageFilename).toImage())

	def setImageData(self, data, height, width, chans):
		self.camera.setImageData(data, height, width, chans)

	def refreshImageData(self):
		'''This can be used after setImageData to notify the graphics card that the image contents has changed.'''
		self.camera.refreshImageData(self)

	def setLayer(self, layerName, layer, selection=False):
		self.layers[layerName] = layer
		if selection and layerName not in self.selectionLayers:
			self.selectionLayers.append(layerName)
		self.updateLayers()
		return self.layers[layerName]

	def setLayers(self, layers):
		self.layers = layers
		self.updateLayers()

	def updateLayers(self):
		self.primitives = self.layers.values()

	def getLayers(self, subset=None):
		if subset is None: return self.layers
		return [self.layers[s] for s in subset]

	def getLayer(self, layerName):
		return self.layers.get(layerName,None)

	def hasLayer(self, layerName):
		return layerName in self.layers

	def clearLayers(self, keepGrid=False):
		if keepGrid:
			layerNames = self.layers.keys()
			for name in layerNames:
				if name != 'grid':
					del(self.layers[name])
		else:
			self.layers.clear()

		self.camera = self.cameras[0]
		self.cameras = [self.cameras[0]]

	def clearSelectionLayers(self):
		for layerName in self.selectionLayers:
			if layerName not in self.layers: continue
			del(self.layers[layerName])

		self.selectionLayers = []

	def drawText(self, x, y, s, font=GLUT.GLUT_BITMAP_TIMES_ROMAN_10, color=None):
		if color is None: color = self.textColor
		if bool(GL.glWindowPos2f):
			GL.glColor3f(color[0],color[1],color[2])
			GL.glWindowPos2f(x, self.height-y)
			GLUT.glutBitmapString(font, s)
		else:
			GL.glDisable(GL.GL_TEXTURE_2D)
			GL.glColor3f(color[0],color[1],color[2])
			GL.glMatrixMode(GL.GL_MODELVIEW)
			GL.glPushMatrix()
			GL.glLoadIdentity()
			GL.glMatrixMode(GL.GL_PROJECTION)
			GL.glPushMatrix()
			GL.glLoadIdentity()
			GLU.gluOrtho2D(0.0, self.width, self.height, 0.0) # (0,0) at top-left
			GL.glRasterPos2f(x, y)
			GLUT.glutBitmapString(font, s)
			GL.glMatrixMode(GL.GL_PROJECTION)
			GL.glPopMatrix()
			GL.glMatrixMode(GL.GL_MODELVIEW)
			GL.glPopMatrix()

	def drawHUD(self):
		if not DRAWOPT_HUD & self.drawOpts: return
		self.drawText(10,48,self.camera.tostring())
		fps = float(len(self.lastTimes))/(time()-self.lastTimes[0])
		self.lastTimes.append(time())
		if len(self.lastTimes) > 20: self.lastTimes.pop(0)
		if fps < 1: self.lastTimes = self.lastTimes[-2:]
		self.drawText(10,72,'%.2f fps' % fps)
		if fps == 0: self.lastTimes = [] # improve accuracy after a pause

	def drawMarquee(self, x0, y0, x1, y1):
		GL.glMatrixMode(GL.GL_MODELVIEW)
		GL.glPushMatrix()
		GL.glLoadIdentity()
		GL.glMatrixMode(GL.GL_PROJECTION)
		GL.glPushMatrix()
		GL.glLoadIdentity()
		GLU.gluOrtho2D(0.0, self.width, 0.0, self.height) # (0,0) at bottom-left
		GL.glLineStipple(1, 0x5555)
		GL.glLineWidth(1.0)
		GL.glEnable(GL.GL_LINE_STIPPLE)
		GL.glEnable(GL.GL_COLOR_LOGIC_OP)
		GL.glLogicOp(GL.GL_INVERT)
		GL.glTranslatef(0.375, 0.375, 0.0)
		GL.glColor3f(1,1,1)
		GL.glBegin(GL.GL_LINE_LOOP)
		GL.glVertex2i(x0,y0)
		GL.glVertex2i(x1,y0)
		GL.glVertex2i(x1,y1)
		GL.glVertex2i(x0,y1)
		GL.glEnd()
		GL.glMatrixMode(GL.GL_PROJECTION)
		GL.glPopMatrix()
		GL.glMatrixMode(GL.GL_MODELVIEW)
		GL.glPopMatrix()
		GL.glDisable(GL.GL_LINE_STIPPLE)
		GL.glDisable(GL.GL_COLOR_LOGIC_OP)

	def initializeGL(self):
		GLUT.glutInit()
		try:
			self.bindBID = GL.glGenBuffers(1) # for streaming video
		except:
			print ('GL: no textures (failed glGenBuffers)')
			self.bindBID = 0

	def resizeGL(self, width, height):
		oldWidth = self.width
		self.width,self.height = max(1,int(width)),max(1,int(height))
		self.aspect = self.width / float(self.height)
		# correct the 2d offset, which is measured in pixels
		scale = self.width / float(oldWidth)
		self.camera.cameraOx *= scale
		self.camera.cameraOy *= scale

	def paintLayer(self, layerName, width, height):
		self.width, self.height = width, height
		GL.glViewport(0, 0, self.width, self.height)
		self.qglClearColor(QtGui.QColor(0, 0, 200, 0))
		GL.glHint(GL.GL_POINT_SMOOTH_HINT, GL.GL_NICEST)
		GL.glHint(GL.GL_LINE_SMOOTH_HINT, GL.GL_NICEST)
		GL.glHint(GL.GL_POLYGON_SMOOTH_HINT, GL.GL_NICEST)
		GL.glEnable(GL.GL_POINT_SMOOTH)
		GL.glEnable(GL.GL_LINE_SMOOTH)
		GL.glEnable(GL.GL_BLEND)
		GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

		GL.glMatrixMode(GL.GL_PROJECTION)
		# 2d operations
		GL.glLoadMatrixf(np.array([[self.camera.cameraZoom,0,0,0],[0,self.camera.cameraZoom,0,0],[0,0,1.0,0],[(2.0*self.camera.cameraOx)/self.width,(2.0*self.camera.cameraOy)/self.height,0,1.0]],dtype=np.float32))
		# set the near and far clipping planes and aspect ratio
		# try to control the near and far based on the interest distance
		GL.glMultMatrixf(np.array([[1,0,0,0],[0,self.aspect,0,0],[0,0,-1.00002,-1],[0,0,self.camera.cameraInterest*-0.0200002,0]],dtype=np.float32))

		GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

		GL.glShadeModel(GL.GL_SMOOTH)
		# draw 3D geometry
		# apply the camera matrix
		GL.glMatrixMode(GL.GL_PROJECTION)
		GL.glMultMatrixd(self.camera.P().T)
		GL.glMatrixMode(GL.GL_MODELVIEW)
		p = self.getLayer(layerName)
		if p is None: return
		p.paintGL(drawOpts=self.drawOpts)

	def paintToImageFile(self, cameraIndex, frame, directory, cameraName=None):
		import os
		self.width, self.height = 1280, 720
		GL.glViewport(0, 0, self.width, self.height)
		self.qglClearColor(QtGui.QColor(*[x*255 for x in COLOURS['Background']]))
		GL.glHint(GL.GL_POINT_SMOOTH_HINT, GL.GL_NICEST)
		GL.glHint(GL.GL_LINE_SMOOTH_HINT, GL.GL_NICEST)
		GL.glHint(GL.GL_POLYGON_SMOOTH_HINT, GL.GL_NICEST)
		GL.glEnable(GL.GL_POINT_SMOOTH)
		GL.glEnable(GL.GL_LINE_SMOOTH)
		#GL.glEnable(GL.GL_POLYGON_SMOOTH) # broken on modern nvidia drivers :-(
		GL.glEnable(GL.GL_BLEND)
		GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

		GL.glMatrixMode(GL.GL_PROJECTION)
		# 2d operations
		GL.glLoadMatrixf(np.array([[self.camera.cameraZoom,0,0,0],[0,self.camera.cameraZoom,0,0],[0,0,1.0,0],[(2.0*self.camera.cameraOx)/self.width,(2.0*self.camera.cameraOy)/self.height,0,1.0]],dtype=np.float32))
		# set the near and far clipping planes and aspect ratio
		# try to control the near and far based on the interest distance
		#self.znear,self.zfar = self.camera.cameraInterest*1e-2,self.camera.cameraInterest*1e3
		#dz = 1.0/(self.znear-self.zfar)
		#GL.glMultMatrixd([[1,0,0,0],[0,self.aspect,0,0],[0,0,(self.znear+self.zfar)*dz,-1],[0,0,2*self.znear*self.zfar*dz,0]])
		GL.glMultMatrixf(np.array([[1,0,0,0],[0,self.aspect,0,0],[0,0,-1.00002,-1],[0,0,self.camera.cameraInterest*-0.0200002,0]],dtype=np.float32))

		GL.glMatrixMode(GL.GL_MODELVIEW)
		GL.glLoadIdentity()
		GL.glDisable(GL.GL_TEXTURE_2D)
		GL.glEnable(GL.GL_DEPTH_TEST)
		GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

		GL.glMatrixMode(GL.GL_MODELVIEW)
		GL.glMultMatrixd(self.camera.P().T)
		GL.glMultMatrixd(np.linalg.inv(self.camera.P(False)).T)
		# self.camera.paintCamera(self, drawOpts=self.drawOpts)
		GL.glLoadIdentity()

		GL.glShadeModel(GL.GL_SMOOTH)
		# draw 3D geometry
		# apply the camera matrix
		GL.glMatrixMode(GL.GL_PROJECTION)
		GL.glMultMatrixd(self.camera.P().T)
		GL.glMatrixMode(GL.GL_MODELVIEW)
		# for p in self.primitives:
			# drawOpts must be passed to primitives (rather than the each primitive maintaining
			# it's draw options state) because different views may display the same primitive with
			# different options.
			# p.paintGL(drawOpts=self.drawOpts)

		GL.glLoadIdentity()
		GL.glMatrixMode(GL.GL_PROJECTION)
		GL.glMultMatrixd(np.linalg.inv(self.camera.P(False)).T)

		# ci = 1
		# for p in self.primitives2D:
		if not self.primitives2D: return
		p = self.primitives2D[0]
		if p:
			p.paintGL(cameraIndex, self.camera.cameraInterest, drawOpts=self.drawOpts)

			import cv2
			if not cameraName:
				cameraName = str(cameraIndex)
			filename = os.path.join(directory, 'Camera_' + cameraName + '_Frame_' + str(frame) + '.jpg')
			buffer = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
			cv_image = cv2.cv.CreateImage((self.width, self.height), cv2.cv.IPL_DEPTH_8U, 3)
			cv2.cv.SetData(cv_image, buffer)
			cv2.cv.Flip(cv_image)
			cv2.cv.SaveImage(filename, cv_image)

			# buffer = (GL.GLubyte * (3 * self.width * self.height))(0)
			# GL.glReadPixels(0, 0, self.width, self.height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, array=buffer)
			# from PIL import Image
			# image = Image.fromstring(mode="RGB", size=(self.width, self.height), data=buffer)
			# image = image.transpose(Image.FLIP_TOP_BOTTOM)
			# if not cameraName:
			# 	cameraName = str(cameraIndex)
			# filename = os.path.join(directory, 'Camera_' + cameraName + '_Frame_' + str(frame) + '.jpg')
			# image.save(filename)

	#@profile
	def paintGL(self):
		self.drawSignal.emit(self)
	
		# since gl is a state machine, we can't assume these settings will persist to the next frame
		GL.glViewport(0, 0, self.width, self.height)
		self.qglClearColor(QtGui.QColor(*[x*255 for x in COLOURS['Background']]))
		GL.glHint(GL.GL_POINT_SMOOTH_HINT, GL.GL_NICEST)
		GL.glHint(GL.GL_LINE_SMOOTH_HINT, GL.GL_NICEST)
		GL.glHint(GL.GL_POLYGON_SMOOTH_HINT, GL.GL_NICEST)
		GL.glEnable(GL.GL_POINT_SMOOTH)
		GL.glEnable(GL.GL_LINE_SMOOTH)
		#GL.glEnable(GL.GL_POLYGON_SMOOTH) # broken on modern nvidia drivers :-(
		GL.glEnable(GL.GL_BLEND)
		GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

		GL.glMatrixMode(GL.GL_PROJECTION)
		# 2d operations
		GL.glLoadMatrixf(np.array([[self.camera.cameraZoom,0,0,0],[0,self.camera.cameraZoom,0,0],[0,0,1.0,0],[(2.0*self.camera.cameraOx)/self.width,(2.0*self.camera.cameraOy)/self.height,0,1.0]],dtype=np.float32))
		# set the near and far clipping planes and aspect ratio
		# try to control the near and far based on the interest distance
		#self.znear,self.zfar = self.camera.cameraInterest*1e-2,self.camera.cameraInterest*1e3
		#dz = 1.0/(self.znear-self.zfar)
		#GL.glMultMatrixd([[1,0,0,0],[0,self.aspect,0,0],[0,0,(self.znear+self.zfar)*dz,-1],[0,0,2*self.znear*self.zfar*dz,0]])
		GL.glMultMatrixf(np.array([[1,0,0,0],[0,self.aspect,0,0],[0,0,-1.00002,-1],[0,0,self.camera.cameraInterest*-0.0200002,0]],dtype=np.float32))

		GL.glMatrixMode(GL.GL_MODELVIEW)
		GL.glLoadIdentity()
		GL.glDisable(GL.GL_TEXTURE_2D)
		GL.glEnable(GL.GL_DEPTH_TEST)
		GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

		GL.glMatrixMode(GL.GL_MODELVIEW)
		GL.glMultMatrixd(self.camera.P().T)
		GL.glMultMatrixd(np.linalg.inv(self.camera.P(False)).T)
		self.camera.paintCamera(self, drawOpts=self.drawOpts)
		GL.glLoadIdentity()

		GL.glShadeModel(GL.GL_SMOOTH)
		# draw 3D geometry
		# apply the camera matrix
		GL.glMatrixMode(GL.GL_PROJECTION)
		GL.glMultMatrixd(self.camera.P().T)
		GL.glMatrixMode(GL.GL_MODELVIEW)
		for p in self.primitives:
			# drawOpts must be passed to primitives (rather than the each primitive maintaining
			# it's draw options state) because different views may display the same primitive with
			# different options.
			p.paintGL(drawOpts=self.drawOpts)

		GL.glLoadIdentity()
		GL.glMatrixMode(GL.GL_PROJECTION)
		GL.glMultMatrixd(np.linalg.inv(self.camera.P(False)).T)

		ci = self.cameras.index(self.camera)-1
		for p in self.primitives2D:
			p.paintGL(ci,self.camera.cameraInterest,drawOpts=self.drawOpts)

		# draw the Heads Up Display
		self.drawHUD()

		# draw the user text
		for text in self.displayText:
			if isinstance(text,dict):
				self.drawText(**text)
			else:
				self.drawText(text[0],text[1],text[2],color=text[3] if len(text) > 3 else None)

		# draw the marquee, if set
		if self.marquee is not None: self.drawMarquee(*self.marquee)

	def getDepth(self, x,y):
		#depth = GL.glReadPixels(x, y, 1, 1, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT)[0][0]
		depth = GL.glReadPixels(x, y, 1, 1, GL.GL_DEPTH_COMPONENT, GL.GL_INT)[0][0]
		# for some reason, we are only getting 20 or so reliable bits. this HACK is a workaround to make clicking less annoying
		# accuracy is ~2 parts in a million
		return (depth | 0xfff)/float(0x7fffffff)

	def select(self, x, y):
		self.paintGL()
		root_depth = self.getDepth(x,y)
		debug=False
		if debug: print root_depth
		if root_depth == 1.0: return None
		GL.glMatrixMode(GL.GL_PROJECTION)
		GL.glLoadIdentity()
		#GLU.gluPickMatrix(x,y,3,3,GL.glGetIntegerv(GL.GL_VIEWPORT))
		GL.glMultMatrixf(np.array([[self.camera.cameraZoom,0,0,0],[0,self.camera.cameraZoom,0,0],[0,0,1,0],[(2.0*self.camera.cameraOx)/self.width,(2.0*self.camera.cameraOy)/self.height,0,1]],dtype=np.float32))
		GL.glMultMatrixf(np.array([[1,0,0,0],[0,self.aspect,0,0],[0,0,-1.00002,-1],[0,0,self.camera.cameraInterest*-0.0200002,0]],dtype=np.float32))
		GL.glMultMatrixd(self.camera.P().T)
		GL.glMatrixMode(GL.GL_MODELVIEW)
		GL.glLoadIdentity()
		GL.glDisable(GL.GL_TEXTURE_2D)
		GL.glEnable(GL.GL_DEPTH_TEST)
		GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
		GL.glMultMatrixd(np.linalg.inv(self.camera.P(False)).T)
		ci = self.cameras.index(self.camera)-1
		for pi,p in enumerate(self.primitives2D):
			p.paintGL(ci,self.camera.cameraInterest,p1=p.len(ci),drawOpts=self.drawOpts)
			dist = self.getDepth(x,y)
			if dist == root_depth:
				lo,hi = 0,p.len(ci)
				while lo+1 < hi:
					GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
					mid = int((lo+1+hi)/2)
					p.paintGL(ci, self.camera.cameraInterest, lo, mid, drawOpts=self.drawOpts)
					dist = self.getDepth(x,y)
					if debug: print (lo,mid,hi,dist,hits)
					if dist == root_depth: hi = mid
					else:
						p.paintGL(ci, self.camera.cameraInterest, mid, hi, drawOpts=self.drawOpts)
						dist2 = self.getDepth(x,y)
						if dist2 == root_depth:
							lo = mid
						else:
							if debug: print 'not in list?',lo,hi,dist,dist2,root_depth
							return ('2d',pi,-1,root_depth)
				print ('selected 2d primitive %d index %d' % (pi,lo), self.primitives2D[pi])
				return ('2d',pi,lo,root_depth)
		GL.glMultMatrixd(self.camera.P(False).T)
		for pi,p in enumerate(self.primitives):
			if debug: print (pi,p)
			if isinstance(p, GLGrid): continue
			GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
			p.paintGL(drawOpts=self.drawOpts)
			dist = self.getDepth(x,y)
			if debug: print dist,root_depth,dist==root_depth,dist-root_depth
			if dist == root_depth:
				if debug: print ('looking for',root_depth,p)
				lo,hi = 0,len(p)
				while lo+1 < hi:
					GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
					mid = int((lo+1+hi)/2)
					p.paintGL(lo, mid, drawOpts=self.drawOpts)
					dist = self.getDepth(x,y)
					if debug: print (lo,mid,hi,dist,root_depth)
					if dist == root_depth: hi = mid
					else:
						p.paintGL(mid, hi, drawOpts=self.drawOpts)
						dist2 = self.getDepth(x,y)
						if debug: print 'here',lo,hi,dist,dist2,root_depth
						if dist2 == root_depth:
							lo = mid
						else:
							if debug: print 'not in list?',lo,hi,dist,dist2,root_depth
							return ('3d',pi,-1,root_depth)
				print ('selected 3d primitive %d index %d' % (pi,lo), self.primitives[pi])
				return ('3d',pi,lo,root_depth)
		# print 'selected nothing'
		return (None,-1,0,root_depth)

	def frame(self):
		'''Move the current camera to focus on the indicated 3D point.'''
		x,y = self.mouseX,self.mouseY
		tmp = self.select(x,y)
		if tmp is None: return None
		(sel_type,pi,pn,root_depth) = tmp
		mat1 = np.array([[self.camera.cameraZoom,0,0,0],[0,self.camera.cameraZoom,0,0],[0,0,1,0],[(2.0*self.camera.cameraOx)/self.width,(2.0*self.camera.cameraOy)/self.height,0,1]],dtype=np.float32).T
		mat2 = np.array([[1,0,0,0],[0,self.aspect,0,0],[0,0,-1.00002,-1],[0,0,self.camera.cameraInterest*-0.0200002,0]],dtype=np.float32).T
		mat3 = self.camera.P()
		P = np.dot(mat1,np.dot(mat2,mat3))
		hw,hh = self.width*0.5,self.height*0.5
		x2d = np.array([x/hw-1,y/hh-1,2*root_depth-1,1],dtype=np.float32)
		x3d = np.linalg.solve(P,x2d)
		self.camera.cameraT = x3d[:3] / x3d[3]
		self.updateGL()

	def endTool(self):
		'''This is the chance for a tool to tidy up before it is lost.'''
		if self.tool == '2d marquee':
			if self.marquee:
				x0,x1 = min(self.marquee[0], self.marquee[2]),max(self.marquee[0], self.marquee[2])
				y0,y1 = min(self.marquee[1], self.marquee[3]),max(self.marquee[1], self.marquee[3])
				print (x0,y0,x1,y1)
				if self.marquee[0] == x0 or self.marquee[1] == y1 : # zoom from top-left
					cameraZoomScale = min(self.width/float(x1-x0+1),self.height/float(y1-y0+1))
					offsetScale = 0.5
				else: # unzoom
					cameraZoomScale = max(float(x1-x0+1)/self.width,float(y1-y0+1)/self.height)
					offsetScale = -0.5/cameraZoomScale
				self.camera.cameraZoom *= cameraZoomScale
				# compute the 2d correction
				dx,dy = ((x0+x1) - self.width), ((y0+y1) - self.height)
				self.camera.cameraOx = (self.camera.cameraOx - dx*offsetScale) * cameraZoomScale
				self.camera.cameraOy = (self.camera.cameraOy - dy*offsetScale) * cameraZoomScale
				self.marquee = None
				self.updateGL()
			else:
				# treat this as a selection operation (same as 'object select toggle') - this makes
				# behaviour consistency between the gl view and the outliner. 
				s = self.select(self.mouseX, self.mouseY)
				if s: self.pickSignal.emit(self,s,False)

		self.tool = None

	def setTool(self, tool):
		'''Change the tool. Store initial state as required and precompute as much as possible.'''
		if self.tool == tool: return # if it's the same tool, do nothing
		self.endTool() # finish the old tool
		self.tool = tool
		# keep a record of where the mouse was when the tool was started
		self.toolX,self.toolY = self.mouseX,self.mouseY
		if self.tool == '2d translate':
			self.toolOriginX,self.toolOriginY = self.camera.cameraOx,self.camera.cameraOy
		elif self.tool == '2d zoom':
			self.toolOriginZoom = self.camera.cameraZoom
		elif self.tool == '3d pan/tilt' or self.tool == '3d pan/tilt view':
			# take into account the roll
			if self.camera.lockedUpright:
				self.toolSinRoll,self.toolCosRoll = 0,1
			else:
				self.toolSinRoll,self.toolCosRoll = sin(np.radians(self.camera.cameraRoll)), cos(np.radians(self.camera.cameraRoll))
		elif self.tool == '3d translate' or self.tool == '3d drag vertex':
			# the screen-space X and Y axes are given, up to scale, by the first two rows of the orientation matrix
			R = self.camera.R(True)*(self.camera.cameraInterest*2.0*tan(np.radians(0.5*self.camera.cameraFovX))/float(self.width * self.camera.cameraZoom))
			self.toolRX, self.toolRY = R[0], R[1]

	def zoomAtMouse(self, x, y, amount):
		cameraZoomScale = pow(1.1,amount)
		self.camera.cameraZoom = self.camera.cameraZoom * cameraZoomScale
		# compute the 2d correction to make the zoom operate at (x,y)
		dx,dy = (x - 0.5*self.width), (y - 0.5*self.height)
		self.camera.cameraOx = (self.camera.cameraOx - dx) * cameraZoomScale + dx
		self.camera.cameraOy = (self.camera.cameraOy - dy) * cameraZoomScale + dy

	def mouseMoveEvent(self, event):
		# apply the mouse movement with the current tool
		if self.tool is None: return
		event.accept()
		oldX,oldY = self.mouseX, self.mouseY
		self.mouseX, self.mouseY = event.x(), (self.height-1)-event.y()
		self.mouseDx, self.mouseDy = self.mouseX - oldX, self.mouseY - oldY
		self.toolDx,self.toolDy = self.mouseX - self.toolX,self.mouseY - self.toolY
		if self.tool == '2d translate':
			self.camera.cameraOx,self.camera.cameraOy = self.toolOriginX + self.toolDx,self.toolOriginY + self.toolDy
		elif self.tool == '2d zoom':
			self.zoomAtMouse(self.toolX, self.toolY, self.mouseDx)
		elif self.tool == '2d marquee':
			# [toolX, toolY]:[mouseX, mouseY]
			self.marquee = [self.toolX, self.toolY, self.mouseX, self.mouseY]
		elif self.tool == '3d pan/tilt' or self.tool == '3d pan/tilt view':
			# take into account the roll
			pre = self.camera.RT()
			self.camera.cameraPan += 0.1*(self.mouseDx * self.toolCosRoll + self.mouseDy * self.toolSinRoll)
			self.camera.cameraTilt += 0.1*(-self.mouseDy * self.toolCosRoll + self.mouseDx * self.toolSinRoll)
			self.camera.cameraTilt = np.clip(self.camera.cameraTilt,-90,90)
			if self.tool == '3d pan/tilt view':
				pre = np.dot(pre[:3,:3].T,-pre[:3,3])
				post = self.camera.RT()
				post = np.dot(post[:3,:3].T,-post[:3,3])
				tmp = pre-post
				self.camera.cameraT += tmp
		elif self.tool == 'fov':
			self.camera.cameraFovX += 0.1*self.mouseDx
		elif self.tool == 'distortion':
			if self.camera.cameraDistortion is None: self.camera.cameraDistortion = (0.0,0.0)
			self.camera.cameraDistortion = (self.camera.cameraDistortion[0] + 0.01*self.mouseDx, self.camera.cameraDistortion[1] + 0.01*self.mouseDy)
		elif self.tool == '3d cyclo':
			self.camera.cameraRoll += 0.1*self.mouseDx
		elif self.tool == '3d translate':
			self.camera.cameraT -= (self.mouseDx * self.toolRX + self.mouseDy * self.toolRY)
		elif self.tool == '3d zoom':
			self.camera.cameraInterest *= pow(2.0,self.mouseDx/-100.)
			self.camera.cameraInterest = min(max(1,self.camera.cameraInterest),1e6) # between 1mm and 1km
		else:
			self.dragSignal.emit(self,self.tool)
		self.updateGL()

	def mousePressEvent(self, event):
		event.accept()
		buttons = event.buttons()
		modifiers = event.modifiers()
		self.mouseX, self.mouseY = event.x(), self.height-1-event.y()
		self.mouseLeft = bool(buttons & QtCore.Qt.LeftButton)
		self.mouseMid = bool(buttons & QtCore.Qt.MidButton)
		self.mouseRight = bool(buttons & QtCore.Qt.RightButton)
		self.mouseShift = bool(modifiers & QtCore.Qt.ShiftModifier)
		self.mouseCtrl = bool(modifiers & QtCore.Qt.ControlModifier)
		self.mouseAlt = bool(modifiers & QtCore.Qt.AltModifier)
		if self.mouseAlt:
			# if the Alt key is pressed, start a 3D operation
			if self.mouseMid or (self.mouseShift and self.mouseLeft): self.setTool('3d translate')
			elif self.mouseLeft and self.mouseCtrl: self.setTool(['3d pan/tilt view','3d cyclo'][self.mouseRight]) # Ctrl+Alt
			elif self.mouseLeft: self.setTool(['3d pan/tilt','3d cyclo'][self.mouseRight])
			elif self.mouseShift and self.mouseRight and self.mouseCtrl: self.setTool('distortion')
			elif self.mouseShift and self.mouseRight: self.setTool('fov')
			elif self.mouseRight: self.setTool('3d zoom')
		elif self.mouseCtrl:
			# if the Ctrl key is pressed, start a 2D operation
			if self.mouseMid or (self.mouseShift and self.mouseLeft): self.setTool('2d translate')
			elif self.mouseRight: self.setTool('2d zoom')
			elif self.mouseLeft: self.setTool('2d marquee')
		else:
			# if neither Ctrl or Alt is pressed, this is a selection operation
			# [however, it's nice to have 2d translate and zoom without modifiers...]
			if self.mouseMid: self.setTool('2d translate')
			elif self.mouseRight: self.setTool('2d zoom')
			elif self.mouseLeft:
				if self.mouseShift:
					s = self.select(self.mouseX, self.mouseY)
					if s: self.pickSignal.emit(self,s,False)
				else: 
					s = self.select(self.mouseX, self.mouseY)
					if s is None: self.pickSignal.emit(self,None,True) # this may set the tool in this view
					else: self.pickSignal.emit(self,s,True)

	def mouseReleaseEvent(self, event):
		event.accept()
		self.mouseX, self.mouseY = event.x(), (self.height-1)-event.y()
		buttons = event.buttons()
		# if all buttons are released, finish the tool
		if buttons == 0: 
			self.setTool(None)
			super(QGLView, self).mouseReleaseEvent(event)
		# otherwise, possibly change the tool
		else: self.mousePressEvent(event)

	def wheelEvent(self, event):
		event.accept()
		self.mouseX, self.mouseY = event.x(), (self.height-1)-event.y()
		self.wheelUp = event.delta() > 0
		self.zoomAtMouse(self.mouseX, self.mouseY, [-1,1][self.wheelUp])
		self.updateGL()

	def keyPressEvent(self, event):
		key = event.key()
		self.keySignal.emit(self,key)

def makePrimitives(vertices=None, altVertices=None, skelDict=None, altSkelDict=None, skels=None):
	from UI import GLPoints3D, GLSkel
	primitives = []
	if vertices is not None:
		points = GLPoints3D(vertices)
		primitives.append(points)
	if altVertices is not None:
		altpoints = GLPoints3D(altVertices)
		altpoints.pointSize = 5
		altpoints.colour = (1.0, 1.0, 0, 0.5)
		primitives.append(altpoints)
	if skelDict is not None:
		skel = GLSkel(skelDict['Bs'], skelDict['Gs'], mvs=skelDict['markerOffsets'], mvis=skelDict['markerParents'])
		skel.boneColour = (1, 1, 0, 1)
		primitives.append(skel)
		if skelDict.has_key('name'): skel.setName(skelDict['name'])
	if altSkelDict is not None:
		skel2 = GLSkel(altSkelDict['Bs'], altSkelDict['Gs'], mvs=altSkelDict['markerOffsets'], mvis=altSkelDict['markerParents'])
		skel2.boneColour = (1, 0, 1, 1)
		primitives.append(skel2)
		if altSkelDict.has_key('name'): skel2.setName(altSkelDict['name'])
	if skels is not None:
		primitives.extend(skels)
	return primitives

def makePrimitives2D(p1 = None, p2 = None):
	from UI import GLPoints2D
	primitives2D = []
	if p1 is not None:
		primitives2D.append(GLPoints2D(p1)) # dark dots
	if p2 is not None:
		primitives2D.append(GLPoints2D(p2))
		primitives2D[-1].colour = (1.0,0.5,0,0.5) # light dots - shown in orange
	return primitives2D

def makeApp(appName=None, appIn=None, win=None):
	import sys
	from PySide import QtGui
	global app
	if appIn is None:
		app = QtGui.QApplication.instance()
		if app is None: app = QtGui.QApplication(sys.argv)
	else: app = appIn
	app.setStyle('plastique')
	if win is None:
		win = QApp.app
		if win is None:
			win = QApp.QApp()
			win.setFocusPolicy(QtCore.Qt.StrongFocus) # get keyboard events
	if appName is not None:
		win.setWindowTitle(appName)
	return app,win

def makeViewer (appName='Imaginarium Viewer',mat=None, md=None, grid=True, primitives=None, primitives2D=None, timeRange=None, \
				callback=None, mats=None, camera_ids=None, movies=None, win=None, appIn=None, \
				pickCallback=None, dragCallback=None, keyCallback=None, drawCallback=None, dirtyCallback=None,
				layers={}, runtime=None):
	'''Helper code to knock up a viewer'''
	from PySide import QtCore
	from UI import GLGrid, GLCameras
	app,win = makeApp(appName=appName,appIn=appIn, win=win)
	v = QApp.view()#.view()
	if mat is not None: v.camera.setP(mat[2], distortion=mat[3], store=True)
	if md is not None: v.camera.setImageData(md['vbuffer'],md['vheight'],md['vwidth'],3)
	if primitives is not None: 
		for pi,p in enumerate(primitives):
			win.setLayer(str(pi), p)
		v.primitives.extend(primitives)
	if primitives2D is not None: v.primitives2D.extend(primitives2D)
	primitives,primitives2D = v.primitives,v.primitives2D
	if callback is not None: # add this last to avoid the callback being called before we're ready
		if timeRange is not None: win.qtimeline.setRange(*timeRange)
		win.qtimeline.cb = callback
	if dirtyCallback is not None:
		win.dirtyCB = dirtyCallback
	if mats is not None:
		cams = win.addCameras(mats, camera_ids, movies)
		layers['cameras'] = cams
	if grid:
		grid = GLGrid()
		primitives.append(grid)
		layers['grid'] = grid
	if pickCallback is not None:
		v.pickSignal.connect(pickCallback)
	if dragCallback is not None:
		v.dragSignal.connect(dragCallback)

	if keyCallback is not None:
		v.keySignal.connect(keyCallback)
		
	if drawCallback is not None:
		v.drawSignal.connect(drawCallback)

	if layers is not None:
		for layerName, layer in layers.iteritems(): win.setLayer(layerName, layer)

	if runtime is not None and timeRange:
		runtime.resetFrame(timeRange[0])

	win.showMaximized()
	app.connect(app, QtCore.SIGNAL('lastWindowClosed()'), app.quit)
	app.exec_()


if __name__ == "__main__":
	import sys
	app = QtGui.QApplication([])
	widget = QGLView()
	widget.setMinimumSize(640, 480)
	widget.show()
	sys.exit(app.exec_())
