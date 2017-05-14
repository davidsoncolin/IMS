#!/usr/bin/env python

from OpenGL import GL, GLU, GLUT
import numpy as np
import math
from UI import DRAWOPT_DEFAULT, DRAWOPT_CAMERAS, DRAWOPT_LABELS, DRAWOPT_AXES

class GLCameras:
	def __init__(self, names, mats):  #, cameras):
		self.setTransforms(names, mats)
		#self.cameras = cameras
		#: camera label colour
		self.colour = (0.2, 0.0, 0.2, 1)
		self.pointSize = 10
		self.font = GLUT.GLUT_BITMAP_HELVETICA_10
		self.nameWidth = None
		self.colours = None
		self.transform = None # world transform...
		#self.cameraVtxArray = np.array([[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,1]],dtype=np.float32)*100
		#self.cameraColourArray = np.array([[1,0,0],[1,0,0],[0,1,0],[0,1,0],[0,0,1]],dtype=np.float32)
		#self.cameraIdxArray = np.array([0,1,2,1,0,3,0,2,4,2,1,4,1,3,4,3,0,4],dtype=np.int32)
		cameraDrawScale = 60
		ch = 1.75
		cpts = np.array([[-1, -ch, 0], [1, -ch, 0], [1, 1, 0], [-1, 1, 0], [-1, -ch, .7], [1, -ch, .7], [1, 1, .7], [-1, 1, .7]], dtype=np.float32)
		#self.cameraPointArray = cpts * cameraDrawScale

		# TODO: put all this in one cameraVtxArray
		self.ringSegments = 12

		scale = 2 * np.pi / self.ringSegments
		t = np.arange(self.ringSegments+1).reshape(-1,1)*scale
		points = np.hstack((np.sin(t),np.cos(t),t*0))
		self.rings = np.vstack(((0.8 * cameraDrawScale) * points,(0.3 * cameraDrawScale) * points, (0.3 * cameraDrawScale) * points + np.array([0,0,-0.5])*cameraDrawScale))

		# frontface, backface, sides
		self.cameraVtxArray = np.array([cpts[0], cpts[1], cpts[1], cpts[2], cpts[2], cpts[3], cpts[3], cpts[0], cpts[4], cpts[5], cpts[5], cpts[6], cpts[6], cpts[7], cpts[7], cpts[4],
										cpts[0], cpts[4], cpts[1], cpts[5], cpts[2], cpts[6], cpts[3], cpts[7] ], dtype=np.float32) * cameraDrawScale

		# axes (should combine this for all cameras like GLSkels)
		self.avs = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]], dtype=np.float32) * cameraDrawScale
		self.acs = np.array([[1,0,0],[1,0,0],[0,1,0],[0,1,0],[0,0,1],[0,0,1]],dtype=np.float32)
		#self.cameraColourArray = np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 1], [0, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=np.float32)

	def setTransforms(self, names, mats):
		self.names, self.mats = names, mats
		self.names = [str(n) for n in self.names]
		numCameras = len(mats)
		self.transforms = np.zeros((numCameras,4,4),dtype=np.float32)
		self.transforms[:,3,3] = 1
		for pi,mat in enumerate(mats):
			#mat = [K,RT,np.dot(K,RT),np.array([k1,k2],dtype=np.float32),d['POSITION'],d['SENSOR_SIZE']
			RT = mat[1]
			position = mat[4]
			self.transforms[pi,:3,:3] = RT[:3,:3].T
			self.transforms[pi,:3,3] = position

	def setTransform(self, transform):
		self.transform = transform

	def __len__(self): return len(self.mats)

	def paintGL(self, p0=0, p1=None, drawOpts=DRAWOPT_DEFAULT):
		'''
		:param drawOpts: OR combination of draw flags. default is :data:`UI.DRAWOPT_DEFAULT`
		'''
		if not DRAWOPT_CAMERAS & drawOpts: return
		if p1 is None: p1 = len(self)
		if self.colour[3] != 1:
			GL.glDisable(GL.GL_DEPTH_TEST)
		# looks much better
		GL.glEnable(GL.GL_BLEND)
		GL.glColor4f(*self.colour)
		if p1 > p0:
			GL.glPointSize(5)
			if self.transform is not None:
				GL.glPushMatrix()
				GL.glMultMatrixf(self.transform.T)
			for pi in xrange(p0,p1):
				GL.glPushMatrix()
				GL.glMultMatrixf(self.transforms[pi].T)
				if self.colours is not None and pi < len(self.colours):
					GL.glColor4f(*self.colours[pi])

				self.drawCamera(drawOpts)

				'''
				GL.glMultMatrixd(np.linalg.inv(self.cameras[pi].K().T))
				x = y = z = self.cameras[pi].cameraInterest
				xn = yn = zn = self.cameras[pi].cameraInterest * 0.200002
				cpts = np.array([[-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z], [-xn, -yn, -zn], [xn, -yn, -zn], [xn, yn, -zn], [-xn, yn, -zn]], dtype=np.float32)
				self.cameraFrustrumArray = np.array([cpts[0], cpts[1], cpts[1], cpts[2], cpts[2], cpts[3], cpts[3], cpts[0], cpts[4], cpts[5], cpts[5], cpts[6], cpts[6], cpts[7], cpts[7], cpts[4],
												cpts[0], cpts[4], cpts[1], cpts[5], cpts[2], cpts[6], cpts[3], cpts[7] ], dtype=np.float32) * .2

				self.cameraFrustrumIdxArray = np.arange(0, len(self.cameraFrustrumArray))
				self.drawFrustrum()
				'''
				GL.glPopMatrix()
			if self.transform is not None:
				GL.glPopMatrix()
		# camera labels
		if self.names is None or not (DRAWOPT_LABELS & drawOpts): return
		if self.nameWidth is None:
			self.nameWidth = [sum([GLUT.glutBitmapWidth(self.font, ord(x)) for x in name]) for name in self.names]
		Mmat = GL.glGetDoublev(GL.GL_MODELVIEW_MATRIX)
		Pmat = GL.glGetDoublev(GL.GL_PROJECTION_MATRIX)
		viewport = GL.glGetIntegerv(GL.GL_VIEWPORT)
		for name, mat, w in zip(self.names, self.mats, self.nameWidth)[p0:p1]:
			try:
				v = mat[4]
				if bool(GL.glWindowPos2f):
					p = GLU.gluProject(v[0],v[1]+10,v[2], Mmat, Pmat, viewport)
					if p[2] > 1 or p[2] < 0: continue # near/far clipping of text
					GL.glWindowPos2f(p[0] - 0.5*w,p[1])
				else:
					GL.glRasterPos3f(v[0],v[1]+10,v[2])
				GLUT.glutBitmapString(self.font, name)
			except ValueError: pass # projection failed

	def drawFrustrum(self):
		GL.glLineWidth(1)
		GL.glDisable(GL.GL_TEXTURE_2D)
		GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
		GL.glVertexPointerf(self.cameraFrustrumArray)
		GL.glDrawElementsui(GL.GL_LINES, self.cameraFrustrumIdxArray)
		GL.glDisableClientState(GL.GL_VERTEX_ARRAY)

	def drawCamera(self, drawOpts):
		# GL.glDisable(GL.GL_TEXTURE_2D)
		GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
		# GL.glEnableClientState(GL.GL_COLOR_ARRAY)
		#GL.glVertexPointerf(self.cameraPointArray)
		#GL.glDrawArrays(GL.GL_POINTS, 0, len(self.cameraPointArray))
		GL.glLineWidth(1)
		GL.glVertexPointerf(self.cameraVtxArray)
		GL.glDrawArrays(GL.GL_LINES, 0, len(self.cameraVtxArray))
		GL.glVertexPointerf(self.rings)
		GL.glDrawArrays(GL.GL_LINE_LOOP, 0, len(self.rings))
		if DRAWOPT_AXES & drawOpts:
			GL.glEnableClientState(GL.GL_COLOR_ARRAY)
			GL.glColorPointerf(self.acs)
			GL.glLineWidth(2)
			GL.glVertexPointerf(self.avs)
			#GL.glLineStipple(1, 0xAAAA)
			#GL.glEnable(GL.GL_LINE_STIPPLE)
			GL.glDrawArrays(GL.GL_LINES, 0, len(self.avs))
			GL.glDisableClientState(GL.GL_COLOR_ARRAY)
			GL.glLineWidth(1)
		GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
		# GL.glEnable(GL.GL_CULL_FACE)
		# GL.glCullFace(GL.GL_BACK)
		# GL.glFrontFace(GL.GL_CCW)
		# GL.glVertexPointerf(self.cameraPointArray)
		# GL.glColorPointerf(self.cameraColourArray)
		# GL.glDrawElementsui(GL.GL_TRIANGLES, self.cameraIdxArray)
		# GL.glDisableClientState(GL.GL_COLOR_ARRAY)


if __name__ == '__main__':
	import sys,os
	directory = os.environ['GRIP_DATA']
	xcp_filename = os.path.join(directory,sys.argv[1])
	from IO import ViconReader
	mats,xcp_data = ViconReader.loadXCP(xcp_filename)

	from UI import QAppCtx
	from UI.QGLViewer import Camera
	from UI.QApp import QApp

	with QAppCtx():
		glWidget = QApp()
		n = len(mats)
		camera_ids = range(len(mats[:n]))
		cameras = []
		for mat, cid in zip(mats, camera_ids)[:n]:
			camera = Camera('Cam_%d' % cid)
			glWidget.view().addCamera(camera)
			cameras.append(camera)
			camera.setP(mat[2], distortion=mat[3])
			camera.setResetData()
		cams = GLCameras(['cam_%d' % x for x in camera_ids], mats[:n])  #, cameras[:n])
		glWidget.view().primitives.append(cams)

		glWidget.resize(640, 480)
		glWidget.show()
	#pointsViewer(np.array([[100,100,100],[200,200,200]],dtype=np.float32), mats=mats)
