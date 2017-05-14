#!/usr/bin/env python

from OpenGL import GL, GLU, GLUT
import numpy as np

from UI import DRAWOPT_DETECTIONS, DRAWOPT_LABELS


class GLPoints2D:
	def __init__(self, verts_bounds, names = None):
		(vertices, bounds) = verts_bounds
		self.vertices = np.array(vertices,dtype=np.float32,copy=True).reshape(-1,2)
		self.bounds = np.array(bounds,dtype=np.int32)
		self.names = names
		self.colour = (0,1,0,0.5)
		self.fontColour = (1, 1, 1, 0.7)
		self.pointSize = 10
		self.font = GLUT.GLUT_BITMAP_HELVETICA_10
		self.colours = np.array([], dtype=np.float32)
		self.visible = True

	def setData(self, vertices, bounds, names=None):
		self.vertices, self.bounds, self.names = vertices, bounds, names
	
	def len(self,ci): 
		if ci < 0 or ci+1 >= len(self.bounds): return 0
		return (self.bounds[ci+1]-self.bounds[ci])

	def paintGL(self, ci, cameraInterest, p0=0, p1=None, drawOpts=DRAWOPT_DETECTIONS):
		'''
		:param drawOpts: OR combination of draw flags. default is :data:`UI.DRAWOPT_DETECTIONS`
		'''
		if ci < 0 or ci+1 >= len(self.bounds): return
		if not DRAWOPT_DETECTIONS & drawOpts or not self.visible: return
		if p1 is None or p1 > self.len(ci): p1 = self.len(ci)
	
		x,s = self.vertices,self.bounds
		x2ds = x[s[ci]:s[ci+1]][p0:p1]
		plot = np.zeros((len(x2ds),3),dtype=np.float32)
		plot[:,:2] = x2ds
		plot[:,2] = -1.0
		plot *= cameraInterest

		GL.glDisable(GL.GL_DEPTH_TEST)
		GL.glPointSize(self.pointSize)
		GL.glEnable(GL.GL_BLEND)

		if self.colours.any():
			GL.glEnableClientState(GL.GL_COLOR_ARRAY)
			GL.glColorPointerf(self.colours[s[ci]:s[ci+1]][p0:p1])
		else:
			GL.glColor4f(*self.colour)

		GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
		GL.glVertexPointerf(plot)
		GL.glDrawArrays(GL.GL_POINTS, 0, len(plot))
		GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
		GL.glEnable(GL.GL_DEPTH_TEST)
		GL.glDisable(GL.GL_BLEND)

		GL.glDisableClientState(GL.GL_COLOR_ARRAY)

		drawLabels = DRAWOPT_LABELS & drawOpts
		if self.names is not None and drawLabels:
			nameWidth = [sum([GLUT.glutBitmapWidth(self.font, ord(x)) for x in name]) for name in self.names]
			GL.glColor4f(*self.fontColour)
			Mmat = GL.glGetDoublev(GL.GL_MODELVIEW_MATRIX)
			Pmat = GL.glGetDoublev(GL.GL_PROJECTION_MATRIX)
			viewport = GL.glGetIntegerv(GL.GL_VIEWPORT)
			for name,v,w in zip(self.names, self.vertices, nameWidth)[s[ci]:s[ci+1]][p0:p1]:
				if bool(GL.glWindowPos2f):
					p = GLU.gluProject(v[0], v[1], -1.0, Mmat, Pmat, viewport)
					GL.glWindowPos2f(p[0] - 0.5 * w, p[1])
				else:
					GL.glRasterPos3f(v[0],v[1]+10,v[2])
				GLUT.glutBitmapString(self.font, name)

