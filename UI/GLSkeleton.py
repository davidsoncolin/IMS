#!/usr/bin/env python

from OpenGL import GL, GLU, GLUT
import numpy as np

from UI import DRAWOPT_DEFAULT, DRAWOPT_LABELS


class GLSkeleton:
	def __init__(self, names, parents, vertices):
		self.names, self.parents, self.vertices = names, parents, np.array(vertices,dtype=np.float32,copy=True).reshape(-1,3)
		self.idx = np.array([[bi,pi] for bi,pi in enumerate(self.parents) if pi != -1],dtype=np.int32).reshape(-1)
		self.graph = None
		self.name  = None
		self.drawNames = False
		self.nameWidths = None
		self.color = (1,1,1)
		self.font = GLUT.GLUT_BITMAP_HELVETICA_18

	def setName(self, name):
		self.name = name
		self.nameWidth = None

	def paintGL(self, p0=0, p1=None, drawOpts=DRAWOPT_DEFAULT):
		'''
		:param drawOpts: OR combination of draw flags. default is :data:`UI.DRAWOPT_DEFAULT`
		'''
		is_click = (p1 is not None)
		if p1 is None: p1 = len(self)
		assert(p0 >= 0 and p1 >= p0 and p1 <= len(self.vertices))
		if p1 <= p0: return # draw nothing

		GL.glDisable(GL.GL_TEXTURE_2D)
		GL.glShadeModel(GL.GL_SMOOTH)
		GL.glColor3f(*self.color)
		GL.glLineWidth(1)
		GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
		GL.glVertexPointerf(self.vertices)
		GL.glDrawElementsui(GL.GL_LINES, self.idx)
		if self.graph is not None:
			GL.glColor3f(0,1,0)
			GL.glLineWidth(5)
			GL.glVertexPointerf(self.vertices)
			GL.glDrawElementsui(GL.GL_LINES, self.graph)
		GL.glColor3f(1,0,0)
		GL.glPointSize(10)
		GL.glVertexPointerf(self.vertices)
		GL.glDrawArrays(GL.GL_POINTS, 0, len(self.vertices))
		if self.name and  DRAWOPT_LABELS & drawOpts:
			GL.glColor3f(*self.color)
			phi = np.max(self.vertices,axis=0)
			plo = np.min(self.vertices,axis=0)
			if bool(GL.glWindowPos2f):
				Mmat = GL.glGetDoublev(GL.GL_MODELVIEW_MATRIX)
				Pmat = GL.glGetDoublev(GL.GL_PROJECTION_MATRIX)
				viewport = GL.glGetIntegerv(GL.GL_VIEWPORT)
				p = GLU.gluProject((phi[0]+plo[0])*0.5,phi[1] + 300.0,(phi[2]+plo[2])*0.5, Mmat, Pmat, viewport)
				if self.nameWidth is None: self.nameWidth = sum([GLUT.glutBitmapWidth(self.font, ord(x)) for x in self.name])
				GL.glWindowPos2f(p[0] - 0.5*self.nameWidth,p[1])
			else:
				GL.glRasterPos3f((phi[0]+plo[0])*0.5,phi[1] + 300.0,(phi[2]+plo[2])*0.5)
			GLUT.glutBitmapString(self.font, self.name)
		if self.names and DRAWOPT_LABELS & drawOpts:
			if self.nameWidths is None:
				self.nameWidths = [sum([GLUT.glutBitmapWidth(self.font, ord(x)) for x in name]) for name in self.names]
			GL.glColor3f(*self.color)
			Mmat = GL.glGetDoublev(GL.GL_MODELVIEW_MATRIX)
			Pmat = GL.glGetDoublev(GL.GL_PROJECTION_MATRIX)
			viewport = GL.glGetIntegerv(GL.GL_VIEWPORT)
			for ni,(name,v,w) in enumerate(zip(self.names, self.vertices, self.nameWidths)):
				if bool(GL.glWindowPos2f):
					p = GLU.gluProject(v[0],v[1]+10+ni*0.1,v[2], Mmat, Pmat, viewport)
					GL.glWindowPos2f(p[0] - 0.5*w,p[1])
				else:
					GL.glRasterPos3f(v,(phi[2]+plo[2])*0.5)
				GLUT.glutBitmapString(self.font, name)
		GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
