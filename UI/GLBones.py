#!/usr/bin/env python

from OpenGL import GL
import numpy as np

from UI import DRAWOPT_BONES


class GLBones:
	def __init__(self, vertices, edges):
		self.vertices, self.edges = np.array(vertices,dtype=np.float32,copy=True).reshape(-1,3), np.array(edges,dtype=np.int32,copy=True).reshape(-1,2)
		self.color = (1,1,1)

	def __len__(self): return int(self.edges.shape[0])

	def paintGL(self, p0=0, p1=None, drawOpts=DRAWOPT_BONES):
		'''
		:param drawOpts: OR combination of draw flags. default is :data:`UI.DRAWOPT_BONES`
		'''
		if p1 is None: p1 = len(self)
		if p0 == p1: return
		if drawOpts & DRAWOPT_BONES:
			GL.glDisable(GL.GL_TEXTURE_2D)
			GL.glShadeModel(GL.GL_SMOOTH)
			GL.glColor3f(*self.color)
			GL.glLineWidth(1)
			GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
			GL.glVertexPointerf(self.vertices)
			GL.glDrawElementsui(GL.GL_LINES, self.edges[p0:p1])
			GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
