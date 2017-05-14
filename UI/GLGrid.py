#!/usr/bin/env python

from OpenGL import GL
from OpenGL import GL, GLU, GLUT
import numpy as np

from UI import DRAWOPT_GRID, COLOURS


class GLGrid:
	def __init__(self, gridShape = (5,4), gridScale = 2000.0):
		self.gridShape,self.gridScale = gridShape,gridScale
		self.gridSizeX,self.gridSizeY = self.gridShape
		self.vertices = np.array([[(x-0.5*self.gridSizeX)*self.gridScale, 0, 0.5*y*self.gridScale] for x in range(self.gridSizeX+1) for y in [-self.gridSizeY,self.gridSizeY]] + [[0.5*x*self.gridScale, 0, (y-0.5*self.gridSizeY)*self.gridScale] for y in range(self.gridSizeY+1) for x in [-self.gridSizeX,self.gridSizeX]],dtype=np.float32)
		self.idx = range((self.gridSizeX+self.gridSizeY+2)*2)
		self.floorIdx = np.array([0,1,self.gridSizeX*2+1,0,self.gridSizeX*2+1,self.gridSizeX*2],dtype=np.uint32)
		self.floorColor = COLOURS['Floor']
		self.gridColour = COLOURS['Grid'][:3]

	def paintGL(self, drawOpts=DRAWOPT_GRID):
		'''
		:param drawOpts: OR combination of draw flags. default is :data:`UI.DRAWOPT_GRID`
		'''
		if not DRAWOPT_GRID & drawOpts: return

		GL.glEnable(GL.GL_BLEND)
		GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
		GL.glVertexPointerf(self.vertices)
		GL.glColor4f(*self.floorColor)
		GL.glEnable(GL.GL_CULL_FACE)
		GL.glCullFace(GL.GL_BACK)
		GL.glFrontFace(GL.GL_CCW)
		GL.glTranslate(0, -0.001 * self.gridScale, 0)
		GL.glDrawElementsui(GL.GL_TRIANGLES, self.floorIdx)
		GL.glTranslate(0, 0.001 * self.gridScale, 0)
		GL.glDisable(GL.GL_CULL_FACE)
		GL.glColor3f(*self.gridColour)
		GL.glLineWidth(0.5)
		GL.glDrawElementsui(GL.GL_LINES, self.idx)

