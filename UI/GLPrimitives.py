#!/usr/bin/env python

from OpenGL import GL, GLU, GLUT
import numpy as np
import sys

from UI import DRAWOPT_DEFAULT, DRAWOPT_POINT_LABELS


class GLPrimitive:
	def __init__(self, attrs=None):
		self.setAttrs(attrs)
		self.visible = True
		self.selectedIndex = -1
		self.font = GLUT.GLUT_BITMAP_HELVETICA_10

	def __len__(self):
		return 1

	def setAttrs(self, attrs):
		self.attrs = attrs
		self.primitiveType = attrs.get('primitiveType', 'sphere')
		self.transform = attrs.get('xform', np.eye(4, 4, dtype=np.float32))
		self.transform = self.transform.T
		self.colour = self.attrs.get('colour', (0.8, 0.8, 0.8, 1.0))

	def paintGL(self, p0=0, p1=None, drawOpts=DRAWOPT_DEFAULT):
		if not self.visible: return
		is_click = (p1 is not None)
		if p1 is None: p1 = len(self)
		if p1 <= p0: return  # draw nothing
		GL.glShadeModel(GL.GL_SMOOTH)

		GL.glPushMatrix()
		GL.glMultMatrixf(self.transform)
		GL.glColor4f(*self.colour)

		GL.glDisable(GL.GL_BLEND)
		GL.glEnable(GL.GL_DEPTH_TEST)

		primType = self.primitiveType.lower()
		if primType == 'sphere':
			radius = self.attrs.get('radius', 100.)
			slices = self.attrs.get('slices', 10)
			stacks = self.attrs.get('stacks', 10)
			if self.attrs.get('wire',None):
				GLUT.glutWireSphere(radius, slices, stacks)
			else:
				GLUT.glutSolidSphere(radius, slices, stacks)

		elif primType == 'cube':
			size = self.attrs.get('size', 100.)
			if self.attrs.get('wire',None):
				GLUT.glutWireCube(size)
			else:
				GLUT.glutSolidCube(size)

		GL.glPopMatrix()
