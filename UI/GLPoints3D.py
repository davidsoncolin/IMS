#!/usr/bin/env python

from OpenGL import GL, GLU, GLUT
import numpy as np
import sys

from UI import DRAWOPT_DEFAULT, DRAWOPT_POINTS, DRAWOPT_EDGES, DRAWOPT_POINT_LABELS


class GLPoints3D:
	def __init__(self, vertices, edges=None, names=None, colour=(0.1,0.8,0.1,1.0), edgeColour=(1.0,1.0,0.0,1.0), drawStyles=None, pointSize=10):
		self.vertices = np.float32(vertices).reshape(-1,3)
		self.drawStyles = drawStyles
		self.names = names
		self.colour = colour
		self.pointSize = pointSize
		self.font = GLUT.GLUT_BITMAP_HELVETICA_10
		self.nameWidth = None
		self.colours = None # can specify colour per-vertex here
		self.normals = None
		self.edges = edges
		self.edgeColour = edgeColour
		self.visible = True
		self.normalTransparency = False
		self.transform = np.eye(4,4,dtype=np.float32)
		self.selectedIndex = -1

	def __len__(self): return len(self.vertices)

	def setVs(self, vertices, drawStyles=None):
		self.vertices = np.array(vertices,dtype=np.float32,copy=True).reshape(-1,3)
		self.drawStyles = drawStyles

	def setPose(self, pose):
		self.transform[:,:] = pose
		
	def setData(self, vertices, names=None, colours=None, normals=None, drawStyles=None):
		self.vertices = np.array(vertices,dtype=np.float32,copy=True).reshape(-1,3)
		self.names, self.colours, self.normals = names, colours, normals
		self.drawStyles = drawStyles

	def drawPoints(self, which, offset=0):
		if len(which):
			if self.colours is not None and len(self.colours) == len(self.vertices):
				GL.glEnableClientState(GL.GL_COLOR_ARRAY)
				GL.glColorPointerf(self.colours)
			GL.glPointSize(self.pointSize)
			GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
			GL.glVertexPointerf(self.vertices)
			#assert np.all(which+offset<len(self.vertices))
			GL.glDrawElementsui(GL.GL_POINTS, which+offset)
			GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
			GL.glDisableClientState(GL.GL_COLOR_ARRAY)

	def drawCrosses(self, which, offset=0):
		numVs = len(which)
		if numVs:
			vs  = np.zeros((numVs*4,3),dtype=np.float32)
			ed = np.zeros((numVs,4),dtype=np.int32)
			verts = self.vertices[which+offset]
			ms = 2
			vs[0*numVs:1*numVs] = verts + [-ms,0,0]
			vs[1*numVs:2*numVs] = verts + [ms,0,0]
			vs[2*numVs:3*numVs] = verts + [0,-ms,0]
			vs[3*numVs:4*numVs] = verts + [0,ms,0]
			ed = np.arange(4*numVs,dtype=np.int32).reshape(4,numVs).T.copy()
			GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
			GL.glVertexPointerf(vs)
			GL.glLineWidth(1.0)
			GL.glDrawElementsui(GL.GL_LINES, ed)
			GL.glDisableClientState(GL.GL_VERTEX_ARRAY)

	def paintGL(self, p0=0, p1=None, drawOpts=DRAWOPT_DEFAULT):
		'''
		:param drawOpts: OR combination of draw flags. default is :data:`UI.DRAWOPT_DEFAULT`
		'''
		if not self.visible: return
		if self.vertices is None or len(self.vertices) == 0: return
		#print self.vertices.shape
		is_click = (p1 is not None)
		if p1 is None: p1 = len(self)
		assert(p0 >= 0 and p1 >= p0 and p1 <= len(self.vertices))
		if p1 <= p0: return # draw nothing
		GL.glDisable(GL.GL_LIGHTING)
		GL.glDisable(GL.GL_TEXTURE_2D)
		GL.glEnable(GL.GL_BLEND)
		GL.glShadeModel(GL.GL_FLAT)

		if self.normals is not None and len(self.normals) and self.normalTransparency:
			nColours = []
			Mmat = GL.glGetDoublev(GL.GL_MODELVIEW_MATRIX)
			Pmat = GL.glGetDoublev(GL.GL_PROJECTION_MATRIX)
			viewport = GL.glGetIntegerv(GL.GL_VIEWPORT)
			cam = GLU.gluUnProject((viewport[2] - viewport[0]) / 2, (viewport[3] - viewport[1]) / 2, 0.0, Mmat, Pmat, viewport)
			cam = np.array(cam, dtype=np.float32)
			diffs = self.vertices - cam
			# diffs *= -1
			for di, diff in enumerate(diffs):
				nd = diff / np.linalg.norm(diff)
				dp = np.dot(nd, self.normals[di])
				alpha = (dp - (-1)) / 2 * (1 - 0.15) + 0.15
				if self.colours:
					nColours.append([self.colours[di][0], self.colours[di][1], self.colours[di][2], alpha])
				else:
					nColours.append([self.colour[0], self.colour[1], self.colour[2], alpha])
			self.colours = np.array(nColours, dtype=np.float32).reshape(-1,4)


		GL.glPushMatrix()
		try:
			GL.glMultMatrixf(self.transform)
			drawPoints = DRAWOPT_POINTS & drawOpts
			if drawPoints:
				# actually draw the points
				if self.drawStyles is None:
					GL.glColor4f(*self.colour)
					self.drawPoints(np.arange(p0,p1,dtype=np.int32))
				else:
					ds = self.drawStyles[p0:p1]
					which_empty = np.where(ds == 0)[0]
					which_pts = np.where(ds == 1)[0]
					which_crosses = np.where(ds == 2)[0]
					GL.glColor4f(*self.colour)
					self.drawPoints(which_empty, p0)
					self.drawPoints(which_pts, p0)
					self.drawCrosses(which_crosses, p0)
				GL.glDisableClientState(GL.GL_COLOR_ARRAY)

			#print self.selectedIndex
			if self.selectedIndex >= p0 and self.selectedIndex < p1:
				GL.glDisable(GL.GL_DEPTH_TEST)
				GL.glColor4f(1,0,0,1)
				if self.drawStyles is None or self.drawStyles[self.selectedIndex] != 2:
					self.drawPoints(np.arange(self.selectedIndex,self.selectedIndex+1,dtype=np.int32))
				else:
					self.drawCrosses(np.arange(self.selectedIndex,self.selectedIndex+1,dtype=np.int32))
				GL.glEnable(GL.GL_DEPTH_TEST)
		
			drawEdges = DRAWOPT_EDGES & drawOpts
			if self.edges is not None and drawEdges and not is_click:
				GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
				GL.glVertexPointerf(self.vertices)
				GL.glColor4f(self.edgeColour[0], self.edgeColour[1], self.edgeColour[2], self.edgeColour[3])
				GL.glLineWidth(1.0)
				GL.glDrawElementsui(GL.GL_LINES, self.edges)
				GL.glDisableClientState(GL.GL_VERTEX_ARRAY)

			GL.glDisable(GL.GL_BLEND)
			GL.glEnable(GL.GL_DEPTH_TEST)

			drawLabels = DRAWOPT_POINT_LABELS & drawOpts
			if self.names is not None and not is_click and drawLabels:
				self.nameWidth = [sum([GLUT.glutBitmapWidth(self.font, ord(x)) for x in name]) for name in self.names]
				GL.glColor4f(*self.colour)

				Mmat = GL.glGetDoublev(GL.GL_MODELVIEW_MATRIX)
				Pmat = GL.glGetDoublev(GL.GL_PROJECTION_MATRIX)
				viewport = GL.glGetIntegerv(GL.GL_VIEWPORT)

				for name,v,w in zip(self.names, self.vertices, self.nameWidth)[p0:p1]:
					if bool(GL.glWindowPos2f):
						p = GLU.gluProject(v[0],v[1]+5,v[2], Mmat, Pmat, viewport)
						GL.glWindowPos2f(p[0] - 0.5*w,p[1])
					else:
						GL.glRasterPos3f(v[0],v[1]+5,v[2])
					GLUT.glutBitmapString(self.font, name)
		except Exception, e:
			print 'ERROR',e,'at line',sys.exc_info()[2].tb_lineno
		finally:
			GL.glPopMatrix()
