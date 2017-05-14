#!/usr/bin/env python

from OpenGL import GL
from OpenGL.arrays import vbo
from OpenGL.GL import shaders
import numpy as np

from UI import (COLOURS, DRAWOPT_ALL, DRAWOPT_BONES, DRAWOPT_JOINTS, DRAWOPT_OFFSET,
					DRAWOPT_AXES, DRAWOPT_POINTS, DRAWOPT_GEOMS, DRAWOPT_LABELS, DRAWOPT_DETECTIONS)


class GLTool:
	'''GLTool is a primitive class that draws a tool in the GLView.
	Examples of tools are: 3D selection, 3D translation, 3D rotation, 3D scale
	The tool has a global transform
	The tool may have a selected component (for example, a single axis)
	The tool may have a 'screen space' component
	'''
	def __init__(self):
		self.setTool(None)
		self.toolType = None
		self.GL_is_initialised = False
		self.selectedColour = [1,1,0]

	def setComponent(self, component):
		self.component = component

	def setTool(self, toolType, transform = np.eye(3,4,dtype=np.float32), component = None):
		self.toolType = toolType
		self.gl_transform = np.eye(4,4,dtype=np.float32)
		self.gl_transform[:4,:3] = transform.T
		self.component = component
		if toolType is None:
			self.numComponents = 0
			return
		if toolType == '3D translation':
			self.numComponents = 3
			self.vs = vbo.VBO(np.array([[0,0,0],[100,0,0],[90,10,0],[90,-10,0],[0,100,0],[-10,90,0],[10,90,0],[0,0,100],[-10,0,90],[10,0,90]],dtype=np.float32), usage='GL_STATIC_DRAW_ARB')
			self.edges = vbo.VBO(np.array([0,1,1,2,1,3,0,4,4,5,4,6,0,7,7,8,7,9],dtype=np.int32), target=GL.GL_ELEMENT_ARRAY_BUFFER, usage='GL_STATIC_DRAW_ARB')
			self.colours = np.array([[1,0,0],[0,1,0],[0,0,1]],dtype=np.float32)
			self.esplits = np.array([0,3,6,9],dtype=np.int32)
		if toolType == '3D rotation':
			self.numComponents = 3
			self.ringSegments = 48
			t = np.arange(self.ringSegments)*(2 * np.pi / self.ringSegments)
			pointsX = np.array([t*0,np.sin(t),np.cos(t)],dtype=np.float32).T
			pointsY = np.array([np.sin(t),t*0,np.cos(t)],dtype=np.float32).T
			pointsZ = np.array([np.sin(t),np.cos(t),t*0],dtype=np.float32).T
			edges = np.arange(2*self.ringSegments+1,dtype=np.int32)[1:]/2
			edges[-1] = 0
			edges = edges.reshape(-1,2)
			print edges.shape, pointsX.shape
			self.vs = vbo.VBO(np.vstack((pointsX,pointsY,pointsZ))*100., usage='GL_STATIC_DRAW_ARB')
			self.edges = vbo.VBO(np.array([edges,edges+self.ringSegments,edges+2*self.ringSegments],dtype=np.int32), target=GL.GL_ELEMENT_ARRAY_BUFFER, usage='GL_STATIC_DRAW_ARB')
			self.colours = np.array([[1,0,0],[0,1,0],[0,0,1]],dtype=np.float32)
			self.esplits = np.array([0,self.ringSegments,2*self.ringSegments,3*self.ringSegments],dtype=np.int32)

	def initializeGL(self):
		self.GL_is_initialised = True

	def __len__(self): return self.numComponents

	def paintGL(self, p0=0, p1=None, drawOpts=DRAWOPT_ALL):
		'''
		:param drawOpts: OR combination of draw flags. default is :data:`UI.DRAWOPT_ALL`
		'''
		doingSelection = (p1 is not None)
		if p1 is None: p1 = len(self)
		if not self.GL_is_initialised: self.initializeGL()
		if p1 == 0: return # don't render if no vertices
		GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
		self.vs.bind()
		GL.glVertexPointerf(self.vs)
		self.edges.bind()
		GL.glLineWidth(2)
		GL.glPushMatrix()
		GL.glMultMatrixf(self.gl_transform)
		Pmat = GL.glGetDoublev(GL.GL_PROJECTION_MATRIX).T
		Mmat = GL.glGetDoublev(GL.GL_MODELVIEW_MATRIX).T
		scale = 0.0006 * np.dot(Pmat[3],np.dot(Mmat,self.gl_transform[:,3])) # make size independent of the distance TODO focal length
		GL.glScale(scale,scale,scale)
		GL.glDisable(GL.GL_DEPTH_TEST)
		GL.glEnable(GL.GL_BLEND)
		for i in xrange(p0,p1):
			e0,e1 = self.esplits[i],self.esplits[i+1]
			if e1 > e0:
				GL.glColor3fv(self.selectedColour if i == self.component else self.colours[i])
				GL.glDrawElements(GL.GL_LINES, (e1-e0)*2, GL.GL_UNSIGNED_INT, self.edges + e0*2*4)
		GL.glEnable(GL.GL_DEPTH_TEST)
		GL.glDisable(GL.GL_BLEND)
		GL.glPopMatrix()
		self.edges.unbind()
		self.vs.unbind()
		GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
	
if __name__ == '__main__':
	def setFrameCB(frame):
		pass
	def pickedCB(view,data,clearSelection=True):
		print view
		print data
		print clearSelection
		if data is None:
			QApp.app.select(None)
		else:
			primitive_type,pn,pi,distance = data
			if primitive_type is '3d':
				p = view.primitives[pn]
				if isinstance(p,GLMeshes):
					name = p.names[pi]
					print "Picked:", name
					QApp.app.select(name)
				elif isinstance(p,GLTool):
					p.setComponent(pi)
					QApp.app.updateGL()
	from UI import GLMeshes # TODO move this colin
	from UI import QApp
	from GCore import State
	from PySide import QtGui
	import sys
	appIn = QtGui.QApplication(sys.argv)
	appIn.setStyle('plastique')
	win = QApp.QApp()
	win.setWindowTitle('Imaginarium Maya File Browser')
	from UI import QGLViewer
	primitives = []
	#QApp.fields = fields
	#outliner = QApp.app.qobjects
	#for i,(o,v) in enumerate(dobjs):
	#	outliner.insertItem(i,o)
	#	outliner.item(i).setData(1,'_OBJ_'+v['name'])
	vs = np.array([[[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],[1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]]],dtype=np.float32)*100.0
	faces = np.array([[[0,1,3,2],[4,6,7,5],[2,3,7,6],[1,0,4,5],[3,1,5,7],[0,2,6,4]]],dtype=np.int32)
	transforms = [np.array([[1,0,0,0],[0,1,0,1000],[0,0,1,0]],dtype=np.float32)]
	p = GLMeshes(names=['my mesh'], verts=vs, faces=faces, transforms=transforms)
	q = GLTool()
	q.setTool('3D rotation', transform = transforms[0], component=None)
	r = GLTool()
	r.setTool('3D translation', transform = transforms[0], component=None)
	primitives = [p,q,r]
	QGLViewer.makeViewer(primitives=primitives, timeRange = (1,100), callback=setFrameCB, pickCallback=pickedCB, appIn=appIn, win=win)
