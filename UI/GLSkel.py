#!/usr/bin/env python

from OpenGL import GL, GLU, GLUT
from OpenGL.arrays import vbo
from OpenGL.GL import shaders
from PySide import QtCore, QtGui, QtOpenGL
from GCore import *
import numpy as np
from UI import *
#from UI import (COLOURS, DRAWOPT_DEFAULT, DRAWOPT_BONES, DRAWOPT_JOINTS, DRAWOPT_OFFSET,DRAWOPT_AXES, DRAWOPT_MARKERS, DRAWOPT_POINTS, DRAWOPT_DETECTIONS, DRAWOPT_GEOMS, DRAWOPT_LABELS)

class GLSkel(object):
	def __init__(self, bones, Gs, vs=None, vts=None, vns=None, tris=None, transformData=None, mvs=None, mvis=None, bone_colour=COLOURS['Bone']):
		super(GLSkel, self).__init__()
		self.numBones = len(bones)
		self.numGs = len(Gs)
		assert(self.numBones == self.numGs)
		self.offset = [0, 0, 0]
		self.nameWidth = None
		self.bone_colour = bone_colour
		self.d = {K_NAME: None,
				K_COLOUR: (0, 0.8, 1, 1),
				K_SELECTED: False,
				K_VISIBLE: True,
				K_DRAW: True}
		self.d.update({K_BONE_COLOUR: COLOURS['Bone'],
						K_MARKER_COLOUR: COLOURS['Marker']})
		self.vs,self.vts,self.vns,self.tris,self.mvs = None,None,None,None,None
		if type(bones[0]) is list:
			bvs = []
			bvis = []
			for bi,bl in enumerate(bones):
				for b in bl:
					bvs.extend([np.zeros(3,dtype=np.float32),b])
					bvis.extend([bi,bi])
			self.bvs = vbo.VBO(np.array(bvs,dtype=np.float32).reshape(-1,3), usage='GL_STATIC_DRAW_ARB')
			self.bvis = vbo.VBO(np.array(bvis,dtype=np.uint32), usage='GL_STATIC_DRAW_ARB')
			self.numBones = len(bvs)/2
		else:
			boneVerts = np.zeros((self.numBones,2,3),dtype=np.float32)
			boneVerts[:,1,:] = bones
			self.bvs = vbo.VBO(np.array(boneVerts,dtype=np.float32).reshape(-1,3), usage='GL_STATIC_DRAW_ARB')
			self.bvis = vbo.VBO(np.array(range(256),dtype=np.uint32)/2, usage='GL_STATIC_DRAW_ARB') # [0,0,1,1,...]
		if vs is not None: self.vs = vbo.VBO(np.array(vs,dtype=np.float32), usage='GL_STATIC_DRAW_ARB')
		if vts is not None: self.vts = vbo.VBO(np.array(vts,dtype=np.float32), usage='GL_STATIC_DRAW_ARB')
		if vns is not None: self.vns = vbo.VBO(np.array(vns,dtype=np.float32), usage='GL_STATIC_DRAW_ARB')
		if tris is not None: self.tris = np.array(tris,dtype=np.int32).reshape(-1,3)
		self.numMarkers = 0
		if mvs is not None and mvis is not None:
			self.setMarkerData(mvs, mvis)
		self.avs = vbo.VBO(np.array([[0,0,0],[50,0,0],[0,50,0],[0,0,50]]*self.numGs,dtype=np.float32).reshape(-1,3), usage='GL_STATIC_DRAW_ARB')
		self.avis = vbo.VBO(np.array(range(self.numGs*4),dtype=np.uint32)/4, usage='GL_STATIC_DRAW_ARB')
		self.aris = vbo.VBO(np.array([[4*i,4*i+1] for i in range(self.numGs)],dtype=np.uint32), target=GL.GL_ELEMENT_ARRAY_BUFFER, usage='GL_STATIC_DRAW_ARB')
		self.agis = vbo.VBO(np.array([[4*i,4*i+2] for i in range(self.numGs)],dtype=np.uint32), target=GL.GL_ELEMENT_ARRAY_BUFFER, usage='GL_STATIC_DRAW_ARB')
		self.abis = vbo.VBO(np.array([[4*i,4*i+3] for i in range(self.numGs)],dtype=np.uint32), target=GL.GL_ELEMENT_ARRAY_BUFFER, usage='GL_STATIC_DRAW_ARB')
		self.transformData = transformData
		#self.transformData = [(ti,vbo.VBO(np.array(tis,dtype=np.uint32), target=GL.GL_ELEMENT_ARRAY_BUFFER, usage='GL_STATIC_DRAW_ARB')) for (ti,tis) in transformData]
		if transformData is not None:
			self.tis, self.vtis = [],np.zeros(len(self.vs),dtype=np.uint32)
			for ti,tis in transformData:
				self.tis.extend(tis)
				self.vtis[tis] = ti
			self.tis = vbo.VBO(np.array(self.tis,dtype=np.uint32), target=GL.GL_ELEMENT_ARRAY_BUFFER, usage='GL_STATIC_DRAW_ARB')
			self.vtis = vbo.VBO(np.array(self.vtis,dtype=np.uint32), usage='GL_STATIC_DRAW_ARB')
		self.bone_colour_states = np.zeros(self.numBones, dtype=np.bool)
		self.transforms = None
		self.setPose(Gs)
		self.image,self.bindImage,self.bindId = None,None,long(0)
		self.GL_is_initialised = False

	def setMarkerData(self, mvs=None, mvis=None):
		self.numMarkers = len(mvs)
		self.mvs = vbo.VBO(np.array(mvs,dtype=np.float32).reshape(-1,3), usage='GL_STATIC_DRAW_ARB')
		self.mvis = vbo.VBO(np.array(mvis,dtype=np.uint32), usage='GL_STATIC_DRAW_ARB')

	def setBoneColour(self, colour):
		self.d[K_BONE_COLOUR] = colour

	def setVisible(self, visible):
		self.d[K_VISIBLE] = visible

	def initializeGL(self):
		VERTEX_SHADER = shaders.compileShader('''
		#version 130
		uniform mat4 myMat[128]; // we support up to 128 bones
		in int bi;
		varying vec3 N;
		void main() {
			gl_TexCoord[0] = gl_TextureMatrix[0] * gl_MultiTexCoord0;
			gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * myMat[bi] * gl_Vertex;
			N = normalize(gl_NormalMatrix * mat3(myMat[bi]) * gl_Normal);
		}''', GL.GL_VERTEX_SHADER)
		FRAGMENT_SHADER = shaders.compileShader('''
		varying vec3 N;
		//uniform sampler2D tex;
		void main() {
			vec3 lightDir = normalize(gl_LightSource[0].position.xyz);
			float NdotL = max(dot(N, lightDir), 0.0);
			//vec4 colour = texture2D(tex, gl_TexCoord[0].st);
			vec4 diffuse = gl_FrontMaterial.diffuse * gl_LightSource[0].diffuse;
			gl_FragColor = vec4(NdotL,NdotL,NdotL,1.0) * diffuse; // *colour
		}''', GL.GL_FRAGMENT_SHADER)
		COLOUR_VERTEX_SHADER = shaders.compileShader('''
		#version 130
		uniform mat4 myMat[128]; // we support up to 128 bones
		uniform bool states[128]; 

		//uniform vec4 coloursb[2]; // I can't make this work
		in vec4 colour;
		in vec4 alt_colour;
		in int bi;

		vec4 colours[2] = vec4[](colour, alt_colour); // i want to pass in coloursb array instead of doing this
		varying vec4 vertex_color;
		void main() {
			gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * myMat[bi] * gl_Vertex;
			vertex_color = colours[int (states[bi])];
		}''', GL.GL_VERTEX_SHADER)
		COLOUR_FRAGMENT_SHADER = shaders.compileShader('''
		varying vec4 vertex_color;
		void main() {
			gl_FragColor = vertex_color;
		}''', GL.GL_FRAGMENT_SHADER)
		self.colour_shader = shaders.compileProgram(COLOUR_VERTEX_SHADER,COLOUR_FRAGMENT_SHADER)
		self.colour_shader_bi = GL.glGetAttribLocation(self.colour_shader, 'bi')
		self.colour_shader_myMat = GL.glGetUniformLocation(self.colour_shader, 'myMat')
		self.colour_shader_colour = GL.glGetAttribLocation(self.colour_shader, 'colour')
		self.colour_shader_alt_colour = GL.glGetAttribLocation(self.colour_shader, 'alt_colour')
		#self.colour_shader_colours = GL.glGetUniformLocation(self.colour_shader, 'coloursb') # not working
		self.colour_shader_states = GL.glGetUniformLocation(self.colour_shader, 'states')
		#self.colour_shader_vertex_color = GL.glGetAttribLocation(self.colour_shader, 'vertex_color')
		self.shader = shaders.compileProgram(VERTEX_SHADER,FRAGMENT_SHADER)
		self.shader_bi = GL.glGetAttribLocation(self.shader, 'bi')
		self.shader_myMat = GL.glGetUniformLocation(self.shader, 'myMat')
		self.font = GLUT.GLUT_BITMAP_HELVETICA_18
		#self.avs = vbo.VBO(np.array([[0,0,0],[50,0,0],[0,0,0],[0,50,0],[0,0,0],[0,0,50]],dtype=np.float32), usage='GL_STATIC_DRAW_ARB')
		#self.avis = vbo.VBO(np.array([0,1,2,3,4,5],dtype=np.int32), target=GL.GL_ELEMENT_ARRAY_BUFFER, usage='GL_STATIC_DRAW_ARB')
		#self.avcs = vbo.VBO(np.array([[1,0,0],[1,0,0],[0,1,0],[0,1,0],[0,0,1],[0,0,1]],dtype=np.float32), usage='GL_STATIC_DRAW_ARB')
		self.GL_is_initialised = True

	def setName(self, name):
		self.d[K_NAME] = name
		self.nameWidth = None
		
	def setImage(self, view, imageFilename):
		self.view = view
		self.image = QtGui.QPixmap(imageFilename).toImage()
		self.imageFlipped = False

	def setPose(self, Gs):
		assert(Gs.shape[0] == self.numGs)
		if self.transforms is None or self.transforms.shape[0] != self.numGs:
			self.transforms = np.zeros((self.numGs,4,4),dtype=np.float32)
			self.transforms[:,3,3] = 1
		self.transforms[:,:,:3] = np.transpose(Gs,axes=[0,2,1])

	def __len__(self): return self.numBones + self.numMarkers + 1

	def hilight(self, pi):
		if pi is None:
			self.bone_colour_states.fill(False)
			return
		self.bone_colour_states[pi]=True

	def paintGL(self, p0=0, p1=None, drawOpts=DRAWOPT_DEFAULT):
		'''
		:param drawOpts: OR combination of draw flags. default is :data:`UI.DRAWOPT_DEFAULT`
		'''
		doingSelection = (p1 is not None)
		if doingSelection: return # TODO, selection not working because of using shaders
		if not self.d['draw'] or not self.d['visible']: return
		if p1 is None: p1 = len(self)
		if not self.GL_is_initialised: self.initializeGL()

		# could store a selected/non-selected colour, switch on selection change and avoid this if.. 
		boneColour = COLOURS['Selected'] if self.d[K_SELECTED] else self.d[K_BONE_COLOUR]

		# TODO: draw offset should be an option of THIS primitive, not a view based draw option?
		drawingOffset = DRAWOPT_OFFSET & drawOpts
		if drawingOffset: GL.glTranslate(self.offset[0], self.offset[1], self.offset[2])
		GL.glEnable(GL.GL_BLEND)
		first_sel = self.numBones
		if self.transforms is not None and p0 < first_sel:
			GL.glUseProgram(self.colour_shader)
			b0,b1 = min(max(0,p0),first_sel), min(max(0,p1),first_sel)
			#GL.glUniform4fv(self.colour_shader_colours, 3, self.colours) # NOT WORKING
			GL.glUniform1iv(self.colour_shader_states,len(self.bone_colour_states), self.bone_colour_states) # set the states so the shader can pick the correct colour per joint
			GL.glVertexAttrib4f(self.colour_shader_colour, *boneColour)
			GL.glVertexAttrib4f(self.colour_shader_alt_colour, *COLOURS['Hilighted'])
			GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
			self.bvs.bind()
			GL.glVertexPointerf(self.bvs)
			GL.glEnableVertexAttribArray(self.colour_shader_bi)
			self.bvis.bind()
			GL.glVertexAttribIPointer(self.colour_shader_bi, 1, GL.GL_UNSIGNED_INT, 0, self.bvis) # write the bvis to bi
			if (DRAWOPT_BONES|DRAWOPT_JOINTS) & drawOpts:
				GL.glLineWidth(1)
				GL.glPointSize(5)
				for t0 in xrange(b0,b1,128): # draw the bones in batches of 128
					t1 = min(t0+128,b1)
					GL.glUniformMatrix4fv(self.colour_shader_myMat, t1-t0, GL.GL_FALSE, self.transforms[t0:t1]) # put the transforms in myMat
					if DRAWOPT_BONES & drawOpts: GL.glDrawArrays(GL.GL_LINES, 2*t0, 2*(t1-t0))
					if DRAWOPT_JOINTS & drawOpts: GL.glDrawArrays(GL.GL_POINTS, 2*t0, 2*(t1-t0))
			self.bvis.unbind()
			GL.glDisableVertexAttribArray(self.colour_shader_bi)
			self.bvs.unbind()
			GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
		if self.mvs and p1 > first_sel and DRAWOPT_MARKERS & drawOpts:
			assert self.numBones < 128, 'Only up to 128 bones are supported for now'
			GL.glUseProgram(self.colour_shader)
			m0,m1 = min(max(0,p0-first_sel),self.numMarkers), min(max(0,p1-first_sel),self.numMarkers)
			GL.glUniformMatrix4fv(self.colour_shader_myMat, len(self.transforms), GL.GL_FALSE, self.transforms) # put the transforms in myMat
			GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
			self.mvs.bind()
			GL.glVertexPointerf(self.mvs)
			GL.glEnableVertexAttribArray(self.colour_shader_bi)
			self.mvis.bind()
			GL.glVertexAttribIPointer(self.colour_shader_bi, 1, GL.GL_UNSIGNED_INT, 0, self.mvis) # write the mvis to bi
			GL.glPointSize(5)
			GL.glVertexAttrib4f(self.colour_shader_colour, *self.d[K_MARKER_COLOUR]) # write the colour
			GL.glDrawArrays(GL.GL_POINTS, m0, m1-m0)
			self.mvis.unbind()
			GL.glDisableVertexAttribArray(self.colour_shader_bi)
			self.mvs.unbind()
			GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
		if p1 == len(self):
			if DRAWOPT_AXES & drawOpts:
				GL.glUseProgram(self.colour_shader)
				GL.glUniformMatrix4fv(self.shader_myMat, len(self.transforms), GL.GL_FALSE, self.transforms) # put the transforms in myMat
				GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
				self.avs.bind()
				GL.glVertexPointerf(self.avs)
				GL.glEnableVertexAttribArray(self.colour_shader_bi)
				self.avis.bind()
				GL.glVertexAttribIPointer(self.colour_shader_bi, 1, GL.GL_UNSIGNED_INT, 0, self.avis) # write the avis to bi
				GL.glLineWidth(2)
				GL.glVertexAttrib4f(self.colour_shader_colour, 1,0,0,1) # red
				GL.glVertexAttrib4f(self.colour_shader_alt_colour, 1,0,0,1) # this doesn't seem clever. need to make the glUniform4fv work then make one call to set the 2 colours
				self.aris.bind()
				GL.glDrawElementsui(GL.GL_LINES, self.aris) # draw the lines
				self.aris.unbind()
				GL.glVertexAttrib4f(self.colour_shader_colour, 0,1,0,1) # green
				GL.glVertexAttrib4f(self.colour_shader_alt_colour, 0,1,0,1)
				self.agis.bind()
				GL.glDrawElementsui(GL.GL_LINES, self.agis) # draw the lines
				self.agis.unbind()
				GL.glVertexAttrib4f(self.colour_shader_colour, 0,0,1,1) # blue
				GL.glVertexAttrib4f(self.colour_shader_alt_colour, 0,0,1,1)
				self.abis.bind()
				GL.glDrawElementsui(GL.GL_LINES, self.abis) # draw the lines
				self.abis.unbind()
				self.avis.unbind()
				GL.glDisableVertexAttribArray(self.colour_shader_bi)
				self.avs.unbind()
				GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
			GL.glUseProgram(self.shader)
			if self.image != self.bindImage:
				if self.bindImage is not None:
					self.deleteTexture(self.bindId)
					self.bindId,self.bindImage = long(0),None
				if self.image is not None:
					global win
					self.bindId = self.view.bindTexture(self.image)
					self.bindImage = self.image
			if self.bindImage is not None:
				GL.glEnable(GL.GL_TEXTURE_2D)
				GL.glBindTexture(GL.GL_TEXTURE_2D, self.bindId)
			GL.glEnable(GL.GL_CULL_FACE)
			GL.glCullFace(GL.GL_BACK)
			GL.glFrontFace(GL.GL_CCW)
			GL.glEnable(GL.GL_LIGHTING)
			GL.glEnable(GL.GL_LIGHT0)
			Pmat = GL.glGetDoublev(GL.GL_PROJECTION_MATRIX)
			lightDir = -Pmat[:3,2] # the direction the camera is looking
			GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, lightDir)
			if self.d[K_COLOUR]: GL.glMaterialfv(GL.GL_FRONT, GL.GL_AMBIENT_AND_DIFFUSE, self.d[K_COLOUR])
			if self.vs is not None:
				GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
				self.vs.bind()
				GL.glVertexPointerf(self.vs)
			if self.vts is not None:
				GL.glEnableClientState(GL.GL_TEXTURE_COORD_ARRAY)
				self.vts.bind()
				GL.glTexCoordPointerf(self.vts)
			if self.vns is not None:
				GL.glEnableClientState(GL.GL_NORMAL_ARRAY)
				self.vns.bind()
				GL.glNormalPointerf(self.vns)
			if self.tris is not None: GL.glDrawElementsui(GL.GL_TRIANGLES, self.tris) # static geom
			if self.transformData is not None and self.transforms is not None:
				GL.glUniformMatrix4fv(self.shader_myMat, len(self.transforms), GL.GL_FALSE, self.transforms) # put the transforms in myMat
				GL.glEnableVertexAttribArray(self.shader_bi)
				self.vtis.bind()
				GL.glVertexAttribIPointer(self.shader_bi, 1, GL.GL_UNSIGNED_INT, 0, self.vtis) # write the vtis to bi
				if DRAWOPT_GEOMS & drawOpts:
					self.tis.bind()
					GL.glDrawElementsui(GL.GL_TRIANGLES, self.tis) # draw the triangles
					self.tis.unbind()
				self.vtis.unbind()
				GL.glDisableVertexAttribArray(self.shader_bi)
			if self.vs is not None:
				self.vs.unbind()
				GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
			if self.vts is not None:
				self.vts.unbind()
				GL.glDisableClientState(GL.GL_TEXTURE_COORD_ARRAY)
			if self.vns is not None:
				self.vns.unbind()
				GL.glDisableClientState(GL.GL_NORMAL_ARRAY)
			GL.glDisable(GL.GL_LIGHTING)
			GL.glDisable(GL.GL_TEXTURE_2D)
		GL.glDisable(GL.GL_BLEND)
		GL.glUseProgram( 0 )
		if self.d[K_NAME] is not None and self.transforms is not None and DRAWOPT_LABELS & drawOpts:
			try:
				GL.glColor4f(*boneColour)
				phi = np.max(self.transforms[:,3,:3],axis=0)
				plo = np.min(self.transforms[:,3,:3],axis=0)
				if bool(GL.glWindowPos2f):
					Mmat = GL.glGetDoublev(GL.GL_MODELVIEW_MATRIX)
					Pmat = GL.glGetDoublev(GL.GL_PROJECTION_MATRIX)
					viewport = GL.glGetIntegerv(GL.GL_VIEWPORT)
					p = GLU.gluProject((phi[0]+plo[0])*0.5,phi[1] + 300.0,(phi[2]+plo[2])*0.5, Mmat, Pmat, viewport)
					if p[2] > 0 and p[2] < 1: # near/far clipping of text
						# TODO, now this won't change if name changes...
						if self.nameWidth is None: self.nameWidth = sum([GLUT.glutBitmapWidth(self.font, ord(x)) for x in self.d[K_NAME]])
						GL.glWindowPos2f(p[0] - 0.5*self.nameWidth,p[1])
						GLUT.glutBitmapString(self.font, self.d[K_NAME])
				else:
					GL.glRasterPos3f((phi[0]+plo[0])*0.5,phi[1] + 300.0,(phi[2]+plo[2])*0.5)
					GLUT.glutBitmapString(self.font, self.d[K_NAME])
			except ValueError: pass # projection failed
		if DRAWOPT_OFFSET & drawOpts: GL.glTranslate(-self.offset[0],-self.offset[1],-self.offset[2])

	#: TODO re-implement!
	'''
	def message(self, s):
		if s == 'toggleGeomAlpha':
			self.colour[3] = [1.0,0.8][self.colour[3] == 1.0]
	'''
if __name__ == '__main__':
	pass