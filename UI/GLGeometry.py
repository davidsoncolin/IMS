#!/usr/bin/env python

from OpenGL import GL
from OpenGL.arrays import vbo
from OpenGL.GL import shaders
from PySide import QtGui
import numpy as np
import pyopencl as cl

from UI import (COLOURS, DRAWOPT_ALL, DRAWOPT_BONES, DRAWOPT_JOINTS, DRAWOPT_OFFSET,
					DRAWOPT_AXES, DRAWOPT_POINTS, DRAWOPT_GEOMS, DRAWOPT_LABELS, DRAWOPT_DETECTIONS)

global CL_ctx, CL_queue
CL_ctx = None
CL_queue = None

class GLGeometry:
	def __init__(self, vs = None, vts = None, vns = None, tris = None, transformData = None, drawStyle = 'smooth',colour = [0.9,0.9,0.9,1.0]):
		self.vs,self.vts,self.vns,self.tris = None,None,None,None
		if vs is not None: self.vs = vbo.VBO(np.array(vs,dtype=np.float32), usage='GL_STATIC_DRAW_ARB')
		if vts is not None: self.vts = vbo.VBO(np.array(vts,dtype=np.float32), usage='GL_STATIC_DRAW_ARB')
		if vns is not None: self.vns = vbo.VBO(np.array(vns,dtype=np.float32), usage='GL_STATIC_DRAW_ARB')
		if tris is not None: self.tris = vbo.VBO(np.array(tris,dtype=np.int32).reshape(-1,3), target=GL.GL_ELEMENT_ARRAY_BUFFER, usage='GL_STATIC_DRAW_ARB')
		self.transformData = transformData
		if self.transformData is not None:
			self.vtis = vbo.VBO(np.array([t[0] for t in self.transformData for x in t[1]],dtype=np.uint32), usage='GL_STATIC_DRAW_ARB')
			self.tis = vbo.VBO(np.array([x for t in self.transformData for x in t[1]],dtype=np.uint32), target=GL.GL_ELEMENT_ARRAY_BUFFER,
			usage='GL_STATIC_DRAW_ARB')
		self.drawStyle = drawStyle # 'wire','smooth','wire_over_smooth'
		self.colour = colour
		self.transforms = None
		self.image,self.bindImage,self.bindId = None,None,long(0)
		self.GL_is_initialised = False
		global CL_ctx, CL_queue
		if CL_ctx is None:
			CL_ctx = cl.create_some_context(False)
			CL_queue = cl.CommandQueue(CL_ctx)
		self.cl_prg = cl.Program(CL_ctx, '''
		__kernel void compute_normals(__global const float *xs_g, __global const int *edgeList_g, __global float *res_g) {
			const int gid = get_global_id(0);
			const int g10 = gid*10;
			const int g3 = gid*3;
			float sx=0,sy=0,sz=0;
			const float x=xs_g[g3],y=xs_g[g3+1],z=xs_g[g3+2];
			int e3 = edgeList_g[g10]*3;
			float ex0 = xs_g[e3]-x, ey0 = xs_g[e3+1]-y, ez0 = xs_g[e3+2]-z;
			for (int i = 1; i < 10; ++i) {
				e3 = edgeList_g[g10+i]*3;
				float ex1 = xs_g[e3]-x, ey1 = xs_g[e3+1]-y, ez1 = xs_g[e3+2]-z;
				sx += ey0*ez1-ey1*ez0;
				sy += ez0*ex1-ez1*ex0;
				sz += ex0*ey1-ex1*ey0;
				ex0=ex1; ey0=ey1; ez0=ez1;
			}
			const float sc = rsqrt(sx*sx+sy*sy+sz*sz+1e-8);
			res_g[g3] = sx*sc;
			res_g[g3+1] = sy*sc;
			res_g[g3+2] = sz*sc;
		}
		''').build()
		self.edgeList = self.trianglesToEdgeList(np.max(tris)+1 if len(tris) else 1, tris)
		self.edgeList_g = cl.Buffer(CL_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.edgeList)
		if vns is None:
			vns = self.computeNormalsFromEdgeList(vs)
			self.vns = vbo.VBO(np.array(vns,dtype=np.float32), usage='GL_STATIC_DRAW_ARB')


	def trianglesToEdgeList(self,numVerts,triangles):
		T = [set() for t in xrange(numVerts)]
		for t0,t1,t2 in triangles:
			T[t0].add((t1,t2))
			T[t1].add((t2,t0))
			T[t2].add((t0,t1))
		S = []
		for vi,es in enumerate(T):
			l = list(es)
			if len(l):
				L = list(l.pop(0))
				for x in xrange(len(l)):
					for pi,p in enumerate(l):
						if p[0] == L[-1]:
							l.pop(pi)
							L.append(p[1])
							break
				l = L
			if l == []: l = [vi]
			if len(l) < 10: l += [vi]*10
			if len(l) > 10: l = l[:10]
			S.append(np.array(l,dtype=np.int32))
		return np.array(S,dtype=np.int32) # Nx10 matrix

	def computeNormalsFromEdgeList(self, xs):
		if not len(xs): return np.zeros_like(xs)
		global CL_ctx, CL_queue
		xs = xs.astype(np.float32)
		xs_g = cl.Buffer(CL_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=xs)
		res_g = cl.Buffer(CL_ctx, cl.mem_flags.WRITE_ONLY, xs.nbytes)
		self.cl_prg.compute_normals(CL_queue, (xs.shape[0],), None, xs_g, self.edgeList_g, res_g)
		res = np.zeros_like(xs)
		cl.enqueue_copy(CL_queue, res, res_g)
		return res
		#numVerts = len(xs)
		#ns = np.zeros((numVerts,3),dtype=np.float32)
		#X = xs[self.edgeList] # Nx10x3
		#X -= xs.reshape(-1,1,3)
		#ns = np.sum(np.cross(X[:,:9,:],X[:,1:,:]),axis=1)
		#ns = ns/(np.sum(ns**2,axis=-1).reshape(-1,1)**0.5 + 1e-8)
		#return ns


	def initializeGL(self):
		# this shader transforms each bone individually
		VERTEX_SHADER = shaders.compileShader('''
		#version 130
		uniform mat4 myMat[128]; // we support up to 128 bones
		in int bi;
		varying vec3 N;
		void main() {
			gl_TexCoord[0] = gl_TextureMatrix[0] * gl_MultiTexCoord0;
			gl_Position = gl_ProjectionMatrix * (gl_ModelViewMatrix * (myMat[bi] * gl_Vertex));
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
		self.shader = shaders.compileProgram(VERTEX_SHADER,FRAGMENT_SHADER)
		self.shader_bi = GL.glGetAttribLocation(self.shader, 'bi')
		self.shader_myMat = GL.glGetUniformLocation(self.shader, 'myMat')

		# this shader generates a face normal
		VERTEX_SHADER2 = shaders.compileShader('''
		#version 130
		varying vec4 v_x;
		varying vec3 N;
		void main() {
			gl_TexCoord[0] = gl_TextureMatrix[0] * gl_MultiTexCoord0;
			v_x = gl_ModelViewMatrix * gl_Vertex;
			gl_Position = gl_ProjectionMatrix * v_x;
			N = normalize(gl_NormalMatrix * gl_Normal);
		}''', GL.GL_VERTEX_SHADER)
		FRAGMENT_SHADER2 = shaders.compileShader('''
		#version 130
		varying vec4 v_x;
		varying vec3 N;
		uniform sampler2D tex;
		void main() {
			//vec3 N = normalize(cross(dFdx(v_x).xyz, dFdy(v_x).xyz));
			vec3 lightDir = normalize(gl_LightSource[0].position.xyz);
			float NdotL = max(0.0,dot(normalize(N), lightDir));
			vec4 colour = texture2D(tex, gl_TexCoord[0].st);
			if (colour.xyz == vec3(0,0,0)) colour = vec4(1,1,1,1);
			vec4 diffuse = gl_FrontMaterial.diffuse * gl_LightSource[0].diffuse;
			vec4 ambient = vec4(0.05,0.05,0.05,0);
			gl_FragColor = vec4(NdotL,NdotL,NdotL,1.0) * diffuse * colour + ambient; // ambient
		}''', GL.GL_FRAGMENT_SHADER)
		self.shader2 = shaders.compileProgram(VERTEX_SHADER2,FRAGMENT_SHADER2)

		#self.font = GLUT.GLUT_BITMAP_HELVETICA_18
		self.GL_is_initialised = True

	def setImage(self, view, imageFilename):
		self.view = view
		self.imageFilename = imageFilename
		self.image = []

	def setPose(self, Gs):
		if self.transforms is None or self.transforms.shape[0] != Gs.shape[0]:
			self.transforms = np.zeros((Gs.shape[0],4,4),dtype=np.float32)
			self.transforms[:,3,3] = 1
		self.transforms[:,:,:3] = np.transpose(Gs,axes=[0,2,1])

	def setVs(self, vs):
		self.vs = vs
		if vs is not None: self.vs = vbo.VBO(np.array(vs,dtype=np.float32), usage='GL_STATIC_DRAW_ARB')
		#if vns is not None: self.vns = vbo.VBO(np.array(vns,dtype=np.float32), usage='GL_STATIC_DRAW_ARB')
		if self.edgeList is not None:
			vns = self.computeNormalsFromEdgeList(vs)
			self.vns = vbo.VBO(np.array(vns,dtype=np.float32), usage='GL_STATIC_DRAW_ARB')

	def __len__(self): return len(self.tris)

	def paintGL(self, p0=0, p1=None, drawOpts=DRAWOPT_ALL):
		'''
		:param drawOpts: OR combination of draw flags. default is :data:`UI.DRAWOPT_ALL`
		'''
		#if not self.d['draw'] or not self.d['visible']: return
		doingSelection = (p1 is not None)
		if p1 is None: p1 = len(self)
		if not self.GL_is_initialised: self.initializeGL()
		if p1 == 0: return # don't render if no vertices
		if self.image != self.bindImage:
			if self.image == []:
				self.image = QtGui.QPixmap(self.imageFilename).toImage()
				self.imageFlipped = False
			if self.bindImage is not None:
				self.deleteTexture(self.bindId)
				self.bindId,self.bindImage = long(0),None
			if self.image is not None:
				global win
				if self.view == None: 	from UI import QApp; self.view = QApp.view() # TODO
				self.bindId = self.view.bindTexture(self.image)
				self.bindImage = self.image
		if self.bindImage is not None:
			GL.glEnable(GL.GL_TEXTURE_2D)
			GL.glBindTexture(GL.GL_TEXTURE_2D, self.bindId)
		GL.glEnable(GL.GL_BLEND)
		GL.glEnable(GL.GL_CULL_FACE)
		GL.glCullFace(GL.GL_BACK)
		GL.glFrontFace(GL.GL_CCW)
		GL.glEnable(GL.GL_LIGHTING)
		GL.glEnable(GL.GL_LIGHT0)
		Pmat = GL.glGetDoublev(GL.GL_PROJECTION_MATRIX)
		lightDir = -Pmat[:3,2] # the direction the camera is looking
		GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, lightDir)
		GL.glShadeModel(GL.GL_SMOOTH)
		if self.colour is not None: GL.glMaterialfv(GL.GL_FRONT, GL.GL_AMBIENT_AND_DIFFUSE, self.colour)
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
		if self.tris is not None and DRAWOPT_GEOMS & drawOpts:
			if not doingSelection: GL.glUseProgram(self.shader2)
			self.tris.bind()
			if self.drawStyle == 'wire':
				GL.glShadeModel(GL.GL_FLAT)
				GL.glLineWidth(1)
				GL.glDrawElements(GL.GL_LINES, (p1-p0)*3, GL.GL_UNSIGNED_INT, self.tris + p0*12)
				GL.glShadeModel(GL.GL_SMOOTH)
			elif self.drawStyle == 'smooth':
				GL.glDrawElements(GL.GL_TRIANGLES, (p1-p0)*3, GL.GL_UNSIGNED_INT, self.tris + p0*12)
				#GL.glDrawElementsui(GL.GL_TRIANGLES, self.tris)
			elif self.drawStyle == 'wire_over_smooth':
				GL.glDrawElements(GL.GL_TRIANGLES, (p1-p0)*3, GL.GL_UNSIGNED_INT, self.tris + p0*12)
				#GL.glDrawElementsui(GL.GL_TRIANGLES, self.tris)
				GL.glShadeModel(GL.GL_FLAT)
				GL.glMaterialfv(GL.GL_FRONT, GL.GL_AMBIENT_AND_DIFFUSE, [0,0,0,1])
				GL.glLineWidth(1)
				GL.glDrawElementsui(GL.GL_LINES, self.tris)
				GL.glMaterialfv(GL.GL_FRONT, GL.GL_AMBIENT_AND_DIFFUSE, self.colour)
				GL.glShadeModel(GL.GL_SMOOTH)
			self.tris.unbind()
		if self.transformData is not None and self.transforms is not None and DRAWOPT_GEOMS & drawOpts:
			if not doingSelection: GL.glUseProgram(self.shader)
			GL.glUniformMatrix4fv(self.shader_myMat, len(self.transforms), GL.GL_FALSE, self.transforms) # put the transforms in myMat
			GL.glEnableVertexAttribArray(self.shader_bi)
			self.vtis.bind()
			GL.glVertexAttribIPointer(self.shader_bi, 1, GL.GL_UNSIGNED_INT, 0, self.vtis) # write the vtis to bi
			self.tis.bind()
			if self.drawStyle == 'wire':
				GL.glShadeModel(GL.GL_FLAT)
				GL.glLineWidth(1)
				GL.glDrawElementsui(GL.GL_LINES, self.tis) # this is wrong
				GL.glShadeModel(GL.GL_SMOOTH)
			elif self.drawStyle == 'smooth':
				GL.glDrawElementsui(GL.GL_TRIANGLES, self.tis)
			elif self.drawStyle == 'wire_over_smooth':
				GL.glDrawElementsui(GL.GL_TRIANGLES, self.tis)
				GL.glShadeModel(GL.GL_FLAT)
				GL.glMaterialfv(GL.GL_FRONT, GL.GL_AMBIENT_AND_DIFFUSE, [0,0,0,1])
				GL.glLineWidth(1)
				GL.glDrawElementsui(GL.GL_LINES, self.tis)
				GL.glMaterialfv(GL.GL_FRONT, GL.GL_AMBIENT_AND_DIFFUSE, self.colour)
				GL.glShadeModel(GL.GL_SMOOTH)
			self.tis.unbind()
			self.vtis.unbind()
			#for ti,tris in self.transformData:
				#if len(tris)==0: continue
				#GL.glUniformMatrix4fv(self.myMat, 1, GL.GL_FALSE, self.transforms[ti])
				#GL.glDrawElementsui(self.drawStyle, tris)
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
		GL.glDisable(GL.GL_BLEND)
		GL.glDisable(GL.GL_TEXTURE_2D)
		GL.glBindTexture(GL.GL_TEXTURE_2D, 0)		
		GL.glUseProgram( 0 )
