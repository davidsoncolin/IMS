#!/usr/bin/env python

from OpenGL import GL
from OpenGL.arrays import vbo
from OpenGL.GL import shaders
from PIL import Image
import numpy as np
import pyopencl as cl
from GCore import State

from UI import (COLOURS, DRAWOPT_DEFAULT, DRAWOPT_BONES, DRAWOPT_JOINTS, DRAWOPT_OFFSET,
					DRAWOPT_AXES, DRAWOPT_POINTS, DRAWOPT_GEOMS, DRAWOPT_LABELS, DRAWOPT_DETECTIONS)

global CL_ctx, CL_queue
CL_ctx = None
CL_queue = None

class GLMeshes:
	'''GLMeshes is a primitive class that efficiently renders a number of geometries, in batches of 128.
	At the moment, there is a single material (drawstyle, colour, texture) for all the geometries (this will improve).
	Each geometry can have a transform.
	Bones are also supported.
	Normals are generated.
	One geometry might be selected (read from State.getSel()) and so drawn in a different style.
	'''
	def __init__(self, names, verts, faces, bones=None, transforms=None, drawStyle = 'smooth', colour = None, vts = None, fts = None, visible = True):
		self.pointSize = 5
		self.selectedIndex = -1
		self.numGeos = len(names)
		self.visible = visible
		assert self.numGeos == len(verts), 'Non-matching parameter lists.'
		assert self.numGeos == len(faces), 'Non-matching parameter lists.'
		if transforms is None: transforms = [None]*self.numGeos
		if bones is None:      bones      = [None]*self.numGeos
		if vts is None: vts = [None]*self.numGeos
		if fts is None: fts = [None]*self.numGeos
		self.transforms = np.zeros((self.numGeos,4,4),dtype=np.float32)
		vs,VTs,es,bs,tris,vtis,vs_mapping,vts_mapping = [],[],[],[],[],[],[],[]
		vsplits,esplits,tsplits,bsplits = [0],[0],[0],[0]
		vs_in_totals = [0]
		for i,(v,f,b,t,vt,ft) in enumerate(zip(verts,faces,bones,transforms,vts,fts)):
			#print (i,names[i])
			voffset = len(vs)
			vt_indices = v_indices = np.arange(len(v),dtype=np.int32)
			if ft is not None:
				f_flat = [x for y in f for x in y]
				ft_flat = [x for y in ft for x in y]
				s = list(set(zip(f_flat, ft_flat)))
				d = dict(zip(s,range(len(s))))
				if len(s): v_indices, vt_indices = np.array(zip(*s),dtype=np.int32)
				else: v_indices, vt_indices = np.zeros((2,0),dtype=np.int32)
				f = [np.array([d[x] for x in zip(*y)],dtype=np.int32) for y in zip(f,ft)]
			vs_in_total = vs_in_totals[-1]
			vs_mapping.extend(v_indices+vs_in_total)
			vts_mapping.extend(vt_indices+vs_in_total)
			vs_in_totals.append(vs_in_total+len(v))
			v = np.array(v,dtype=np.float32).reshape(-1,3)[v_indices]
			if vt is None: vt = np.zeros((len(vt_indices),2),dtype=np.float32) # TODO missing verts
			#if np.any(vt_indices < 0): print 'WARNING GLMeshes geom ',i,names[i],vt_indices
			vt = np.array(vt,dtype=np.float32).reshape(-1,2)[vt_indices]
			vs.extend(v) # TODO is this slow? faster to use np.concatenate?
			VTs.extend(vt)
			vtis.extend([i]*len(v))
			if b is not None: bs.extend(np.array(b,dtype=np.int32).reshape(-1,2)+voffset)
			self.transforms[i] = np.eye(4)
			if t is not None: self.transforms[i,:,:3] = t.T
			if len(f) == 2 and f[1][0] == 0: # assume this is faces and splits
				f0 = np.array(f[0],dtype=np.int32)+voffset
				for c0,c1 in zip(f[1][:-1],f[1][1:]):
					fc = f0[c0:c1]
					es.append((fc[-1],fc[0]))
					es.append((fc[0],fc[1]))
					for fi in xrange(2,len(fc)):
						tris.append((fc[0],fc[fi-1],fc[fi]))
						es.append((fc[fi-1],fc[fi]))
			else:
				try:
					# see if the mesh is regular
					fr = np.array(f,dtype=np.int32)+voffset # will fail if not rectangular ints
					numFaces,faceSize = fr.shape # will fail if not size 2
					e = np.zeros((numFaces,faceSize,2),dtype=np.int32)
					t = np.zeros((numFaces,faceSize-2,3),dtype=np.int32)
					e[:,0,0] = fr[:,-1]
					e[:,0,1] = fr[:,0]
					e[:,1,0] = fr[:,0]
					e[:,1,1] = fr[:,1]
					t[:,:,0] = fr[:,0].reshape(-1,1)
					for fi in xrange(2,faceSize):
						e[:,fi,0] = fr[:,fi-1]
						e[:,fi,1] = fr[:,fi]
						t[:,fi-2,1] = fr[:,fi-1]
						t[:,fi-2,2] = fr[:,fi]
					e = e.reshape(-1,2)
					t = t.reshape(-1,3)
					es.extend(e)
					tris.extend(t)
				except Exception as e:
					for fc in f:
						if len(fc) > 2:
							fc = np.array(fc,dtype=np.int32)+voffset
							es.append((fc[-1],fc[0]))
							es.append((fc[0],fc[1]))
							for fi in xrange(2,len(fc)):
								tris.append((fc[0],fc[fi-1],fc[fi]))
								es.append((fc[fi-1],fc[fi]))
			vsplits.append(len(vs))
			esplits.append(len(es))
			bsplits.append(len(bs))
			tsplits.append(len(tris))
		self.vsplits = np.array(vsplits,dtype=np.int32)
		self.esplits = np.array(esplits,dtype=np.int32)
		self.bsplits = np.array(bsplits,dtype=np.int32)
		self.tsplits = np.array(tsplits,dtype=np.int32)
		self.vs_mapping = np.array(vs_mapping,dtype=np.int32)
		self.vts_mapping = np.array(vts_mapping,dtype=np.int32)
		self.vs_in_totals = vs_in_totals
		self.names = names
		vs = np.array(vs,dtype=np.float32).reshape(-1,3)
		tris = np.array(tris,dtype=np.int32).reshape(-1,3)
		edges = np.array(es,dtype=np.int32).reshape(-1,2)
		bones = np.array(bs,dtype=np.int32).reshape(-1,2)
		VTs = np.array(VTs,dtype=np.float32).reshape(-1,2)
		#print ('lens',len(vs), len(tris), len(edges), (np.min(tris),np.max(tris)) if len(tris) else 'None', (np.min(edges), np.max(edges)) if len(edges) else 'None', (np.min(bones), np.max(bones)) if len(bones) else 'None')
		self.num_in_verts = vs_in_totals[-1]
		self.vs = vbo.VBO(vs, usage='GL_STATIC_DRAW_ARB')
		self.tris = vbo.VBO(tris, target=GL.GL_ELEMENT_ARRAY_BUFFER, usage='GL_STATIC_DRAW_ARB')
		self.edges = vbo.VBO(edges, target=GL.GL_ELEMENT_ARRAY_BUFFER, usage='GL_STATIC_DRAW_ARB')
		self.bones = vbo.VBO(bones, target=GL.GL_ELEMENT_ARRAY_BUFFER, usage='GL_STATIC_DRAW_ARB')
		self.vtis = vbo.VBO(np.array(vtis,dtype=np.int32), usage='GL_STATIC_DRAW_ARB')
		assert len(vtis) == len(vs)
		self.vts,self.vns = None,None
		# TODO, deal with input textures and normals
		if vts is not None:
			self.vts = vbo.VBO(VTs, usage='GL_STATIC_DRAW_ARB')
		#if vns is not None: self.vns = vbo.VBO(np.array(vns,dtype=np.float32), usage='GL_STATIC_DRAW_ARB')
		self.drawStyle = drawStyle # 'wire','smooth','wire_over_smooth'
		self.colour = colour
		self.image,self.bindImage,self.bindId = None,None,None
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
				if (xs_g[e3] > 1e10) continue;
				float ex1 = xs_g[e3]-x, ey1 = xs_g[e3+1]-y, ez1 = xs_g[e3+2]-z;
				sx += ey0*ez1-ey1*ez0;
				sy += ez0*ex1-ez1*ex0;
				sz += ex0*ey1-ex1*ey0;
				ex0=ex1; ey0=ey1; ez0=ez1;
			}
			const float sum = sx*sx+sy*sy+sz*sz;
			if (sum < 1e-8) { sx = 0; sy = 0; sz = 0; }
			else {
				const float sc = rsqrt(sum);
				sx *= sc;
				sy *= sc;
				sz *= sc;
			}
			res_g[g3] = sx;
			res_g[g3+1] = sy;
			res_g[g3+2] = sz;
		}
		''').build()
		self.edgeList = self.trianglesToEdgeList(tris, len(vs))
		self.edgeList_g = cl.Buffer(CL_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.edgeList)
		if self.vns is None:
			vns = self.computeNormalsFromEdgeList(vs)
			self.vns = vbo.VBO(np.array(vns,dtype=np.float32), usage='GL_STATIC_DRAW_ARB')

	def trianglesToEdgeList(self,triangles, numVerts=0):
		import ISCV
		return ISCV.trianglesToEdgeList(triangles, numVerts)

	def test_trianglesToEdgeList(self,triangles, numVerts=0):
		'''Convert a list of triangle indices to an array of up-to-10 neighbouring vertices per vertex (following anticlockwise order).'''
		if numVerts is None: numVerts = np.max(triangles)+1 if len(triangles) else 1
		if numVerts < 1: numVerts = 1 # avoid empty arrays
		T = [dict() for t in xrange(numVerts)]
		P = [dict() for t in xrange(numVerts)]
		for t0,t1,t2 in triangles:
			T[t0][t1],T[t1][t2],T[t2][t0] = t2,t0,t1
			P[t1][t0],P[t2][t1],P[t0][t2] = t2,t0,t1
		S = np.zeros((numVerts,10),dtype=np.int32)
		for vi,(Si,es,ps) in enumerate(zip(S,T,P)):
			Si[:] = vi
			if not es: continue
			v = min(es.keys())
			while v in ps: v = ps.pop(v)
			for li in xrange(10):
				Si[li] = v
				if v not in es: break
				v = es.pop(v,vi)
		import ISCV
		ret = ISCV.trianglesToEdgeList(triangles, numVerts)
		assert np.all(S == ret),repr(S)+repr(ret)
		return ret

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
		VERTEX_SHADER_NOTEX = shaders.compileShader('''
		#version 130
		uniform mat4 myMat[128]; // we support up to 128 transforms
		uniform int bo;
		in int bi;
		out vec3 N;
		void main() {
			mat4 mat = myMat[bi-bo];
			gl_TexCoord[0] = gl_TextureMatrix[0] * gl_MultiTexCoord0;
			gl_Position = gl_ProjectionMatrix * (gl_ModelViewMatrix * (mat * gl_Vertex));
			N = gl_NormalMatrix * (mat3(mat) * gl_Normal);
			if (gl_Vertex.x > 1e10) gl_Position.xyzw = vec4(0,0,1e20,0); // special value, to hide a vertex!
		}''', GL.GL_VERTEX_SHADER)
		FRAGMENT_SHADER_NOTEX = shaders.compileShader('''
		#version 130
		in vec3 N;
		uniform sampler2D tex;
		void main() {
			vec3 lightDir = normalize(gl_LightSource[0].position.xyz);
			float NdotL = max(0.1, abs(dot(normalize(N), lightDir))); // add ambient light
			if (N == vec3(0,0,0)) NdotL = 1.0; // this allows the shader to draw edges (which have gl_Normal=0)
			vec4 diffuse = gl_FrontMaterial.diffuse * gl_LightSource[0].diffuse;
			gl_FragColor = vec4(NdotL,NdotL,NdotL,1.0) * diffuse;
		}''', GL.GL_FRAGMENT_SHADER)
		self.shader_notex = shaders.compileProgram(VERTEX_SHADER_NOTEX,FRAGMENT_SHADER_NOTEX)
		self.shader_notex_bi = GL.glGetAttribLocation(self.shader_notex, 'bi')
		self.shader_notex_myMat = GL.glGetUniformLocation(self.shader_notex, 'myMat')
		self.shader_notex_bo = GL.glGetUniformLocation(self.shader_notex, 'bo')
		
		VERTEX_SHADER_TEX = shaders.compileShader('''
		#version 130
		uniform mat4 myMat[128]; // we support up to 128 transforms
		uniform int bo;
		in int bi;
		out vec3 N;
		void main() {
			mat4 mat = myMat[bi-bo];
			gl_TexCoord[0] = gl_TextureMatrix[0] * gl_MultiTexCoord0;
			gl_Position = gl_ProjectionMatrix * (gl_ModelViewMatrix * (mat * gl_Vertex));
			N = gl_NormalMatrix * (mat3(mat) * gl_Normal);
			if (gl_Vertex.x > 1e10) gl_Position.xyzw = vec4(0,0,1e20,0); // special value, to hide a vertex!
		}''', GL.GL_VERTEX_SHADER)
		FRAGMENT_SHADER_TEX = shaders.compileShader('''
		#version 130
		in vec3 N;
		uniform sampler2D tex;
		void main() {
			vec3 lightDir = normalize(gl_LightSource[0].position.xyz);
			float NdotL = max(0.1, abs(dot(normalize(N), lightDir))); // add ambient light
			if (N == vec3(0,0,0)) NdotL = 1.0; // this allows the shader to draw edges (which have gl_Normal=0)
			vec4 colour = texture2D(tex, gl_TexCoord[0].st);
			gl_FragColor = vec4(NdotL,NdotL,NdotL,1.0) * colour;
		}''', GL.GL_FRAGMENT_SHADER)
		self.shader_tex = shaders.compileProgram(VERTEX_SHADER_TEX,FRAGMENT_SHADER_TEX)
		self.shader_tex_bi = GL.glGetAttribLocation(self.shader_tex, 'bi')
		self.shader_tex_myMat = GL.glGetUniformLocation(self.shader_tex, 'myMat')
		self.shader_tex_bo = GL.glGetUniformLocation(self.shader_tex, 'bo')
		
		VERTEX_SHADER_XFORM = shaders.compileShader('''
			#version 130
			uniform mat4 myMat[128]; // we support up to 128 transforms
			out vec3 N;
			void main() {
				int mi = gl_VertexID / 6;
				int vi = gl_VertexID % 6;
				mat4 mat = myMat[mi];
				vec4 tmp = gl_ProjectionMatrix * (gl_ModelViewMatrix * (mat * vec4(0,0,0,1)));
				float scale = abs(0.1 * inversesqrt(dot(mat[0].xyz,mat[0].xyz)+dot(mat[1].xyz,mat[1].xyz)+dot(mat[2].xyz,mat[2].xyz)) * tmp.w); // ~8pc of the window
				N = vec3(vi==4 || vi==5, vi==2 || vi==3, vi==0 || vi==1);
				gl_Position = gl_ProjectionMatrix * (gl_ModelViewMatrix * (mat * vec4(vi==5?scale:0,vi==3?scale:0,vi==1?scale:0,1)));
			}''', GL.GL_VERTEX_SHADER)
		FRAGMENT_SHADER_XFORM = shaders.compileShader('''
			#version 130
			in vec3 N;
			void main() {
				gl_FragColor = vec4(N,1);
			}''', GL.GL_FRAGMENT_SHADER)
		self.shader_xform = shaders.compileProgram(VERTEX_SHADER_XFORM,FRAGMENT_SHADER_XFORM)
		self.shader_xform_myMat = GL.glGetUniformLocation(self.shader_xform, 'myMat')

		VERTEX_SHADER_POINT = shaders.compileShader('''
			#version 130
			uniform mat4 myMat[128]; // we support up to 128 transforms
			void main() {
				int mi = gl_VertexID;
				mat4 mat = myMat[mi];
				gl_Position = gl_ProjectionMatrix * (gl_ModelViewMatrix * (mat * vec4(0,0,0,1)));
			}''', GL.GL_VERTEX_SHADER)
		FRAGMENT_SHADER_POINT = shaders.compileShader('''
			#version 130
			void main() {
				vec4 diffuse = gl_FrontMaterial.diffuse * gl_LightSource[0].diffuse;
				gl_FragColor = diffuse;
			}''', GL.GL_FRAGMENT_SHADER)
		self.shader_point = shaders.compileProgram(VERTEX_SHADER_POINT,FRAGMENT_SHADER_POINT)
		self.shader_point_myMat = GL.glGetUniformLocation(self.shader_point, 'myMat')

		#self.font = GLUT.GLUT_BITMAP_HELVETICA_18
		try:
			self.bindBID = GL.glGenBuffers(1) # for streaming video
		except:
			print ('GL: no textures (failed glGenBuffers)')
			self.bindBID = 0
		self.GL_is_initialised = True

	def setImageFilename(self, imageFilename):
		self.image = Image.open(imageFilename)

	def setImage(self, image):
		self.image = image

	def create_bindImage(self):
		'''
		Replace the texture on the graphics card with self.image.
		Set self.image to None to only delete the existing texture.
		'''
		if self.bindImage is not None: GL.glDeleteTextures([self.bindId])
		self.bindId,self.bindImage = None,None
		if self.image is not None:
			self.bindImage = self.image
			self.bindId = GL.glGenTextures(1)
			self.refreshImage()

	def bindImage_data(self):
		if isinstance(self.bindImage, Image.Image):
			(w,h),c = self.bindImage.size,len(self.bindImage.getbands())
			return self.bindImage.tostring(),(h,w,c)
		if isinstance(self.bindImage, np.ndarray):
			return self.bindImage,self.bindImage.shape
		return self.bindImage

	def refreshImage(self):
		'''Update the texture on the graphics card.'''
		if self.bindImage is not None and self.bindImage is self.image and self.bindId is not None:
			data,(h,w,c) = self.bindImage_data()
			GL.glBindTexture(GL.GL_TEXTURE_2D, self.bindId)
			GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, self.bindBID)
			GL.glBufferData(GL.GL_PIXEL_UNPACK_BUFFER, h*w*c, data, GL.GL_STREAM_DRAW)
			GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
			GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
			GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, w, h, 0, [GL.GL_RGB,GL.GL_RGBA][c-3], GL.GL_UNSIGNED_BYTE, None)
			GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)
			GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
		self.imageInvalidated = False

	def setPose(self, Gs):
		assert self.transforms.shape[0] == self.numGeos, 'Number of transforms must agree.'
		if self.numGeos == 0: return
		self.transforms[:,:,:3] = np.transpose(Gs,axes=[0,2,1])
		self.transforms[:,:,3] = np.array([0,0,0,1],dtype=np.float32)

	def setVs(self, vs, name = None, vts = None):
		if name is None:
			if vs is not None:
				assert len(vs) == self.num_in_verts, 'GLMeshes Updated with bad data'+repr(len(vs))+' vs '+repr(self.num_in_verts)
				self.vs[:] = vs[self.vs_mapping]
				vns = self.computeNormalsFromEdgeList(self.vs)
				self.vns[:] = vns
			if vts is not None: 
				self.vts[:] = vts[self.vts_mapping]
		else:
			assert name in self.names, '{} not in GLMesh'.format(name)
			ni = self.names.index(name)
			if vs is not None:
				self.vs[self.vsplits[ni]:self.vsplits[ni+1]] = vs[self.vs_mapping[self.vsplits[ni]:self.vsplits[ni+1]]-self.vs_in_totals[ni]]
				vns = self.computeNormalsFromEdgeList(self.vs)
				self.vns[:] = vns
			if vts is not None:
				self.vts[self.vsplits[ni]:self.vsplits[ni+1]] = vts[self.vts_mapping[self.vsplits[ni]:self.vsplits[ni+1]]-self.vs_in_totals[ni]]
			#vns = self.computeNormalsFromEdgeList(vs, ni) # TODO
			#self.vns[self.vsplits[ni]:self.vsplits[ni+1]] = vns

	def __len__(self): return self.numGeos

	def paintGL(self, p0=0, p1=None, drawOpts=DRAWOPT_DEFAULT):
		'''
		:param drawOpts: OR combination of draw flags. default is :data:`UI.DRAWOPT_DEFAULT`
		'''
		#if not self.d['draw'] or not self.d['visible']: return
		if not self.visible: return
		doingSelection = (p1 is not None)
		if p1 is None: p1 = len(self)
		if not self.GL_is_initialised: self.initializeGL()
		if p1 == 0: return # don't render if no vertices
		GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
		self.vs.bind()
		GL.glVertexPointerf(self.vs)
		if self.image is not self.bindImage:
			self.create_bindImage()
		self.shader,self.shader_bi,self.shader_myMat,self.shader_bo = \
			self.shader_notex,self.shader_notex_bi,self.shader_notex_myMat,self.shader_notex_bo
		if self.bindImage is not None:
			GL.glEnable(GL.GL_TEXTURE_2D)
			GL.glBindTexture(GL.GL_TEXTURE_2D, self.bindId)
			self.shader,self.shader_bi,self.shader_myMat,self.shader_bo = \
				self.shader_tex,self.shader_tex_bi,self.shader_tex_myMat,self.shader_tex_bo
		if self.vts is not None:
			GL.glEnableClientState(GL.GL_TEXTURE_COORD_ARRAY)
			self.vts.bind()
			GL.glTexCoordPointerf(self.vts)
		if self.vns is not None:
			GL.glEnableClientState(GL.GL_NORMAL_ARRAY)
			self.vns.bind()
			GL.glNormalPointerf(self.vns)
		sel = State.getSel()
		if sel is None:
			self.selectedIndex,self.selectedName = -1,None
		if sel is not None:
			if self.selectedIndex == -1 or self.selectedName != sel:
				try:
					self.selectedIndex,self.selectedName = self.names.index(sel),sel
				except:
					self.selectedIndex,self.selectedName = -1,sel
		#if sel is not None: print ('selection',sel,sel[5:], self.selectedIndex, self.names[self.selectedIndex], self.names[:5])
		GL.glEnable(GL.GL_BLEND)
		GL.glShadeModel(GL.GL_SMOOTH)
		GL.glEnable(GL.GL_LIGHTING)
		GL.glEnable(GL.GL_LIGHT0)
		Pmat = GL.glGetDoublev(GL.GL_PROJECTION_MATRIX)
		lightDir = -Pmat[:3,2] # the direction the camera is looking
		GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, lightDir)
		#GL.glEnable(GL.GL_CULL_FACE)
		GL.glCullFace(GL.GL_BACK)
		GL.glFrontFace(GL.GL_CCW)
		GL.glEnableVertexAttribArray(self.shader_bi)
		self.vtis.bind()
		GL.glVertexAttribIPointer(self.shader_bi, 1, GL.GL_UNSIGNED_INT, 0, self.vtis) # write the vtis to bi
		ranges = []
		for t0 in xrange(p0,p1,128): # draw the geometries in batches of 128
			t1 = min(t0+128,p1)
			if self.selectedIndex in xrange(t0,t1):
				if t0 < self.selectedIndex: ranges.append((t0,self.selectedIndex))
				if t1 > self.selectedIndex+1: ranges.append((self.selectedIndex+1,t1))
			else: ranges.append((t0,t1))
		#print ('ranges',ranges)
		if self.selectedIndex >= p0 and self.selectedIndex < p1: # draw the selected object
			GL.glUseProgram(self.shader)
			i = self.selectedIndex
			GL.glUniformMatrix4fv(self.shader_myMat, 1, GL.GL_FALSE, self.transforms[i]) # put the transforms in myMat
			GL.glUniform1i(self.shader_bo, i) # put the offset
			if drawOpts & DRAWOPT_GEOMS:
				tr0,tr1 = self.tsplits[i],self.tsplits[i+1]
				if tr1 > tr0:
					GL.glMaterialfv(GL.GL_FRONT, GL.GL_AMBIENT_AND_DIFFUSE, [1,0,0,1])
					self.tris.bind()
					GL.glDrawElements(GL.GL_TRIANGLES, (tr1-tr0)*3, GL.GL_UNSIGNED_INT, self.tris + tr0*3*4)
					self.tris.unbind()
				e0,e1 = self.esplits[i],self.esplits[i+1]
				if e1 > e0:
					GL.glMaterialfv(GL.GL_FRONT, GL.GL_AMBIENT_AND_DIFFUSE, [0,0,0,1])
					self.edges.bind()
					GL.glLineWidth(0.5)
					GL.glDrawElements(GL.GL_LINES, (e1-e0)*2, GL.GL_UNSIGNED_INT, self.edges + e0*2*4)
					self.edges.unbind()
			if drawOpts & DRAWOPT_BONES: # draw bones
				b0,b1 = self.bsplits[i],self.bsplits[i+1]
				if b1 > b0:
					GL.glLineWidth(1)
					GL.glMaterialfv(GL.GL_FRONT, GL.GL_AMBIENT_AND_DIFFUSE, [1,1,0,1])
					self.bones.bind()
					GL.glDrawElements(GL.GL_LINES, (b1-b0)*2, GL.GL_UNSIGNED_INT, self.bones + b0*2*4)
					self.bones.unbind()
			if drawOpts & DRAWOPT_JOINTS: # draw joints
				#GL.glUseProgram(self.shader_point)
				#GL.glDrawArrays(GL.GL_POINTS, 0, 1)
				b0,b1 = self.bsplits[i],self.bsplits[i+1]
				if b1 > b0:
					GL.glPointSize(self.pointSize)
					GL.glMaterialfv(GL.GL_FRONT, GL.GL_AMBIENT_AND_DIFFUSE, [1,0,0,1])
					self.bones.bind()
					GL.glDrawElements(GL.GL_POINTS, (b1-b0)*2, GL.GL_UNSIGNED_INT, self.bones + b0*2*4)
					self.bones.unbind()
			if drawOpts & DRAWOPT_AXES: # draw axes TODO does this depend on len(vs) > 6 ?
				GL.glUseProgram(self.shader_xform)
				GL.glLineWidth(2)
				GL.glUniformMatrix4fv(self.shader_xform_myMat, 1, GL.GL_FALSE, self.transforms[i]) # put the transforms in myMat
				GL.glDrawArrays(GL.GL_LINES, 0, 6)
		GL.glUseProgram(self.shader)
		if drawOpts & DRAWOPT_GEOMS and (self.drawStyle == 'smooth' or self.drawStyle == 'wire_over_smooth'): # draw triangles
			GL.glMaterialfv(GL.GL_FRONT, GL.GL_AMBIENT_AND_DIFFUSE, self.colour if self.colour is not None else [0.9,0.9,0.9,1.0])
			self.tris.bind()
			for t0,t1 in ranges:
				tr0,tr1 = self.tsplits[t0],self.tsplits[t1]
				if tr1 > tr0:
					GL.glUniformMatrix4fv(self.shader_myMat, t1-t0, GL.GL_FALSE, self.transforms[t0:t1]) # put the transforms in myMat
					GL.glUniform1i(self.shader_bo, t0) # put the offset
					GL.glDrawElements(GL.GL_TRIANGLES, (tr1-tr0)*3, GL.GL_UNSIGNED_INT, self.tris + tr0*3*4)
			self.tris.unbind()
		if drawOpts & DRAWOPT_GEOMS and (self.drawStyle == 'wire' or self.drawStyle == 'wire_over_smooth'): # draw edges
			GL.glLineWidth(1)
			GL.glMaterialfv(GL.GL_FRONT, GL.GL_AMBIENT_AND_DIFFUSE, [0,0,0,1])
			self.edges.bind()
			for t0,t1 in ranges:
				e0,e1 = self.esplits[t0],self.esplits[t1]
				if e1 > e0:
					GL.glUniformMatrix4fv(self.shader_myMat, t1-t0, GL.GL_FALSE, self.transforms[t0:t1]) # put the transforms in myMat
					GL.glUniform1i(self.shader_bo, t0) # put the offset
					GL.glDrawElements(GL.GL_LINES, (e1-e0)*2, GL.GL_UNSIGNED_INT, self.edges + e0*2*4)
			self.edges.unbind()
		if drawOpts & DRAWOPT_BONES: # draw bones
			GL.glUseProgram(self.shader_notex)
			GL.glLineWidth(1)
			bone_colour = self.colour if self.colour is not None else [1,1,1,1.]
			GL.glMaterialfv(GL.GL_FRONT, GL.GL_AMBIENT_AND_DIFFUSE, bone_colour)
			self.bones.bind()
			for t0,t1 in ranges:
				b0,b1 = self.bsplits[t0],self.bsplits[t1]
				if b1 > b0:
					GL.glUniformMatrix4fv(self.shader_notex_myMat, t1-t0, GL.GL_FALSE, self.transforms[t0:t1]) # put the transforms in myMat
					GL.glUniform1i(self.shader_notex_bo, t0) # put the offset
					GL.glDrawElements(GL.GL_LINES, (b1-b0)*2, GL.GL_UNSIGNED_INT, self.bones + b0*2*4)
			self.bones.unbind()
		if drawOpts & DRAWOPT_JOINTS: # draw joints
			GL.glUseProgram(self.shader_notex)
			GL.glPointSize(self.pointSize)
			bone_colour = self.colour if self.colour is not None else [0,1,1,1]
			GL.glMaterialfv(GL.GL_FRONT, GL.GL_AMBIENT_AND_DIFFUSE, bone_colour)
			self.bones.bind()
			for t0,t1 in ranges:
				b0,b1 = self.bsplits[t0],self.bsplits[t1]
				if b1 > b0:
					GL.glUniformMatrix4fv(self.shader_notex_myMat, t1-t0, GL.GL_FALSE, self.transforms[t0:t1]) # put the transforms in myMat
					GL.glUniform1i(self.shader_notex_bo, t0) # put the offset
					GL.glDrawElements(GL.GL_POINTS, (b1-b0)*2, GL.GL_UNSIGNED_INT, self.bones + b0*2*4)
			self.bones.unbind()
		if drawOpts & DRAWOPT_AXES: # draw axes TODO does this depend on len(vs) > 6 ?
			GL.glUseProgram(self.shader_xform)
			GL.glLineWidth(2)
			for t0,t1 in ranges:
				GL.glUniformMatrix4fv(self.shader_xform_myMat, t1-t0, GL.GL_FALSE, self.transforms[t0:t1]) # put the transforms in myMat
				GL.glDrawArrays(GL.GL_LINES, 0, 6*(t1-t0))
		GL.glDisableVertexAttribArray(self.shader_bi)
		self.vtis.unbind()
		GL.glUseProgram( 0 )
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
		self.vs.unbind()
		GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
