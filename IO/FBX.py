#!/usr/bin/env python

import zlib
import struct
import os, sys
import numpy as np
import IO # TODO StreamReader

global g_fields
g_fields = {}

def read_bool(data, offset):
	v = bool(data[offset])
	return v,offset+1
	
def read_byte(data, offset):
	v = ord(data[offset])
	return v,offset+1

def read_int16(data, offset):
	v = struct.unpack_from('<h',data[offset:offset+2])[0]
	return v,offset+2

def read_int32(data, offset):
	v = struct.unpack_from('<i',data[offset:offset+4])[0]
	return v,offset+4

def read_uint32(data, offset):
	v = struct.unpack_from('<I',data[offset:offset+4])[0]
	return v,offset+4

def read_int64(data, offset):
	v = struct.unpack_from('<q',data[offset:offset+8])[0]
	return v,offset+8

def read_float(data, offset):
	v = struct.unpack_from('<f',data[offset:offset+4])[0]
	return v,offset+4

def read_double(data, offset):
	v = struct.unpack_from('<d',data[offset:offset+8])[0]
	return v,offset+8
	
def read_string(data, offset):
	size,offset = read_uint32(data, offset)
	return data[offset:offset+size],offset+size

def read_shortstring(data, offset):
	size,offset = read_byte(data, offset)
	return data[offset:offset+size],offset+size

def read_array(dt, data, offset):
	length,offset = read_uint32(data, offset)
	is_compressed,offset = read_uint32(data, offset)
	s,offset = read_string(data, offset)
	if is_compressed:
		assert is_compressed == 1
		s = zlib.decompress(s, 0, np.dtype(dt).itemsize*length)
	ret = np.fromstring(s, dtype=dt)
	assert len(ret) == length
	return ret,offset

g_FBX_types       = {ord('B'):read_bool, ord('C'):read_byte, ord('F'):read_float, ord('D'):read_double,\
					 ord('Y'):read_int16, ord('I'):read_int32, ord('L'):read_int64, ord('R'):read_string}
g_FBX_array_types = {ord('b'):np.bool,   ord('c'):np.uint8,  ord('f'):np.float32, ord('d'):np.float64,\
					 ord('y'):np.int16,   ord('i'):np.int32,   ord('l'):np.int64}

def read_attr(data,offset):
	global g_FBX_types, g_FBX_array_types
	data_type,offset = read_byte(data, offset)
	if data_type in g_FBX_types: return g_FBX_types[data_type](data, offset)
	if data_type in g_FBX_array_types: return read_array(g_FBX_array_types[data_type], data, offset)
	if data_type == ord('S'):
		s,offset = read_string(data,offset)
		return s.replace('\0\x01','::'),offset
	print 'unknown type',data_type,chr(data_type)
	raise

def read_header(data,offset):
	assert data[offset:offset+23] == 'Kaydara FBX Binary  \0\x1a\0'
	return offset+23

def read_node(data,offset):
	global g_ns,g_nsz
	end_offset,num_attrs,attrs_bytes = struct.unpack_from(g_ns,data[offset:offset+g_nsz])  # 4GB filesize bug: fixed in version >= 7500
	name,offset = read_shortstring(data, offset+g_nsz)
	if end_offset is 0:
		assert num_attrs == 0, repr(num_attrs)
		assert attrs_bytes == 0, repr(attrs_bytes)
		assert name == '', repr(name)
		return (None,None),offset
	node = {}
	attrs_end_offset = offset + attrs_bytes
	for i in range(num_attrs):
		attr,offset = read_attr(data, offset)
		node.setdefault('attrs',[]).append(attr)
	assert offset == attrs_end_offset
	# special case: immediately simplify a node that contains a single attribute to the value
	if offset == end_offset and num_attrs == 1: return (name,node['attrs'][0]),offset
	while offset < end_offset:
		(childname,child),offset = read_node(data, offset)
		if child is None: break
		node.setdefault(childname,[]).append(child)
	# special case: decode ['Properties70'][0]['P'] to ['props']
	# Properties70 attributes encode a name, class, type, flags(?), value
	if 'Properties70' in node:
		p70 = node.pop('Properties70')
		assert len(p70) == 1 and p70[0].keys() == ['P']
		node['props'] = ps = {}
		gui_map = {'Enum':'enum','ReferenceProperty':'float','Number':'float','Integer':'int','Short':'int','bool':'bool', 'Bool':'bool', 'enum':'enum', 'Url':'string', 'KString':'string', 'Compound':'compound', 'DateTime':'datetime', 'Time':'time', 'object':'object', 'Visibility':'bool', 'Visibility Inheritance':'bool', 'Blob':'Blob', 'charptr':'string', 'Action':'float', 'ULongLong':'long','XRefUrl':'string','Matrix Transformation':'matrix','Vector2':'float2', 'float':'float','vec_int':'vec_int'}
		for p in p70[0]['P']:
			o = p.pop('attrs')
			gui = o[2]
			if gui == '': gui = o[1]
			nvs = len(o[4:])
			# 'U' in flags -> positive floats
			if gui not in gui_map and nvs and type(o[4]) is float: gui_map[gui] = 'float' if nvs ==1 else 'float'+str(nvs)
			if gui not in gui_map: print 'TODO',o; gui_map[gui] = gui
			gui = gui_map[gui]
			enum=None
			if nvs == 2 and gui=='enum':
				nvs = 1
				enum = o[5].split('~')
			if '+' in o[3] and not nvs>1: print 'unexpected',o
			if '+' not in o[3] and nvs>1 and not (nvs==3 and gui[-1]=='3'): print 'unexpected2',o,gui,nvs
			if nvs > 1 and not gui[-1].isdigit(): gui = 'vec_'+gui
			prop = (o[4] if nvs == 1 else o[4:])
			global g_fields
			#if name not in g_fields or o[0] not in g_fields[name]: print 'FIELD INFO',name,o
			g_fields.setdefault(name,{}).setdefault(o[0],{}).update([('class',o[1]),('type',gui),('flags',o[3])])
			if enum: g_fields[name][o[0]]['enum'] = enum
			if 'BinaryData' in p: prop = p.pop('BinaryData')
			assert not p,repr(p) # now it's empty
			ps[o[0]] = prop
	assert offset == end_offset, repr(offset)+'!='+repr(end_offset)
	return (name,node),offset

def load(fn):
	data = open(fn,'rb').read()
	offset = read_header(data, 0)
	version,offset = read_uint32(data, offset)
	doc = {'_version':version}
	global g_ns,g_nsz
	g_ns,g_nsz = '<III',12
	if version >= 7500: g_ns,g_nsz = '<QQQ',24
	while 1:
		(name,node),offset = read_node(data, offset)
		if node is None: break
		assert name not in doc
		doc[name] = node
	footer,offset = data[offset:],len(data) # footer
	return doc

def decode(fbx_payload):
	ret = {}
	glob = fbx_payload['GlobalSettings']
	globs = glob['props']
	ret['globs'] = globs
	#assert globs['TimeSpanStop'] == 46186158000L, repr(globs['TimeSpanStop']) # 1919027552091
	timeScale = 1.0/46186158000 # seconds units = 7697693 * 6000
	conn = fbx_payload['Connections']
	assert conn.keys() == ['C']
	conn = conn['C']
	object_parent,object_prop_parent,prop_object_parent,prop_prop_parent = {},{},{},{}
	for c in conn:
		assert c.keys() == ['attrs']
		c = c['attrs']
		ctype = c[0]
		if ctype == 'OO': object_parent[c[1]] = c[2]
		elif ctype == 'OP': object_prop_parent.setdefault(c[1],{})[c[3]] = c[2]
		elif ctype == 'PO': prop_object_parent.setdefault(c[1],{})[c[2]] = c[3]
		elif ctype == 'PP': prop_prop_parent.setdefault(c[1],{}).setdefault(c[2],{})[c[4]] = c[3]
		else: assert False,repr(ctype)
	ret['object_parent'] = object_parent
	ret['object_prop_parent'] = object_prop_parent
	ret['prop_object_parent'] = prop_object_parent
	ret['prop_prop_parent'] = prop_prop_parent
	
	defs = fbx_payload['Definitions']
	count = defs['Count'][0]
	version = defs['Version'][0]
	dots = defs['ObjectType']
	defn = {}
	for dot in dots:
		tclass = dot['attrs'][0]
		count = count - dot['Count'][0]
		if 'PropertyTemplate' in dot:
			defn[tclass] = dot['PropertyTemplate'][0]['props']
	assert count == 0
	ret['defs'] = defn

	objects = fbx_payload['Objects']
	objs = {}
	ignores = set(['props','attrs','KeyTime','KeyValueFloat'])
	for node,nodes in objects.iteritems():
		for nd in nodes:
			k,t,h = nd['attrs']
			val = {'attrs':nd.get('props',{})}
			if node == 'AnimationCurve':
				kts = np.float32(nd['KeyTime'][0] * timeScale)
				kvs = nd['KeyValueFloat'][0]
				assert len(kts) == len(kvs)
				val['curve'] = np.float32((kts,kvs)) # 2xN
			name = t.split('::')[0].split(':')[-1] if '::' in t else t
			val.update({'type':node,'id':k,'fullname':t,'name':name,'class':h})
			for v in nd.keys():
				if v not in ignores:
					tmp = nd[v]
					if len(tmp)==1: tmp = tmp[0]
					val['attrs'][v] = tmp
			assert k not in objs
			if k in object_parent: val['parent'] = object_parent[k]
			if k in object_prop_parent: val['pparent'] = object_prop_parent[k]
			#else: assert node == 'AnimationStack', repr(node) # ?
			objs[k] = val
	ret['objs'] = objs
	return ret

def mobu_rmat(xyz_degrees, ro=0):
	#['xyz','xzy','yzx','yxz','zxy','zyx'][ro] NB right-to-left matrix order because our matrices are the transpose of mobu
	sx,sy,sz = np.sin(np.radians(xyz_degrees))
	cx,cy,cz = np.cos(np.radians(xyz_degrees))
	mx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]],dtype=np.float32)
	my = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]],dtype=np.float32)
	mz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]],dtype=np.float32)
	m1 = [mz,my,mx,mz,my,mx][ro]
	m2 = [my,mz,mz,mx,mx,my][ro]
	m3 = [mx,mx,my,my,mz,mz][ro]
	return np.dot(m1,np.dot(m2,m3))

def matrix_mult(p,c):
	ret = np.dot(p[:3,:3],c[:3,:])
	ret[:3,3] += p[:3,3]
	return ret

def matrix_inverse(m):
	ret = np.zeros((3,4),dtype=np.float32)
	try:
		ret[:,:3] = np.linalg.inv(m[:3,:3])
	except:
		print '???exception in matrix_inverse',list(ret.ravel()) # TODO HACK
		ret[:,:3] = np.eye(3) #m[:3,:3].T
	ret[:,3] = -np.dot(ret[:,:3],m[:3,3])
	return ret
	
def sample(mat, t):
	return mat[1][np.searchsorted(mat[0][:-1], t)] # HACK
	return np.interp(t, mat[0], mat[1]) # TODO doesn't work for periodic mat[1] (angles)

def extract_geo(node):
	if node['type'] != 'Geometry': return None
	# node keys are:'name', 'parent', 'id', 'type', 'class', 'fullname', 'attrs'
	# node['attrs'] : 'LayerElementNormal', 'GeometryVersion', 'LayerElementUV', 'Vertices', 'Edges', 'Layer', 'PolygonVertexIndex', 'LayerElementMaterial'
	if node['class'] == 'Shape': # blendshape
		#print node['attrs'].keys() #['Vertices', 'Version', 'Normals', 'Indexes']
		return {'verts':np.float32(node['attrs']['Vertices']).reshape(-1,3)*10.0,'indices':np.int32(node['attrs']['Indexes'])}
	if node['class'] == 'NurbsCurve':
		#['NurbsCurveVersion', 'KnotVector', 'Form', 'Dimension', 'Points', 'Rational', 'GeometryVersion', 'Type', 'Order']
		vs = np.float32(node['attrs']['Points']).reshape(-1,4)[:,:3]*10.0
		return {'verts':vs,'edges':[[i,(i+1) % len(vs)] for i in range(len(vs)-1 if node['attrs']=='Open' else len(vs))]}
	print node['type'],node['class'],node['attrs'].keys()
	vs, vis = node['attrs']['Vertices'], node['attrs']['PolygonVertexIndex']
	#es = node['Edges']
	#if len(es) % 2 != 0: print 'unexpected',len(es),es # 3983
	layer, layer_uv, layer_mtl = node['attrs']['Layer'], node['attrs']['LayerElementUV'], node['attrs']['LayerElementMaterial']
	vs = np.float32(vs).reshape(-1,3)*10.0 # Vertices
	#es = np.int32(es).reshape(-1,2) # Edges
	vis = np.int32(vis) # Polys (index into vs, last edge has minus)
	faces = []
	face = []
	for vi in vis:
		if vi < 0:
			face.append(-1-vi)
			faces.append(face)
			face = []
		else: face.append(vi)
	return {'verts':vs, 'faces':faces}

def extract_animation(fbx, skel_dict):
	if skel_dict is None: return None
	objs = fbx['objs']
	nodeObjs = [(k,v) for k,v in sorted(objs.iteritems())[::-1] if v['type'] == 'AnimationCurve']
	jointIds = skel_dict['jointIds']
	num_joints = len(jointIds)
	translations = [[None,None,None] for i in jointIds]
	rotations = [[None,None,None] for i in jointIds]
	scalings = [[None,None,None] for i in jointIds] # TODO not hooked up yet
	deform_percents = [[None] for i in jointIds] # TODO not hooked up yet
	visibility = [[True] for i in jointIds] # TODO not hooked up yet
	param_map = {'Lcl Translation':translations, 'Lcl Rotation':rotations, 'Lcl Scaling':scalings, 'DeformPercent':deform_percents,
			  'Visibility':visibility}
	chan_map = {'d|X':0, 'd|Y':1, 'd|Z':2, 'd|DeformPercent':0, 'd|Visibility':0}
	for kid,kval in nodeObjs:
		try:
			curve = kval['curve']
			[(cn,pid),] = kval['pparent'].items()
			pval = fbx['objs'][pid]
			loop = set()
			while pid not in jointIds:
				if pid in loop: raise Exception('loop detected'+str(pid))
				loop.add(pid)
				while 'pparent' not in pval:
					print 'DEBUG',pval['type'],pval['name']
					pval = objs[pval['parent']]
				if len(pval['pparent'].items()) != 1:
					print 'UNEX',pval['pparent'], kval['name']
				(ppn,pid) = pval['pparent'].items()[0]
				#print ppn, # d|Y 2843320234096 Lcl Translation 2847226443280 # TODO other params here: MaxTranslation
				pval = fbx['objs'][pid]
			jid = jointIds.index(pid)
			param_map[ppn][jid][chan_map[cn]] = curve
		except Exception as E:
			print 'ERR',E
	jcs,jcss,jcns = [],[0],[]
	for L,jn,ts,rs,ro in zip(skel_dict['Ls'],skel_dict['jointNames'],translations,rotations,skel_dict['jointROs']):
		for cn,t in zip('xyz',ts):
			if t is not None:
				ci = ord(cn)-ord('x')
				jcs.append(ci)
				jcns.append(jn+':t'+cn)
				L[ci,3] = 0 # override local translation; TODO this won't work if there's a pretranslation?
		jcss.append(len(jcs))
		for cn in ro:
			ci = ord(cn)-ord('x')
			r = rs[ci]
			if r is not None:
				jcs.append(ci+3)
				jcns.append(jn+':r'+cn)
		jcss.append(len(jcs))
	skel_dict['jointChans'] = np.array(jcs, dtype=np.int32)
	skel_dict['jointChanSplits'] = np.array(jcss, dtype=np.int32)
	skel_dict['chanNames'] = jcns
	skel_dict['numChans'] = len(jcs)
	skel_dict['chanValues'] = np.zeros(len(jcs), dtype=np.float32)
	#print jcs[:10],jcss[:10],jcns[:10],skel_dict['jointParents'][:10],skel_dict['Gs'][:3]
	#print jcns
	#from pprint import pprint
	#pprint(zip(skel_dict['jointROs'],skel_dict['jointNames']))
	return {'t':translations,'r':rotations}

def extract_skeleton(fbx):
	objs = fbx['objs']
	#print set([x['type'] for x in objs.itervalues()])
	skel_types = set(['Model','Deformer','Geometry','CollectionExclusive'])
	nodeObjs = [(k,v.get('parent',0),v) for k,v in sorted(objs.iteritems())[::-1] if v['type'] in skel_types]
	def depth_first_children(js, parent):
		for jid,pid,jval in nodeObjs:
			if pid != parent: continue
			js.append((jid,pid,jval))
			depth_first_children(js, jid)
		return js
	js = depth_first_children([], 0)
	if js == []: return None
	jids,jps,jvals = zip(*js)
	jointIds = list(jids)
	jointParents = [jointIds.index(pid) if pid != 0 else -1 for pid in jps]
	jointGeos = map(extract_geo, jvals)
	jointNames = [v['name'] for v in jvals]
	jointTypes = [v['type'] for v in jvals]
	#for jt,jv in zip(jointTypes, jvals):
	#	if jt == 'Deformer':
	#		from pprint import pprint
	#		pprint(jv)
	#exit()
	numJoints = len(jointIds)
	jointChans = []
	jointChanSplits = np.zeros(len(jointNames)*2+1, dtype=np.int32)
	dofNames = []
	numDofs = 0
	is_bone = [v.get('class','')=='LimbNode' for v in jvals]
	ros = [v['attrs'].get('RotationOrder',0) for v in jvals]
	rs = [v['attrs'].get('PreRotation',[0,0,0]) for v in jvals]
	r2s = [v['attrs'].get('Lcl Rotation',[0,0,0]) for v in jvals]
	ts = [v['attrs'].get('Lcl Translation',[0,0,0]) for v in jvals]
	Ls = np.zeros((numJoints,3,4), dtype=np.float32)
	for r,r2,ro,t,L in zip(rs,r2s,ros,ts,Ls):
		L[:3,:3] = mobu_rmat(r) #np.dot(mobu_rmat(r),mobu_rmat(r2,ro))
		L[:,3] = np.float32(t)*10.0
	#	if n.has_key('Transform'):
	#		L[:3,:4] = n['TransformLink'].reshape(4,4)[:4,:3].T * [1,1,1,10]
	Gs = np.zeros((numJoints,3,4), dtype=np.float32)
	for p,L,G,n in zip(jointParents,Ls,Gs,jvals):
		if n['attrs'].has_key('TransformLink'): # TransformLink is the global matrix of the bone at bind time # TODO this is not right
			TL_new = n['attrs']['TransformLink'].reshape(4,4)[:4,:3].T * [1,1,1,10]
			T_new = n['attrs']['Transform'].reshape(4,4)[:4,:3].T * [1,1,1,10]
			assert np.allclose(matrix_mult(T_new,TL_new), np.eye(3,4))
			L[:] = matrix_mult(matrix_inverse(Gs[p]),TL_new)
		#if n['attrs'].has_key('Transform'): # Transform is the global matrix of the mesh at bind time
		#	L[:] = matrix_mult(L,n['attrs']['Transform'].reshape(4,4)[:4,:3].T * [1,1,1,10])
		G[:] = matrix_mult(Gs[p], L) if p != -1 else L
	Bs = Ls[:,:,3].copy()
	return { 'name'           : 'skel',
			 'jointIds'       : jointIds,
			 'jointTypes'     : jointTypes,
			 'numJoints'      : numJoints,
			 'jointNames'     : jointNames,  # list of strings
			 'jointIndex'     : dict(zip(jointNames,range(numJoints))), # dict of string:int
			 'jointParents'   : np.int32(jointParents),
			 'jointChans'     : np.int32(jointChans), # 0 to 5 : tx,ty,tz,rx,ry,rz
			 'jointChanSplits': np.int32(jointChanSplits),
			 'chanNames'      : dofNames,   # list of strings
			 'chanValues'     : np.zeros(numDofs,dtype=np.float32),
			 'numChans'       : int(numDofs),
			 'Bs'             : Bs,
			 'Ls'             : Ls,
			 'Gs'             : Gs,
			 'jointROs'       : [['xyz','xzy','yzx','yxz','zxy','zyx'][ro] for ro in ros],
			 'jointGeos'      : jointGeos,
			 'isBone'         : is_bone
			}

def skelDictToMesh(skelDict):
	Vs, Bs, Ts, Names, Faces = [], [], [], [], []
	if skelDict is None: return dict(names=Names,verts=Vs,faces=Faces,bones=Bs,transforms=Ts)
	for ji, jn in enumerate(skelDict['jointIds']):
		bs = [[0,0,0]]
		children_inds = np.where(skelDict['jointParents'] == ji)[0]
		if skelDict['isBone'][ji]:
			for ci in children_inds:
				bs.append(skelDict['Bs'][ci])
		Vs.append(bs)
		Bs.append([(0,i) for i in range(1,len(bs))])
		if len(children_inds) == 0: Bs[-1].append((0,0)) # TODO is this a workaround for a bug in GLMeshes?
		Ts.append(skelDict['Gs'][ji])
		Faces.append([])
		Names.append('/fbx/'+str(jn))
		geo = skelDict['jointGeos'][ji]
		if geo is not None:
			offset = len(bs)
			Vs[-1] = list(geo['verts']) + list(np.int32(Vs[-1])+offset)
			Bs[-1] = list(np.int32(Bs[-1])+offset)
			if 'faces' in geo: Faces[-1].extend(list(geo['faces']))
			if 'edges' in geo: Bs[-1].extend(list(geo['edges']))
	return dict(names=Names,verts=Vs,faces=Faces,bones=Bs,transforms=Ts)

def skel_to_nodes(skel_dict):
	ret = []
	if skel_dict is None: return ret
	jointNames = skel_dict['jointNames']
	jointIds = skel_dict['jointIds']
	jointParents = skel_dict['jointParents']
	jointTypes = skel_dict['jointTypes']
	for ji,(n,p,c) in enumerate(zip(jointNames,jointParents,list(jointParents[1:])+[-1])):
		jid = jointIds[ji]
		ps = '' if p == -1 else ' '*(ret[p][0].index('-'))
		ret.append((ps+(' -+' if c==ji else ' -')+n+':'+jointTypes[ji],'/fbx/'+str(jid)))
	return ret

def set_frame_CB(fi):
	view = QApp.view()
	skel_mesh = view.getLayer('skel')
	global g_anim_dict, g_skel_dict
	t = g_anim_dict['t']
	r = g_anim_dict['r']
	chan_values = g_skel_dict['chanValues']
	jcs = g_skel_dict['jointChans']
	jcss = g_skel_dict['jointChanSplits']
	num_joints = g_skel_dict['numJoints']
	anim = []
	time_sec = fi / 120. # TODO time range, fps
	for ji in range(num_joints):
		for ti in range(jcss[2*ji],jcss[2*ji+1]):
			anim.append(sample(t[ji][jcs[ti]], time_sec)*10.0)
		for ri in range(jcss[2*ji+1],jcss[2*ji+2]):
			anim.append(np.radians(sample(r[ji][jcs[ri]-3], time_sec)))
	#print ji,anim[:10]
	g_skel_dict['chanValues'][:] = anim
	from GCore import Character
	Character.updatePoseAndMeshes(g_skel_dict, skel_mesh, None)
	#print g_skel_dict['Gs'][:3]
	view.updateGL()

def pickedCB(view,data,clearSelection=True):
	print 'pickedCB',view
	print data
	print clearSelection
	if data is None:
		QApp.app.select(None)
	else:
		primitive_type,pn,pi,distance = data
		if primitive_type is '3d':
			p = view.primitives[pn]
			if isinstance(p,GLMeshes):
				global g_skel_dict
				name = str(g_skel_dict['jointIds'][pi])
				print "Picked:", name
				QApp.app.select('/fbx/'+name)

if __name__ == '__main__':
	import sys
	from GCore import State
	from UI import GLMeshes
	payload = load(sys.argv[1])
	#from pprint import pprint; pprint(payload, open('fbx.txt','w'))
	
	fbx = decode(payload)
	IO.save('fbx.tmp',fbx)
	skel_dict = extract_skeleton(fbx)
	anim_dict = extract_animation(fbx, skel_dict)
	#print skel_dict['jointNames']
	skel_mesh = GLMeshes(**skelDictToMesh(skel_dict))

	global g_anim_dict, g_skel_dict
	g_anim_dict = anim_dict
	g_skel_dict = skel_dict

	#State.setKey('/doc',payload)

	#fbx = decode(payload)
	#State.setKey('/fbx',fbx)
	#for k,v in fbx.iteritems(): State.setKey(k,v)
	
	from UI import QApp, QGLViewer, GLMeshes, GLPoints3D
	app,win = QGLViewer.makeApp(appName='Imaginarium FBX')
	#outliner = QApp.app.qobjects
	#for gi,(key,value) in enumerate(fbx.items()):
	#	outliner.addItem(str(key)+'='+repr(value)[:200], data=str(key), index=gi)

	display_nodes = skel_to_nodes(skel_dict)
	#print zip(*display_nodes)[1]
	#for gi,(disp,key) in enumerate(display_nodes): outliner.addItem(disp, data='_OBJ_'+key, index=gi)
	#global g_fields
	QApp.fields = dict([(k,sorted(v.items())) for k,v in g_fields.iteritems()])

	for gi,(k,v) in enumerate(fbx['objs'].items()): State.setKey('/fbx/'+str(k),v)
	QApp.app.qoutliner.set_root('/fbx')
	#for gi,key in enumerate(sorted(State.allKeys())): outliner.addItem(key+'='+repr(State.getKey(key))[:80], data=key, index=gi)

	QGLViewer.makeViewer(timeRange=(0,8000), callback=set_frame_CB, layers={'skel':skel_mesh}, pickCallback=pickedCB) #, dragCallback=dragCB, keyCallback=keyCB, drawCallback=drawGL_cb, layers=layers)
