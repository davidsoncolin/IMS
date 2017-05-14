import numpy as np
import ISCV

global g_all_skels, g_joint_index, g_src_anim
g_all_skels = []
g_joint_index = 19
g_src_anim = None

'''usage:
	skels,nodes = Character.mayaToSkelDicts(ma_filename)
	all_skels = []
	gl_primitives = []
	for skel_dict in skels:
		skel_mesh = GLMeshes(**Character.skelDictToMesh(skel_dict))
		geom_mesh = GLMeshes(**skel_dict['geom_dict'])
		gl_primitives.append(skel_mesh)
		gl_primitives.append(geom_mesh)
		all_skels.append((skel_dict,skel_mesh,geom_mesh))
	...
	for skel_dict, skel_mesh, geom_mesh in all_skels:
		new_chans = ...
		Character.updatePoseAndMeshes(skel_dict, skel_mesh, geom_mesh, new_chans)
'''

def addGeomDictToSkelDict(skel_dict, geom_dict, shape_weights = None):
	'''integrate skinning geometry data into a skel_dict.'''
	skel_dict['geom_dict'] = geom_dict
	if shape_weights is not None:
		skel_dict['shape_weights'] = [shape_weights[name] for name in geom_dict['names']]
	vsplits = np.array(map(len, geom_dict['verts']),dtype=np.int32)
	vsplits = np.array([np.sum(vsplits[:i]) for i in xrange(len(vsplits)+1)],dtype=np.int32)
	skel_dict['geom_Vs'] = np.zeros((vsplits[-1],3),dtype=np.float32)
	skel_dict['geom_Gs'] = np.array([np.eye(3,4) for x in geom_dict['verts']],dtype=np.float32)
	skel_dict['geom_vsplits'] = vsplits # useful to have around

def pose_skeleton_with_chan_mats(chan_mats, Gs, skel_dict, cvs=None, x_mat=None):
	Xm = skel_dict.get('rootMat',np.eye(3, 4, dtype=np.float32)) if x_mat is None else x_mat
	if cvs is None: cvs = skel_dict['chanValues']
	ISCV.pose_skeleton_with_chan_mats(chan_mats, Gs, skel_dict['Ls'], skel_dict['jointParents'], skel_dict['jointChans'], skel_dict['jointChanSplits'], cvs, Xm)

def pose_skeleton(Gs, skel_dict, cvs=None, x_mat=None):
	Xm = skel_dict.get('rootMat',np.eye(3, 4, dtype=np.float32)) if x_mat is None else x_mat
	if cvs is None: cvs = skel_dict['chanValues']
	ISCV.pose_skeleton(Gs, skel_dict['Ls'], skel_dict['jointParents'], skel_dict['jointChans'], skel_dict['jointChanSplits'], cvs, Xm)

def matrix_mult(p,c):
	ret = np.dot(p[:3,:3],c[:3,:])
	ret[:,3] += p[:3,3]
	return ret

def updatePose(skel_dict, cvs=None, x_mat=None):
	'''update the pose of a skel_dict.'''
	pose_skeleton(skel_dict['Gs'], skel_dict, cvs, x_mat)
	if 'geom_dict' in skel_dict and 'shape_weights' in skel_dict:
		geom_dict = skel_dict['geom_dict']
		Vs = skel_dict['geom_Vs']
		vsplits = skel_dict['geom_vsplits']
		Vs[:] = 0
		for si,(smats,joint_list) in enumerate(skel_dict['shape_weights']):
			if smats is None: # this is just a geometry; joint_list should be a singleton
				for joint_name, bind_mat in joint_list.iteritems():
					ji = skel_dict['jointIndex'].get(joint_name,-1)
					if ji == -1: print 'warn'; continue
					mat = skel_dict['Gs'][ji]
					matrix_mult(mat,bind_mat)
					skel_dict['geom_Gs'][si,:3,:] = mat
					Vs[vsplits[si]:vsplits[si+1]] = geom_dict['verts'][si] # the original values
			else:
				Vs_si = Vs[vsplits[si]:] # a view on the vertices
				for joint_name, mi in joint_list.iteritems():
					ji = skel_dict['jointIndex'].get(joint_name,-1)
					if ji == -1 or mi not in smats:
						print 'wat?',ji, joint_name, mi, mi in smats, skel_dict['jointIndex'].keys();
						continue
					mat = skel_dict['Gs'][ji]
					vis,vs = smats[mi]
					Vs_si[vis] += np.dot(vs, (mat*[10,10,10,1]).T) # TODO fix the scaling in MAReader.evaluate_skinClusters

def updatePoseAndMeshes(skel_dict, skel_mesh, geom_mesh, cvs=None, x_mat=None):
	'''update the pose of a skel_dict and the meshes.'''
	updatePose(skel_dict, cvs, x_mat)
	skel_mesh.setPose(skel_dict['Gs'])
	if 'geom_dict' in skel_dict:
		geom_mesh.setVs(skel_dict['geom_Vs'])
		geom_mesh.setPose(skel_dict['geom_Gs'])

def make_geos(skel_dict, x_mat=None):
	from UI.GLMeshes import GLMeshes
	mesh_dict = skelDictToMesh(skel_dict)
	skel_mesh = GLMeshes(**mesh_dict)
	geom_mesh = GLMeshes(**skel_dict['geom_dict'])
	updatePoseAndMeshes(skel_dict, skel_mesh, geom_mesh, skel_dict['chanValues'], x_mat)
	return (mesh_dict, skel_mesh, geom_mesh)
		
def shape_weights_mapping(skel_dict, tgt_skel_dict, name_mapping):
	jointIndex,Gs,jointParents = skel_dict['jointIndex'],skel_dict['Gs'],skel_dict['jointParents']
	tgt_jointIndex,tgt_Gs,tgt_jointParents = tgt_skel_dict['jointIndex'],tgt_skel_dict['Gs'],tgt_skel_dict['jointParents']
	jointNames = skel_dict['jointNames']
	joint_mapping = [tgt_jointIndex[name_mapping[jointNames[ji]]] for ji in xrange(len(jointNames))]
	bones = [np.array([1,0,0],dtype=np.float32) for x in jointParents]
	bones2 = [np.array([1,0,0],dtype=np.float32) for x in jointParents]
	for ji,jpi in enumerate(jointParents):
		tji,tjpi = joint_mapping[ji],joint_mapping[jpi]
		tmp = Gs[ji,:,3] - Gs[jpi,:,3]
		tmp2 = tgt_Gs[tji,:,3] - tgt_Gs[tjpi,:,3]
		if np.linalg.norm(tmp) > 0 and np.linalg.norm(tmp2) > 0:
			bones[jpi] = tmp
			bones2[jpi] = tmp2
	#for ji in xrange(len(jointParents)):
		#if len(np.where(jointParents == ji)[0]) != 1: bones[ji] = bones2[ji] = np.array([1,0,0],dtype=np.float32)
	#for tji in xrange(len(tgt_jointParents)):
		#if tji not in joint_mapping: continue
		#ji = joint_mapping.index(tji)
		#if len(np.where(tgt_jointParents == tji)[0]) != 1: bones[ji] = bones2[ji] = np.array([1,0,0],dtype=np.float32)
	from GCore import Retarget
	rots = [Retarget.rotateBetweenVectors(b,b2) for b,b2 in zip(bones,bones2)] #* (np.linalg.norm(b2)/np.linalg.norm(b))
	
	Vs = np.zeros((skel_dict['geom_Vs'].shape[0],4),dtype=np.float32)
	tgt_shape_weights = []
	for si,(smats,joint_list) in enumerate(skel_dict['shape_weights']):
		tgt_smats, tgt_joint_list = {},{}
		if smats is None: continue
		for joint_name, mi in joint_list.iteritems():
			ji = jointIndex.get(joint_name,-1)
			if ji == -1 or mi not in smats: continue
			tgt_joint_name = name_mapping[joint_name]
			tgt_joint_list.setdefault(tgt_joint_name,[]).append(mi)
			tji = tgt_jointIndex[tgt_joint_name]
			R = np.eye(4,4,dtype=np.float32)
			R[:3,:3] = np.dot(np.linalg.inv(tgt_Gs[tji,:3,:3]),np.dot(rots[ji],Gs[ji,:3,:3]))
			vis,vs = smats[mi]
			vs = np.dot(vs,R.T)
			assert mi not in tgt_smats
			tgt_smats[mi] = (vis,vs)
		for jn,mis in tgt_joint_list.iteritems():
			if len(mis) > 1:
				# multiple joints map to the a single joint. we must be careful, because vertices could be duplicated
				Vs[:] = 0
				root = int(np.min(mis))
				root_joint_name = joint_list.keys()[joint_list.values().index(root)]
				pji = jointIndex[root_joint_name]
				tji = tgt_jointIndex[name_mapping[root_joint_name]]
				M = np.linalg.inv(10*tgt_Gs[tji,:3,:3])
				for mi in mis:
					joint_name = joint_list.keys()[joint_list.values().index(mi)]
					vis,vs = tgt_smats.pop(mi)
					Vs[vis] += vs
					ji = jointIndex[joint_name]
					Vs[vis,:3] += np.dot(M,(Gs[ji,:,3] - Gs[pji,:,3]))*(vs[:,3].reshape(-1,1))
				# now find all the non-zero values
				vis = np.nonzero(Vs[:,3])[0]
				vs = Vs[vis]
				mis = [root]
				tgt_smats[mis[0]] = (vis,vs)
			tgt_joint_list[jn] = int(mis[0])
		tgt_shape_weights.append((tgt_smats,tgt_joint_list))
	return tgt_shape_weights

def skelDictToMesh(skelDict):
	Vs, Bs, Ts, Names, Faces = [], [], [], [], []
	for ji, joint in enumerate(skelDict['jointNames']):
		bs = [[0,0,0]]
		children_inds = np.where(skelDict['jointParents'] == ji)[0]
		for ci in children_inds: bs.append(skelDict['Ls'][ci][:,3])
		Vs.append(bs)
		Bs.append([(0,i) for i in range(1,len(bs))])
		Ts.append(skelDict['Gs'][ji])
		Faces.append([])
		Names.append(joint)
	return dict(names=Names,verts=Vs,faces=Faces,bones=Bs,transforms=Ts)

def makeGeomDict(mesh_data, shape_weights):
	geomInd = [ni for ni,name in enumerate(mesh_data['names']) if name in shape_weights]
	Vs, Bs, Ts, Names, Faces, Vts, Fts = [], [], [], [], [], [], []
	for gi in geomInd:
		Vs.append(np.array(mesh_data['verts'][gi],dtype=np.float32))
		Bs.append(mesh_data['bones'][gi])
		Ts.append(mesh_data['transforms'][gi])
		Names.append(mesh_data['names'][gi])
		Faces.append(mesh_data['faces'][gi])
		Vts.append(mesh_data['vts'][gi])
		Fts.append(mesh_data['fts'][gi])
	geom_dict = dict(names=Names,verts=Vs,faces=Faces,bones=Bs,transforms=Ts,vts=Vts,fts=Fts)
	return geom_dict

def mayaToSkelDicts(ma_filename):
	from IO import MAReader
	nodes,nodeLists = MAReader.read_MA(ma_filename)
	MAReader.evaluate_scenegraph(nodeLists)
	ma_primitives,ma_primitives2D,mats,camera_ids,movies = MAReader.construct_geos(nodeLists)
	grip_dict = MAReader.extract_GRIP(nodeLists)
	shape_weights = grip_dict['shape_weights']
	skels = grip_dict['skels']
	for skel_dict in skels:
		geom_dict = makeGeomDict(ma_primitives[0], shape_weights) # TODO this can't be right, shape_weights should be per skel
		addGeomDictToSkelDict(skel_dict, geom_dict, shape_weights)
	return skels,nodes

class Character:
	Type = 'Character'
	def __init__(self, skeleton = None, geometry = None):
		from UI.GLMeshes import GLMeshes
		self.state_data = {'type':self.Type,'attrs':{}}
		self.state_data['attrs']['skeleton'] = skeleton
		self.state_data['attrs']['geometry'] = geometry
		if skeleton is not None and 'skel_dict' in skeleton:
			self.scale = 1.
			self.skel_dict = skeleton['skel_dict']
			if 'primitive' in skeleton:
				self.skel_primitive = skeleton['primitive']
				self.state_data['attrs']['skeleton'].pop('primitive')
			else:
				self.skel_primitive = GLMeshes(**skelDictToMesh(self.skel_dict))
			self.anim_dict = skeleton['anim_dict'] if 'anim_dict' in skeleton else None
		if geometry is None or 'geom_dict' not in geometry:
			assert False, "No Geometry entered or incorrectly formatted data. Geometry should be a dict with a mandatory key \"geom_dict\" and optional key \"shape_weights\""
		self.geom_dict = geometry['geom_dict']
		self.geom_mesh = GLMeshes(**self.geom_dict)
		self.has_skin_weights = geometry.has_key('shape_weights')
		addGeomDictToSkelDict(self.skel_dict, self.geom_dict, shape_weights = geometry.get('shape_weights',None))

	def setJointChanValues(self, joint_chan_values, indices=None):
		if indices is not None:
			self.skel_dict['chanValues'][indices] = joint_chan_values
		else:
			self.skel_dict['chanValues'][:] = joint_chan_values

	def updatePose(self, cvs=None, x_mat=None, geo=True):
		if geo and self.has_skin_weights and self.geom_mesh.visible:
			updatePoseAndMeshes(self.skel_dict, self.skel_primitive, self.geom_mesh, cvs, x_mat)
		else:
			pose_skeleton(self.skel_dict['Gs'], self.skel_dict, cvs, x_mat)
			self.skel_primitive.setPose(self.skel_dict['Gs'])

	def scaleCharacter(self, scale):
		self.skel_dict['Ls'][0,:,:] = scale * self.skel_dict['Ls'][0,:,:] / self.scale
		self.scale = scale
		self.updatePose()

def setFrameCB(new_frame):
	global g_all_skels, g_joint_index, g_src_anim
	if new_frame == 1: g_joint_index = g_joint_index + 1
	for skel_dict,skel_mesh,geom_mesh in g_all_skels:
		num_chans = skel_dict['numChans']
		ji = g_joint_index % num_chans
		new_chans = np.zeros(num_chans, dtype=np.float32)
		if g_src_anim is None:
			angle = new_frame * np.pi/100.
			new_chans[ji] = np.pi * np.sin(angle) / 2
		else:
			new_chans[:] = g_src_anim[new_frame%len(g_src_anim)]
		updatePoseAndMeshes(skel_dict, skel_mesh, geom_mesh, new_chans)
	QApp.view().updateGL()

def pickedCB(view,data,clearSelection=True):
	from UI.GLMeshes import GLMeshes
	if data is None:
		QApp.app.select(None)
	else:
		primitive_type,pn,pi,distance = data
		if primitive_type is '3d':
			p = view.primitives[pn]
			if isinstance(p,GLMeshes):
				name = p.names[pi]
				print "Picked:", name
				QApp.app.select('_OBJ_'+name)

def scaleCB(char):
	from PySide.QtGui import QInputDialog
	val, ok = QInputDialog.getDouble(None, "Set Scale", "Scale: ", 1.0, 0.1, 100.0, 2)
	if ok: char.scaleCharacter(val)

if __name__ == '__main__':
	import os, sys
	from IO import MAReader
	import IO
	from UI import QApp
	from PySide import QtGui
	from UI import QGLViewer
	from UI.GLMeshes import GLMeshes
	global g_all_skels, g_joint_index, g_src_anim
	ma_filename = sys.argv[1]
	skels,nodes = mayaToSkelDicts(ma_filename)

	gl_primitives = []
	for skel_dict in skels:
		skel_mesh = GLMeshes(**skelDictToMesh(skel_dict))
		geom_mesh = GLMeshes(**skel_dict['geom_dict'])
		gl_primitives.append(skel_mesh)
		gl_primitives.append(geom_mesh)
		g_all_skels.append((skel_dict,skel_mesh,geom_mesh))
		#char = Character(skeleton={'primitive':skel_mesh,'skel_dict':skel_dict}, geometry={'geom_dict':geom_dict,'shape_weights':shape_weights})

	appIn = QtGui.QApplication(sys.argv)
	appIn.setStyle('plastique')
	win = QApp.QApp()
	win.setWindowTitle('Imaginarium Character Demo')
	QApp.fields, dnodes = MAReader.maya_to_state(nodes)
	#outliner = QApp.app.qobjects
	#for i,(o,v) in enumerate(dnodes): outliner.addItem(o, data='_OBJ_'+v['name'], index=i)

	#win.addMenuItem({'menu':'&Scale','item':'&Scale Character','tip':'Scale the Character','cmd':scaleCB,'args':[char]})

	timeRange=(1,100) if g_src_anim is None else (1,len(g_src_anim)-1)
	QGLViewer.makeViewer(primitives=gl_primitives, timeRange=timeRange, callback=setFrameCB,
						pickCallback=pickedCB, appIn=appIn, win=win)


