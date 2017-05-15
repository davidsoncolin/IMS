#!/usr/bin/env python

import os,sys
import struct
import numpy as np
import IO
import ISCV
from GCore import Calibrate, Character

def unpackGreyScaleLine(s):
	l = IO.unpack_from('<H',s,0)[0]
	offset,eos = 2,len(s)
	ret = []
	for x in xrange(l):
		t1,offset = IO.unpack_from('<H',s,offset)[0], offset+2
		t2,offset = IO.unpack_from('<H',s,offset)[0], offset+2
		l = IO.unpack_from('<H',s,offset)[0]
		t3,offset = s[offset+2:offset+2+l], offset+2+l
		ret.append((t1,t2,t3))
	tmp,offset = IO.unpack_from('<i',s,offset)
	#assert(tmp == 0) # I think this is padding
	assert(offset == eos)
	ret.append(tmp)
	return ret

def loadX2D(fn):
	'''Parse an x2d file into a structure of some sort'''
	data = open(fn,'rb').read()
	return ISCV.decode_X2D(data)

def extractCameraInfo(x2d):
	'''Returns for each camera the Width,Height and ID.'''
	field = x2d['header'][11]
	#assert(field[0] == 0)
	return np.array( [x[:3] for x in field[-1]], dtype=np.int32)

def frameCentroidsToDets(frame, mats=None):
	'''Extract the centroids for given cameras and undistorts them. Returns a list of x2ds and splits per camera.'''
	x2ds_raw_data, x2ds_splits = frame[0],frame[1]
	if mats is None: return x2ds_raw_data[:,:2].copy(), x2ds_splits
	return Calibrate.undistort_dets(x2ds_raw_data, x2ds_splits, mats)

#def saveX2D(fn, payload):
#	open(fn,'wb').write(encodeX2D(payload))

def loadVSS(fn):
	'''Decode a Vicon Skeleton file (VST format). VSK is labeling skeleton. VSS is solving skeleton.'''
	import xml.etree.cElementTree as ET
	import numpy as np
	dom = ET.parse(fn)
	parameters = dom.findall('Parameters')[0]
	params = dict([(p.get('NAME'),p.get('VALUE')) for p in parameters])
	sticks = dom.findall('MarkerSet')[0].find('Sticks')
	sticksPairs = [(x.get('MARKER1'),x.get('MARKER2')) for x in sticks]
	sticksColour= [np.fromstring(x.get('RGB1', '255 255 255'), dtype=np.uint8, sep=' ') for x in sticks]
	hasTargetSet = True
	try: markers = dom.findall('TargetSet')[0].find('Targets')
	except: markers = dom.findall('MarkerSet')[0].find('Markers'); hasTargetSet = False
	markerOffsets = [x.get('POSITION').split() for x in markers]
	def ev(x,params):
		for k,v in params.items(): x = x.replace(k,v)
		return float(x) # eval(x)
	markerOffsets = [[ev(x,params) for x in mp] for mp in markerOffsets]
	markerColour= [np.fromstring(col, dtype=np.uint8, sep=' ') for col in \
						[x.get('MARKER', x.get('RGB')) for x in dom.findall('MarkerSet')[0].find('Markers')]]
	colouredMarkers = [x.get('MARKER', x.get('NAME')) for x in dom.findall('MarkerSet')[0].find('Markers')]
	markerNames = [x.get('MARKER', x.get('NAME')) for x in markers]
	markerWeights = [float(x.get('WEIGHT')) if hasTargetSet else 1.0 for x in markers]
	markerParents = [x.get('SEGMENT') for x in markers]
	skeleton = dom.findall('Skeleton')[0]
	# skeleton is defined as a tree of Segments
	# Segment contains Joint and Segment
	# Joint is JointDummy(0)/JointHinge(1)/JointHardySpicer(2)/JointBall(3)/JointFree(6), containing JointTemplate
	def ap(skeleton, parent, skel):
		for seg in skeleton:
			if seg.tag == 'Segment':
				skel.append([seg.get('NAME'),parent,seg.attrib])
				ap(seg, len(skel)-1, skel)
			else:
				skel[parent].extend([seg.tag,seg.attrib,{} if len(seg) == 0 else seg[0].attrib])
		return skel
	# recursively parse the skeleton
	root = ap(skeleton, -1, [])
	assert(len(markerParents) == len(markerOffsets))
	def cqToR(rs, R):
		'''Given a compressed quaternion, form a 3x3 rotation matrix.'''
		angle = np.dot(rs,rs)**0.5
		scale = (np.sin(angle*0.5)/angle if angle > 1e-8 else 0.5)
		q = np.array([rs[0]*scale,rs[1]*scale,rs[2]*scale,np.cos(angle*0.5)], dtype=np.float32)
		q = np.outer(q, q)*2
		R[:3,:3] = [
			[1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3]],
			[    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3]],
			[    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1]]]
	def float3(x): return np.array(map(lambda x:ev(x,params), x.split()),dtype=np.float32)
	def mats(x):
		preT = x.get('PRE-POSITION', '0 0 0')
		postT = x.get('POST-POSITION', '0 0 0')
		preR = x.get('PRE-ORIENTATION', '0 0 0')
		postR = x.get('POST-ORIENTATION', '0 0 0')
		pre = np.zeros((3,4),dtype=np.float32)
		post = np.zeros((3,4),dtype=np.float32)
		pre[:,3] = float3(preT)
		post[:,3] = float3(postT)
		cqToR(float3(preR), pre[:3,:3])
		cqToR(float3(postR), post[:3,:3])
		return pre,post
	name = fn.rpartition('/')[2].partition('.')[0]
	numBones = len(root)
	jointNames = [r[0] for r in root]
	markerParents = np.array([jointNames.index(mp) for mp in markerParents],dtype=np.int32)
	jointNames[0] = 'root' # !!!! WARNING !!!!
	jointParents = [r[1] for r in root]
	jointData = [mats(r[4]) for r in root]
	jointTypes = [r[3] for r in root] # JointDummy(0)/JointHinge(1)/JointHardySpicer(2)/JointBall(3)/JointFree(6)
	#jointTemplates = [mats(r[5]) for r in root] # JointTemplate ... contains the same data as jointTypes
	jointAxes = [r[4].get('AXIS',r[4].get('AXIS-PAIR',r[4].get('EULER-ORDER','XYZ'))) for r in root] # order
	jointTs = [r[4].get('T',None) for r in root]
	Gs = np.zeros((numBones,3,4),dtype=np.float32) # GLOBAL mats
	Ls = np.zeros((numBones,3,4),dtype=np.float32) # LOCAL mats
	Bs = np.zeros((numBones,3),dtype=np.float32) # BONES
	for ji,pi in enumerate(jointParents):
		if pi == -1: Ls[ji] = jointData[ji][0]
		else: np.dot(jointData[pi][1][:,:3],jointData[ji][0],out=Ls[ji]); Ls[ji,:,3] += jointData[pi][1][:,3]
	dofNames = []
	jointChans = [] # tx=0,ty,tz,rx,ry,rz
	jointChanSplits = [0]
	# TODO: locked channels
	for ji,(jt,T) in enumerate(zip(jointTypes,jointTs)):
		jointChanSplits.append(len(jointChans))
		if jt == 'JointDummy': assert(T is None)
		elif jt == 'JointHinge':
			assert(T == '* ')
			jointChans.append(jointAxes[ji].split().index('1')+3)
		elif jt == 'JointHardySpicer':
			assert(T == '* * ')
			ja = jointAxes[ji].split()
			jointChans.append(ja.index('1',3))
			jointChans.append(ja.index('1')+3)
		elif jt == 'JointBall':
			assert(T == '* * * ')
			ja = jointAxes[ji]
			jointChans.append(ord(ja[0])-ord('X')+3)
			jointChans.append(ord(ja[1])-ord('X')+3)
			jointChans.append(ord(ja[2])-ord('X')+3)
		elif jt == 'JointFree':
			assert(T == '* * * * * * ' or T is None) # version 1 of the file apparently doesn't fill this!
			ja = jointAxes[ji]
			jointChans.append(0)
			jointChans.append(1)
			jointChans.append(2)
			jointChanSplits[-1] = len(jointChans)
			jointChans.append(ord(ja[0])-ord('X')+3)
			jointChans.append(ord(ja[1])-ord('X')+3)
			jointChans.append(ord(ja[2])-ord('X')+3)
		for jc in jointChans[jointChanSplits[-2]:]:
			dofNames.append(jointNames[ji]+':'+'tx ty tz rx ry rz'.split()[jc])
		jointChanSplits.append(len(jointChans))
	numDofs = len(dofNames)
	# fill Gs
	chanValues = np.zeros(numDofs,dtype=np.float32)
	rootMat = np.eye(3, 4, dtype=np.float32)

	# fill Bs; TODO add dummy joints to store the extra bones (where multiple joints have the same parent)
	for ji,pi in enumerate(jointParents):
		if pi != -1: Bs[pi] = Ls[ji,:,3]
	Bs[np.where(Bs*Bs<0.01)] = 0 # zero out bones < 0.1mm

	# TODO: compare skeleton with ASF exported version
	skel_dict = {
			'markerOffsets'  : np.array(markerOffsets, dtype=np.float32),
			'markerParents'  : markerParents,
			'markerNames'    : markerNames,
			'markerNamesUnq' : colouredMarkers,
			'markerColour'   : markerColour,
			'markerWeights'  : np.array(markerWeights,dtype=np.float32),
			'numMarkers'     : len(markerNames),
			'sticks'         : sticksPairs,
			'sticksColour'   : sticksColour,
			'name'           : str(name),
			'numJoints'      : int(numBones),
			'jointNames'     : jointNames,  # list of strings
			'jointIndex'     : dict([(k,v) for v,k in enumerate(jointNames)]), # dict of string:int
			'jointParents'   : np.array(jointParents,dtype=np.int32),
			'jointChans'     : np.array(jointChans,dtype=np.int32), # 0 to 5 : tx,ty,tz,rx,ry,rz
			'jointChanSplits': np.array(jointChanSplits,dtype=np.int32),
			'chanNames'      : dofNames,   # list of strings
			'chanValues'     : np.zeros(numDofs,dtype=np.float32),
			'numChans'       : int(numDofs),
			'Bs'             : np.array(Bs, dtype=np.float32),
			'Ls'             : np.array(Ls, dtype=np.float32),
			'Gs'             : np.array(Gs, dtype=np.float32),
			'rootMat'        : rootMat,
			}
	Character.pose_skeleton(skel_dict['Gs'], skel_dict)
	return skel_dict

def loadXCP(fn):
	'''Decode a calibration file into a list of dicts per camera. Decode each dict into a list of numbers in our format.
	Reorder the cameras by DEVICEID.'''
	import xml.etree.cElementTree as ET
	import numpy as np
	dom = ET.parse(fn)
	ret = [dict(y.items()+y.find('KeyFrames')[0].items()) for y in dom.findall('Camera')]
	for ci,c in enumerate(ret): c['LABEL'] = 'CAM_%02d'%(ci+1)
	units = {
			'PRINCIPAL_POINT'    : lambda x:np.array(x.split(),dtype=np.float32),
			'VICON_RADIAL'       : lambda x:np.array(x.split(),dtype=np.float32),
			'ORIENTATION'        : lambda x:np.array(x.split(),dtype=np.float32),
			'POSITION'           : lambda x:np.array(x.split(),dtype=np.float32),
			'SENSOR_SIZE'        : lambda x:np.array(x.split(),dtype=np.int32),
			'PIXEL_ASPECT_RATIO' : float,
			'ACTIVE_THRESHOLD'   : float,
			'FOCAL_LENGTH'       : float,
			'IMAGE_ERROR'        : float,
			#'SKEW'               : float,
			'FRAME'              : int,
			'DEVICEID'           : int }
	for d in ret:
		for k,v in units.items(): d[k] = v(d[k])
	mats = []
	# jig the camera to have x-coordinates in the range -1 to 1
	for d in ret:
		hsx,hsy = d['SENSOR_SIZE'][0]*0.5,d['SENSOR_SIZE'][1]*0.5
		pfl = d['FOCAL_LENGTH']
		fx = pfl / hsx
		assert(d['PIXEL_ASPECT_RATIO'] == 1.0) # if this assertion trips, please test the next line
		fy = fx / d['PIXEL_ASPECT_RATIO']
		skew = 0. #d['SKEW'] / hsx
		ox,oy = d['PRINCIPAL_POINT']
		ox = (ox - hsx) / hsx
		oy = (oy - hsy) / hsx
		# Vicon pixels are [hsx 0 hsx; 0 -hsx hsy; 0 0 -1] * ours. this scales k1 and k2
		# because Vicon looks down the positive z-axis, we want to flip the z-axis. for parity, we need to flip y too.
		# so, P = diag(1,-1,-1)*K*RT = [diag(1,-1,-1)*K*diag(1,-1,-1)]*[diag(1,-1,-1)*RT]. this puts some minus signs in K & RT.
		K = np.array([[fx, -skew, -ox],[0, fy, oy], [0, 0, 1]], dtype=np.float32)
		qx,qy,qz,qw = d['ORIENTATION']
		RT = np.array([ [qw*qw+qx*qx-qy*qy-qz*qz, 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy), 0],
						[-2*(qx*qy+qw*qz), -(qw*qw-qx*qx+qy*qy-qz*qz), -2*(qy*qz-qw*qx), 0],
						[-2*(qx*qz-qw*qy), -2*(qy*qz+qw*qx), -(qw*qw-qx*qx-qy*qy+qz*qz), 0]],dtype = np.float32)
		RT[:,3] = -np.dot(RT[:3,:3], d['POSITION'])
		k1 = d['VICON_RADIAL'][0] * hsx*hsx
		k2 = d['VICON_RADIAL'][1] * hsx*hsx*hsx*hsx
		mats.append([K,RT,np.dot(K,RT),np.array([k1,k2],dtype=np.float32),d['POSITION'],d['SENSOR_SIZE']]) # 9+12+12+2+3+2 = 40 numbers
	order = np.argsort([c['DEVICEID'] for c in ret])
	return [mats[oi] for oi in order],[ret[oi] for oi in order]

def r_to_quat(R):
	[[m00,m01,m02],[m10,m11,m12],[m20,m21,m22]] = R
	tr = m00 + m11 + m22

	if (tr > 0):
		S = (tr+1.0)**0.5 * 2.0 # S=4*qw
		qw = 0.25 * S
		qx = (m21 - m12) / S
		qy = (m02 - m20) / S
		qz = (m10 - m01) / S
	elif ((m00 > m11) and (m00 > m22)):
		S = (1.0 + m00 - m11 - m22)**0.5 * 2.0 # S=4*qx
		qw = (m21 - m12) / S
		qx = 0.25 * S
		qy = (m01 + m10) / S
		qz = (m02 + m20) / S
	elif (m11 > m22):
		S = (1.0 + m11 - m00 - m22)**0.5 * 2.0 # S=4*qy
		qw = (m02 - m20) / S
		qx = (m01 + m10) / S
		qy = 0.25 * S;
		qz = (m12 + m21) / S
	else:
		S = (1.0 + m22 - m00 - m11)**0.5 * 2.0 # S=4*qz
		qw = (m10 - m01) / S
		qx = (m02 + m20) / S
		qy = (m12 + m21) / S
		qz = 0.25 * S
	return qw,qx,qy,qz

def saveXCP(filename, mats, camera_ids):
	'''Write an XCP file, hopefully vicon will accept it. TODO missing threshold grid, rms.'''
	data = '<?xml version="1.0" encoding="UTF-8" standalone="no" ?>\r\n<Cameras NAME="" VERSION="1.0">\r\n\r\n'
	for ci,((K,RT,P,k,T,wh),camera_id) in enumerate(zip(mats,camera_ids)):
		rms = 0.2 # not recorded
		hsx,hsy = wh[0]*0.5,wh[1]*0.5
		fx = K[0,0]
		pfl = fx*hsx
		ox,oy = -K[0,2]*hsx + hsx,K[1,2]*hsx + hsy
		R = RT[:3,:3] * [[1],[-1],[-1]]
		qw,qx,qy,qz = r_to_quat(R)
		tx,ty,tz = T
		k1 = k[0]/(hsx**2)
		k2 = k[1]/(hsx**4)
		if type(camera_id) is str: camera_id = int(camera_id.rpartition(':')[-1].rpartition('_')[-1]) # don't pass a string in here!
		t160 = {'dtype':'T160','sensor':'AM63','type':'DxCore160','display':'T160','d2':'T160', 'threshold':'" r 31752 0" GRID_SIZE="294 108" TILE_SIZE="16 32"'}
		t40 = {'dtype':'T40','sensor':'AM41','type':'DxCore41', 'display':'T40-S', 'd2':'T40_S', 'threshold':'" r 31536 0" GRID_SIZE="292 108" TILE_SIZE="8 16"'}
		info = {'id':camera_id,'uid':ci+1,'w':wh[0],'h':wh[1],'fl':pfl,'rms':rms,'qx':qx,'qy':qy,'qz':qz,'qw':qw,'tx':tx,'ty':ty,'tz':tz,'ox':ox,'oy':oy,'k1':k1,'k2':k2}
		if wh[0] == 4704 and wh[1] == 3456: info.update(t160)
		elif wh[0] == 2336 and wh[1] == 1728: info.update(t40)
		else: print ('WARNING: unknown resolution ',wh)
		data += '  <Camera ACTIVE_THRESHOLD="1" DEVICEID="{id}" DISPLAY_TYPE="{display}" ISDV="0" NAME="{d2}_{id}" PIXEL_ASPECT_RATIO="1" SENSOR="{sensor}" SENSOR_SIZE="{w} {h}" SKEW="0" SYSTEM="" TYPE="{type}" USERID="{uid}">\r\n    <ThresholdGrid BIT_DEPTH="1" DATA={threshold}/>\r\n    <ControlFrames/>\r\n    <KeyFrames>\r\n'.format(**info)
		data += '      <KeyFrame FOCAL_LENGTH="{fl}" FRAME="0" IMAGE_ERROR="{rms}" ORIENTATION="{qx} {qy} {qz} {qw}" POSITION="{tx} {ty} {tz}" PRINCIPAL_POINT="{ox} {oy}" VICON_RADIAL="{k1} {k2}"/>\r\n'.format(**info)
		data += '    </KeyFrames>\r\n  </Camera>\r\n\r\n'
	data += '</Cameras>\r\n'
	open(filename,'wb').write(data)

def load_xcp_and_x2d(xcp_filename, x2d_filename, raw=False):
	'''Load an x2d, xcp pair and make a valid data structure.
	The returned cameras are in the order of the x2d file -- as it happens, this is in order of deviceid.
	If any particular camera is not in the xcp then it's initialised on the positive x-axis.
	If a camera is only in the xcp then it is discarded.'''

	from GCore import Calibrate

	print ('loading xcp')
	vicon_mats,xcp_data = loadXCP(xcp_filename)
	xcp_camera_ids = np.array([int(x['DEVICEID']) for x in xcp_data],dtype=np.int32)
	camera_names = ['%s:%s'%(x['LABEL'],x['DEVICEID']) for x in xcp_data]
	camera_vicon_errors = np.array([x['IMAGE_ERROR'] for x in xcp_data],dtype=np.float32)
	#print (camera_names)
	#print ('vicon_errors',camera_vicon_errors,np.min(camera_vicon_errors),np.max(camera_vicon_errors),np.mean(camera_vicon_errors))
	print ('loading x2d',x2d_filename)
	x2d_dict = loadX2D(x2d_filename)
	cameras_info = extractCameraInfo(x2d_dict) # W,H,ID per camera
	x2d_cids = cameras_info[:,2]
	x2ds = [(x[0][:,:2].copy(),x[1]) if raw else frameCentroidsToDets(x, vicon_mats) for x in x2d_dict['frames']]

	if not(np.all(xcp_camera_ids == x2d_cids)):
		print ('WARNING not all the cameras from the x2d were in the xcp file?') # TODO, we should report this
		print (xcp_camera_ids, x2d_cids)
		vicon_mats = [vicon_mats[list(xcp_camera_ids).index(ci)] if ci in list(xcp_camera_ids) else Calibrate.makeUninitialisedMat(ci,(w,h)) for w,h,ci in cameras_info]
		camera_names = ['CAM_%s'%x for x in x2d_cids]
		xcp_camera_ids = [f for f in x2d_cids]
	Ps = np.array([m[2]/(np.sum(m[2][0,:3]**2)**0.5) for m in vicon_mats],dtype=np.float32)
	headerMetadata = readX2DMetadata(x2d_dict)
	return Ps, vicon_mats, x2d_cids, camera_names, x2ds, x2d_dict['header']

def readX2DMetadata(x2d_data):
	meta = {}
	# metaData found by brute force
	meta['capture_rate_fps'] = x2d_data['header'][11][2][0]
	meta['capture_rate_code'] = x2d_data['header'][11][2][1]
	meta['number_of_cameras'] = x2d_data['header'][0][1]
	meta['number_of_frames'] = x2d_data['header'][1][1]
	# these bits may need more thought - a defined struct?
	# attr dict??
	meta['camera_meta_list'] = x2d_data['header'][11][4]
	meta['timecode'] = x2d_data['header'][12][6] if len(x2d_data['header']) > 12 else ''

	return meta

if __name__ == '__main__':
	from IO import ASFReader
	ret = loadX2D(sys.argv[1])
	print (ret['header'])
	exit()
	IO.save('out.tis2d',ret)
