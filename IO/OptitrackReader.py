#!/usr/bin/env python

import sys
import struct
import numpy as np
import IO

import ISCV

def rle_compress(data):
	'''compress a binary image by run-length encoding.'''
	shape = data.shape
	data = data.reshape(-1)
	tmp = data - np.concatenate(([0],data[:-1]))
	return np.array(np.where(tmp != 0)[0],dtype=np.int32),shape
	
def rle_decompress(rle, shape):
	'''decompress a run-length encoding into a binary image.'''
	ret = np.zeros(shape,dtype=np.bool)
	data = ret.reshape(-1)
	for c0,c1 in zip(rle[::2],rle[1::2]): data[c0:c1] = True
	return ret

def read_float(data,s):
	return struct.unpack('<d',data[s:s+8])[0],s+8

def read_int(data,s):
	return struct.unpack('<i',data[s:s+4])[0],s+4

def read_half(data,s):
	return struct.unpack('<h',data[s:s+2])[0],s+2

def read_bool(data,s):
	return [False,True][ord(data[s])],s+1

def read_mask(data,s):
	assert(data[s] == chr(2)); s+= 1
	assert(data[s] == chr(1)); s+= 1
	height_a,s = read_int(data,s)
	width_a,s = read_int(data,s)
	chans_a,s = read_int(data,s)
	chans_b,s = read_int(data,s)
	size_bytes,s = read_int(data,s)
	end_s = s + size_bytes
	assert(data[s] == chr(1)); s+= 1
	height,s = read_int(data,s)
	width,s = read_int(data,s)
	chans,s = read_int(data,s)
	size,s = read_int(data,s)
	assert(width*height == size*8)
	assert(chans == 1)
	block = []
	while data[s] != chr(0xef) or data[s+1] != chr(0xbe):
		v = ord(data[s]); s += 1
		count = 1
		if v == 0 or v == 255: count,s = read_int(data,s)
		v = [v&1,v&2,v&4,v&8,v&0x10,v&0x20,v&0x40,v&0x80]
		block.extend(v*count)
	s += 4
	assert(s == end_s)
	assert(len(block) == size*8)
	return np.array(block,dtype=np.bool).reshape(height,width),s

def read_addendum(data,s):
	#assert(data[s] == chr(0x0a))
	ret = []
	func = {'h':read_half,'i':read_int,'d':read_float,'b':read_bool}
	for t in 'hibddddiiiddbddddddiiidiiiiibiiiddiddihiiibiiihibiibibddidiiibiididdiiiid':
		v,s=func[t](data,s)
		print t,v
		ret.append(v)
	return ret,s

def read_CAL(filename):
	'''Reader for .cal format (optitrack).'''
	data = open(filename,'rb').read()
	s = 0
	fourcc,s = read_int(data,s)
	assert(fourcc == 844258806)
	version,s = read_int(data,s)
	assert(version == 20)
	num_cameras,s = read_int(data,s)
	#print num_cameras
	cameras = []
	for ci in xrange(num_cameras):
		#print ' '.join(map(lambda x:hex(x)[-2:].replace('x','0'),map(ord,data[s:s+10])))
		camera_tag,s = read_half(data,s)
		assert(camera_tag == 0x4)
		cam = {}
		# NB because the images are square, we can't be sure which is width and which is height.
		# NB Undecoded CAMERA_0.77 and CAMERA_0.93 parameters are not constants
		for t,k in zip('iiiiiiiiiibiiiiibbbbdbibbiiidiiiiddddddddddddrmiiiid',['INDEX','ID','i11','i0_a','i0_b','HEIGHT','WIDTH',\
					   '?i225_600','?i167_200','strobe_output_Watts','b0_a','i4','fps','i0_c','i0_d','i100','b1_a','b0_b','b0_c','b0_d','d0','b0_e','i3_b',\
					   'b1_b','b1_c','i2','?i1_4_a','i2000','d0.6000000238418579','i48','i64','i4','i2','OX','OY','FX','FY',\
					   'K1','K2','K3','P1','P2','TX','TY','TZ','R','MASK','i64','i1_b','i1_c','i1_d','d0.5']):
			if t == 'm':
				v,s = read_mask(data,s)
				rle,shape = rle_compress(v)
				v2 = rle_decompress(rle, shape)
				assert(np.all(v == v2))
				v = rle,shape
			elif t == 'r':
				v = np.fromstring(data[s:s+9*8],dtype=np.float64).reshape(3,3)
				s += 9*8
			else:
				assert(data[s] == chr(1)); s += 1
				if t == 'i': v,s = read_int(data,s)
				elif t == 'b':v,s = read_bool(data,s)
				elif t == 'd':v,s = read_float(data,s)
				else: raise
			# if we get a file with different values, cause a crash
			# we want to know what these values might mean!
			if k.startswith('?'): print k,v
			if k.startswith('i'): assert(v == int(k[1:].partition('_')[0])); continue
			if k.startswith('d'): assert(v == float(k[1:].partition('_')[0])); continue
			if k.startswith('b0'): assert(v is False); continue
			if k.startswith('b1'): assert(v is True); continue
			cam[k] = v
		cameras.append(cam)
	# no idea what this data is
	addenda = []
	num_addenda,s = read_int(data,s)
	print 'num_addenda',num_addenda
	if 0:
		for ai in xrange(num_addenda):
			addendum,s = read_addendum(data,s)
			addenda.append(addendum)
	#assert(s == len(data))
	return {'filename':filename,'cameras':cameras,'addenda':addenda}

def load_CAL(filename):
	ret = read_CAL(filename)['cameras']
	#print ret[0]; exit()
	mats = []
	for d in ret:
		hsx,hsy = d['WIDTH']*0.5,d['HEIGHT']*0.5
		sensor = np.array([d['WIDTH'],d['HEIGHT']],dtype=np.int32)
		fx = d['FX'] / hsx
		fy = d['FY'] / hsy
		skew = 0. #d['SKEW'] / hsx
		ox,oy = d['OX'],d['OY']
		ox = (ox - hsx) / hsx
		oy = (oy - hsy) / hsx
		K = np.array([[fx, skew, -ox],[0, fy, oy], [0, 0, 1]], dtype=np.float32)
		RT = np.zeros((3,4),dtype=np.float32)
		RT[:3,:3] = d['R'].T
		pos = np.array([d['TX'],d['TY'],d['TZ']],dtype=np.float32)*1000.
		RT[:,3] = -np.dot(RT[:3,:3], pos)
		k1 = d['K1'] * 0.25 # their image goes from -0.5 to +0.5 whereas ours goes from -1 to +1
		k2 = d['K2'] * (0.25*0.25)
		mats.append([K,RT,np.dot(K,RT),np.array([k1,k2],dtype=np.float32),pos,sensor]) # 9+12+12+2+3+2 = 40 numbers
	order = np.argsort([c['INDEX'] for c in ret])
	print 'order',order#np.array([c['INDEX'] for c in ret])[order]
	return [mats[oi] for oi in order],[ret[oi] for oi in order]

if __name__ == '__main__':
	import os
	import sys
	import IO
	import ASFReader
	filename = sys.argv[1]
	cal = load_CAL(filename)
	cams = cal['cameras']
	keys = cams[0].keys()
	#for key in keys:
	#	vals = [c[key] for c in cams]
	#	print key,vals
	IO.save('cal.imscal',cal)
	import pylab as pl
	mask = cal['cameras'][0]['MASK']	
	mask = rle_decompress(mask[0],mask[1])
	print mask.shape
	pl.imshow(mask)
	pl.hold()
	pl.show()
