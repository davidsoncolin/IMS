#!/usr/bin/env python

import os
import sys
import struct
import numpy as np
from collections import deque
from operator import mul

# A basic IO toolkit that serializes dicts, lists, tuples, (simple) numpy arrays, None, strings, bools, ints, longs, floats.
# It also restores references, for example it can deal with a = {}; a['a']=a; which would otherwise cause infinite recursion.
# encode(data) : returns a string that represents the data object
# decode(encode(data)+'remain') = data,'remain' : returns the original data and any excess string
# wrap(data) : returns a string that represents the data object
# unwrap(wrap(data)+'remain') = data,'remain' : returns the original data and any excess string
# for streams, read 8 characters and use streamSize to get the packet size.

def extend(o,d): o.extend(d)
class StreamWriter:
	def __init__(self, o, extend = extend): self.o,self._extend = o,extend
	def extend(self, d): self._extend(self.o, d)

def unpack_from(st, s, offset):
	if isinstance(s,StreamReader):
		return struct.unpack_from(st,str(s[offset:offset+12]),0)
	return struct.unpack_from(st, s, offset)

class StreamReader:
	'''Behaves like a str, but reads the file as requested, to avoid holding the whole file in memory.'''
	_stream = None
	_stream_size = 0
	_block = ''
	_block_start = 0
	_chunk_size = 128*1024*1024
	# self._block_start + len(self._block) == self._stream.tell()

	def __init__(self, s):
		self._stream = s
		self._stream.seek(0,2); self._stream_size = int(self._stream.tell())
		self.read_block(0, 0)
		if self._stream_size < self._chunk_size: # fallthrough for small files
			self.__len__ = self._block.__len__
			self.__getitem__ = self._block.__getitem__
			self.__getslice__ = self._block.__getslice__

	def __len__(self): return self._stream_size

	def read_block(self, i, size):
		self._stream.seek(i)
		self._block_start = int(self._stream.tell())
		self._block = self._stream.read(max(self._chunk_size, size))

	def __getitem__(self, i):
		if i < 0: i += self._stream_size
		if i < 0 or i >= self._stream_size: return ''[1] # exception
		if i < self._block_start or i >= self._block_start+len(self._block): self.read_block(i, 1)
		return self._block[i-self._block_start]

	def __getslice__(self, i, j):
		if i < 0: i += self._stream_size
		if j < 0: j += self._stream_size
		i,j = max(i,0),min(j,self._stream_size)
		if i >= self._stream_size or j <= i: return ''
		if i < self._block_start or j > self._block_start+len(self._block): self.read_block(i, j-i)
		return self._block[i-self._block_start : j-self._block_start]

def readStream(s_read, cache = {}): # cache holds static data between calls; eg s_read = socket.recv
	if not cache.has_key('s'): cache['s'] = ''
	if len(cache['s']) < 12: cache['s'] += s_read(12-len(cache['s']))
	if len(cache['s']) < 12: return None
	size,o = streamSize(cache['s'])
	if len(cache['s']) < o+size: cache['s'] += s_read(o+size-len(cache['s']))
	if len(cache['s']) < o+size: return None
	data,cache['s'] = unwrap(cache['s'])
	return data

def writeStream(s_write, data): # eg: s_write = socket.send
	s = StreamWriter(None, extend = lambda x,y:s_write(y))
	_wrap(s, data)

def streamSize(s):
	if s[:4] == 'TIMS': return unpack_from('<Q',s,4)[0],12
	assert(s[:4] == 'TISF')
	return unpack_from('<I',s,4)[0],8

def _wrap(s, data):
	s.extend('TIMS\x00\x00\x00\x00\x00\x00\x00\x00')
	return encode(data, s)

def wrap(data):
	s =  _wrap(bytearray(), data)
	s[4:12] = struct.pack('<Q',len(s)-12)
	return str(s)
	
def unwrap(s):
	l,o = streamSize(s)
	data,offset = decode(s,o)
	if l != 0:
		return data,s[offset:]
	else:
		return data,''

def encodeString(s,data):
	assert isinstance(data,str)
	s.extend(struct.pack('<I',len(data)))
	s.extend(data)

def decodeString(s,offset,refs):
	l = unpack_from('<I',s,offset)[0]
	ret = s[offset+4:offset+4+l]
	refs.append(ret)
	return ret, offset+4+l

def decodeStringList(s,offset,refs):
	ret,offset = decodeString(s,offset,[])
	ret = ret.split(chr(0))
	refs.append(ret)
	return ret,offset

class Marker(str): pass
M = {tuple:('(',[Marker(')')]),list:('[',[Marker(']')]),set:('|',[Marker('_')])}
transtypes = {type(None),bool}
transvals = {None:'\0',False:'0',True:'1'}
# NOTE this will crash by design if data exceeds 64 bits
# NOTE we don't try to preserve np.float32 and np.float64 objects, just cast them to floats
basetypes = {int:('i',struct.Struct('<i').pack) if sys.maxint==0x7fffffff else ('q',struct.Struct('<q').pack),long:('q',struct.Struct('<q').pack),float:('d',struct.Struct('<d').pack),np.float32:('d',struct.Struct('<d').pack),np.float64:('d',struct.Struct('<d').pack),np.int32:('i',struct.Struct('<i').pack),np.int64:('q',struct.Struct('<q').pack)}
intpack = struct.Struct('<I').pack
longpack = struct.Struct('<Q').pack

def encode(data, s = None, refs = None):
	if s is None: s = bytearray()
	if refs is None: refs = {}
	queue = deque([data]) # avoid recursion by using a queue
	while queue:
		data = queue.popleft()
		t,did = type(data),id(data)
		if t is Marker      : s.extend(str(data))
		elif t in transtypes: s.extend(transvals[data])
		elif t in basetypes : v,f = basetypes[t]; s.extend(v+f(data))
		elif did in refs    : s.extend('r'+intpack(refs[did][0]))
		else:
			refs[did] = (len(refs),data) # NOTE we maintain a reference inside the tuple, just in case it's a temporary
			if t is str      : s.extend('"'+intpack(len(data))+data)
			elif t is unicode: data = data.encode('utf8'); s.extend('u'+intpack(len(data))+data)
			elif t is dict :
				v = data.values()
				if len(v) > 1 and isinstance(v[0],np.ndarray) and isinstance(v[1],np.ndarray): # concatenate a list of arrays into a single array
					try: 
						v1 = np.array(v,dtype=v[0].dtype)
						assert np.all(v==v1)
						v = v1
					except: pass # an exception means it can't be concatenated
				s.extend('{')
				queue.extendleft([Marker('}'),v,data.keys()])
			elif t in M:
				if t is list :
					try: # a string list is encoded differently
						v = chr(0).join(data)
						assert data == v.split(chr(0))
						s.extend('S'+intpack(len(v))+v)
						continue
					except: pass # an exception means its not a string list
				m0,m1 = M[t]
				s.extend(m0)
				queue.extendleft(m1)
				queue.extendleft(reversed(deque(data)))
			elif t is np.ndarray:
				v = data.dtype.str
				s.extend('<'+intpack(len(v))+v+'(')
				for v in data.shape: s.extend('q'+longpack(v))
				s.extend(')'); s.extend(data.tostring()); s.extend('>')
			else:
				assert False, 'IO.encode: unknown type'+repr(t)
	return s

def decodeNumpy(s,offset,refs):
	rid = len(refs); refs.append(0) # special case: we have to reserve the space for the numpy in the ref list
	dt,offset = decodeString(s,offset,[]) # special case: no refs, missing 's' indicating string
	dt = np.dtype(dt)
	shape,offset = decode(s,offset)
	size = int(dt.itemsize)*reduce(mul,map(int,shape),1) # ensure size is an int, even if shape is an int32 (don't use np.prod)
	ret = np.fromstring(s[offset:offset+size], dtype=dt).reshape(shape)
	assert(s[offset+size] == '>')
	refs[rid] = ret
	return ret,offset+size+1

def decode(s, offset = 0, refs = None):
	if isinstance(s,bytearray): s = str(s)
	if refs is None: refs = []
	basetypes = {'i':('<i',4),'q':('<q',8),'d':('<d',8)}
	transtypes = {'\0':None,'0':False,'1':True}
	rv,queue = [],[]
	fixes = {}
	while rv or not queue:
		k = s[offset]
		offset = offset+1
		if k in transtypes: queue.append(transtypes[k])
		elif k in basetypes: b0,b1=basetypes[k]; queue.append(unpack_from(b0,s,offset)[0]); offset = offset + b1
		elif k == 'r' :
			ri = unpack_from('<I',s,offset)[0]
			ref = refs[ri]
			if ref is None: # the reference is a tuple which hasn't been created yet; later on we can try to fix this..
				fixes.setdefault(ri,[]).append(map(len,rv)+[len(queue)])
			queue.append(ref); offset = offset + 4
		elif k == '"' :
			ret,offset = decodeString(s,offset,refs)
			queue.append(ret)
		elif k == 'u' :
			ret,offset = decodeString(s,offset,refs)
			ret = ret.decode('utf8')
			queue.append(ret)
		elif k == 'S' : 
			ret,offset = decodeStringList(s,offset,refs)
			queue.append(ret)
		elif k == '#' : # for backwards compatibility only; we don't like recursion!
			ret = {}
			refs.append(ret)
			ks,offset = decode(s,offset,refs)
			vs,offset = decode(s,offset,refs)
			ret.update(zip(ks,vs))
			queue.append(ret)
		elif k == '{' :
			ret = {}
			refs.append(ret); rv.append(queue); queue = [ret]
		elif k == '}' :
			ret = queue[0]
			try:
				assert len(queue) == 3 and len(queue[1]) == len(queue[2])
				ret.update(zip(queue[1],queue[2]))
			except:
				# very backward compatibility
				ret.update(zip(queue[1::2],queue[2::2]))
			queue = rv.pop(); queue.append(ret)
		elif k == '(' :
			ret = len(refs) # because it's immutable, we can only reserve a place for the tuple
			refs.append(None); rv.append(queue); queue = [ret]
		elif k == ')' :
			ri = queue[0]
			ret = tuple(queue[1:])
			refs[ri] = ret
			queue = rv.pop(); queue.append(ret)
			if ri in fixes: # try to find all the references and fix them
				updepth = len(rv)+1
				for indlist in fixes[ri]:
					node = ret
					for i in indlist[updepth:-1]:
						if type(node) is list: node = node[i]
						elif type(node) is tuple: node = node[i-1]
						elif type(node) is dict: assert(False)
						elif type(node) is set: assert(False)
					assert(node[indlist[-1]] is None)
					node[indlist[-1]] = ret # must be a list
		elif k == '[' :
			ret = []
			refs.append(ret); rv.append(queue); queue = ret
		elif k == ']' :
			ret = queue
			queue = rv.pop(); queue.append(ret)
		elif k == '|' :
			ret = set()
			refs.append(ret); rv.append(queue); queue = [ret]
		elif k == '_' :
			ret = queue[0]
			ret.update(queue[1:])
			queue = rv.pop(); queue.append(ret)
		elif k == '<' :
			ret,offset = decodeNumpy(s,offset,refs)
			queue.append(ret)
		else:
			# TODO: Skip unknown keys ?
			assert False,'unexpected key '+repr(k)
	return queue[0],offset

def load(fn, version = None):
	print 'loading',fn
	fs = os.path.getsize(fn)
	(header,payload),v = unwrap(StreamReader(open(fn,'rb')) if (fs > (1<<30)) else open(fn,'rb').read()) # bigger than 1GB
	assert(len(v) == 0)
	if version != None: assert(header['version'] == version)
	return header,payload

def save(fn, payload):
	print 'saving',fn
	header = {'version':'0.2','studio':'Imaginarium'}
	f = open(fn,'wb')
	writeStream(f.write, (header,payload))

def test():
	for x in [[],(),{},set(),{0},{0:1},[0],(1),'hi',u"Kl\xc3ft",['s','u'],0.123,dict(zip(['a','b'],np.eye(2,dtype=np.float32))),[1,2]]:
		assert repr(decode(str(encode(x)))[0]) == repr(x),'failed test'+repr(decode(str(encode(x)))[0])+' vs '+repr(x)
	a = {}
	b = {}
	a['a'] = a
	a['b'] = b
	b['a'] = a
	b['b'] = b
	a['c'] = (a,b)
	a['d'] = [b,[b,a]]
	a['e'] = (None, False, True, [0,1], (True,False), (np.arange(10,dtype=np.int32),))
	a['f'] = (a['e'],)
	a['g'] = ['a','b','c','d','e','f','g']
	a['h'] = (set([0,1]),np.array([[1,2],[3,4]],dtype=np.float32))
	save('a',a)
	A = load('a')[1]
	print ('A=',repr(A))
	print ('a=',repr(a))
	assert(str(A) == str(a))
	assert(repr(A) == repr(a))
	assert(A['a'] is A)
	assert(A['a']['b'] is A['b'])
	assert(A['b']['b'] is A['b'])
	assert(A['b']['a'] is A)
	assert(type(A['c']) is tuple)
	assert(len(A['c']) is 2)
	assert(A['c'][0] is A)
	assert(A['c'][1] is A['b'])
	assert(A['d'][0] is A['b'])
	assert(type(A['d']) is list)
	assert(len(A['d']) is 2)
	assert(len(A['d'][1]) is 2)
	assert(A['d'][1][0] is A['b'])
	assert(A['d'][1][1] is A)
	assert(type(A['e']) is tuple)
	assert(str(A['e']) == str(a['e']))
	assert(A['f'][0] is A['e'])
	assert(A['g'] == ['a','b','c','d','e','f','g'])

if __name__ == '__main__':
	if len(sys.argv)==1: test()
	else:
		header, payload = load(sys.argv[1])
		np.set_printoptions(threshold=100000000)
		print ('from numpy import *')
		print ('header=',repr(header))
		print ('payload=',repr(payload))
	
