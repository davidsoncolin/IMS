# Copyright (c) 2010 Leif Johnson <leif@leifjohnson.net>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''A Python library for reading and writing C3D files.'''

def read(c3d_filename):
	import numpy as np
	c3d_reader = Reader(open(c3d_filename, 'rb'))
	c3d_frames_read = c3d_reader.read_frames()
	c3d_frames_read = [f for f in c3d_frames_read]
	numFrames = len(c3d_frames_read)
	numPoints = len(c3d_frames_read[0][0])
	c3d_frames = np.zeros((numFrames, numPoints, 4))
	for fi,(f,_) in enumerate(c3d_frames_read): c3d_frames[fi,:,:] = f
	ret = {'frames':np.array(c3d_frames,dtype=np.float32), 'fps': c3d_reader.frame_rate() }
	# these groups and keys are found in my c3d file, but no guarantees
	try:
		point = c3d_reader.group('POINT')
		point_params = point.params
		label_keys = sorted([x for x in point_params.keys() if x.startswith('LABELS')])
		ret['labels'] = ''.join([point_params[l].bytes for l in label_keys]).split()
		ret['units'] = point_params['UNITS'].bytes.strip() # 'mm'
		ret['rate'] = point.get_float('RATE')
		ret['num_frames'] = point.get_uint16('FRAMES')
		assert(len(ret['labels']) == numPoints)
		assert(ret['rate'] == c3d_reader.frame_rate())
		assert(ret['num_frames'] == numFrames)
	except: print ('(W) Some POINT keys not found in ', c3d_filename); pass
	try:
		subjects = c3d_reader.group('SUBJECTS')
		ret['subject_names'] = subjects.params['NAMES'].bytes.split()
		ret['subject_prefixes'] = subjects.params['LABEL_PREFIXES'].bytes.split()
	except: print ('(W) Some SUBJECTS keys not found in ', c3d_filename); pass
	try:
		timecode = c3d_reader.group('TIMECODE')
		ret['tc'] = np.fromstring(timecode.params['TIMECODES'].bytes,dtype=np.dtype('<H'))
		ret['tc_standard'] = timecode.params['STANDARD'].bytes.strip() # 'PAL'
		ret['tc_subframes'] = timecode.get_uint16('SUBFRAMESPERFRAME') # 4
		ret['tc_offset'] = timecode.get_uint16('OFFSETS')
	except: print ('(W) Some TIMECODE keys not found in ', c3d_filename); pass
	#analog = c3d_reader.group('ANALOG') # not used
	return ret

import array
import numpy
import struct
import logging
import operator

def DECfloat_to_float(f):
	'''Convert from DEC floating point format to something sensible'''
	bits = struct.pack('<f',f)
	return struct.unpack('<f',bits[2]+bits[3]+bits[0]+bits[1])[0]*0.25

def DECfloat_to_float_map(points):
	'''Convert an array from DEC floating point format to something sensible'''
	# flip alternate bytes in place
	if len(points) == 0: return numpy.array([],dtype=numpy.float32)
	tmp = numpy.frombuffer(points,dtype=numpy.dtype('H')).byteswap(True)
	# now we have big-endian floats, multiplied by 4
	tmp = numpy.frombuffer(points,dtype=numpy.dtype('>f'))
	return numpy.array(tmp*0.25,dtype=numpy.float32) # return a native data type

class Header(object):
	'''Header information from a C3D file.'''

	BINARY_FORMAT = '<BBHHHHHfHHf270sHH214s'

	def __init__(self, handle=None):
		self.label_block = 0
		self.parameter_block = 2
		self.data_block = 3

		self.point_count = 50
		self.analog_count = 0

		self.first_frame = 1
		self.last_frame = 1
		self.sample_per_frame = 0
		self.frame_rate = 60.0

		self.max_gap = 0
		self.scale_factor = -1.0
		self.long_event_labels = False

		if handle:
			self.read(handle)

	def write(self, handle):
		'''Write binary header data to a file handle.

		This method writes exactly 512 bytes to the beginning of the file.
		'''
		handle.seek(0)
		handle.write(struct.pack(self.BINARY_FORMAT,
								 self.parameter_block,
								 0x50,
								 self.point_count,
								 self.analog_count,
								 self.first_frame,
								 self.last_frame,
								 self.max_gap,
								 self.scale_factor,
								 self.data_block,
								 self.sample_per_frame,
								 self.frame_rate,
								 '',
								 self.long_event_labels and 0x3039 or 0x0,
								 self.label_block,
								 ''))

		logging.info('''wrote C3D header information:
  parameter_block: %(parameter_block)s
      point_count: %(point_count)s
     analog_count: %(analog_count)s
      first_frame: %(first_frame)s
       last_frame: %(last_frame)s
          max_gap: %(max_gap)s
     scale_factor: %(scale_factor)s
       data_block: %(data_block)s
 sample_per_frame: %(sample_per_frame)s
       frame_rate: %(frame_rate)s
long_event_labels: %(long_event_labels)s
      label_block: %(label_block)s''' % self.__dict__)

	def read(self, handle):
		'''Read and parse binary header data from a file handle.

		This method reads exactly 512 bytes from the beginning of the file.
		'''
		handle.seek(0)
		(self.parameter_block,
		 magic,
		 self.point_count,
		 self.analog_count,
		 self.first_frame,
		 self.last_frame,
		 self.max_gap,
		 self.scale_factor,
		 self.data_block,
		 self.sample_per_frame,
		 self.frame_rate,
		 _,
		 self.long_event_labels,
		 self.label_block,
		 _) = struct.unpack(self.BINARY_FORMAT, handle.read(512))

		assert magic == 80, 'C3D magic %d != 80 !' % magic

		logging.info('''loaded C3D header information:
  parameter_block: %(parameter_block)s
      point_count: %(point_count)s
     analog_count: %(analog_count)s
      first_frame: %(first_frame)s
       last_frame: %(last_frame)s
          max_gap: %(max_gap)s
     scale_factor: %(scale_factor)s
       data_block: %(data_block)s
 sample_per_frame: %(sample_per_frame)s
       frame_rate: %(frame_rate)s
long_event_labels: %(long_event_labels)s
      label_block: %(label_block)s''' % self.__dict__)


class Param(object):
	'''We represent a single named parameter from a C3D file.'''

	def __init__(self,
				 name,
				 desc='',
				 data_size=1,
				 dimensions=None,
				 bytes=None,
				 handle=None):
		'''Set up a new parameter with at least a name.

		name: The name of the parameter.
		desc: The description of the parameter.
		data_size: The number of bytes that are in the binary representation of
		the data for this parameter. Use -1 if this parameter represents
		string data.
		dimensions: The dimensions of the data for this parameter. This is
		primarily used for string data ; if you want to use strings, they must
		all be passed in the "bytes" variable as a space-delimited string.
		Each field in the string must be the same length, and that length must
		be passed in the dimensions list.
		bytes: The raw bytes for this parameter. Use struct.pack() to construct
		this value, or just pass the raw string data for string parameters.
		handle: If provided, the data for the parameter will be read from this
		file handle.
		'''
		self.name = name
		self.desc = desc

		self.negative_data_size = data_size < 0
		self.data_size = abs(data_size)
		self.dimensions = dimensions or []
		self.bytes = bytes

		if handle:
			self.read(handle)

	def __repr__(self):
		return '<Param: %s>' % self.desc

	def binary_size(self):
		'''Return the number of bytes needed to store this parameter.'''
		return (
			1 + # group_id
			2 + # next offset marker
			1 + len(self.name) + # size of name and name bytes
			1 + # data size
			1 + len(self.dimensions) + # size of dimensions and dimension bytes
			reduce(operator.mul, self.dimensions, 1) * self.data_size + # data
			1 + len(self.desc) # size of desc and desc bytes
			)

	def write(self, handle):
		'''Write binary data for this parameter to a file handle.

		This writes data at the current position in the file.
		'''
		size = self.data_size
		if self.negative_data_size:
			size = -size
		handle.write(struct.pack('b', size))
		handle.write(struct.pack('B', len(self.dimensions)))
		handle.write(struct.pack('B' * len(self.dimensions), *self.dimensions))
		if self.bytes:
			handle.write(self.bytes)
		handle.write(struct.pack('B', len(self.desc)))
		handle.write(self.desc)

		kwargs = self.__dict__.copy()
		if self.bytes:
			kwargs['bytes'] = self.bytes[:40]

		logging.debug('''wrote C3D parameter information:
      name: %(name)s
      desc: %(desc)s
 data_size: %(data_size)s
dimensions: %(dimensions)s
     bytes: %(bytes)r''' % kwargs)

	def read(self, handle):
		'''Read binary data for this parameter from a file handle.

		This reads exactly enough data from the current position in the file to
		initialize the parameter.
		'''
		self.data_size, = struct.unpack('b', handle.read(1))
		if self.data_size < 0:
			self.negative_data_size = True
			self.data_size = abs(self.data_size)

		count, = struct.unpack('B', handle.read(1))
		self.dimensions = [
			struct.unpack('B', handle.read(1))[0] for _ in xrange(count)]

		count = reduce(operator.mul, self.dimensions, 1)
		self.bytes = None
		if self.data_size * count:
			self.bytes = handle.read(self.data_size * count)
			assert(len(self.bytes) == self.data_size * count) # CBD
		else:
			logging.debug('zero data_size * count !')

		c = handle.read(1)
		if c == '': self.desc = ''
		else:
			size, = struct.unpack('B', c)
			self.desc = size and handle.read(size) or ''

		logging.debug('loaded C3D parameter information: %s', str(self))

	def __str__(self):
		'''Return a friendly string representation of this parameter.'''
		kwargs = self.__dict__.copy()
		kwargs['shaped'] = ''
		if self.bytes:
			kwargs['bytes'] = '%s%s' % (self.bytes[:10], ['', '...'][len(self.bytes) > 10])
			if len(self.dimensions) == 2:
				cols, rows = self.dimensions
				lines = ['']
				for r in range(rows):
					lines.append(repr(self.bytes[r * cols:(r+1) * cols]))
				kwargs['shaped'] = '\n'.join(lines)
			if len(self.dimensions) == 0:
				fmt = '<%s' % {2: 'H', 4: 'f'}.get(len(self.bytes), 'B')
				val = struct.unpack(fmt, self.bytes)[0]
				global c3d_processor
				if fmt == '<f' and c3d_processor == 85: val = DECfloat_to_float(val)
				kwargs['shaped'] = ' -> %s' % val

		return '''
      name: %(name)s
      desc: %(desc)s
 data_size: %(data_size)s
dimensions: %(dimensions)s
	 bytes: %(bytes)r%(shaped)s''' % kwargs


class Group(object):
	'''A group of parameters from a C3D file.'''

	def __init__(self, name=None, desc=None):
		self.name = name
		self.desc = desc
		self.params = {}

	def __repr__(self):
		return '<Group: %s>' % self.desc

	def add_param(self, name, **kwargs):
		self.params[name.upper()] = Param(name.upper(), **kwargs)
		if 'handle' not in kwargs:
			logging.debug('added parameter %s: %s', name, kwargs)

	def binary_size(self):
		'''Return the number of bytes to store this group and its parameters.'''
		return (
			1 + # group_id
			1 + len(self.name) + # size of name and name bytes
			2 + # next offset marker
			1 + len(self.desc) + # size of desc and desc bytes
			sum(p.binary_size() for p in self.params.itervalues()))

	def get_int8(self, key):
		return struct.unpack('b', self.params[key].bytes)[0]

	def get_uint8(self, key):
		return struct.unpack('B', self.params[key].bytes)[0]

	def get_int16(self, key):
		return struct.unpack('h', self.params[key].bytes)[0]

	def get_uint16(self, key):
		return struct.unpack('H', self.params[key].bytes)[0]

	def get_int32(self, key):
		return struct.unpack('i', self.params[key].bytes)[0]

	def get_uint32(self, key):
		return struct.unpack('I', self.params[key].bytes)[0]

	def get_float(self, key):
		rv = struct.unpack('f', self.params[key].bytes)[0]
		global c3d_processor
		if c3d_processor == 85: rv = DECfloat_to_float(rv)
		return rv

	def get_string(self, key, offset=0):
		return self.params[key].bytes.split()[offset]


class Manager(object):
	'''A base class for managing C3D file metadata.'''

	def __init__(self, header=None, groups=None):
		self.header = header or Header()
		self._groups = groups or {}

	def check_group(self, group_id, name=None, desc=None):
		'''Add a new parameter group.'''
		group = self._groups.get(group_id)

		if group is None:
			logging.debug('added C3D parameter group #%d: %s: %s', group_id, name, desc)
			group = self._groups[group_id] = Group(name, desc)
		else:
			logging.debug('using C3D parameter group %s: %s', group.name, group.desc)

		if name is not None:
			name = name.upper()
			if name in self._groups:
				raise NameError('group name %s was used more than once' % name)
			self._groups[name] = group
			group.name = name
			group.desc = desc

		return group

	def group(self, name):
		'''Get the parameter group with a given name.'''
		return self._groups.get(name.upper(), None)

	def groups(self):
		'''Get all the (name, group) pairs in our file.'''
		return self._groups.iteritems()

	def parameter_blocks(self):
		'''Compute the size (in 512B blocks) of the parameter section.'''
		bytes = 4
		for name, group in self._groups.iteritems():
			bytes += group.binary_size()
		blocks, overflow = divmod(bytes, 512)
		if overflow:
			blocks += 1
		return blocks

	def frame_rate(self):
		return self.header.frame_rate

	def scale_factor(self):
		return self.group('POINT').get_float('SCALE')

	def points_per_frame(self):
		return self.group('POINT').get_uint16('USED')

	num_points = points_per_frame

	def analog_per_frame(self):
		return self.group('ANALOG').get_uint16('USED')

	num_analog = analog_per_frame

	def start_field(self):
		return self.group('TRIAL').get_uint32('ACTUAL_START_FIELD')

	def end_field(self):
		return self.group('TRIAL').get_uint32('ACTUAL_END_FIELD')


class Reader(Manager):
	'''This class provides methods for reading the data in a C3D file.

	A C3D file contains metadata and frame-based data describing 3D motion.

	You can iterate over the frames in the file by calling
	"read_frames(handle)" after construction:

	>>> r = C3D.Reader(open('capture.c3d', 'rb'))
	>>> for points, analog in r.read_frames():
	...	 print points.shape, 'points in this frame'
	...	 frames.append(points, analog)
	'''

	def __init__(self, handle):
		'''Initialize this C3D file by reading header and parameter data.
		'''
		super(Reader, self).__init__(Header(handle))
		self._handle = handle
		self._read_metadata()

	def _read_metadata(self):
		'''Read group and parameter data from our file handle.'''
		self._handle.seek((self.header.parameter_block - 1) * 512)

		# metadata header
		start = self._handle.tell()
		buf = self._handle.read(4)
		global c3d_processor
		_, __, num_records, c3d_processor = struct.unpack('BBBB', buf)
		# in my file, num_parameters = 8, but the 26 parameters!
		# apparently there's a terminating parameter with zeros, so use that instead
		if c3d_processor == 85:
			logging.debug('DEC floats')
			self.header.scale_factor = DECfloat_to_float(self.header.scale_factor)
			self.header.frame_rate = DECfloat_to_float(self.header.frame_rate)
			logging.info('corrected scale_factor = %s', self.header.scale_factor)
			logging.info('corrected frame_rate = %s', self.header.frame_rate )

		buf = self._handle
		while True: # CBD bugfix
			chars_in_name, group_id = struct.unpack('bb', buf.read(2))
			if group_id == 0 or chars_in_name == 0: # a terminating parameter
				break

			name = buf.read(abs(chars_in_name)).upper()

			offset_to_next, = struct.unpack('h', buf.read(2))

			if group_id < 0:
				group_id = abs(group_id)
				size, = struct.unpack('B', buf.read(1))
				desc = size and buf.read(size) or ''
				g = self.check_group(group_id, name, desc)
				logging.debug('%s group takes up %d bytes', name, g.binary_size())
			else:
				g = self.check_group(group_id)
				g.add_param(name, handle=buf)
				logging.debug('%s parameter takes up %d bytes', name, g.params[name].binary_size())

			logging.debug('consumed %d bytes of metadata', 2 + abs(chars_in_name) + offset_to_next)
		logging.debug('total bytes = %d, records = %d (header stated %d)' % ((self._handle.tell()-start), numpy.ceil((self._handle.tell()-start)/512.), num_records))
		logging.debug('read %d parameter groups', len(self._groups) // 2)

	def read_frames(self):
		'''Iterate over the data frames from our C3D file handle.

		This generates a sequence of (points, analog) ordered pairs, one
		ordered pair per frame. The first element of each frame contains a numpy
		array of 4D "points" and the second element of each frame contains a
		numpy array of 1D "analog" values that were probably recorded
		simultaneously. The four dimensions in the point data are typically
		(x, y, z) and a "confidence" estimate for the point.
		'''
		# find out where we seek to start reading frame data.
		start_block = self.group('POINT').get_uint16('DATA_START')
		if start_block != self.header.data_block:
			logging.info('start_block %d != data_block %d',
						 start_block, self.header.data_block)
			if not start_block:
				logging.info('setting start_block = data_block')
				start_block = self.header.data_block

		# read frame and analog data in either float or int format.
		format = 'fi'[self.group('POINT').get_float('SCALE') >= 0]
		ppf = self.points_per_frame()
		apf = None
		try: apf = self.analog_per_frame() # may not exist
		except: logging.debug('analog_per_frame does not exist')

		self._handle.seek((start_block - 1) * 512)
		start = self._handle.tell()
		f = 0
		numFrames = self.group('POINT').get_uint16('FRAMES')
		try: numFrames = self.end_field() - self.start_field() + 1
		except: logging.debug('end_field or start_field does not exist')
		for f in xrange(numFrames):
			points = array.array(format)
			points.fromfile(self._handle, 4 * ppf)
			if format == 'f' and c3d_processor == 85: points = DECfloat_to_float_map(points)
			analog = array.array(format)
			if apf != None:
				analog.fromfile(self._handle, apf)
				if format == 'f' and c3d_processor == 85: analog = DECfloat_to_float_map(analog)
			yield (numpy.array(points).reshape((ppf, 4)), numpy.array(analog))
			if f and not f % 10000:
				logging.debug('consumed %d frames from %dkB of frame data', f, (self._handle.tell() - start) / 1000)

		logging.info('iterated over %d frames', f)


class Writer(Manager):
	'''This class manages the task of writing metadata and frames to a C3D file.

	>>> r = C3D.Reader(open('data.c3d', 'rb'))
	>>> frames = smooth_frames(r.read_frames())
	>>> w = C3D.Writer(open('smoothed.c3d', 'wb'))
	>>> w.write_from_reader(frames, r)
	'''

	def __init__(self, handle):
		super(Writer, self).__init__()
		self._handle = handle

	def _pad_block(self):
		'''Pad the file with 0s to the end of the next block boundary.'''
		extra = self._handle.tell() % 512
		if extra:
			logging.debug('padding with %d zeros', 512 - extra)
			self._handle.write('\x00' * (512 - extra))

	def write_metadata(self):
		'''Write metadata for this file to our file handle.'''
		# header
		self.header.write(self._handle)
		self._pad_block()
		assert self._handle.tell() == 512
		logging.debug('produced %d bytes of header data', self._handle.tell())

		# groups
		self._handle.write(struct.pack('BBBB', 0, 0, self.parameter_blocks(), 84))
		id_groups = sorted((i, g) for i, g in self.groups() if isinstance(i, int))
		for group_id, group in id_groups:
			self._write_group(group_id, group)

		# padding
		self._pad_block()
		while self._handle.tell() != 512 * (self.header.data_block - 1):
			self._handle.write('\x00' * 512)

		logging.debug('produced %d bytes of metadata', self._handle.tell())

	def _write_group(self, group_id, group):
		'''Write a single parameter group, with parameters, to our file handle.

		group_id: The numerical ID of the group.
		group: The Group object to write to the handle.
		'''
		logging.debug('writing C3D parameter group #%d: %s: %s',
					 group_id, group.name, group.desc)
		self._handle.write(struct.pack('bb', len(group.name), -group_id))
		self._handle.write(group.name)
		self._handle.write(struct.pack('h', 3 + len(group.desc)))
		self._handle.write(struct.pack('B', len(group.desc)))
		self._handle.write(group.desc)
		logging.debug('writing group info yields offset %d', self._handle.tell())
		for name, param in group.params.iteritems():
			self._handle.write(struct.pack('bb', len(name), group_id))
			self._handle.write(name)
			self._handle.write(struct.pack('h', param.binary_size() - 2 - len(name)))
			param.write(self._handle)
			logging.debug('writing %d bytes yields offset %d', 4 + len(name) + param.binary_size(), self._handle.tell())
		logging.debug('group %s ends at byte offset %d', group.name, self._handle.tell())

	def write_frames(self, frames):
		'''Write the given list of frame data to our file handle.

		frames: A sequence of (points, analog) tuples, each containing data for one frame.
		'''
		assert self._handle.tell() == 512 * (self.header.data_block - 1)
		format = 'fi'[self.group('POINT').get_float('SCALE') >= 0]
		for p, a in frames:
			point = array.array(format)
			point.extend(p.flatten())
			point.tofile(self._handle)
			analog = array.array(format)
			analog.extend(a)
			analog.tofile(self._handle)
		self._pad_block()

	def write_like_phasespace(self, frames, frame_count, point_frame_rate=480.0, analog_frame_rate=0.0, point_scale_factor=-1.0,
							point_units='mm  ', gen_scale=1.0, ):
		'''Write a set of frames to a file so it looks like Phasespace wrote it.

		frames: The sequence of frames to write.
		frame_count: The number of frames to write.
		point_frame_rate: The frame rate of the data.
		analog_frame_rate: The number of analog samples per frame.
		point_scale_factor: The scale factor for point data.
		point_units: The units that the point numbers represent.
		'''
		try:
			points, analog = iter(frames).next()
		except StopIteration:
			return

		# POINT group
		ppf = len(points)
		point_group = self.check_group(1, 'POINT', 'POINT group')
		point_group.add_param('USED', desc='Number of 3d markers', data_size=2, bytes=struct.pack('H', ppf))
		point_group.add_param('FRAMES', desc='frame count', data_size=2, bytes=struct.pack('H', min(65535, frame_count)))
		point_group.add_param('DATA_START', desc='data block number', data_size=2, bytes=struct.pack('H', 0))
		point_group.add_param('SCALE', desc='3d scale factor', data_size=4, bytes=struct.pack('f', point_scale_factor))
		point_group.add_param('RATE', desc='3d data capture rate', data_size=4, bytes=struct.pack('f', point_frame_rate))
		point_group.add_param('X_SCREEN', desc='X_SCREEN parameter', data_size=-1, dimensions=[2], bytes='+X')
		point_group.add_param('Y_SCREEN', desc='Y_SCREEN parameter', data_size=-1, dimensions=[2], bytes='+Z')
		point_group.add_param('UNITS', desc='3d data units', data_size=-1, dimensions=[len(point_units)], bytes=point_units)
		point_group.add_param('LABELS', desc='labels', data_size=-1, dimensions=[5, ppf], bytes=''.join('M%03d ' % i for i in xrange(ppf)))
		point_group.add_param('DESCRIPTIONS', desc='descriptions', data_size=-1, dimensions=[16, ppf], bytes=' ' * 16 * ppf)

		# ANALOG group
		apf = len(analog)
		analog_group = self.check_group(2, 'ANALOG', 'ANALOG group')
		analog_group.add_param('USED', desc='analog channel count', data_size=2, bytes=struct.pack('H', apf))
		analog_group.add_param('RATE', desc='analog frame rate', data_size=4, bytes=struct.pack('f', analog_frame_rate))
		analog_group.add_param('GEN_SCALE', desc='analog general scale factor', data_size=4, bytes=struct.pack('f', gen_scale))
		analog_group.add_param('SCALE', desc='analog channel scale factors', data_size=4, dimensions=[0])
		analog_group.add_param('OFFSET', desc='analog channel offsets', data_size=2, dimensions=[0])

		# TRIAL group
		trial_group = self.check_group(3, 'TRIAL', 'TRIAL group')
		trial_group.add_param('ACTUAL_START_FIELD', desc='actual start frame', data_size=2, dimensions=[2], bytes=struct.pack('I', 1))
		trial_group.add_param('ACTUAL_END_FIELD', desc='actual end frame', data_size=2, dimensions=[2], bytes=struct.pack('I', frame_count))

		# sync parameter information to header.
		blocks = self.parameter_blocks()
		point_group.params['DATA_START'].bytes = struct.pack('H', 2 + blocks)

		self.header.data_block = 2 + blocks
		self.header.frame_rate = point_frame_rate
		self.header.last_frame = min(frame_count, 65535)
		self.header.point_count = ppf
		self.header.analog_count = apf
		self.header.scale_factor = point_scale_factor

		self.write_metadata()
		self.write_frames(frames)

	def write_from_reader(self, frames, reader):
		'''Write a file with the same metadata and number of frames as a Reader.

		frames: A sequence of frames to write.
		reader: Copy metadata from this reader to the output file.
		'''
		self.write_like_phasespace(frames, reader.end_field(), reader.frame_rate())
