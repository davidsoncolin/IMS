import struct
import numpy as np
from time import time,ctime

def read_uint32(d):
	(data, frame_offset) = d
	v = struct.unpack_from('<I',data[frame_offset:frame_offset+4])[0]
	return v,(data,frame_offset+4)

def read_uint64(d):
	(data, frame_offset) = d
	v = struct.unpack_from('<Q',data[frame_offset:frame_offset+8])[0]
	return v,(data,frame_offset+8)

def read_string(d):
	size,d = read_uint32(d)
	(data, frame_offset) = d
	v = data[frame_offset:frame_offset+size]
	return v,(data,frame_offset+size)
	
def read_dict(d):
	num,d = read_uint32(d)
	head = {}
	for it in range(num):
		k,d = read_string(d)
		v,d = read_string(d)
		assert k not in head
		head[k]=v
	return head,d

def convert_to_mov(pico_filename, mov_filename):
	raw_data = open(pico_filename, 'rb').read()
	outfile = open(mov_filename,'wb')
	data = (raw_data,0)
	_101,data = read_uint64(data)
	assert _101 == 101,repr(_101)+'expected 101'
	data_first_frame,data = read_uint64(data)
	data_last_frame,data = read_uint64(data)
	#print hex(_101),hex(data_first_frame),hex(data_last_frame),len(raw_data)
	vs = []
	for it in range(8): #58):
		v,data = read_uint32(data)
		vs.append(v)
	print vs
	# 0:end frame-30ish, 1:start frame+30ish, 
	# 2:start frame, 3:end frame, 4:channel (2), 5:unknown (4189859218,962611082)
	# 6:last I-frame, 7:0
	data = (raw_data,0x100)
	head,data = read_dict((raw_data,0x100))
	#from pprint import pprint
	#pprint (head)
	print 'capture name:',head['this_capture.capture_name']
	print 'capture time:',head['this_capture.hal_time_ms'], ':', ctime(int(head['this_capture.hal_time_ms'])*0.001)

	assert(data[1] == data_first_frame)

	frame_offset = data_first_frame

	first_frame_vals = None
	#vals[:14] = (100, 764813648, 291635812, 0, 27000000, 540000, 0, 1543, 171448332, 0, 512, 369502501, 0, 0)
	#assert vals[23:25] == (4113, 0)
	#assert vals[27:] == (808910848, 38656, 0, 0, 0)
	frame_marker = '\x64\x00\x00\x00'
	block_marker = '\x00\x00\x00\x01' # annex-b format block marker; will convert to AVCC format
	def half_pack(l): return np.array(l,dtype='>H').tostring()
	def int_pack(l): return np.array(l,dtype='>I').tostring()
	def mov_pack(s,o=4): return int_pack(len(s)+o) + s
	header_data = mov_pack('ftypqt  \x20\x14\x02\x11qt  ' + int_pack([0,0,0,1]),0) + 'mdat' + int_pack([0,0,0])
	outfile.write(header_data)
	write_size = len(header_data)
	frame_starts = []
	frame_types = []
	# 38 byte header
	frame_header =  '\x28\xee\x07\xf3\x90\xd2\x6b\xac\x92\x11\x06\x93\x5d\x64\xa2\x42' +\
					'\x56\xba\xff\xd7\xff\xeb\xf5\xff\xaf\xd5\x44\x12\xb5\xd7\xfe\xbf' +\
					'\xff\x5f\xaf\xfd\x7e\xbc'
	prev_tod = 0
	while frame_offset < data_last_frame:
		#print hex(frame_offset)
		if raw_data[frame_offset:frame_offset+4] != frame_marker:
			frame_start = raw_data.index(frame_marker, frame_offset)
			print '### skipped',frame_start-frame_offset,'at',frame_offset
			frame_offset = frame_start
		vals = np.fromstring(raw_data[frame_offset:frame_offset+128], dtype='<i')
		frame_offset += 128
		###print len(frame_starts),hex(frame_offset),hex(data_last_frame),vals
		if first_frame_vals is None: first_frame_vals = vals
		assert np.all(vals[:14] == first_frame_vals[:14])
		assert np.all(vals[23:26] == first_frame_vals[23:26]),repr(vals[23:26])+repr(first_frame_vals[23:26])
		assert np.all(vals[27:] == first_frame_vals[27:]),repr(vals[27:])+repr(first_frame_vals[27:])
		#print vals[14:18],vals[18:23],vals[26]
		frame_number,chans,width,height,bit_rate,frame_type,frame_size,_4,size_184 = vals[14:23]
		assert bit_rate == 15000000 #<PicoLiveConversionDescriptor Name="....pico" Channel="2" BitRate="15000000" FrameRate="50" SizeX="1280" SizeY="720" OutputVideoFileName="....mp4" OutputFormat="mp4" StartCaptureTimeCode="12:03:59:08" StopCaptureTimeCode="12:03:59:09" TimeCodeStandard="25Hz PAL">
		assert _4 == 4
		assert frame_type in [1,2,3] # I,P,B frames [intraframe, predicted, bidirectional]
		#print frame_number,chans,width,height,frame_type,frame_size,vals[25:27]
		#if frame_type == 1: print '.'.join(map(hex,map(ord,raw_data[frame_offset:frame_offset+128])))
		assert size_184 == frame_size-184
		payload_offset,payload_data = 0,raw_data[frame_offset:frame_offset+frame_size]
		
		# payload data is delivered in packets of 188 bytes
		assert len(payload_data) % 188 == 0
		payload_data = np.fromstring(payload_data, dtype='B').reshape(-1,188)
		if not np.all(payload_data[:,0] == 71): # try to avoid a crash in a corrupted file
			which = np.where(payload_data[:,0] != 71)[0]
			first = which[0]
			print 'cut to',first,'of',len(payload_data)
			payload_data = payload_data[:first]
		assert np.all(payload_data[:,0] == 71), repr(payload_data[:,0])
		head = payload_data[:,:4]
		data = payload_data[:,4:]
		assert set(head[:,2]).union([0,1,17,31]) == set([0,1,17,31]), repr(set(head[:,2]))
		assert set(head[:,1]).union([16,17,48,64,65,80,81,112]) == set([16,17,48,64,65,80,81,112]), repr(set(head[:,1])) # b0,b4,b5,b6=0x71
		keeps = np.where(head[:,2] == 0x11)[0]
		if 0: # this data seems to be a ~2.56 GHz clock
			clocks = np.where(head[:,2] == 1)[0]
			if len(clocks): print float(np.fromstring(data[clocks[0],:8].tostring(),'>Q'))/2.56e9,data[clocks[0],:8]
		ran = head[keeps,3]&0xf
		ran2 = (np.arange(len(ran)) + ran[0])&0xf
		assert np.all(ran == ran2), repr(zip(ran,ran2))
		cull_head,cull_data = head[keeps[-1]],data[keeps[-1]]
		tmp = head[keeps,1]&0x40 # b6 marks the first and last packets
		assert np.all(tmp[[0,-1]] == 0x40)
		assert np.all(tmp[1:-1] == 0)
		keeps = keeps[:-1]
		last = keeps[-1]
		head_last,data_last = head[last],data[last]
		keeps = keeps[:-1]
		head,data = head[keeps],data[keeps]
		cut_size = data_last[0]
		if head_last[3] & 0x20: # when this flag is set, the tail packet is padded
			assert np.all(data_last[2:cut_size+1] == 0xff)
			data_last = data_last[cut_size+1:]
		payload_data = data.reshape(-1).tostring() + data_last.tostring()

		frame_data = ''
		assert payload_data[:7] == '\x00\x00\x01\xe0\x00\x00\x81'
		if payload_data[7] == chr(192): tc = payload_data[15:19]
		else: assert payload_data[7] == chr(128); tc = payload_data[10:14]
		tod = int(np.fromstring(tc,'>I'))
		th,tl = divmod(tod,65536)
		tod = (th//2)*65536 + tl
		if prev_tod == 0: prev_tod = tod
		diff = tod-prev_tod
		assert diff in [0,3600,3750], diff
		prev_tod = tod
		tod,tsc = divmod(tod-1,60 * 180000)
		tod,tmn = divmod(tod,60)
		thr = tod # max is 3 hours
		#print 'tod',prev_tod,len(frame_starts),'{thr:02}:{tmn:02}:{tsc:02.3f}'.format(thr=thr,tmn=tmn,tsc=tsc/180000.) #,map(ord,payload_data[:19])
		while block_marker in payload_data[payload_offset:]:
			block_start = payload_data.index(block_marker, payload_offset)+len(block_marker)
			assert payload_data[block_start-4:block_start] == block_marker
			if block_marker in payload_data[block_start:]:
				block_end = payload_data.find(block_marker, block_start)
			else:
				block_end = len(payload_data)
			payload_offset = block_end
			block_data = payload_data[block_start:block_end]
			block_size = block_end - block_start
			# order is 9,6,1/33 or 9_16,39,40,6,1/33 with an occasional 12 at the end
			if block_data[0] == chr(12): # tail-end dump block, the only thing that should come after the frame...
				assert block_data[-1] == chr(128)
				assert block_data[1:-1] == chr(255)*(len(block_data)-2)
				continue
			assert frame_data == '', map(ord,block_data)
			if block_data[0] == chr(9): # tag: [9,16] [9,80] [9,48]
				assert block_size == 2
				if block_data[1] == chr(16):
					assert payload_data[block_end+4] == chr(39)
				continue
			if block_data[0] == chr(39): # unknown 
				assert block_size in [48,52], block_size
				assert payload_data[block_start-6] == chr(9) # always preceeded by a [9,16] tag
				assert payload_data[block_start-5] == chr(16) # always preceeded by a [9,16] tag
				assert payload_data[block_end+4] == chr(40) # always followed by a PPS header
				continue 
			if block_data[0] == chr(40): # PPS header
				assert block_size == 38, block_size
				frame_header = block_data
				continue
			if block_data[0] == chr(6): # heartbeat?
				assert block_size in [11,12,13,22], block_size
				assert ord(payload_data[block_end+4]) in [1,33] # always followed by frame
				continue
			assert ord(block_data[0]) in [1,33]
			if block_data.count('\xff') >= len(block_data)/100: # more than 1% of data is 0xff?
				print 'suspicious; dumping', len(block_data), map(ord,block_data), block_size, map(ord,payload_data[block_start:block_end])
				print head,data
				continue
			frame_data = block_data
		frame_starts.append(write_size)
		frame_types.append(frame_type)
		assert len(frame_header) == 38
		write_data = mov_pack(frame_header,0) + mov_pack(frame_data,0)
		outfile.write(write_data)
		write_size += len(write_data)
		frame_offset += frame_size
	assert frame_offset == data_last_frame, (frame_offset,len(raw_data))
	num_frames = len(frame_starts)
	frame_starts.append(write_size) # frame_starts includes the EOF
	mvhd_str = mov_pack('mvhd' + int_pack([0,0,0,50,num_frames,0x10000,0x1000000,0,0,0x10000,0,0,0,0x10000,0,0,0,0x40000000,0,0,0,0,0,0,3]))
	trak_str = 'trak'+mov_pack('tkhd'+ int_pack([15,0,0,1,0,num_frames,0,0,0])+'\x00\x00'+int_pack([1,0,0,0,1,0,0,0,0x4000,1280,720])+'\x00\x00')
	clef_str = mov_pack('clef'+'\x00\x00\x00\x00\x05\x00\x00\x00\x02\xd0\x00\x00')
	prof_str = mov_pack('prof'+'\x00\x00\x00\x00\x05\x00\x00\x00\x02\xd0\x00\x00')
	enof_str = mov_pack('enof'+'\x00\x00\x00\x00\x05\x00\x00\x00\x02\xd0\x00\x00')
	tapt_str = mov_pack('tapt'+clef_str+prof_str+enof_str)
	edts_str = mov_pack('edts' + mov_pack('elst'+int_pack([0,1,num_frames,0,0x10000])))
	tref_str = mov_pack('tref' + mov_pack('tmcd'+int_pack([2])))
	mdia_str = 'mdia'+mov_pack('mdhd' + int_pack([0,0,0,50,num_frames,0x7fff0000]))+\
		mov_pack('hdlr' + int_pack(0) +'mhlrvideappl'+ '\x10\x00\x00\x00\x00\x01\x02\x3f'+'\x19Apple Video Media Handler')
	minf_str = 'minf'+\
		mov_pack('vmhd'+ int_pack([1, 0x8000, 0x80008000])) +\
		mov_pack('hdlr' + int_pack(0) + 'dhlrurl ' + int_pack([0,0,0]) + '\x0bDataHandler') + \
		mov_pack('dinf' + mov_pack('dref' + int_pack([0,1]) + mov_pack('url ' + int_pack(1))))
	stbl_str = 'stbl' + \
		mov_pack('stsd' + int_pack([0,1]) +\
			mov_pack('avc1' + half_pack([0,0,0,1,1,0]) + 'vmsl' + half_pack([0,0,0,512,1280,720,72,0,72,0,0,0,1]) + \
				'\x04H264' + int_pack([0,0,0,0,0,0,0])+'\x18\xff\xff' +
				mov_pack('avcC' + '\x01\x64\x00\x28\xff\xe1' + # version profile compat level nula numSPS
					'\x00\x30' + # SPS-size http://aviadr1.blogspot.co.uk/2010/05/h264-extradata-partially-explained-for.html
					'\x27\x64\x00\x28\xad\x00\xec\x05\x00\x5b\xb0\x11\x00\x00\x03\x00' +
					'\x01\x00\x00\x03\x00\x64\xe0\x20\x00\x0d\xf8\x2c\x00\x01\xbf\x05' +
					'\xdf\x7b\x83\x01\x00\x00\x6f\xc1\x60\x00\x0d\xf8\x2c\xfb\xdc\x0a' +
					'\x01\x00\x26' + # numPPS PPS-size
					'\x28\xee\x07\xf3\x90\xd2\x6b\xac\x92\x11\x06\x93\x5d\x64\xa2\x42' +
					'\x56\xba\xff\xd7\xff\xeb\xf5\xff\xaf\xd5\x44\x12\xb5\xd7\xfe\xbf' +
					'\xff\x5f\xaf\xfd\x7e\xbc'))) +\
			mov_pack('stts'+int_pack([0, 1, num_frames, 1])) +\
			mov_pack('stss' + int_pack([0,(num_frames+23)//24]+[f for f in range(1,num_frames+1,24)]))
	ctts_str = 'ctts' + int_pack([0,num_frames])
	assert len(frame_types) == num_frames
	ctts_str += np.array([[0,0],[1,0],[1,0],[1,-3]],dtype='>i')[frame_types].tostring()
	stsc_str = 'stsc' + int_pack([0,1,1,1,1])
	stsz_str = 'stsz' + int_pack([0,0,num_frames]) + int_pack(np.diff(frame_starts))
	co64_str = 'co64' + int_pack([0,num_frames])
	co64_str += np.array(frame_starts[:-1],dtype='>Q').tostring()
	trak2_str = 'trak' + mov_pack('tkhd' + int_pack([15,0,0,2,0,num_frames,0,0,0])+'\x00\x00'+int_pack([1,0,0,0,1,0,0,0,0x4000,1280,720])+'\x00\x00') +\
	mov_pack('edts'+mov_pack('elst' + int_pack([0,1,num_frames,0,0x10000]))) + \
	mov_pack('mdia' + \
	mov_pack('mdhd' + int_pack([0,0,0,50,num_frames,0x7fff0000])) + \
	mov_pack('hdlr' + int_pack(0) + 'mhlrtmcd'+int_pack([0,0,0]) + '\x0fTimeCodeHandler') +\
	mov_pack('minf' +\
	mov_pack('gmhd' + mov_pack('gmin' + int_pack([0,0x408000,0x80008000,0]))+mov_pack('tmcd' + mov_pack('tcmi'+int_pack([0,0,0xc0000,0,0xff,0xff00ff]) + '\x0dLucida Grande'))) + \
	mov_pack('hdlr' + int_pack(0) + 'dhlrurl '+int_pack([0,0,0]) + '\x0bDataHandler')+\
	mov_pack('dinf' + \
	mov_pack('dref' + int_pack([0,1]) + \
	mov_pack('url ' + int_pack(1)))) +\
	mov_pack('stbl' + \
	mov_pack('stsd' + int_pack([0,1]) + \
	mov_pack('tmcd' + int_pack([0,1,0,0,50,2,0x19000000]))) + \
	mov_pack('stts' + int_pack([0,1,1,num_frames]))+\
	mov_pack('stsc' + int_pack([0,1,1,1,1]))+\
	mov_pack('stsz' + int_pack([0,4,1]))+\
	mov_pack('stco' + int_pack([0,1,48])))))
	tmp = ''.join(map(mov_pack,(ctts_str,stsc_str,stsz_str,co64_str)))
	mdia_str += mov_pack(minf_str + mov_pack(stbl_str+tmp))
	moov_str = 'moov' + mvhd_str + mov_pack(trak_str+tapt_str+edts_str+tref_str+mov_pack(mdia_str))
	outfile.write(mov_pack(moov_str + mov_pack(trak2_str)))
	print 'wrote out.mov'
	outfile.seek(44)
	outfile.write(struct.pack('>I',write_size-32))
	outfile.write(struct.pack('>I',0x14a1c0)) # TODO ??? What is this ??? 0x10b4e2 ?? 0x14a1c0

if __name__ == '__main__':
	import sys
	pico_fn, out_fn = sys.argv[1:]
	convert_to_mov(pico_fn, out_fn)
