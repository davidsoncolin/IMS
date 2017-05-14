import ctypes
try:
	import av
	av.t = ctypes.c_int
except:
	print 'using av_64'
	import av_64 as av
	av.t = ctypes.c_long
import numpy as np
import pygame

global g_av_is_initialised
g_av_is_initialised = False

def setVolume(md, level):
	if level == 0:
		alookup = None
	else:
		alookup = np.arange(65536,dtype=np.int16)
		for it in range(level): alookup = np.int16(np.sin(np.pi*0.5*(alookup/65535.))*65535.)
	md['alookup'] = alookup

#@profile
def resampleAudio(audio, md):
	'''Convert input audio samples to 48000,s16,stereo.'''
	fmt = md['asamplefmt']
	fmt,planar = fmt%5,fmt/5
	adtype = [np.uint8,np.int16,np.int32,np.float32,np.float64][fmt]
	bytes_per_inchannel_sample = [1,2,4,4,8][fmt]
	audio = [np.fromstring(a,dtype=adtype) for a in audio]
	if planar: audio = [a.reshape(md['ainchannels'],-1).T.reshape(-1) for a in audio] # change planar to interlaced before concatenating
	if fmt > av.AV_SAMPLE_FMT_S32: # convert FLT and DBL to S32
		audio = [np.int32(a*0x7fffffff) for a in audio]
		fmt,adtype,bytes_per_inchannel_sample = av.AV_SAMPLE_FMT_S32,np.int32,4
	audio = np.concatenate(audio).tostring()
	ai = ctypes.cast(audio,ctypes.POINTER(ctypes.c_short))
	bytes_per_in_sample = md['ainchannels'] * bytes_per_inchannel_sample
	bytes_per_out_sample = md['aoutchannels'] * 2
	num_in_samples = len(audio)/bytes_per_in_sample
	audio_out_size = bytes_per_out_sample*np.ceil(num_in_samples*48000./md['asamplerate'])
	audio_out = ' '*int(audio_out_size) #(ctypes.c_short * (audio_out_size/2))()
	ao = ctypes.cast(audio_out,ctypes.POINTER(ctypes.c_short))
	num_out_samples = av.audio_resample(md['aresample'], ao, ai, num_in_samples)
	assert(num_out_samples*bytes_per_out_sample <= audio_out_size),repr(num_out_samples*bytes_per_out_sample)+'<='+repr(audio_out_size)
	audio_out = np.fromstring(audio_out,dtype=np.int16)
	audio_out = audio_out[:num_out_samples*bytes_per_out_sample/2]
	if md['alookup'] is not None: audio_out = md['alookup'][audio_out]
	if md['aoutchannels'] != 2: # the resampler only outputs stereo if the input is stereo/mono; otherwise we take the first two channels
		tmp = audio_out
		audio_out = np.zeros(len(tmp)*2/md['ainchannels'],dtype=np.int16)
		audio_out[::2] = tmp[::md['ainchannels']]
		audio_out[1::2] = tmp[1::md['ainchannels']]
	return audio_out

#@profile
def readFrame(md, seekFrame=None, playingAudio=True, assumedFps=30000./1001., debugging=False):
	pVCodecCtx, pVFrame, videoStream = md['vcodec'], md['vframe'], md['vstream']
	pACodecCtx, pAFrame, audioStream = md['acodec'], md['aframe'], md['astream']
	pTcCodecCtx, pDFrame, tcStream = md['dcodec'], md['dframe'], md['dstream']
	seekFrame = max(0,seekFrame)
	if md['vcodec'] is not None and seekFrame > md['vmaxframe']: return False
	if seekFrame == md['frameNumber']:
		if videoStream is not None:
			av.sws_scale(md['sws'], pVFrame.contents.data, pVFrame.contents.linesize, 0, md['vheight'],
						 md['vrgb'].contents.data, md['vrgb' ].contents.linesize)
		return True # same frame; do nothing
	if md['frameNumber'] is not None and (seekFrame <= md['frameNumber'] or seekFrame >= md['frameNumber']+10):
		if audioStream is not None:
			seekTarget = int(seekFrame * md['aduration'] + md['aoffset'])
			tmp = av.av_seek_frame(md['aformat'], audioStream, min(max(seekTarget,md['aoffset']),md['aformat'].contents.duration), av.AVSEEK_FLAG_ANY)
			md['aplayedtoframe'] = 0 # cancel the player
		if videoStream is not None:
			#print 'seekFrame',seekFrame,'last frame',md['frameNumber']
			seekTarget = max(md['voffset'],int((seekFrame) * md['vduration'] + md['voffset'])) # 32 for old cara files
			#print 'seek',seekTarget
			av.avcodec_flush_buffers(pVCodecCtx)
			tmp = av.av_seek_frame(md['vformat'], videoStream, seekTarget, av.AVSEEK_FLAG_BACKWARD if seekTarget != md['voffset'] else av.AVSEEK_FLAG_BACKWARD)
			assert (tmp >= 0), 'ERROR:movie seek failed with code '+repr(tmp)
	packet = av.AVPacket()
	if audioStream is not None:
		if md['frameNumber'] is not None and seekFrame > md['frameNumber'] and seekFrame < min(md['frameNumber']+10,md['aplayedtoframe']):
			#print 'skip',seekFrame
			md['frameNumber'] = seekFrame
		else:
			while (av.av_read_frame(md['aformat'], packet)>=0):
				if packet.stream_index==audioStream: # From the audio stream?
					if md['aduration'] == 0:
						md['aduration'] = int(packet.duration)
						md['aduration'] = int(md['asamplerate']/float(assumedFps)) # needed for WAV
						if debugging: print 'setting frame aduration',md['aduration']
					if md['aduration'] != 0:
						frameFinished = av.t(0)
						decoded = av.avcodec_decode_audio4(pACodecCtx, pAFrame, ctypes.pointer(frameFinished), packet)
						if decoded < 1 or not frameFinished:
							print 'error decoding audio'; av.av_free_packet(packet); continue
						data_size = av.av_samples_get_buffer_size(None, md['ainchannels'], pAFrame.contents.nb_samples, pACodecCtx.contents.sample_fmt, 1)
						if (data_size < 0): break # no audio
						# HACK HACK stereo data in planes is split over two buffers. the length is wrong. I don't know why.
						if md['ainchannels'] == 2 and md['asamplefmt'] >= 5:
							tmp = ctypes.string_at(pAFrame.contents.data[0],data_size/2) + ctypes.string_at(pAFrame.contents.data[1],data_size/2)
						else:
							tmp = ctypes.string_at(pAFrame.contents.data[0],data_size)
						if playingAudio: md['adata'].append(tmp)
						foundFrame = (pAFrame.contents.pkt_pts - md['aoffset']) / md['aduration']
						if md['frameNumber'] is None: md['aoffset'] = pAFrame.contents.pkt_pts
						if (foundFrame > seekFrame) or md['aplayedtoframe'] is None:
							av.av_free_packet(packet)
							md['aplayedtoframe'] = foundFrame
							if videoStream is None: md['frameNumber'] = seekFrame
							break
				av.av_free_packet(packet)
	if videoStream is not None:
		while (av.av_read_frame(md['vformat'], packet)>=0):
			''' # testing timecode
			if packet.stream_index == tcStream: # Is this a packet from the timecode stream?
				# not sure what to expect here. seems to be one packet. don't know how to decode it.. or what info it might contain.
				print "tcstream packet :"
				for x in dir(packet): print x,packet.__getattribute__(x)
			'''
			if packet.stream_index == videoStream: # Is this a packet from the video stream?
				if md['vduration'] == 0:
					md['vduration'] = int(packet.duration)
					if debugging: print 'setting frame vduration',md['vduration']
				if md['vduration'] != 0:
					#print 'packet duration,pts',packet.duration,packet.pts
					frameFinished = av.t(0)
					if (av.avcodec_decode_video2(pVCodecCtx, pVFrame, ctypes.pointer(frameFinished), packet) < 1):
						print 'error decoding video'; av.av_free_packet(packet); continue
					if not frameFinished: continue
					foundFrame = (pVFrame.contents.pkt_pts - md['voffset']) / md['vduration']
					if pVFrame.contents.pkt_pts < 0: foundFrame = seekFrame
					#print 'v',foundFrame,pVFrame.contents.pkt_pts,md['vduration']
					if foundFrame >= seekFrame:
						if md['frameNumber'] is None:
							md['voffset'] = pVFrame.contents.pkt_pts
							if debugging: print 'setting voffset',md['voffset']
						av.sws_scale(md['sws'], pVFrame.contents.data, pVFrame.contents.linesize, 0, md['vheight'],
									 md['vrgb'].contents.data, md['vrgb' ].contents.linesize)
						av.av_free_packet(packet)
						md['frameNumber'] = foundFrame
						break
			av.av_free_packet(packet)
	if md['frameNumber'] == seekFrame:
		if playingAudio and len(md['adata']):
			audio = resampleAudio(md['adata'],md)
			amixer = pygame.mixer.Sound(buffer(audio))
			channel = md.get('achannel',None)
			if channel is None or channel.get_queue() is not None: channel = pygame.mixer.find_channel(True)
			#pygame.mixer.stop()
			channel.queue(amixer)
			md['achannel'] = channel
			md['amixer'].append(amixer)
			# we keep a reference to these buffers for a while to prevent crashes
			if len(md['amixer'])>=20: md['amixer'] = md['amixer'][10:]
			md['adata'] = []
		return True

	return False

def open_file(filename, audio=True, frame_offset=0, debugging=False, volume_ups=0):
	'''Setup a file decoding nightmare. We actually create two nightmares, one for the audio and one for the video.
	This should allow us to synchronize without buffering (though it could be IO heavy, depending on how clever the file caching is).'''
	global g_av_is_initialised
	if not g_av_is_initialised:
		g_av_is_initialised = True
		av.av_register_all()
		av.av_log_set_level(av.AV_LOG_VERBOSE if debugging else av.AV_LOG_QUIET)
	aformat,aresample,ainchannels,aoutchannels,asamplerate,aduration = None,None,0,2,48000,0
	if debugging: print "startAudio"
	try:
		if not audio: raise Exception('unwanted')
		for it in range(10):
			try:
				aformat = av.avformat_alloc_context()
				if av.avformat_open_input(aformat, filename, None, None):
					raise Exception('error opening audio file %s' % filename)
				tmp = av.avformat_find_stream_info(aformat, None)
				if tmp >= 0: break # if we don't do this, some fields won't be filled in :-(
			except Exception as e:
				print '.',e
		if it == 9: raise Exception('error retrieving audio stream info %s' % filename)
		streams = [aformat.contents.streams[stream].contents.codec.contents.codec_type for stream in xrange(aformat.contents.nb_streams)]
		pygame.mixer.init(frequency=48000, size=-16, channels=2, buffer=512)
		audioStreams = np.where(np.array(streams) == av.AVMEDIA_TYPE_AUDIO)[0]
		audioStream = audioStreams[0] # the first audio stream
		audioStream = audioStreams[-1] # the last audio stream
		pACodecCtx = aformat.contents.streams[audioStream].contents.codec
		audioCodec = av.avcodec_find_decoder(pACodecCtx.contents.codec_id)
		av.avcodec_open2(pACodecCtx, audioCodec, None)
		ainchannels = pACodecCtx.contents.channels
		asamplerate = pACodecCtx.contents.sample_rate
		asamplefmt = pACodecCtx.contents.sample_fmt
		print 'ainfo',ainchannels,asamplerate, asamplefmt
		aoutchannels = max(2,ainchannels)
		fmt = min(asamplefmt%5,av.AV_SAMPLE_FMT_S32)
		aresample = av.av_audio_resample_init(aoutchannels, ainchannels, 48000, asamplerate, av.AV_SAMPLE_FMT_S16, fmt, 16, 10, 0, 0.8)
		#print aresample,aresample.contents # fails if NULL
		pAFrame = av.av_frame_alloc() # holds the codec frame
		#for x in dir(aformat.contents): print x,aformat.contents.__getattribute__(x)
		#for x in dir(pACodecCtx.contents): print x,pACodecCtx.contents.__getattribute__(x)
	except Exception, e:
		if e.args[0] == 'unwanted' :
			if debugging:
				print "No Audio Requested"
		else:
			print 'no audio: ', e
		audioStream,pACodecCtx,audioCodec,pAFrame,asamplefmt = None,None,None,None,None
	if debugging: print 'endAudio\nstartVideo'
	try:
		for it in range(10):
			print '.',
			try:
				vformat = av.avformat_alloc_context()
				if av.avformat_open_input(vformat, filename, None, None):
					raise Exception('error opening video file %s' % filename)
				streams = [vformat.contents.streams[stream].contents.codec.contents.codec_type for stream in xrange(vformat.contents.nb_streams)]
				tmp = av.avformat_find_stream_info(vformat, None)
				if tmp >= 0: break # if we don't do this, some fields won't be filled in :-(
			except:
				pass
		if it == 9: raise Exception('error retrieving video stream info %s' % filename)
		videoStream = streams.index(av.AVMEDIA_TYPE_VIDEO) # the first video stream
		av.av_dump_format(vformat, 0, filename, False)
		pVCodecCtx = vformat.contents.streams[videoStream].contents.codec
		#for x in dir(pVCodecCtx.contents): print x,pVCodecCtx.contents.__getattribute__(x)
		videoCodec = av.avcodec_find_decoder(pVCodecCtx.contents.codec_id)
		av.avcodec_open2(pVCodecCtx, videoCodec, None)
		pVFrame = av.av_frame_alloc() # holds the codec frame
		pVFrameRGB = av.av_frame_alloc() # holds the decoded frame
		vwidth,vheight = pVCodecCtx.contents.width,pVCodecCtx.contents.height
		vwidth,vheight = int((vwidth+7)/8)*8, int((vheight+7)/8)*8
		numBytes = vwidth * vheight * 3
		vbuffer=ctypes.ARRAY(ctypes.c_uint8, numBytes)() # make a buffer; TODO what about buffer alignment?
		av.avpicture_fill( ctypes.cast(pVFrameRGB,ctypes.POINTER(av.AVPicture)), vbuffer, av.PIX_FMT_RGB24, vwidth, vheight)
		vfps = vformat.contents.streams[videoStream].contents.avg_frame_rate.num / float(vformat.contents.streams[videoStream].contents.avg_frame_rate.den+1e-8)
		if vfps == 0.0: vfps = 30.0
		if debugging: print 'vfps', vfps
		aduration = asamplerate/vfps # NOTE, we believe in this case the audio packet pts is SAMPLES, not TICKS. aduration is SAMPLES PER FRAME
		vmaxframe = vformat.contents.streams[videoStream].contents.nb_frames
		if vmaxframe <= 0: vmaxframe = 8000
	except Exception, e:
		print 'no video: ',e
		videoStream,pVCodecCtx,videoCodec,pVFrame,pVFrameRGB,vwidth,vheight = None,None,None,None,None,0,0
		vmaxframe=0
		vbuffer=''
		vfps = None
	# given we have a video stream, attempt to read timecode from data stream
	try:
		tcStream = streams.index(av.AVMEDIA_TYPE_DATA)
		pTcCodecCtx = vformat.contents.streams[tcStream].contents.codec
		pTcFrame = av.av_frame_alloc() # holds the codec frame
		#for x in dir(dformat.contents.streams[timecodeStream].contents): print x,dformat.contents.streams[timecodeStream].contents.__getattribute__(x)
		#timebase = dformat.contents.streams[timecodeStream].contents.time_base
		#print 'tb',timebase, timebase.num, timebase.den
		metadataPointer = vformat.contents.streams[tcStream].contents.metadata
		tag = ctypes.POINTER(av.AVDictionaryEntry)()
		e = av.av_dict_get(metadataPointer, "timecode", tag, av.AV_DICT_IGNORE_SUFFIX)
		if debugging: print 'e',e, 'k',e.contents.key, 'v',e.contents.value
		timecode = e.contents.value
	except Exception, e:
		print "no timecode:", e
		pTcCodecCtx, tcStream, pTcFrame = None, None, None
		timecode = None

	if pVCodecCtx is not None:
		sws = av.sws_getCachedContext(None,vwidth,vheight,pVCodecCtx.contents.pix_fmt,vwidth,vheight,av.PIX_FMT_RGB24,av.SWS_BICUBIC,None,None,None)
	else:
		sws = None

	md = {'dstream':tcStream,'dframe':pTcFrame,'dcodec':pTcCodecCtx,
		  'adata':[], 'amixer':[],'aformat':aformat, 'acodec':pACodecCtx, 'aframe':pAFrame, 'astream':audioStream, 'aduration':aduration,\
		  'aresample':aresample, 'alookup':None, 'ainchannels':ainchannels, 'aoutchannels':aoutchannels, 'asamplerate':asamplerate, 'asamplefmt':asamplefmt, 'aplayedtoframe':None, 'aoffset':0, \
		  'vformat':vformat, 'vcodec':pVCodecCtx, 'vframe':pVFrame, 'vrgb':pVFrameRGB, 'vbuffer':vbuffer, 'vwidth':vwidth, 'vheight':vheight, 'vstream':videoStream, 'vduration':0, 'voffset':0, 'vmaxframe':vmaxframe, \
		  'sws':sws, 'frameNumber':None, 'timecode':timecode, 'fps':vfps}
	setVolume(md, volume_ups)
	readFrame(md, frame_offset, debugging=debugging)
	#print md
	return md

if __name__ == '__main__':
	from UI import QGLViewer,QApp,GLMeshes
	import sys

	def set_frame_cb(frame):
		global md
		readFrame(md, seekFrame=frame)
		img = np.frombuffer(md['vbuffer'],dtype=np.uint8).reshape(md['vheight'],md['vwidth'],3)
		image_mesh = QApp.app.getLayer('image_mesh')
		image_mesh.setImage(img)
		view = QApp.view()
		view.refreshImageData()
		view.updateGL()

	if len(sys.argv) > 1:
		filename = sys.argv[1]
		md = open_file(filename)
		if len(sys.argv) > 2: setVolume(md,int(sys.argv[2]))
		img = np.frombuffer(md['vbuffer'],dtype=np.uint8).reshape(md['vheight'],md['vwidth'],3)
		h,w = md['vheight']/2,md['vwidth']/2
		img_vs = [[-w,-h,0],[w,-h,0],[w,h,0],[-w,h,0]]
		img_fs = [[0,1,2,3]]
		img_ts = np.array([[1,0,0,0],[0,1,0,1000],[0,0,1,0]], dtype=np.float32)
		img_vts = [[0,1],[1,1],[1,0],[0,0]]
		image_mesh = GLMeshes(names=['image_mesh'],verts=[img_vs],faces=[img_fs],transforms=[img_ts],bones = [[]], vts=[img_vts])
		image_mesh.setImage(img)
		QGLViewer.makeViewer(timeRange=(0,md['vmaxframe'],1,md['fps']), callback=set_frame_cb, layers={'image_mesh':image_mesh})
		exit()
		#'Imaginarium Movie Playback Tool'
	if len(sys.argv) > 1:
		import cv2
		appname = sys.argv.pop(0)
		filename = sys.argv.pop(0)
		rotate = False
		if filename == '-r180': rotate = True; filename = sys.argv.pop(0)
		md1 = open_file(filename, audio=False)
		img1 = np.frombuffer(md1['vbuffer'],dtype=np.uint8).reshape(md1['vheight'],md1['vwidth'],3)
		fn = filename.rpartition('.')[0]+'.%04d.png'
		assert fn != filename and fn != ''
		fi,fo = 0,None
		if sys.argv: fi = int(sys.argv.pop(0))
		if sys.argv: fo = int(sys.argv.pop(0))
		while readFrame(md1, fi):
			if rotate: img1[:] = img1.ravel()[::-1].reshape(img1.shape)
			cv2.imwrite(fn % fi, img1)
			fi += 1
			print '\rwrote',fi,'frames',
			sys.stdout.flush()
			if fi == fo: break
		exit()
