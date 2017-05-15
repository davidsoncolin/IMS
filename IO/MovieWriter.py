'''
unfortunately, this code is just WIP
'''

import ctypes
#try:
if 1:
	import av
	av.t = ctypes.c_int
#except:
#	print 'using av_64'
#	import av_64 as av
#	av.t = ctypes.c_long
import numpy as np

def disp(v):
	from pprint import pprint; pprint([(x,v.contents.__getattribute__(x)) for x in dir(v.contents)])

	
#static void open_video(AVFormatContext *fmt_ctx, AVStream *video_st)
def open_video(fmt_ctx, video_st, width=640, height=480):
	codec_ctx = video_st.contents.codec
	codec = av.avcodec_find_encoder(codec_ctx.contents.codec_id)
	assert codec, 'codec not found'

	# find the video encoder
	#codec_ctx = av.avcodec_alloc_context3(codec)

	# open the codec
	#avd = ctypes.POINTER(av.AVDictionary)()
	#av.av_dict_set(avd, '-vpre' , 'medium', 0)
	av.avcodec_open2(codec_ctx, codec, None)
	#av.av_dict_free(avd)

	pVFrameRGB = av.avcodec_alloc_frame() # holds the decoded frame
	vbuffer=ctypes.ARRAY(ctypes.c_uint8, width * height * 3)() # make a buffer
	av.avpicture_fill(ctypes.cast(pVFrameRGB,ctypes.POINTER(av.AVPicture)), vbuffer, av.PIX_FMT_RGB24, width, height)
	pVFrameRGB.contents.width = width
	pVFrameRGB.contents.height = height
	pVFrameRGB.contents.format = av.PIX_FMT_RGB24
	pVFrameRGB.contents.pts = 0
	pVFrameRGB.contents.pkt_pts = 0
	
	#disp(codec_ctx)

	## allocate the encoded raw picture
	#picture = av.avcodec_alloc_frame()
	#assert picture, 'failed to alloc frame'
	#size = av.avpicture_get_size(codec_ctx.contents.pix_fmt, width, height)
	#av.avpicture_fill(ctypes.cast(picture,ctypes.POINTER(av.AVPicture)), av.av_malloc(size), codec_ctx.contents.pix_fmt, width, height)
	#assert picture, 'Could not allocate picture'

	# if the output format is not YUV420P, then a temporary YUV420P
	#  picture is needed too. It is then converted to the required
	#  output format
	#tmp_picture = None
	#if codec_ctx.contents.pix_fmt != av.AV_PIX_FMT_YUV420P:
		#tmp_picture = av.alloc_picture(av.AV_PIX_FMT_YUV420P, codec_ctx.contents.width, codec_ctx.contents.height)
		#assert tmp_picture, 'Could not allocate temporary picture'
		
	return {'vrgb':pVFrameRGB, 'vbuffer':vbuffer}

def open_movie(filename, vwidth=640, vheight=480, fps=(30000,1001), bit_rate=400000):
	# initialize libavcodec, and register all codecs and formats
	av.av_register_all()
	av.avcodec_register_all()
	fmt = av.av_guess_format(None, filename, None)
	assert format, 'failed to guess format for:'+filename
	codec_id = fmt.contents.video_codec # 28 = mp4

	fmt_ctx = av.avformat_alloc_context()
	assert fmt_ctx, 'memory error'

	fmt_ctx.contents.oformat = fmt
	fmt_ctx.contents.filename = filename

	sws = av.sws_getCachedContext(None,vwidth,vheight,av.AV_PIX_FMT_RGB24,vwidth,vheight,av.AV_PIX_FMT_YUV420P,av.SWS_POINT,None,None,None)
	pVFrameRGB = av.av_frame_alloc() # holds the raw frame
	numBytes = vwidth * vheight * 3
	vbuffer=ctypes.ARRAY(ctypes.c_uint8, numBytes)() # make a buffer; TODO what about buffer alignment???
	av.avpicture_fill(ctypes.cast(pVFrameRGB,ctypes.POINTER(av.AVPicture)), vbuffer, av.PIX_FMT_RGB24, vwidth, vheight)
	pVFrame = av.av_frame_alloc() # holds the encoded frame
	av.avpicture_alloc(ctypes.cast(pVFrame,ctypes.POINTER(av.AVPicture)), av.AV_PIX_FMT_YUV420P, vwidth, vheight)

	# add the audio and video streams using the default format codecs and initialize the codecs
	video_st,audio_st,vbuffer = None,None,None
	if codec_id != av.CODEC_ID_NONE:
		codec = av.avcodec_find_encoder(codec_id)
		assert codec, 'no codec'
		video_st = av.avformat_new_stream(fmt_ctx, None)
		assert video_st, 'Could not alloc stream'
		codec_ctx = video_st.contents.codec
		codec_ctx.contents.codec_id = codec_id
		codec_ctx.contents.codec_type = av.AVMEDIA_TYPE_VIDEO
		codec_ctx.contents.bit_rate = bit_rate
		codec_ctx.contents.width = vwidth
		codec_ctx.contents.height = vheight
		# NOTE time_base = 1/fps
		codec_ctx.contents.time_base.num,codec_ctx.contents.time_base.den = fps[::-1]
		video_st.contents.time_base.num,video_st.contents.time_base.den = fps[::-1]
		codec_ctx.contents.gop_size = 12 # emit one intra frame every twelve frames at most
		codec_ctx.contents.pix_fmt = pVFrame.contents.pix_fmt # av.AV_PIX_FMT_YUV420P
		codec_ctx.max_b_frames = 2
		# some formats want stream headers to be separate
		if fmt_ctx.contents.oformat.contents.flags & av.AVFMT_GLOBALHEADER:
			codec_ctx.contents.flags = codec_ctx.contents.flags | av.CODEC_FLAG_GLOBAL_HEADER

		av.avcodec_open2(codec_ctx, codec, None)
	#if fmt.contents.audio_codec != av.CODEC_ID_NONE:
	#	audio_st = add_audio_stream(fmt_ctx, fmt.contents.audio_codec)

	# set the output parameters (must be done even if no parameters).
	#assert av.av_set_parameters(fmt_ctx, None) >= 0, 'Invalid output format parameters'

	#av.av_dump_format(fmt_ctx, 0, filename, 1)

	# now that all the parameters are set, we can open the audio and video codecs and allocate the necessary encode buffers
	#if video_st: md.update(open_video(fmt_ctx, video_st))
	#if audio_st: open_audio(fmt_ctx, audio_st)

	if not (fmt.contents.flags & av.AVFMT_NOFILE):
		ok = av.avio_open2(fmt_ctx.contents.pb, filename, av.AVIO_FLAG_WRITE, None, None)
		assert ok >= 0, 'Could not open '+filename

	av.avformat_write_header(fmt_ctx, None)
	
	md = {'fmt_ctx':fmt_ctx,'audio_st':audio_st,'video_st':video_st, 'vframe':pVFrame, 'vrgb':pVFrameRGB, 'vbuffer':vbuffer, 'pkt':av.AVPacket(), 'sws':sws, 'vwidth':vwidth, 'vheight':vheight, 'vbuffer':vbuffer}
	return md

def write_frame(md):
	fmt_ctx,audio_st,video_st = md['fmt_ctx'],md['audio_st'],md['video_st']
	# compute current audio and video time
	if audio_st:
		audio_pts = audio_st.contents.pts.val * audio_st.contents.time_base.num / float(audio_st.contents.time_base.den)
	else:
		audio_pts = 0.0

	if video_st:
		video_pts = video_st.contents.pts.val * video_st.contents.time_base.num / float(video_st.contents.time_base.den + 1e-10)
	else:
		video_pts = 0.0


	# write interleaved audio and video frames
	if not video_st or (video_st and audio_st and audio_pts < video_pts):
		write_audio_frame(fmt_ctx, audio_st)
	elif video_st:
		vrgb, vbuffer, pkt = md['vrgb'], md['vbuffer'], md['pkt']
		pVFrame = md['vframe']
		av.sws_scale(md['sws'], vrgb.contents.data, vrgb.contents.linesize, 0, md['vheight'], pVFrame.contents.data, pVFrame.contents.linesize)
		print 'write_video_frame'
		codec_ctx = video_st.contents.codec
		av.av_init_packet(pkt)
		pkt.stream_index = video_st.contents.index
		#from pprint import pprint; pprint([(x,codec_ctx.contents.__getattribute__(x)) for x in dir(codec_ctx.contents)])
		#from pprint import pprint; pprint([(x,pkt.__getattribute__(x)) for x in dir(pkt)])
		got_packet = av.t(0)
		print 'here'
		out_size = av.avcodec_encode_video2(codec_ctx, pkt, pVFrame, ctypes.pointer(got_packet))
		print 'NEVER GOT HERE', out_size, got_packet
		
		## encode the image
		#out_size = av.avcodec_encode_video(codec_ctx, vrgb, video_outbuf_size, picture)
		## if zero size, it means the image was buffered 
		#if out_size > 0:

			#if codec_ctx.contents.coded_frame.contents.pts != av.AV_NOPTS_VALUE:
				#pkt.pts= av.av_rescale_q(codec_ctx.contents.coded_frame.contents.pts, codec_ctx.contents.time_base, video_st.contents.time_base)
			#if codec_ctx.contents.coded_frame.contents.key_frame:
				#pkt.flags = pkt.flags | av.AV_PKT_FLAG_KEY
			#pkt.stream_index = video_st.contents.index
			#pkt.data = vrgb
			#pkt.size = out_size

		if got_packet:
			# write the compressed frame in the media file
			ret = av.av_interleaved_write_frame(fmt_ctx, pkt)
		else:
			ret = 0
		av.av_free_packet(pkt)

def close_movie(md):
	fmt_ctx,audio_st,video_st = md['fmt_ctx'],md['audio_st'],md['video_st']

	# write the trailer, if any.  the trailer must be written
	# before you close the CodecContexts open when you wrote the
	# header; otherwise write_trailer may try to use memory that
	# was freed on av_codec_close()
	av.av_write_trailer(fmt_ctx)

	# close each codec
	if video_st: av.close_video(fmt_ctx, video_st)
	if audio_st: av.close_audio(fmt_ctx, audio_st)

	# free the streams
	for i in range(fmt_ctx.contents.nb_streams):
		av.av_freep(fmt_ctx.contents.streams[i].contents.codec)
		av.av_freep(fmt_ctx.contents.streams[i])

	if not (fmt.contents.flags & AVFMT_NOFILE):
		# close the output file
		av.url_fclose(fmt_ctx.contents.pb)

	# free the stream
	av.av_free(fmt_ctx)

	
if __name__ == '__main__':
	md = open_movie('out.mp4')
	write_frame(md)
	close_movie(md)
