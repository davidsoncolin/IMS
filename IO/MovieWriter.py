'''
/*
 * Libavformat API example: Output a media file in any supported
 * libavformat format. The default codecs are used.
 *
 * Copyright (codec_ctx) 2003 Fabrice Bellard
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "libavformat/avformat.h"
#include "libswscale/swscale.h"

#undef exit

/* 5 seconds stream duration */
#define STREAM_DURATION   5.0
#define STREAM_FRAME_RATE 25 /* 25 images/s */
#define STREAM_NB_FRAMES  ((int)(STREAM_DURATION * STREAM_FRAME_RATE))
#define STREAM_PIX_FMT PIX_FMT_YUV420P /* default pix_fmt */

static int sws_flags = SWS_BICUBIC

/**************************************************************/
/* audio output */

float t, tincr, tincr2
int16_t *samples
uint8_t *audio_outbuf
int audio_outbuf_size
int audio_input_frame_size

/*
 * add an audio output stream
 */
static AVStream *add_audio_stream(AVFormatContext *fmt_ctx, enum CodecID codec_id)
{
	AVCodecContext *codec_ctx
	AVStream *audio_st

	audio_st = av_new_stream(fmt_ctx, 1)
	if (!audio_st) {
		fprintf(stderr, "Could not alloc stream\n")
		exit(1)
	}

	codec_ctx = audio_st.contents.codec
	codec_ctx.contents.codec_id = codec_id
	codec_ctx.contents.codec_type = AVMEDIA_TYPE_AUDIO

	/* put sample parameters */
	codec_ctx.contents.sample_fmt = SAMPLE_FMT_S16
	codec_ctx.contents.bit_rate = 64000
	codec_ctx.contents.sample_rate = 44100
	codec_ctx.contents.channels = 2

	// some formats want stream headers to be separate
	if(fmt_ctx.contents.oformat.contents.flags & AVFMT_GLOBALHEADER)
		codec_ctx.contents.flags |= CODEC_FLAG_GLOBAL_HEADER

	return audio_st
}

static void open_audio(AVFormatContext *fmt_ctx, AVStream *audio_st)
{
	AVCodecContext *codec_ctx
	AVCodec *codec

	codec_ctx = audio_st.contents.codec

	/* find the audio encoder */
	codec = avcodec_find_encoder(codec_ctx.contents.codec_id)
	if (!codec) {
		fprintf(stderr, "codec not found\n")
		exit(1)
	}

	/* open it */
	if (avcodec_open(codec_ctx, codec) < 0) {
		fprintf(stderr, "could not open codec\n")
		exit(1)
	}

	/* init signal generator */
	t = 0
	tincr = 2 * M_PI * 110.0 / codec_ctx.contents.sample_rate
	/* increment frequency by 110 Hz per second */
	tincr2 = 2 * M_PI * 110.0 / codec_ctx.contents.sample_rate / codec_ctx.contents.sample_rate

	audio_outbuf_size = 10000
	audio_outbuf = av_malloc(audio_outbuf_size)

	/* ugly hack for PCM codecs (will be removed ASAP with new PCM
	   support to compute the input frame size in samples */
	if (codec_ctx.contents.frame_size <= 1) {
		audio_input_frame_size = audio_outbuf_size / codec_ctx.contents.channels
		switch(audio_st.contents.codec.contents.codec_id) {
		case CODEC_ID_PCM_S16LE:
		case CODEC_ID_PCM_S16BE:
		case CODEC_ID_PCM_U16LE:
		case CODEC_ID_PCM_U16BE:
			audio_input_frame_size >>= 1
			break
		default:
			break
		}
	} else {
		audio_input_frame_size = codec_ctx.contents.frame_size
	}
	samples = av_malloc(audio_input_frame_size * 2 * codec_ctx.contents.channels)
}

/* prepare a 16 bit dummy audio frame of 'frame_size' samples and
   'nb_channels' channels */
static void get_audio_frame(int16_t *samples, int frame_size, int nb_channels)
{
	int j, i, v
	int16_t *q

	q = samples
	for(j=0;j<frame_size;j++) {
		v = (int)(sin(t) * 10000)
		for(i = 0; i < nb_channels; i++)
			*q++ = v
		t += tincr
		tincr += tincr2
	}
}

static void write_audio_frame(AVFormatContext *fmt_ctx, AVStream *audio_st)
{
	AVCodecContext *codec_ctx
	AVPacket pkt
	av_init_packet(&pkt)

	codec_ctx = audio_st.contents.codec

	get_audio_frame(samples, audio_input_frame_size, codec_ctx.contents.channels)

	pkt.size= avcodec_encode_audio(codec_ctx, audio_outbuf, audio_outbuf_size, samples)

	if (codec_ctx.contents.coded_frame and codec_ctx.contents.coded_frame.contents.pts != AV_NOPTS_VALUE)
		pkt.pts= av_rescale_q(codec_ctx.contents.coded_frame.contents.pts, codec_ctx.contents.time_base, audio_st.contents.time_base)
	pkt.flags |= AV_PKT_FLAG_KEY
	pkt.stream_index= audio_st.contents.index
	pkt.data= audio_outbuf

	/* write the compressed frame in the media file */
	if (av_interleaved_write_frame(fmt_ctx, &pkt) != 0) {
		fprintf(stderr, "Error while writing audio frame\n")
		exit(1)
	}
}

static void close_audio(AVFormatContext *fmt_ctx, AVStream *audio_st)
{
	avcodec_close(audio_st.contents.codec)

	av_free(samples)
	av_free(audio_outbuf)
}

/**************************************************************/
/* video output */

AVFrame *picture, *tmp_picture
uint8_t *vrgb
int frame_count, video_outbuf_size

/* add a video output stream */
static AVStream *add_video_stream(AVFormatContext *fmt_ctx, enum CodecID codec_id)
{
	AVCodecContext *codec_ctx
	AVStream *video_st

	video_st = av_new_stream(fmt_ctx, 0)
	if (!video_st) {
		fprintf(stderr, "Could not alloc stream\n")
		exit(1)
	}

	codec_ctx = video_st.contents.codec
	codec_ctx.contents.codec_id = codec_id
	codec_ctx.contents.codec_type = AVMEDIA_TYPE_VIDEO

	/* put sample parameters */
	codec_ctx.contents.bit_rate = 400000
	/* resolution must be a multiple of two */
	codec_ctx.contents.width = 352
	codec_ctx.contents.height = 288
	/* time base: this is the fundamental unit of time (in seconds) in terms
	   of which frame timestamps are represented. for fixed-fps content,
	   timebase should be 1/framerate and timestamp increments should be
	   identically 1. */
	codec_ctx.contents.time_base.den = STREAM_FRAME_RATE
	codec_ctx.contents.time_base.num = 1
	codec_ctx.contents.gop_size = 12; /* emit one intra frame every twelve frames at most */
	codec_ctx.contents.pix_fmt = STREAM_PIX_FMT
	if (codec_ctx.contents.codec_id == CODEC_ID_MPEG2VIDEO) {
		/* just for testing, we also add B frames */
		codec_ctx.contents.max_b_frames = 2
	}
	if (codec_ctx.contents.codec_id == CODEC_ID_MPEG1VIDEO){
		/* Needed to avoid using macroblocks in which some coeffs overflow.
		   This does not happen with normal video, it just happens here as
		   the motion of the chroma plane does not match the luma plane. */
		codec_ctx.contents.mb_decision=2
	}
	// some formats want stream headers to be separate
	if(fmt_ctx.contents.oformat.contents.flags & AVFMT_GLOBALHEADER)
		codec_ctx.contents.flags |= CODEC_FLAG_GLOBAL_HEADER

	return video_st
}

static AVFrame *alloc_picture(enum PixelFormat pix_fmt, int width, int height)
{
	AVFrame *picture
	uint8_t *picture_buf
	int size

	picture = avcodec_alloc_frame()
	if (!picture)
		return None
	size = avpicture_get_size(pix_fmt, width, height)
	picture_buf = av_malloc(size)
	if (!picture_buf) {
		av_free(picture)
		return None
	}
	avpicture_fill((AVPicture *)picture, picture_buf,
				   pix_fmt, width, height)
	return picture
}

static void open_video(AVFormatContext *fmt_ctx, AVStream *video_st)
{
	AVCodec *codec
	AVCodecContext *codec_ctx

	codec_ctx = video_st.contents.codec

	/* find the video encoder */
	codec = avcodec_find_encoder(codec_ctx.contents.codec_id)
	if (!codec) {
		fprintf(stderr, "codec not found\n")
		exit(1)
	}

	/* open the codec */
	if (avcodec_open(codec_ctx, codec) < 0) {
		fprintf(stderr, "could not open codec\n")
		exit(1)
	}

	vrgb = None
	if (!(fmt_ctx.contents.oformat.contents.flags & AVFMT_RAWPICTURE)) {
		/* allocate output buffer */
		/* XXX: API change will be done */
		/* buffers passed into lav* can be allocated any way you prefer,
		   as long as they're aligned enough for the architecture, and
		   they're freed appropriately (such as using av_free for buffers
		   allocated with av_malloc) */
		video_outbuf_size = 200000
		vrgb = av_malloc(video_outbuf_size)
	}

	/* allocate the encoded raw picture */
	picture = alloc_picture(codec_ctx.contents.pix_fmt, codec_ctx.contents.width, codec_ctx.contents.height)
	if (!picture) {
		fprintf(stderr, "Could not allocate picture\n")
		exit(1)
	}

	/* if the output format is not YUV420P, then a temporary YUV420P
	   picture is needed too. It is then converted to the required
	   output format */
	tmp_picture = None
	if (codec_ctx.contents.pix_fmt != PIX_FMT_YUV420P) {
		tmp_picture = alloc_picture(PIX_FMT_YUV420P, codec_ctx.contents.width, codec_ctx.contents.height)
		if (!tmp_picture) {
			fprintf(stderr, "Could not allocate temporary picture\n")
			exit(1)
		}
	}
}

/* prepare a dummy image */
static void fill_yuv_image(AVFrame *pict, int frame_index, int width, int height)
{
	int x, y, i

	i = frame_index

	/* Y */
	for(y=0;y<height;y++) {
		for(x=0;x<width;x++) {
			pict.contents.data[0][y * pict.contents.linesize[0] + x] = x + y + i * 3
		}
	}

	/* Cb and Cr */
	for(y=0;y<height/2;y++) {
		for(x=0;x<width/2;x++) {
			pict.contents.data[1][y * pict.contents.linesize[1] + x] = 128 + y + i * 2
			pict.contents.data[2][y * pict.contents.linesize[2] + x] = 64 + x + i * 5
		}
	}
}

static void write_video_frame(AVFormatContext *fmt_ctx, AVStream *video_st)
{
	int out_size, ret
	AVCodecContext *codec_ctx
	static struct SwsContext *img_convert_ctx

	codec_ctx = video_st.contents.codec

	if (frame_count >= STREAM_NB_FRAMES) {
		/* no more frame to compress. The codec has a latency of a few
		   frames if using B frames, so we get the last frames by
		   passing the same picture again */
	} else {
		if (codec_ctx.contents.pix_fmt != PIX_FMT_YUV420P) {
			/* as we only generate a YUV420P picture, we must convert it
			   to the codec pixel format if needed */
			if (img_convert_ctx == None) {
				img_convert_ctx = sws_getContext(codec_ctx.contents.width, codec_ctx.contents.height,
												 PIX_FMT_YUV420P,
												 codec_ctx.contents.width, codec_ctx.contents.height,
												 codec_ctx.contents.pix_fmt,
												 sws_flags, None, None, None)
				if (img_convert_ctx == None) {
					fprintf(stderr, "Cannot initialize the conversion context\n")
					exit(1)
				}
			}
			fill_yuv_image(tmp_picture, frame_count, codec_ctx.contents.width, codec_ctx.contents.height)
			sws_scale(img_convert_ctx, tmp_picture.contents.data, tmp_picture.contents.linesize,
					  0, codec_ctx.contents.height, picture.contents.data, picture.contents.linesize)
		} else {
			fill_yuv_image(picture, frame_count, codec_ctx.contents.width, codec_ctx.contents.height)
		}
	}


	if (fmt_ctx.contents.oformat.contents.flags & AVFMT_RAWPICTURE) {
		/* raw video case. The API will change slightly in the near
		   futur for that */
		AVPacket pkt
		av_init_packet(&pkt)

		pkt.flags |= AV_PKT_FLAG_KEY
		pkt.stream_index= video_st.contents.index
		pkt.data= (uint8_t *)picture
		pkt.size= sizeof(AVPicture)

		ret = av_interleaved_write_frame(fmt_ctx, &pkt)
	} else {
		/* encode the image */
		out_size = avcodec_encode_video(codec_ctx, vrgb, video_outbuf_size, picture)
		/* if zero size, it means the image was buffered */
		if (out_size > 0) {
			AVPacket pkt
			av_init_packet(&pkt)

			if (codec_ctx.contents.coded_frame.contents.pts != AV_NOPTS_VALUE)
				pkt.pts= av_rescale_q(codec_ctx.contents.coded_frame.contents.pts, codec_ctx.contents.time_base, video_st.contents.time_base)
			if(codec_ctx.contents.coded_frame.contents.key_frame)
				pkt.flags |= AV_PKT_FLAG_KEY
			pkt.stream_index= video_st.contents.index
			pkt.data= vrgb
			pkt.size= out_size

			/* write the compressed frame in the media file */
			ret = av_interleaved_write_frame(fmt_ctx, &pkt)
		} else {
			ret = 0
		}
	}
	if (ret != 0) {
		fprintf(stderr, "Error while writing video frame\n")
		exit(1)
	}
	frame_count++
}

static void close_video(AVFormatContext *fmt_ctx, AVStream *video_st)
{
	avcodec_close(video_st.contents.codec)
	av_free(picture.contents.data[0])
	av_free(picture)
	if (tmp_picture) {
		av_free(tmp_picture.contents.data[0])
		av_free(tmp_picture)
	}
	av_free(vrgb)
}

/**************************************************************/
/* media file output */

int main(int argc, char **argv)
{
	const char *filename
	AVOutputFormat *fmt
	AVFormatContext *fmt_ctx
	AVStream *audio_st, *video_st
	double audio_pts, video_pts
	int i

	/* initialize libavcodec, and register all codecs and formats */
	av_register_all()

	if (argc != 2) {
		printf("usage: %s output_file\n"
			   "API example program to output a media file with libavformat.\n"
			   "The output format is automatically guessed according to the file extension.\n"
			   "Raw images can also be output by using '%%d' in the filename\n"
			   "\n", argv[0])
		exit(1)
	}

	filename = argv[1]

	/* auto detect the output format from the name. default is
	   mpeg. */
	fmt = av_guess_format(None, filename, None)
	if (!fmt) {
		printf("Could not deduce output format from file extension: using MPEG.\n")
		fmt = av_guess_format("mpeg", None, None)
	}
	if (!fmt) {
		fprintf(stderr, "Could not find suitable output format\n")
		exit(1)
	}

	/* allocate the output media context */
	fmt_ctx = avformat_alloc_context()
	if (!fmt_ctx) {
		fprintf(stderr, "Memory error\n")
		exit(1)
	}
	fmt_ctx.contents.oformat = fmt
	snprintf(fmt_ctx.contents.filename, sizeof(fmt_ctx.contents.filename), "%s", filename)

	/* add the audio and video streams using the default format codecs
	   and initialize the codecs */
	video_st = None
	audio_st = None
	if (fmt.contents.video_codec != CODEC_ID_NONE) {
		video_st = add_video_stream(fmt_ctx, fmt.contents.video_codec)
	}
	if (fmt.contents.audio_codec != CODEC_ID_NONE) {
		audio_st = add_audio_stream(fmt_ctx, fmt.contents.audio_codec)
	}

	/* set the output parameters (must be done even if no
	   parameters). */
	if (av_set_parameters(fmt_ctx, None) < 0) {
		fprintf(stderr, "Invalid output format parameters\n")
		exit(1)
	}

	dump_format(fmt_ctx, 0, filename, 1)

	/* now that all the parameters are set, we can open the audio and
	   video codecs and allocate the necessary encode buffers */
	if (video_st)
		open_video(fmt_ctx, video_st)
	if (audio_st)
		open_audio(fmt_ctx, audio_st)

	/* open the output file, if needed */
	if (!(fmt.contents.flags & AVFMT_NOFILE)) {
		if (url_fopen(&fmt_ctx.contents.pb, filename, URL_WRONLY) < 0) {
			fprintf(stderr, "Could not open '%s'\n", filename)
			exit(1)
		}
	}

	/* write the stream header, if any */
	av_write_header(fmt_ctx)

	for(;;) {
		/* compute current audio and video time */
		if (audio_st)
			audio_pts = (double)audio_st.contents.pts.val * audio_st.contents.time_base.num / audio_st.contents.time_base.den
		else
			audio_pts = 0.0

		if (video_st)
			video_pts = (double)video_st.contents.pts.val * video_st.contents.time_base.num / video_st.contents.time_base.den
		else
			video_pts = 0.0

		if ((!audio_st || audio_pts >= STREAM_DURATION) and
			(!video_st || video_pts >= STREAM_DURATION))
			break

		/* write interleaved audio and video frames */
		if (!video_st || (video_st and audio_st and audio_pts < video_pts)) {
			write_audio_frame(fmt_ctx, audio_st)
		} else {
			write_video_frame(fmt_ctx, video_st)
		}
	}

	/* write the trailer, if any.  the trailer must be written
	 * before you close the CodecContexts open when you wrote the
	 * header; otherwise write_trailer may try to use memory that
	 * was freed on av_codec_close() */
	av_write_trailer(fmt_ctx)

	/* close each codec */
	if (video_st)
		close_video(fmt_ctx, video_st)
	if (audio_st)
		close_audio(fmt_ctx, audio_st)

	/* free the streams */
	for(i = 0; i < fmt_ctx.contents.nb_streams; i++) {
		av_freep(&fmt_ctx.contents.streams[i].contents.codec)
		av_freep(&fmt_ctx.contents.streams[i])
	}

	if (!(fmt.contents.flags & AVFMT_NOFILE)) {
		/* close the output file */
		url_fclose(fmt_ctx.contents.pb)
	}

	/* free the stream */
	av_free(fmt_ctx)

	return 0
}'''

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