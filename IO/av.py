from ctypes import *

_libraries = {}
STRING = c_char_p
WSTRING = c_wchar_p
uint8_t = c_uint8
uint16_t = c_uint16
uint32_t = c_uint32
uint64_t = c_uint64
int8_t = c_int8
int16_t = c_int16
int32_t = c_int32
int64_t = c_int64

try:
	# ubuntu 16.04
	_libraries['libavcodec.so'] = CDLL('libavcodec.so')
	_libraries['libavformat.so'] = CDLL('libavformat.so')
	_libraries['libswscale.so'] = CDLL('libswscale.so')
	_libraries['libavutil.so'] = CDLL('libavcodec.so')
except:
	_libraries['libavcodec.so'] = CDLL('avcodec-56.dll')# compatible with ffmpeg-2.8.6-1
	_libraries['libavformat.so'] = CDLL('avformat-56.dll')#
	_libraries['libswscale.so'] = CDLL('swscale-3.dll')#
	_libraries['libavutil.so'] = CDLL('avutil-54.dll')# compatible with ffmpeg-2.8.6-1

AVPixelFormat = c_int # enum
AVPictureType = c_int # enum
AVSampleFormat = c_int # enum
CodecID = c_int # enum
AVMediaType = c_int # enum
AVColorPrimaries = c_int # enum
AVColorTransferCharacteristic = c_int # enum
AVColorSpace = c_int # enum
AVColorRange = c_int # enum
AVChromaLocation = c_int # enum
AVLPCType = c_int # enum
AVAudioServiceType = c_int # enum
AVFieldOrder = c_int # enum
AVStreamParseType = c_int # enum
AVDurationEstimationMethod = c_int # enum
AVPacketSideDataType = c_int # enum
AVDiscard = c_int # enum
AV_NUM_DATA_POINTERS = 8
AVMEDIA_TYPE_VIDEO = 0
AVMEDIA_TYPE_AUDIO = 1
AVMEDIA_TYPE_DATA = 2
AVSEEK_FLAG_ANY = 4 # Variable c_int '4'
AVSEEK_FLAG_BACKWARD = 1 # Variable c_int '1'
AV_DICT_IGNORE_SUFFIX = 2 # Variable c_int '2'
AV_LOG_QUIET = -8 # Variable c_int '-0x00000000000000008'
AV_LOG_VERBOSE = 40 # Variable c_int '40'
AV_SAMPLE_FMT_S16 = 1
AV_SAMPLE_FMT_S32 = 2
AV_SAMPLE_FMT_FLT = 3
AV_TIME_BASE = 1000000 # Variable c_int '1000000'
PIX_FMT_RGB24 = 2
SWS_FAST_BILINEAR = 1 # Variable c_int '1'
SWS_BILINEAR = 2 # Variable c_int '2'
SWS_BICUBIC = 4 # Variable c_int '4'
SWS_POINT = 16 # Variable c_int '16'
SWS_BICUBLIN = 64 # Variable c_int '64'

AV_PIX_FMT_NONE = -1 # AVPixelFormat
AV_PIX_FMT_YUV420P = 0 # planar YUV 4:2:0, 12bpp, (1 Cr & Cb sample per 2x2 Y samples)
AV_PIX_FMT_YUYV422 = 1 # packed YUV 4:2:2, 16bpp, Y0 Cb Y1 Cr
AV_PIX_FMT_RGB24 = 2   # packed RGB 8:8:8, 24bpp, RGBRGB...
AV_PIX_FMT_BGR24 = 3   #  packed RGB 8:8:8, 24bpp, BGRBGR...

AVFMT_NOFILE = 1

CODEC_ID_NONE = 0  # AVCodecTag
CODEC_ID_MOV_TEXT = 0x08
CODEC_ID_MPEG4 = 0x20
CODEC_ID_H264 = 0x21
CODEC_ID_AAC = 0x40
CODEC_ID_MPEG2VIDEO = 0x61
CODEC_ID_MJPEG = 0x6C

CODEC_FLAG_GLOBAL_HEADER = 0x00400000

AVFMT_GLOBALHEADER = 0x0040
AVFMT_RAWPICTURE = 0x0020
AVIO_FLAG_READ = 1
AVIO_FLAG_WRITE = 2
AV_CODEC_ID_H264 = 28

class AVDictionary(Structure): pass
class AVFrame(Structure): pass
class AVCodecContext(Structure): pass
class AVCodec(Structure): pass
class RcOverride(Structure): pass
class AVIndexEntry(Structure): pass
class AVProbeData(Structure): pass
class AVCodecParserContext(Structure): pass
class AVCodecParser(Structure): pass
class AVProfile(Structure): pass
class AVCodecDefault(Structure): pass
class AVFrac(Structure): pass
class AVCodecInternal(Structure): pass
class AVDictionaryEntry(Structure): pass
class AVHWAccel(Structure): pass
class AVPacketSideData(Structure): pass
class AVBuffer(Structure): pass
class AVBufferRef(Structure): pass
class AVPacket(Structure): pass
class AVPacketList(Structure): pass
class N8AVStream4DOT_33E(Structure): pass
class AVOption(Structure): pass
class AVClass(Structure): pass
class AVPaletteControl(Structure): pass
class AVRational(Structure): pass
class AVPanScan(Structure): pass
class AVIOInterruptCB(Structure): pass
class AVChapter(Structure): pass
class AVProgram(Structure): pass
class AVIOContext(Structure): pass
class AVOutputFormat(Structure): pass
class AVInputFormat(Structure): pass
class AVFormatContext(Structure): pass
class AVCodecDescriptor(Structure): pass
class FFFrac(Structure): pass
class AVStream(Structure): pass
class AVPicture(Structure): pass
class ReSampleContext(Structure): pass
class SwsContext(Structure): pass
class SwsVector(Structure): pass
class SwsFilter(Structure):  pass
class ByteIOContext(Structure): pass

#AVIndexEntry._fields_ = [
	#('pos', int64_t),
	#('timestamp', int64_t),
	#('flags', c_int, 2),
	#('size', c_int, 30),
	#('min_distance', c_int),
#]

#AVProfile._fields_ = [
	#('profile', c_int),
	#('name', STRING),
#]

AVOutputFormat._fields_ = [
	('name', STRING),
	('long_name', STRING),
	('mime_type', STRING),
	('extensions', STRING),
	('audio_codec', c_int), # enum AVCodecID
	('video_codec', c_int), # enum AVCodecID
	('subtitle_codec', c_int), #enum AVCodecID
	('flags', c_int), # AVFMT_NOFILE, AVFMT_NEEDNUMBER, AVFMT_GLOBALHEADER, AVFMT_NOTIMESTAMPS, AVFMT_VARIABLE_FPS, AVFMT_NODIMENSIONS, AVFMT_NOSTREAMS, AVFMT_ALLOW_FLUSH, AVFMT_TS_NONSTRICT, AVFMT_TS_NEGATIVE
	# ...
]

AVFrac._fields_ = [
	('val', int64_t),
	('num', int64_t),
	('den', int64_t),
]

AVDictionaryEntry._fields_ = [
	('key', STRING),
	('value', STRING),
]

#AVPacketSideData._fields_ = [
	#('data', POINTER(uint8_t)),
	#('size', c_int),
	#('type', AVPacketSideDataType),
#]

#AVBufferRef._fields_ = [
	#('buffer', POINTER(AVBuffer)),
    #('data', POINTER(uint8_t)),
    #('size', c_int)
#]

AVPacket._fields_ = [
	('buf', POINTER(AVBufferRef)),
	('pts', int64_t),
	('dts', int64_t),
	('data', POINTER(uint8_t)),
	('size', c_int),
	('stream_index', c_int),
	('flags', c_int),
	('side_data', POINTER(AVPacketSideData)),
	('side_data_elems', c_int),
	('duration', c_int),
	('destruct', CFUNCTYPE(None, POINTER(AVPacket))),
	('priv', c_void_p),
	('pos', int64_t),
	('convergence_duration', int64_t),
]

#N8AVStream4DOT_33E._fields_ = [
	#('last_dts', int64_t),
	#('duration_gcd', int64_t),
	#('duration_count', c_int),
	#('rfps_duration_sum', int64_t),
	#('duration_error', c_double * 2 * (30*12+30+3+6)),
	#('codec_info_duration', int64_t),
	#('codec_info_duration_fields', int64_t),
	#('found_decoder', c_int),
	#('last_duration', int64_t),
	#('fps_first_dts', int64_t),
	#('fps_first_dts_idx', c_int),
	#('fps_last_dts', int64_t),
	#('fps_last_dts_idx', c_int),
#]

AVRational._fields_ = [
	('num', c_int),
	('den', c_int),
]

AVCodecContext._fields_ = [
	('av_class', POINTER(AVClass)),
	('log_level_offset', c_int),
	('codec_type', AVMediaType),
	('codec', POINTER(AVCodec)),
	('codec_name', c_char * 32),
	('codec_id', CodecID),
	('codec_tag', c_uint),
	('stream_codec_tag', c_uint),
	('priv_data', c_void_p),
	('internal', POINTER(AVCodecInternal)),
	('opaque', c_void_p),
	('bit_rate', c_int),
	('bit_rate_tolerance', c_int),
	('global_quality', c_int),
	('compression_level', c_int),
	('flags', c_int),
	('flags2', c_int),
	('extradata', POINTER(uint8_t)),
	('extradata_size', c_int),
	('time_base', AVRational),
	('ticks_per_frame', c_int),
	('delay', c_int),
	('width', c_int),
	('height', c_int),
	('coded_width', c_int),
	('coded_height', c_int),
	('gop_size', c_int),
	('pix_fmt', AVPixelFormat),
	('me_method', c_int),
	('draw_horiz_band', CFUNCTYPE(None,     POINTER(AVCodecContext), POINTER(AVFrame), POINTER(c_int), c_int, c_int, c_int)),
	('get_format', CFUNCTYPE(AVPixelFormat, POINTER(AVCodecContext), POINTER(AVPixelFormat))),
	('max_b_frames', c_int),
	('b_quant_factor', c_float),
	('rc_strategy', c_int),
	('b_frame_strategy', c_int),
	('b_quant_offset', c_float),
	('has_b_frames', c_int),
	('mpeg_quant', c_int),
	('i_quant_factor', c_float),
	('i_quant_offset', c_float),
	('lumi_masking', c_float),
	('temporal_cplx_masking', c_float),
	('spatial_cplx_masking', c_float),
	('p_masking', c_float),
	('dark_masking', c_float),
	('slice_count', c_int),
	('prediction_method', c_int),
	('slice_offset', POINTER(c_int)),
	('sample_aspect_ratio', AVRational),
	('me_cmp', c_int),
	('me_sub_cmp', c_int),
	('mb_cmp', c_int),
	('ildct_cmp', c_int),
	('dia_size', c_int),
	('last_predictor_count', c_int),
	('pre_me', c_int),
	('me_pre_cmp', c_int),
	('pre_dia_size', c_int),
	('me_subpel_quality', c_int),
	('dtg_active_format', c_int),
	('me_range', c_int),
	('intra_quant_bias', c_int),
	('inter_quant_bias', c_int),
	('slice_flags', c_int),
	('xvmc_acceleration', c_int),
	('mb_decision', c_int),
	('intra_matrix', POINTER(uint16_t)),
	('inter_matrix', POINTER(uint16_t)),
	('scenechange_threshold', c_int),
	('noise_reduction', c_int),
	('me_threshold', c_int),
	('mb_threshold', c_int),
	('intra_dc_precision', c_int),
	('skip_top', c_int),
	('skip_bottom', c_int),
	('border_masking', c_float),
	('mb_lmin', c_int),
	('mb_lmax', c_int),
	('me_penalty_compensation', c_int),
	('bidir_refine', c_int),
	('brd_scale', c_int),
	('keyint_min', c_int),
	('refs', c_int),
	('chromaoffset', c_int),
	('scenechange_factor', c_int),
	('mv0_threshold', c_int),
	('b_sensitivity', c_int),
	('color_primaries', AVColorPrimaries),
	('color_trc', AVColorTransferCharacteristic),
	('colorspace', AVColorSpace),
	('color_range', AVColorRange),
	('chroma_sample_location', AVChromaLocation),
	('slices', c_int),
	('field_order', AVFieldOrder),
	('sample_rate', c_int),
	('channels', c_int),
	('sample_fmt', AVSampleFormat),
	('frame_size', c_int),
	('frame_number', c_int),
	('block_align', c_int),
	('cutoff', c_int),
	('request_channels', c_int),
	('channel_layout', uint64_t),
	('request_channel_layout', uint64_t),
	('audio_service_type', AVAudioServiceType),
	('request_sample_fmt', AVSampleFormat),
	('get_buffer', CFUNCTYPE(c_int, POINTER(AVCodecContext), POINTER(AVFrame))),
	('release_buffer', CFUNCTYPE(None, POINTER(AVCodecContext), POINTER(AVFrame))),
	#('reget_buffer', CFUNCTYPE(None, POINTER(AVCodecContext), POINTER(AVFrame))),
	#('get_buffer2', CFUNCTYPE(c_int, POINTER(AVCodecContext), POINTER(AVFrame))), ###?
	#('refcounted_frames', c_int),
	#('qcompress', c_float),
	#('qblur', c_float),
	#('qmin', c_int),
	#('qmax', c_int),
	#('max_qdiff', c_int),
	#('rc_qsquish', c_float),
	#('rc_qmod_amp', c_float),
	#('rc_qmod_freq', c_int),
	#('rc_buffer_size', c_int),
	#('rc_override_count', c_int),
	#('rc_override', POINTER(RcOverride)),
	#('rc_eq', STRING),
	#('rc_max_rate', c_int),
	#('rc_min_rate', c_int),
	#('rc_buffer_aggressivity', c_float),
	#('rc_initial_cplx', c_float),
	#('rc_max_available_vbv_use', c_float),
	#('rc_min_vbv_overflow_use', c_float),
	#('rc_initial_buffer_occupancy', c_int),
	#('coder_type', c_int),
	#('context_model', c_int),
	#('lmin', c_int),
	#('lmax', c_int),
	#('frame_skip_threshold', c_int),
	#('frame_skip_factor', c_int),
	#('frame_skip_exp', c_int),
	#('frame_skip_cmp', c_int),
	#('trellis', c_int),
	#('min_prediction_order', c_int),
	#('max_prediction_order', c_int),
	#('timecode_frame_start', int64_t),
	#('rtp_callback', CFUNCTYPE(None, POINTER(AVCodecContext), c_void_p, c_int, c_int)),
	#('mv_bits', c_int),
	#('header_bits', c_int),
	#('i_tex_bits', c_int),
	#('p_tex_bits', c_int),
	#('i_count', c_int),
	#('p_count', c_int),
	#('skip_count', c_int),
	#('misc_bits', c_int),
	#('frame_bits', c_int),
	#('stats_out', STRING),
	#('stats_in', STRING),
	#('workaround_bugs', c_int),
	#('strict_std_compliance', c_int),
	#('error_concealment', c_int),
	#('debug', c_int),
	#('debug_mv', c_int),
	#('error_recognition', c_int),
	#('reordered_opaque', int64_t),
	#('hwaccel', POINTER(AVHWAccel)),
	#('hwaccel_context', c_void_p),
	#('error', uint64_t * AV_NUM_DATA_POINTERS),
	#('dct_algo', c_int),
	#('idct_algo', c_int),
	#('bits_per_coded_sample', c_int),
	#('lowres', c_int),
	#('coded_frame', POINTER(AVFrame)),
	#('thread_count', c_int),
	#('thread_type', c_int),
	#('active_thread_type', c_int),
	#('thread_safe_callbacks', c_int),
	#('execute', CFUNCTYPE(c_int, POINTER(AVCodecContext), CFUNCTYPE(c_int, POINTER(AVCodecContext), c_void_p), c_void_p, POINTER(c_int), c_int, c_int)),
	#('execute2', CFUNCTYPE(c_int, POINTER(AVCodecContext), CFUNCTYPE(c_int, POINTER(AVCodecContext), c_void_p, c_int, c_int), c_void_p, POINTER(c_int), c_int)),
	#('thread_opaque', c_void_p),
	#('nsse_weight', c_int),
	#('profile', c_int),
	#('level', c_int),
	#('skip_loop_filter', AVDiscard),
	#('skip_idct', AVDiscard),
	#('skip_frame', AVDiscard),
	#('subtitle_header', POINTER(uint8_t)),
	#('subtitle_header_size', c_int),
	#('error_rate', c_int),
	#('pkt', POINTER(AVPacket)),
	#('vbv_delay', uint64_t),
	#('side_data_only_packets', c_int),
	#('initial_padding', c_int),
	#('framerate', AVRational),
	#('sw_pix_fmt', AVPixelFormat),
	#('pkt_timebase', AVRational),
	#('codec_descriptor', POINTER(AVCodecDescriptor)),
	#('pts_correction_num_faulty_pts', int64_t),
	#('pts_correction_num_faulty_dts', int64_t),
	#('pts_correction_last_pts', int64_t),
	#('pts_correction_last_dts', int64_t),
	#('sub_charenc', STRING),
	#('sub_charenc_mode', c_int),
	#('skip_alpha', c_int),
	#('seek_preroll', c_int),
	#('debug_mv', c_int),
	#('chroma_intra_matrix', POINTER(uint16_t)),
	#('dump_separator', POINTER(uint8_t)),
	#('codec_whitelist', STRING),
	#('properties', c_uint)
]

AVFrame._fields_ = [
	('data', POINTER(uint8_t) * AV_NUM_DATA_POINTERS),
	('linesize', c_int * AV_NUM_DATA_POINTERS),
	('extended_data', POINTER(POINTER(uint8_t))),
	('width', c_int),
	('height', c_int),
	('nb_samples', c_int),
	('format', c_int),
	('key_frame', c_int),
	('pict_type', AVPictureType),
	('base', POINTER(uint8_t) * AV_NUM_DATA_POINTERS),
	('sample_aspect_ratio', AVRational),
	('pts', int64_t),
	('pkt_pts', int64_t),
	#('pkt_dts', int64_t),
	#('coded_picture_number', c_int),
	#('display_picture_number', c_int),
	#('quality', c_int),
	#('reference', c_int),
	#('qscale_table', POINTER(int8_t)),
	#('qstride', c_int),
	#('qscale_type', c_int),
	#('mbskip_table', POINTER(uint8_t)),
	#('motion_val', POINTER(int16_t * 2) * 2),
	#('mb_type', POINTER(uint32_t)),
	#('dct_coeff', POINTER(c_short)),
	#('ref_index', POINTER(int8_t) * 2),
	#('opaque', c_void_p),
	#('error', uint64_t * AV_NUM_DATA_POINTERS),
	#('type', c_int),
	#('repeat_pict', c_int),
	#('interlaced_frame', c_int),
	#('top_field_first', c_int),
	#('palette_has_changed', c_int),
	#('buffer_hints', c_int),
	#('pan_scan', POINTER(AVPanScan)),
	#('reordered_opaque', int64_t),
	#('hwaccel_picture_private', c_void_p),
	#('owner', POINTER(AVCodecContext)),
	#('thread_opaque', c_void_p),
	#('motion_subsample_log2', uint8_t),
	##... many more
]

AVStream._fields_ = [
	('index', c_int),
	('id', c_int),
	('codec', POINTER(AVCodecContext)),
	('priv_data', c_void_p),
	('pts', AVFrac),
	('time_base', AVRational),
	('start_time', int64_t),
	('duration', int64_t),
	('nb_frames', int64_t),
	('disposition', c_int),
	('discard', AVDiscard),
	('sample_aspect_ratio', AVRational),
	('metadata', POINTER(AVDictionary)),
	('avg_frame_rate', AVRational),
	#('attached_pic', AVPacket),
	#('side_data', POINTER(AVPacketSideData)),
	#('nb_side_data', c_int),
	#('event_flags', c_int),
	#('info', POINTER(N8AVStream4DOT_33E)),
	#('pts_wrap_bits', c_int),
	#('first_dts', int64_t),
	#('cur_dts', int64_t),
	#('last_IP_pts', int64_t),
	#('last_IP_duration', c_int),
	#('probe_packets', c_int),
	#('codec_info_nb_frames', c_int),
	#('need_parsing', AVStreamParseType),
	#('parser', POINTER(AVCodecParserContext)),
	#('last_in_packet_buffer', POINTER(AVPacketList)),
	#('probe_data', AVProbeData),
	#('pts_buffer', int64_t * 17),
	#('index_entries', POINTER(AVIndexEntry)),
	#('nb_index_entries', c_int),
	#('index_entries_allocated_size', c_uint),
	#('r_frame_rate', AVRational),
	#('stream_identifier', c_int),
	#('interleaver_chunk_size', int64_t),
	#('interleaver_chunk_duration', int64_t),
	#('request_probe', c_int),
	#('skip_to_keyframe', c_int),
	#('skip_samples', c_int),
	#('start_skip_samples', int64_t),
	#('first_discard_sample', int64_t),
	#('last_discard_sample', int64_t),
	#('nb_decoded_frames', c_int),
	#('mux_ts_offset', int64_t),
    #('pts_wrap_reference', int64_t),
    #('pts_wrap_behavior', c_int),
    #('update_initial_durations_done', c_int),
    #('pts_reorder_error', int64_t * 17),
    #('pts_reorder_error_count', uint8_t * 17),
    #('last_dts_for_order_check', int64_t),
    #('dts_ordered', uint8_t),
    #('dts_misordered', uint8_t),
    #('inject_global_side_data', c_int),
    #('recommended_encoder_configuration', POINTER(int8_t)),
    #('display_aspect_ratio', AVRational),
	#('priv_pts', POINTER(FFFrac)),
]

AVFormatContext._fields_ = [
	('av_class', POINTER(AVClass)),
	('iformat', POINTER(AVInputFormat)),
	('oformat', POINTER(AVOutputFormat)),
	('priv_data', c_void_p),
	('pb', POINTER(AVIOContext)),
	('ctx_flags', c_int),###
	('nb_streams', c_uint),
	('streams', POINTER(POINTER(AVStream))),
	('filename', c_char * 1024),
	('start_time', int64_t),
	('duration', int64_t),
	#('bit_rate', c_int),
	#('packet_size', c_uint),
	#('max_delay', c_int),
	#('flags', c_int),
	#('probesize', c_uint),
	#('max_analyze_duration', c_int),
	#('key', POINTER(uint8_t)),
	#('keylen', c_int),
	#('nb_programs', c_uint),
	#('programs', POINTER(POINTER(AVProgram))),
	#('video_codec_id', CodecID),
	#('audio_codec_id', CodecID),
	#('subtitle_codec_id', CodecID),
	#('max_index_size', c_uint),
	#('max_picture_buffer', c_uint),
	#('nb_chapters', c_uint),
	#('chapters', POINTER(POINTER(AVChapter))),
	#('metadata', POINTER(AVDictionary)),
	#('start_time_realtime', int64_t),
	#('fps_probe_size', c_int),
	#('error_recognition', c_int),
	#('interrupt_callback', AVIOInterruptCB),
	#('debug', c_int),
	#('max_interleave_delta', int64_t),
	#('strict_std_compliance', c_int),
	#('event_flags', c_int),
	#('max_ts_probe', c_int),
	#('avoid_negative_ts', c_int),
	#('ts_id', c_int),
	#('audio_preload', c_int),
	#('max_chunk_duration', c_int),
	#('max_chunk_size', c_int),
	#('use_wallclock_as_timestamps', c_int),
	#('avio_flags', c_int),
	#('duration_estimation_method', AVDurationEstimationMethod),
	#('skip_initial_bytes', int64_t),
	#('correct_ts_overflow', c_uint),
	#('seek2any', c_int),
	#('flush_packets', c_int),
	## ...many more
]

#AVPicture._fields_ = [
	#('data', POINTER(uint8_t) * AV_NUM_DATA_POINTERS),
	#('linesize', c_int * AV_NUM_DATA_POINTERS),
#]

#SwsVector._fields_ = [
	#('coeff', POINTER(c_double)),
	#('length', c_int),
#]

#SwsFilter._fields_ = [
	#('lumH', POINTER(SwsVector)),
	#('lumV', POINTER(SwsVector)),
	#('chrH', POINTER(SwsVector)),
	#('chrV', POINTER(SwsVector)),
#]

audio_resample = _libraries['libavcodec.so'].audio_resample
audio_resample.restype = c_int
audio_resample.argtypes = [POINTER(ReSampleContext), POINTER(c_short), POINTER(c_short), c_int]

av_audio_resample_init = _libraries['libavcodec.so'].av_audio_resample_init
av_audio_resample_init.restype = POINTER(ReSampleContext)
av_audio_resample_init.argtypes = [c_int, c_int, c_int, c_int, AVSampleFormat, AVSampleFormat, c_int, c_int, c_int, c_double]

#int av_dict_set(AVDictionary **pm, const char *key, const char *value, int flags)
av_dict_set = _libraries['libavutil.so'].av_dict_set
av_dict_set.restype = c_int
av_dict_set.argtypes = [POINTER(POINTER(AVDictionary)), STRING, STRING, c_int]

#AVDictionaryEntry *av_dict_get(const AVDictionary *m, const char *key, const AVDictionaryEntry *prev, int flags)
av_dict_get = _libraries['libavutil.so'].av_dict_get
av_dict_get.restype = POINTER(AVDictionaryEntry)
av_dict_get.argtypes = [POINTER(AVDictionary), STRING, POINTER(AVDictionaryEntry), c_int]

#void av_dict_free(AVDictionary **m)
av_dict_free = _libraries['libavutil.so'].av_dict_free
av_dict_free.restype = None
av_dict_free.argtypes = [POINTER(POINTER(AVDictionary))]

av_dump_format = _libraries['libavformat.so'].av_dump_format
av_dump_format.restype = None
av_dump_format.argtypes = [POINTER(AVFormatContext), c_int, STRING, c_int]

av_free_packet = _libraries['libavcodec.so'].av_free_packet
av_free_packet.restype = None
av_free_packet.argtypes = [POINTER(AVPacket)]

av_log_set_level = _libraries['libavutil.so'].av_log_set_level
av_log_set_level.restype = None
av_log_set_level.argtypes = [c_int]

av_init_packet = _libraries['libavcodec.so'].av_init_packet
av_init_packet.restype = None
av_init_packet.argtypes = [POINTER(AVPacket)]

av_picture_copy = _libraries['libavcodec.so'].av_picture_copy
av_picture_copy.restype = None
av_picture_copy.argtypes = [POINTER(AVPicture), POINTER(AVPicture), AVPixelFormat, c_int, c_int]

av_read_frame = _libraries['libavformat.so'].av_read_frame
av_read_frame.restype = c_int
av_read_frame.argtypes = [POINTER(AVFormatContext), POINTER(AVPacket)]

av_register_all = _libraries['libavformat.so'].av_register_all
av_register_all.restype = None
av_register_all.argtypes = []

#void avcodec_register_all(void)
avcodec_register_all = _libraries['libavcodec.so'].avcodec_register_all
avcodec_register_all.restype = None
avcodec_register_all.argtypes = []

av_samples_get_buffer_size = _libraries['libavutil.so'].av_samples_get_buffer_size
av_samples_get_buffer_size.restype = c_int
av_samples_get_buffer_size.argtypes = [POINTER(c_int), c_int, c_int, AVSampleFormat, c_int]

av_seek_frame = _libraries['libavformat.so'].av_seek_frame
av_seek_frame.restype = c_int
av_seek_frame.argtypes = [POINTER(AVFormatContext), c_int, int64_t, c_int]

#AVFrame *av_frame_alloc(void)
av_frame_alloc = _libraries['libavcodec.so'].av_frame_alloc
av_frame_alloc.restype = POINTER(AVFrame)
av_frame_alloc.argtypes = []

#void av_frame_free(AVFrame **frame)
av_frame_free = _libraries['libavcodec.so'].av_frame_free
av_frame_free.restype = POINTER(POINTER(AVFrame))
av_frame_free.argtypes = []

avcodec_decode_audio4 = _libraries['libavcodec.so'].avcodec_decode_audio4
avcodec_decode_audio4.restype = c_int
avcodec_decode_audio4.argtypes = [POINTER(AVCodecContext), POINTER(AVFrame), POINTER(c_int), POINTER(AVPacket)]

avcodec_decode_video2 = _libraries['libavcodec.so'].avcodec_decode_video2
avcodec_decode_video2.restype = c_int
avcodec_decode_video2.argtypes = [POINTER(AVCodecContext), POINTER(AVFrame), POINTER(c_int), POINTER(AVPacket)]

avcodec_find_decoder = _libraries['libavcodec.so'].avcodec_find_decoder
avcodec_find_decoder.restype = POINTER(AVCodec)
avcodec_find_decoder.argtypes = [CodecID]

avcodec_flush_buffers = _libraries['libavcodec.so'].avcodec_flush_buffers
avcodec_flush_buffers.restype = None
avcodec_flush_buffers.argtypes = [POINTER(AVCodecContext)]

avcodec_open2 = _libraries['libavcodec.so'].avcodec_open2
avcodec_open2.restype = c_int
avcodec_open2.argtypes = [POINTER(AVCodecContext), POINTER(AVCodec), POINTER(POINTER(AVDictionary))]

avformat_alloc_context = _libraries['libavformat.so'].avformat_alloc_context
avformat_alloc_context.restype = POINTER(AVFormatContext)
avformat_alloc_context.argtypes = []

avformat_find_stream_info = _libraries['libavformat.so'].avformat_find_stream_info
avformat_find_stream_info.restype = c_int
avformat_find_stream_info.argtypes = [POINTER(AVFormatContext), POINTER(POINTER(AVDictionary))]

avformat_open_input = _libraries['libavformat.so'].avformat_open_input
avformat_open_input.restype = c_int
avformat_open_input.argtypes = [POINTER(POINTER(AVFormatContext)), STRING, POINTER(AVInputFormat), POINTER(POINTER(AVDictionary))]

avpicture_fill = _libraries['libavcodec.so'].avpicture_fill
avpicture_fill.restype = c_int
avpicture_fill.argtypes = [POINTER(AVPicture), POINTER(uint8_t), AVPixelFormat, c_int, c_int]

avpicture_get_size = _libraries['libavcodec.so'].avpicture_get_size
avpicture_get_size.restype = c_int
avpicture_get_size.argtypes = [AVPixelFormat, c_int, c_int]

sws_getCachedContext = _libraries['libswscale.so'].sws_getCachedContext
sws_getCachedContext.restype = POINTER(SwsContext)
sws_getCachedContext.argtypes = [POINTER(SwsContext), c_int, c_int, AVPixelFormat, c_int, c_int, AVPixelFormat, c_int, POINTER(SwsFilter), POINTER(SwsFilter), POINTER(c_double)]

sws_scale = _libraries['libswscale.so'].sws_scale
sws_scale.restype = c_int
sws_scale.argtypes = [POINTER(SwsContext), POINTER(POINTER(uint8_t)), POINTER(c_int), c_int, c_int, POINTER(POINTER(uint8_t)), POINTER(c_int)]

#AVOutputFormat * 	av_guess_format (const char *short_name, const char *filename, const char *mime_type)
av_guess_format = _libraries['libavformat.so'].av_guess_format
av_guess_format.restype = POINTER(AVOutputFormat)
av_guess_format.argtypes = [STRING, STRING, STRING]

#AVStream *avformat_new_stream(AVFormatContext *s, const AVCodec *c);
avformat_new_stream = _libraries['libavformat.so'].avformat_new_stream
avformat_new_stream.restype = POINTER(AVStream)
avformat_new_stream.argtypes = [POINTER(AVFormatContext), POINTER(AVCodec)]


#AVCodec * avcodec_find_encoder (enum AVCodecID id)
avcodec_find_encoder = _libraries['libavcodec.so'].avcodec_find_encoder
avcodec_find_encoder.restype = POINTER(AVCodec)
avcodec_find_encoder.argtypes = [c_int]

#void * av_malloc (size_t size)
av_malloc = _libraries['libavutil.so'].av_malloc
av_malloc.restype = POINTER(uint8_t)
av_malloc.argtypes = [int64_t]

#int av_image_get_buffer_size	(	enum AVPixelFormat 	pix_fmt,int 	width,int 	height,int 	align )	
av_image_get_buffer_size = _libraries['libavutil.so'].av_image_get_buffer_size
av_image_get_buffer_size.restype = c_int
av_image_get_buffer_size.argtypes = [c_int, c_int, c_int, c_int]

#int avpicture_alloc(AVPicture *picture, enum AVPixelFormat pix_fmt, int width, int height);
avpicture_alloc = _libraries['libavutil.so'].avpicture_alloc
avpicture_alloc.restype = c_int
avpicture_alloc.argtypes = [POINTER(AVPicture), c_int, c_int, c_int]

if 0:
	#int 	url_fopen (ByteIOContext **s, const char *url, int flags)
	url_fopen  = _libraries['libavutil.so'].url_fopen
	url_fopen .restype = c_int
	url_fopen .argtypes = [POINTER(POINTER(ByteIOContext)), STRING, c_int]

#int avformat_write_header(AVFormatContext *s, AVDictionary **options);
avformat_write_header = _libraries['libavformat.so'].avformat_write_header
avformat_write_header.restype = c_int
avformat_write_header.argtypes = [POINTER(AVFormatContext), POINTER(POINTER(AVDictionary))]

#int av_write_trailer(AVFormatContext *s);
av_write_trailer = _libraries['libavformat.so'].av_write_trailer
av_write_trailer.restype = c_int
av_write_trailer.argtypes = [POINTER(AVFormatContext)]



#int avcodec_encode_video2(AVCodecContext *avctx, AVPacket *avpkt, const AVFrame *frame, int *got_packet_ptr);
avcodec_encode_video2 = _libraries['libavcodec.so'].avcodec_encode_video2
avcodec_encode_video2.restype = c_int
avcodec_encode_video2.argtypes = [POINTER(AVCodecContext), POINTER(AVPacket), POINTER(AVFrame), POINTER(c_int)]

#AVCodecContext *avcodec_alloc_context3(const AVCodec *codec);
avcodec_alloc_context3 = _libraries['libavcodec.so'].avcodec_alloc_context3
avcodec_alloc_context3.restype = POINTER(AVCodecContext)
avcodec_alloc_context3.argtypes = [POINTER(AVCodec)]

#int avio_open2(AVIOContext **s, const char *url, int flags, const AVIOInterruptCB *int_cb, AVDictionary **options);
avio_open2 = _libraries['libavformat.so'].avio_open2
avio_open2.restype = c_int
avio_open2.argtypes = [POINTER(POINTER(AVIOContext)), STRING, c_int, POINTER(AVIOInterruptCB), POINTER(POINTER(AVDictionary))]

#int avio_close(AVIOContext *s);
avio_close = _libraries['libavformat.so'].avio_close
avio_close.restype = c_int
avio_close.argtypes = [POINTER(AVIOContext)]