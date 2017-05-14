import struct
from time import sleep
import numpy as np
from SocketServer import SocketServer
from GCore import Character

G_MAPPING_HACK = {}

radToDegrees = 180.0 / np.pi

G_MESSAGE_TYPE = 0

''' Component Types '''

ComponentTypeID_container_begin = 0
ComponentTypeID_parameter_begin = 1
ComponentTypeID_container_end = 0xffffffff

''' Container Hints '''

ContainerHintID_unspecified = 0
ContainerHintID_position = 1 << 0
ContainerHintID_rotation = 1 << 1
ContainerHintID_scale = 1 << 2
ContainerHintID_globals = 1 << 3
ContainerHintID_transformation = 1 << 4 | ContainerHintID_position | ContainerHintID_rotation | ContainerHintID_scale
ContainerHintID_actor = 1 << 5 | ContainerHintID_position | ContainerHintID_rotation | ContainerHintID_scale
ContainerHintID_bone = 1 << 6 | ContainerHintID_position | ContainerHintID_rotation | ContainerHintID_scale
ContainerHintID_locator = 1 << 7 | ContainerHintID_position | ContainerHintID_rotation | ContainerHintID_scale
ContainerHintID_camera = 1 << 8 | ContainerHintID_position | ContainerHintID_rotation | ContainerHintID_scale
ContainerHintID_light = 1 << 9 | ContainerHintID_position | ContainerHintID_rotation | ContainerHintID_scale
ContainerHintID_takename = 1 << 10
ContainerHintID_user = 1500000

''' Parameter Hints '''

ParameterHintID_unspecified = 0
ParameterHintID_translation_xyz = 1
ParameterHintID_rotation_euler_xyz = 2
ParameterHintID_rotation_quat_xyzw = 3
ParameterHintID_rotation_33matrix = 4
ParameterHintID_scaling_33matrix = 5
ParameterHintID_transform_44matrix = 6
ParameterHintID_vertex_index_array = 7
ParameterHintID_vertex_position_array = 8
ParameterHintID_timecode_frame = 9
ParameterHintID_timecode_second = 10
ParameterHintID_timecode_rate = 11
ParameterHintID_camera_horizontal_fov = 12
ParameterHintID_camera_vertical_fov = 13
ParameterHintID_camera_fov = 14
ParameterHintID_camera_focal_length = 15
ParameterHintID_camera_focus_distance = 16
ParameterHintID_camera_near_clip = 17
ParameterHintID_camera_far_clip = 18
ParameterHintID_camera_film_width = 19
ParameterHintID_camera_film_height = 20
ParameterHintID_takename = 21
ParameterHintID_scaling_xyz = 22
ParameterHintID_user = 1000

''' Parameter Types '''

ParameterDataTypeID_float32 = 0
ParameterDataTypeID_double64 = 1
ParameterDataTypeID_int8 = 2
ParameterDataTypeID_int16 = 3
ParameterDataTypeID_int32 = 4
ParameterDataTypeID_int64 = 5
ParameterDataTypeID_uint8 = 6
ParameterDataTypeID_uint16 = 7
ParameterDataTypeID_uint32 = 8
ParameterDataTypeID_uint64 = 9
ParameterDataTypeID_string = 10

def initServer(host='', port=6500):
	TIS_Server = SocketServer()
	TIS_Server.Start(host, port)
	return TIS_Server

def createHeader(payloadSize=0):
	ret = struct.pack('<II', G_MESSAGE_TYPE, payloadSize)
	# print len(ret), "header"
	return ret

def beginContainer(name, hintId, flags = 0):
	ret = struct.pack('<II' + 's' * len(name), 	ComponentTypeID_container_begin, len(name) + 1, *name)
	ret += struct.pack('<cII', '\0', hintId, flags)
	# print len(ret), "begin"
	return ret

def endContainer():
	ret = struct.pack('<I', ComponentTypeID_container_end)
	# print len(ret), "end"
	return ret

def addParameter(name, hintId, flags = 74528, typeId = 0, count = 1, value = (1,2)):
	ret = struct.pack('<II' + 's' * len(name), ComponentTypeID_parameter_begin, len(name) + 1, *name)
	ret += struct.pack('<cII','\0', hintId, flags)
	ret += struct.pack('<II' + 'f'*count, typeId, count, *value)
	return ret

def stripVicon(jointName):
	return jointName[4:] if jointName[:4] == "VSS_" else jointName

def eulerFromMat(mat):
	rotation = np.zeros(3, dtype=np.float32)
	if np.abs(mat[2, 0]) <> 1:
		rotation[0] = -np.arcsin(mat[2, 0])
		rotation[1] = np.arctan2(mat[2, 1], mat[2, 2])
		rotation[2] = np.arctan2(mat[1, 0], mat[0, 0])
	else:
		if mat[2, 0] < 0:
			rotation[0] = np.pi / 2.0
			rotation[2] = np.arctan2(mat[0, 1], mat[0, 2])
		else:
			rotation[0] = -np.pi / 2.0
			rotation[2] = np.arctan2(-mat[0, 1], -mat[0, 2])
	return rotation

def addJoint(ji, skelDict, scale=1.0, level=0):
	data = bytearray()
	swizzle = [1, 0, 2]

	jointName = stripVicon(skelDict['jointNames'][ji])

	jointName = G_MAPPING_HACK.get(jointName, jointName)

	data += beginContainer(jointName, ContainerHintID_bone)

	translation = (skelDict['Gs'][ji][:3,3] / 10.0)

	rotation = skelDict['Gs'][ji][:3,:3]

	data += addParameter("rotation", ParameterHintID_rotation_euler_xyz, count=3, value=np.array(eulerFromMat(rotation)[swizzle] * radToDegrees))
	data += addParameter("translation", ParameterHintID_translation_xyz, count=3, value=translation * scale)
	data += addParameter("scale", ParameterHintID_scaling_xyz, count=3, value = np.ones(3, dtype=np.float32) * scale) # TODO: Currently ignored in Unreal, fix
	children_inds = [i for i in xrange(skelDict['numJoints']) if skelDict['jointParents'][i] == ji]
	for ci in children_inds:
		data += addJoint(ci, skelDict, scale=scale, level=level + 1)
	return data + endContainer()


def serialiseSkeleton(name, skelDict, scale=1.0):
	data = bytearray()
	data += beginContainer(name, ContainerHintID_actor)
	rootIndex = np.where(skelDict['jointParents'] == -1)[0][0]
	data += addJoint(rootIndex, skelDict, scale=scale)
	data += endContainer()
	return createHeader(len(data)) + data

def getBlendshapeData(shapes, values):
	assert len(shapes) == len(values)
	data = bytearray()
	data += beginContainer("globals", ContainerHintID_globals)
	for si, shape in enumerate(shapes):
		data += addParameter(shape, ParameterHintID_user, flags=1000, value=[values[si]])
	data += endContainer()
	return createHeader(len(data)) + data

if __name__ == '__main__':
	import numpy as np
	import ASFReader

	conn, addr = initServer()

	ted_shapes = ["head_GEO_UW", "head_GEO_UH", "head_GEO_SH", "head_GEO_F", "head_GEO_M", "head_GEO_B",
					"head_GEO_Blink", "head_GEO_browsUp", "head_GEO_browsDown", "head_GEO_happy"]
	i = 0
	while 1:
		values = np.zeros(len(ted_shapes))
		values[9] = 100 * np.sin(np.pi * (i / 100.))
		conn.sendall(getBlendshapeData(ted_shapes, values))

		i = (i + 1) % 101

		sleep(1.0 / 60.0)

	conn.close()