# keys for 'scene state' dicts
K_NAME = 'name'
K_COLOUR = 'colour'
K_VISIBLE = 'visible'
K_DRAW = 'draw'  # internal use, should this primitive draw or not
K_SELECTED = 'selected'
K_TYPE = 'objectType'
K_PRIMITIVE = 'primitive'  # internal use, should this primitive draw or not
K_STARTINDEX = 'startIndex'
K_FILENAME = 'filename'

K_BONE_COLOUR = 'boneColour'
K_MARKER_COLOUR = 'markerColour'

ALL_KEYS = [k for k in locals().keys() if k.startswith('K_')]

# types of 'objects'
T_NULL = 'null'
T_MOTION = 'motion'
T_SKELETON = 'skeleton'  # has no motion
T_CAMERAS = 'cameras'
