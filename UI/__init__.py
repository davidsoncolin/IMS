import sys
from PySide import QtCore, QtGui

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

DRAWOPT_AXES = 0x1
DRAWOPT_BONES = 0x1 << 1
DRAWOPT_JOINTS = 0x1 << 2
DRAWOPT_GEOMS = 0x1 << 3
DRAWOPT_GEOMALPHA = 0x1 << 4
DRAWOPT_MARKERS = 0x1 << 5
DRAWOPT_OFFSET = 0x1 << 6
DRAWOPT_CAMERAS = 0x1 << 7
DRAWOPT_GRID = 0x1 << 8
DRAWOPT_LABELS = 0x1 << 9
DRAWOPT_HUD = 0x1 << 10
DRAWOPT_DETECTIONS = 0x1 << 11
DRAWOPT_POINTS = 0x1 << 12
DRAWOPT_IMAGE = 0x1 << 13
DRAWOPT_EDGES = 0x1 << 14
DRAWOPT_POINT_LABELS = 0x1 << 15

DRAWOPTS = {'Axes': DRAWOPT_AXES,
			'Bones': DRAWOPT_BONES,
			'Cameras': DRAWOPT_CAMERAS,
			'Detections': DRAWOPT_DETECTIONS,
			'Edges': DRAWOPT_EDGES,
			'Geometry': DRAWOPT_GEOMS,
			'Geometry Alpha': DRAWOPT_GEOMALPHA,
			'Grid': DRAWOPT_GRID,
			'HUD': DRAWOPT_HUD,
			'Image': DRAWOPT_IMAGE,
			'Joints': DRAWOPT_JOINTS,
			'Markers': DRAWOPT_MARKERS,
			'Offset': DRAWOPT_OFFSET,
			'Points': DRAWOPT_POINTS,
			'Labels': DRAWOPT_LABELS,
			'Point Labels': DRAWOPT_POINT_LABELS,
			}

DRAWOPT_NONE = 0
DRAWOPT_ALL = sum(DRAWOPTS.values())
DRAWOPT_DEFAULT = DRAWOPT_ALL ^ (DRAWOPT_POINT_LABELS | DRAWOPT_AXES)
COLOURS = {'Floor':[0.2,0.5,0.1,0.5],
		'Grid':[0,0,0,1],
		'Background':[0.39,0.59,0.98,1 ],
		'Bone':[0.3,0.42,0.66,1],
		'Marker':[0,1,0,1],
		'Camera':(0,0,0,1),
		'HUDText':(1,1,1,1),
		'Selected':(0,1,0,1),
		'Active':(1,.6,.3,1), # target skeleton in retargeting operation
		'Hilighted':(1,1,1,1)
		}

# toms colours
COLOURS.update({'Floor':[.55,.6,.57,0.7],
		'Grid':[0,0,0,1],
		'Background':[.6,.6,.6,1 ],
		'Selected':(.5,1,.5,1),
		})


#: when the app creates actions it will get the shortcut from this dict. key is the action name
SHORTCUTS = {
			'Auto Match': 'a',
			'Geometry Alpha': 'a',
			'Bones': 'b',
			'Cameras': 'c',
			'Detections': 'd',
			'Edges' : 'e',
			'Frame Selected': 'f',
			'Grid': 'g',
			'HUD': 'h',
			'Image': 'i',
			'Joints': 'j',
			'UnMap Joints': 'k',
			'Labels': 'l',
			'Map Joints': 'l',
			'Markers': 'm',
			'Offset': 'o',
			'Points': 'p',
			'Geometry': 'q',
			'Axes': 'x',
			'Next Camera': ']',
			'Previous Camera': '[',
			'Reset 3D': '0',
			'Reset 2D': '1',
			"&New": 'Ctrl+N',
			"Import": 'Ctrl+Shift+O',
			'Exit': 'Ctrl+Q',
			'Play': ' ',  #QtGui.QKeySequence(QtCore.Qt.Key_MediaTogglePlayPause),
			'Step Forward': '.',#QtGui.QKeySequence(QtCore.Qt.Key_MediaNext),
			'Step Backward': ',',#QtGui.QKeySequence(QtCore.Qt.Key_MediaPrevious),
			'Backward one second': 'Shift+,',  #QtGui.QKeySequence(QtCore.Qt.Key_MediaNext),
			'Forward one second': 'Shift+.',  #QtGui.QKeySequence(QtCore.Qt.Key_MediaPrevious),
			'Go to range start': 'Home',  #QtGui.QKeySequence(QtCore.Qt.Key_MediaNext),
			'Go to range end': 'End',  #QtGui.QKeySequence(QtCore.Qt.Key_MediaPrevious),
			'Delete': 'Del',  #QtGui.QKeySequence(QtCore.Qt.Key_MediaPrevious),
			'Toggle Fullsize': 'F10',
			'Clear All': 'Ctrl+D',
			}

import GCore
from GLGrid import GLGrid
from GLPoints3D import GLPoints3D
from GLPoints2D import GLPoints2D
from GLSkel import GLSkel
from GLSkeleton import GLSkeleton
from GLCameras import GLCameras
from GLGeometry import GLGeometry
from GLBones import GLBones
from GLMeshes import GLMeshes
from GLPrimitives import GLPrimitive

def createAction(name, parent, commands, tip=None, checkable=False, checked=False, icon=None, globalContext=False):
	ac = QtGui.QAction(name, parent)
	if name in SHORTCUTS:
		ac.setShortcut(SHORTCUTS[name])
		#ac.setShortcut(QtGui.QKeySequence(SHORTCUTS[name]))
		ac.setShortcutContext(QtCore.Qt.WidgetWithChildrenShortcut)
	for command in commands:
		ac.triggered.connect(command)
	ac.setCheckable(checkable)
	ac.setChecked(checked)
	if tip is not None:
		ac.setStatusTip(tip)
		ac.setToolTip(tip)
	if icon is not None:
		ac.setIcon(icon)
	if globalContext:
		ac.setShortcutContext(QtCore.Qt.ApplicationShortcut)
	# make sure the key press is picked up when the parent has focus
	parent.addAction(ac)
	return ac

from PySide import QtGui

def errorDialog(title, short, long=None):
	box = QtGui.QMessageBox()
	box.setWindowTitle(title)
	box.setText(short)
	box.setStandardButtons(QtGui.QMessageBox.Abort)
	box.setIcon(QtGui.QMessageBox.Critical)
	if long:
		box.setInformativeText(long)
	box.exec_()

from QTimeline import QTimeline
from QGLViewer import QGLView

__all__ = ['QTimeline', 'QGLView']


class QAppCtx(object):
	'''
	Convenience context manager for a generic qapp.

	.. code-block:: python

		#Make ctrl+C work
		import signal
		signal.signal(signal.SIGINT, signal.SIG_DFL)

		#app
		import sys
		app = QtGui.QApplication(sys.argv)

		##########################
		#Your custom code here...
		...
		##########################

		#Run the app
		sys.exit(app.exec_())

	Usage:

	.. code-block:: python

		if __name__ == "__main__":
			from UI import QAppCtx
			with QAppCtx() as app:
				dialog = MyDialog()
				dialog.show()
	'''
	def __init__(self):
		'''
		'''
		self.app = None

	def __enter__(self):
		'''
		Do the setup. Returns the app
		'''
		# Make ctrl+C work
		import signal
		signal.signal(signal.SIGINT, signal.SIG_DFL)

		self.app = QtGui.QApplication(sys.argv)
		self.app.setStyle('plastique')
		f = self.app.font()
		f.setPointSize(8)
		self.app.setFont(f)

		return self.app

	def __exit__(self, exceptionType, exceptionValue, traceback):
		'''
		Exec the app and exit.
		'''
		if not exceptionType:
			sys.exit(self.app.exec_())
