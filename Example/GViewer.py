#import OpenGL
#OpenGL.FULL_LOGGING = True
import functools
import logging
from PySide.QtOpenGL import QGLFormat

logging.basicConfig()
LOG = logging.getLogger(__name__)
from time import time
from PySide import QtCore, QtGui
from UI import QAppCtx, QGLView, QTimeline, createAction
from UI.QGLViewer import Camera
from UI import GLGrid, GLPoints2D, GLPoints3D, GLSkel, GLCameras
from UI import DRAWOPTS, DRAWOPT_ALL, DRAWOPT_DEFAULT, DRAWOPT_AXES
import IO

class CameraSwitchMenu(QtGui.QMenu):
	''' menu that allows selecting the current camera from a list
	the menu has a pointer to a :class:`UI.QGLViewer.QGLView` and reads it's cameras attribute directly.
	it will then set the current camera in the view by directly setting it's camera attribute
	the camera list is built dynamically when the menu is about to be shown
	'''
	def __init__(self, name, parent=None):
		if not parent:
			# windows pyside 1.1.2 will crash if this is not constructed with a parent
			raise RuntimeError("CameraSwitchMenu must be passed a parent widget")
		super(CameraSwitchMenu, self).__init__(name, parent=parent)

		#: pointer to the :class:`UI.QGLView` that i'll build the camera list from
		self.glview = None

		# add the actions before showing. (this can't be in the showEvent or the menu is not
		# resized properly and may potentially go off the bottom of the screen)
		self.aboutToShow.connect(self._addActions)

	@QtCore.Slot()
	def _addActions(self):
		''' create an action for each camera whenever the menu is about shown '''
		for camera in self.glview.cameras:
			ca = QtGui.QAction(camera.name, self)
			ca.setCheckable(True)
			ca.setChecked(self.glview.camera == camera)
			ca.triggered.connect(functools.partial(self.setCamera, camera))
			self.addAction(ca)

	def hideEvent(self, event):
		''' remove all the cameras on hide (the list is freshly created on show) '''
		super(CameraSwitchMenu, self).hideEvent(event)
		for action in self.actions():
			self.removeAction(action)

	@QtCore.Slot()
	def setCamera(self, camera):
		''' connected to the camera actions to switch the current camera in the view '''
		self.glview.camera = camera
		self.glview.updateGL()


class GPanel(QtGui.QFrame):
	''' this is the base class for all GRIP panels. '''
	def __init__(self, parent=None):
		super(GPanel, self).__init__(parent=parent)
		self.setFrameStyle(QtGui.QFrame.Sunken)
		# 1px frame around the panel that has focus
		self.setStyleSheet("""QFrame:focus {border: 1px solid #FFFFFF;}""")

		# menus
		self.menuBar = QtGui.QMenuBar(self)

		layout = QtGui.QVBoxLayout()
		layout.setContentsMargins(1, 0, 1, 1)
		layout.setSpacing(0)
		layout.addWidget(self.menuBar)
		self.setLayout(layout)


class QGLPanel(GPanel):
	''' contains a QGLView and menus'''
	def __init__(self, grid=True, parent=None):
		super(QGLPanel, self).__init__(parent=parent)
		self.setFrameStyle(QtGui.QFrame.Sunken)

		# mask of drawable stuff (axes/bones/labels etc). don't show axes by default
		self.drawOpts = DRAWOPT_DEFAULT

		#: The QGLView widget, access this directly, there's no .view() getter
		self.view = QGLView(parent=self)

		# add grid
		if grid:
			self.view.primitives.append(GLGrid())
		self.view.drawOpts = self.drawOpts
	
		cameraMenu = QtGui.QMenu("&Camera")
		self.showMenu = QtGui.QMenu("Show")

		# the camera switch menu needs a pointer to the view so that it can get the list of cameras
		# and switch the current one
		camSwitchMenu = CameraSwitchMenu("Switch", parent=self)
		camSwitchMenu.glview = self.view
		cameraMenu.addMenu(camSwitchMenu)

		#cameraMenu.addAction(createAction('Next Camera', self, [functools.partial(self.view.cycleCamera, 1)]))
		#cameraMenu.addAction(createAction('Previous Camera', self, [functools.partial(self.view.cycleCamera, -1)]))
		cameraMenu.addAction(createAction('Reset', self, [self.resetCamera3D, self.resetCamera2D]))
		cameraMenu.addAction(createAction('Reset 3D', self, [self.resetCamera3D]))
		cameraMenu.addAction(createAction('Reset 2D', self, [self.resetCamera2D]))

		# show menu
		self.showMenu.addAction(createAction('Frame Selected', self, [self.view.frame]))
		self.showMenu.addAction(createAction('Show All', self, [functools.partial(self.setAllDrawOptions, True)]))
		self.showMenu.addAction(createAction('Show None', self, [functools.partial(self.setAllDrawOptions, False)]))
		for opt in sorted(DRAWOPTS):
			a = createAction(opt, self, [functools.partial(self.toggleDrawOption, DRAWOPTS[opt])], checkable=True, checked=bool(DRAWOPTS[opt] & self.drawOpts))
			self.showMenu.addAction(a)

		self.menuBar.addMenu(cameraMenu)
		self.menuBar.addMenu(self.showMenu)

		self.layout().addWidget(self.view)
		self.layout().setStretch(1, 1)

	def frame(self):
		self.view.frame()
		self.view.updateGL()

	def resetCamera3D(self):
		self.view.camera.reset3D()
		self.view.updateGL()

	def resetCamera2D(self):
		self.view.camera.reset2D()
		self.view.updateGL()

	def setAllDrawOptions(self, state):
		'''set all draw options on or off

		:param bool state: True = on'''
		for a in self.showMenu.actions(): a.setChecked(state)
		self.view.drawOpts = [0, DRAWOPT_ALL][state]
		self.view.updateGL()

	def toggleDrawOption(self, opt):
		'''toggle the provided draw option
		
		:param int opt: one or a combination of :`data:UI.DRAWOPTS` values'''
		self.view.drawOpts = self.view.drawOpts ^ opt
		self.view.updateGL()


class QGViewer(QtGui.QMainWindow):
	def __init__(self, parent=None):
		super(QGViewer, self).__init__(parent=parent)

		self.setWindowTitle('Imaginarium Viewer')
		self.setMinimumWidth(640)
		self.setMinimumHeight(480)

		self.menus = {}
		#: list of all the GL* things to draw. the views all refer to this
		self.primitives = []
		self.primitives2D = []

		self.setTabPosition(QtCore.Qt.TopDockWidgetArea, QtGui.QTabWidget.North)
		self.setTabPosition(QtCore.Qt.RightDockWidgetArea, QtGui.QTabWidget.East)
		self.setTabPosition(QtCore.Qt.LeftDockWidgetArea, QtGui.QTabWidget.West)
		self.setTabPosition(QtCore.Qt.BottomDockWidgetArea, QtGui.QTabWidget.North)

		menuBar = QtGui.QMenuBar(self)
		self.setMenuBar(menuBar)
		self.getOrCreateMenu('&File')

		self.createWidgets()
		self.createMenus()
		self.createLayout()
		self.createConnections()

		self.new()

	def createMenus(self):
		self.getOrCreateMenu('&File').addAction(createAction('&New', self, [self.new], tip='Create a new scene'))
		# hack. maybe make a getOrCreateAction thing like the menus so i can get them by name
		# (this is so i can insert the import action before it, so it's not very robust or good)
		self._exitAction = createAction('Exit', self, [self.close], tip='Quit the application')
		self.getOrCreateMenu('&File').addAction(self._exitAction)
		self.getOrCreateMenu('&Edit')
		self.getOrCreateMenu('&View').addAction(createAction('Show status bar', self, [lambda:self.statusBar().setVisible(not self.statusBar().isVisible())],checkable=True, checked=True,  tip='Toggle the help/status bar on or off'))

	def createWidgets(self):
		# widgets
		self._mainWidget = QtGui.QWidget(self)
		self.setCentralWidget(self._mainWidget)
		self.statusBar().showMessage("Starting up...", 1000) # this shows the status bar too
		self.timeline = QTimeline(self)
		self.addDock('Timeline', self.timeline,
					QtCore.Qt.BottomDockWidgetArea | QtCore.Qt.TopDockWidgetArea,
					QtCore.Qt.BottomDockWidgetArea,
					QtGui.QDockWidget.DockWidgetMovable | QtGui.QDockWidget.DockWidgetVerticalTitleBar,
					useTitle=False)
		self._panel = QGLPanel(grid=False)
		self.view().primitives = self.primitives
		self.view().primitives2D = self.primitives2D

	def createConnections(self):
		# when the frame rate changes, reset the lastTimes cache in the view so it can update
		# rate hud display more accurately sooner
		self.timeline.rateChanged.connect(lambda: self.view().__setattr__('lastTimes', [time()]))

	def createLayout(self):
		# layout
		self.viewLayout = QtGui.QVBoxLayout()
		self.viewLayout.setContentsMargins(0, 0, 0, 0)
		self.viewLayout.setSpacing(0)
		self.viewLayout.addWidget(self._panel)
		self._mainWidget.setLayout(self.viewLayout)

	def getOrCreateMenu(self, menuName, before=None):
		try:
			return self.menus[menuName]
		except KeyError:
			if before:
				m = QtGui.QMenu(menuName)
				self.menuBar().insertMenu(self.menus[before].menuAction(), m)
				self.menus[menuName] = m
			else:
				self.menus[menuName] = self.menuBar().addMenu(menuName)
			return self.menus[menuName]

	def addDock(self, title, obj, allowed, start, features, useTitle=True):
		dock = QtGui.QDockWidget(self)
		if useTitle: dock.setWindowTitle(title)
		dock.setObjectName(title) # required so that save settings doesn't barf
		dock.setWidget(obj)
		dock.setAllowedAreas(allowed)
		dock.setFeatures(features)
		dock.layout().setContentsMargins(0, 0, 0, 0)
		dock.layout().setSpacing(0)
		self.addDockWidget(start, dock)
		self.getOrCreateMenu('&View').addAction(createAction('Show %s' % title, self, [lambda:dock.setVisible(not dock.isVisible())], checkable=True, checked=True, tip='Toggle whether the %s is displayed' % title))

	def view(self):
		return self._panel.view
	
	def updateGL(self):
		self.view().updateGL()

	def refreshImageData(self):
		self.view().refreshImageData()

	def new(self):
		del self.primitives[:]
		self.view().points = None
		self.primitives.append(GLGrid())
		self.view().camera.reset2D()
		self.view().camera.reset3D()
		self.timeline.setRange(1, 100)
		self.updateGL()

	def addPoints2D(self, points):
		'''
		add 2d points to the viewer.

		:type points: :class:`numpy.array` (Nx2)
		:param points: array of vertices
		'''
		glPoints = GLPoints2D(points)
		self.primitives2D.append(glPoints)
		return self.primitives2D[-1]

	def addPoints3D(self, points):
		'''
		add 3d points to the viewer.

		:type points: :class:`numpy.array` (Nx3)
		:param points: array of vertices
		'''
		glPoints = GLPoints3D(points)
		self.primitives.append(glPoints)
		return self.primitives[-1]

	def addCameras(self, mats, camera_ids, movies=None):
		if movies == None: movies = [None]*len(mats)
		for mat, cid, md in zip(mats, camera_ids, movies):
			camera = Camera(cid)
			camera.setP(mat[2], distortion=mat[3])
			camera.setResetData()
			if md != None: camera.setImageData(md['vbuffer'],md['vheight'],md['vwidth'],3)
			self.view().addCamera(camera)
		cams = GLCameras(camera_ids, mats)
		self.primitives.append(cams)

if __name__ == '__main__':
	with QAppCtx():
		dialog = QGViewer()
		dialog.show()