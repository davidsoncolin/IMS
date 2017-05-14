#!/usr/bin/env python

'''Qapp : Qt application wrapper. main window, menus, hotkeys, widgets, resizing, docking etc.
 - holds widgets, triggers updates as needed
 - responsible for loading/saving a state file; on load, update the Qview with camera list
 - add a Qview, a layer manager for the layers, a field editor for the selected layer
 - create a camera from an image sequence (drag-drop also possible); update list of cameras in Qview
 - render to a file sequence (reinterlace options)
 - when a layer is selected
   * update the layer field editor
 - when a layer is added/moved/deleted/edited/selected
   * rerender the layer on this frame and update the Qview
'''
from PySide.QtOpenGL import QGLFormat

global app, fields # the one and only app object
app = None
fields = {'none':{}}
def view(): return app.view() # shortcut


from GCore import State
import UI
from UI import QFields, QGLViewer, QStateTreeView, createAction
from PySide import QtCore, QtGui
from UI import GLGrid, GLPoints2D, GLPoints3D, GLSkel, GLCameras
from UI import DRAWOPTS, DRAWOPT_ALL, DRAWOPT_DEFAULT, DRAWOPT_AXES
import functools


class CameraSwitchMenu(QtGui.QMenu):
	''' menu that allows selecting the current camera from a list
	the menu has a pointer to a :class:`UI.QGLViewer.QGLView` and reads it's cameras attribute directly.
	it will then set the current camera in the view by directly setting it's camera attribute
	the camera list is built dynamically when the menu is about to be shown
	'''
	def __init__(self, name, view, parent=None):
		if not parent:
			# windows pyside 1.1.2 will crash if this is not constructed with a parent
			raise RuntimeError("CameraSwitchMenu must be passed a parent widget")
		super(CameraSwitchMenu, self).__init__(name, parent=parent)

		self.view = view #: pointer to the :class:`UI.QGLView` that i'll build the camera list from

		# add the actions before showing. (this can't be in the showEvent or the menu is not
		# resized properly and may potentially go off the bottom of the screen)
		self.aboutToShow.connect(self._addActions)

	@QtCore.Slot()
	def _addActions(self):
		''' create an action for each camera whenever the menu is about shown '''
		for camera in self.view.cameras:
			ca = QtGui.QAction(camera.name, self)
			ca.setCheckable(True)
			ca.setChecked(self.view.camera == camera)
			ca.triggered.connect(functools.partial(self.parent().setCamera, camera))
			self.addAction(ca)

	def hideEvent(self, event):
		''' remove all the cameras on hide (the list is freshly created on show) '''
		super(CameraSwitchMenu, self).hideEvent(event)
		for action in self.actions():
			self.removeAction(action)

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


class CameraPanels(QtGui.QFrame):
	def __init__(self, parent=None, mainView=None, rows=3, cols=5):
		super(CameraPanels, self).__init__(parent=parent)
		self.setFrameStyle(QtGui.QFrame.Sunken)
		self.setStyleSheet("""QFrame:focus {border: 1px solid #FFFFFF;}""")
		self.setWindowTitle('Camera Viewer')

		self.mainView = mainView
		self.rows, self.cols = rows, cols
		self.numCells = self.rows * self.cols
		self.panels = {}

		self.gridLayout = QtGui.QGridLayout()
		self.rebuildPanels()
		self.setGeometry(100, 100, 1000, 800)

	def resetCurrPos(self):
		self.currPos = 0

	def rebuildPanels(self):
		self.currPos = 0
		for i in range(self.numCells):
			self.setCameraLayer(rebuild=True)

		self.setLayout(self.gridLayout)

	def setCameraLayer(self, camIdx=None, pos=None, rebuild=False):
		import math
		if pos is None:
			row = int(math.floor(float(self.currPos) / float(self.cols)))
			col = self.currPos % self.cols
		else:
			row, col = pos

		if self.currPos not in self.panels or rebuild:
			self.panels[self.currPos] = CameraPanel(mainView=self.mainView)
			self.gridLayout.addWidget(self.panels[self.currPos], row, col)

		if camIdx is not None and camIdx < len(self.panels[self.currPos].view.cameras):
			# TODO: Find by name instead of index?
			self.panels[self.currPos].view.camera = self.panels[self.currPos].view.cameras[camIdx]
			# print('> setCameraLayer (%d): Setting entry [%d, %d] to camera index %d' % (self.currPos, row, col, camIdx))

		self.currPos = (self.currPos + 1) % self.numCells


class CameraPanel(GPanel):
	def __init__(self, parent=None, mainView=None):
		super(CameraPanel, self).__init__(parent=parent)

		self.view = QGLViewer.QGLView()
		if mainView is not None:
			self.view.cameras = mainView.cameras
			self.view.camera = mainView.cameras[0]
			# self.view.primitives = mainView.primitives
			self.view.primitives2D = mainView.primitives2D

		camPanel = QGLPanel(self.view, parent=mainView)
		self.layout().addWidget(camPanel)
		self.view.updateGL()


class QGLPanel(GPanel):
	''' contains a QGLView and menus'''
	def __init__(self, view, grid=True, parent=None):
		super(QGLPanel, self).__init__(parent=parent)
		self.setFrameStyle(QtGui.QFrame.Sunken)

		# mask of drawable stuff (axes/bones/labels etc). don't show axes by default
		self.drawOpts = DRAWOPT_DEFAULT

		#: The QGLView widget, access this directly, there's no .view() getter
		self.view = view

		# add grid
		if grid:
			g = GLGrid()
			self.view.primitives.append(g)
			self.view.setLayer('grid', g)
		self.view.drawOpts = self.drawOpts
	
		cameraMenu = QtGui.QMenu("&Camera")
		self.showMenu = QtGui.QMenu("Show")

		# the camera switch menu needs a pointer to the view so that it can get the list of cameras
		# and switch the current one
		camSwitchMenu = CameraSwitchMenu('Switch', self.view, parent=self)
		cameraMenu.addMenu(camSwitchMenu)

		global app
		cameraMenu.addAction(createAction('Next Camera', self, [functools.partial(self.cycleCamera, 1)]))
		cameraMenu.addAction(createAction('Previous Camera', self, [functools.partial(self.cycleCamera, -1)]))
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
		self.view.drawOpts ^=  opt
		self.view.updateGL()

	def setCamera(self, camera):
		''' connected to the camera actions to switch the current camera in the view '''
		self.view.camera = camera
		self.parent().refresh()

	def cycleCamera(self, direction=1):
		cams = self.view.cameras
		self.setCamera(cams[(cams.index(self.view.camera)+direction)%len(cams)])

class QView(QtGui.QWidget):

	def __init__(self, child, parent=None):
		self.child = child
		QtGui.QWidget.__init__(self, parent)
		self.menuBar = QtGui.QMenuBar(self)
		self.menuBar.setSizePolicy(QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed))
		self.viewMenu = self.menuBar.addMenu('&View')
		self.camraMenu = self.menuBar.addMenu('&Camera')
		child.setMinimumSize(300,300)

		qlayout = QtGui.QGridLayout()
		qlayout.addWidget(self.menuBar, 0, 0)
		qlayout.addWidget(self.child, 1, 0)
		self.setLayout(qlayout)

class QApp(QtGui.QMainWindow):

	edit_signal = QtCore.Signal(str,str,object)

	def __init__(self):
		QtGui.QMainWindow.__init__(self)
		global app
		assert(app is None)
		app = self
		self.value_is_adjusting = False
		self.dirtyCB = None
		self.filename = None
		self.menus = {}
		self.move(0,0)
		self.resize(640, 480)
		self.setWindowTitle(State.appName())
		self.setFocusPolicy(QtCore.Qt.StrongFocus)  # get keyboard events
		self.blockUpdate = False
		self.set_widget = {}
		self.trigger_calls = {}
		self.preLoadCB = None
		self.loadCB = None
		self.saveCB = None

		self.addMenuItem({'menu':'&File','item':'&New','tip':'New scene','cmd':self.new})
		self.addMenuItem({'menu':'&File','item':'&Open','shortcut':'Ctrl+O','tip':'Open state','cmd':self.load})
		self.addMenuItem({'menu':'&File','item':'&Save','shortcut':'Ctrl+S','tip':'Save state','cmd':self.save})
		self.addMenuItem({'menu':'&File','item':'Save &As','tip':'Save state as','cmd':self.saveAs})
		self.addMenuItem({'menu':'&File','item':'&Quit','shortcut':'Ctrl+Q','tip':'Exit application','cmd':self.quit})
		self.undoItem = self.addMenuItem({'menu':'&Edit','item':'&Undo','shortcut':'Ctrl+Z','tip':'Undo last command','cmd':self.undo})
		self.redoItem = self.addMenuItem({'menu':'&Edit','item':'Re&do','shortcut':'Ctrl+Shift+Z','tip':'Redo last command','cmd':self.redo})
		self.clearUndoItem = self.addMenuItem({'menu':'&Edit','item':'&Clear undo','tip':'Clear the undo stack','cmd':self.clearUndo})
		self.getOrCreateMenu('&View')
		#self.addMenuItem({'menu':'&Create','item':'&Image','tip':'Create image','cmd':self.loadImage})
		self.qglview = QGLViewer.QGLView()
		self.qview      = QGLPanel(self.qglview, parent=self)
		self.qfields    = QFields.QFieldsEditor(self)
		self.qoutliner  = QStateTreeView.QStateTreeView()
		self.qoutliner.selectionModel().currentChanged.connect(functools.partial(self.selectCB2, caller=self.qoutliner))

		# Add Ops list
		self.qnodes = UI.QCore.QNodeWidget(parent=self)
		self.qnodes.item_selected.connect(functools.partial(self.selectCB, caller=self.qnodes))
		self.qnodes.data_changed.connect(self.dataChangedCB)
		self.runtime = None
		self.cameraPanels = None

		# Add Python Console
		self.qpython    = UI.QCore.PythonConsole()
		
		#self.qlayers   = Qlayers.Qlayers(self)
		self.qtimeline = UI.QTimeline(self)
		self.setCentralWidget(self.qview)
		ar = (QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea | QtCore.Qt.BottomDockWidgetArea | QtCore.Qt.TopDockWidgetArea)
		self.attributeEditor = self.addDock('Attribute Editor', self.qfields,   ar, QtCore.Qt.RightDockWidgetArea)

		# Add docks and tab the Ops, outliner, and script together
		self.opNodes = self.addDock('Ops', self.qnodes,   ar, QtCore.Qt.RightDockWidgetArea)
		self.outliner = self.addDock('Outliner', self.qoutliner, ar, QtCore.Qt.RightDockWidgetArea)
		self.pythonDock = self.addDock('Console',  self.qpython, ar, QtCore.Qt.BottomDockWidgetArea)
		self.tabifyDockWidget(self.opNodes, self.outliner)
		self.tabifyDockWidget(self.outliner, self.pythonDock)

		self.timelineDock = self.addDock('Timeline',        self.qtimeline, ar, QtCore.Qt.BottomDockWidgetArea)

		self.setTabPosition(QtCore.Qt.TopDockWidgetArea,    QtGui.QTabWidget.North)
		self.setTabPosition(QtCore.Qt.RightDockWidgetArea,  QtGui.QTabWidget.East )
		self.setTabPosition(QtCore.Qt.LeftDockWidgetArea,   QtGui.QTabWidget.West )
		self.setTabPosition(QtCore.Qt.BottomDockWidgetArea, QtGui.QTabWidget.North)
		self.updateMenus()

		# Set the ops dock as the default visible one (enable this when everyone is happy to proceed)
		self.opNodes.show()
		self.opNodes.raise_()
		# self.outliner.show()
		# self.outliner.raise_()

		self.setAcceptDrops(True)

	def addOpItem(self, name, data, silent=False):
		#print 'addOpItem',name,data
		self.qnodes.addItem(name, data=data)
		if silent: return
		State.push('Add Op %s' % name) # TODO this method shouldn't be here
		self.updateMenus()

	def clearOpItems(self):
		self.qnodes.clear()

	def setLayer(self, layerName, layer, selection=False):
		return self.qglview.setLayer(layerName, layer, selection)

	def setLayers(self, layers):
		self.qglview.setLayers(layers)

	def updateLayers(self):
		self.qglview.updateLayers()

	def getLayers(self, *args):
		return self.qglview.getLayers(*args)

	def getLayer(self, layerName):
		return self.qglview.getLayer(layerName)

	def hasLayer(self, layerName):
		return self.qglview.hasLayer(layerName)

	def clearLayers(self, keepGrid=False):
		self.qglview.clearLayers(keepGrid)

	def clearSelectionLayers(self):
		self.qglview.clearSelectionLayers()

	def triggerSlot(self,name):
		print (self.trigger_calls[name])

	def selectCB2(self, index, index2=None, caller=None):
		#print 'selectCB2',index,index2,self.blockUpdate,caller.model().itemFromIndex(index)
		if self.blockUpdate: return
		v = caller.model().itemFromIndex(index).data()
		if v is None: return
		QApp.select(str(v))

	def selectCB(self, index, caller=None):
		if self.blockUpdate: return
		v = caller.getItem(index, QtCore.Qt.UserRole)
		QApp.select(str(v))

	def dataChangedCB(self, *args):
		hints = args[0]
		if 'flush' in hints and hints['flush']:
			self.runtime.flush(hard=False)

	def addWidget(self,widget,dock_name,connections,dock_area = QtCore.Qt.LeftDockWidgetArea):
		self.set_widget[dock_name] = widget
		ar = (QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea | QtCore.Qt.BottomDockWidgetArea | QtCore.Qt.TopDockWidgetArea)
		self.addDock(dock_name, self.set_widget[dock_name],ar,dock_area)
		for connection in connections:
			for action in connections[connection]:
				if isinstance(action,str):
					action = eval(action)
				getattr(self.set_widget[dock_name],connection).connect(action)

	def addDock(self, title, obj, allowed, start):
		dock = QtGui.QDockWidget(title, self)
		dock.setWidget(obj)
		dock.setAllowedAreas(allowed)
		self.addDockWidget(start, dock)
		self.getOrCreateMenu('&View').addAction(dock.toggleViewAction())
		return dock

	def setFilename(self, filename):
		self.filename = filename
		self.setWindowTitle(State.appName() + ' - ' + self.filename.replace('\\','/').rpartition('/')[-1])
		self.updateMenus()

	def getOrCreateMenu(self, menu):
		if not self.menus.has_key(menu): self.menus[menu] = self.menuBar().addMenu(menu)
		return self.menus[menu]

	def cb_then_update(self, func):
		func()
		self.updateMenus()
		
	def addMenuItem(self, menu_dict):
		fileMenu = self.getOrCreateMenu(menu_dict['menu'])
		action = QtGui.QAction(menu_dict['item'], self)
		if 'shortcut' in menu_dict: action.setShortcut(menu_dict['shortcut'])
		if 'tip' in menu_dict: action.setStatusTip(menu_dict['tip'])
		cmd = eval(menu_dict['cmd']) if isinstance(menu_dict['cmd'],str) else menu_dict['cmd']
		if 'args' in menu_dict: cmd = functools.partial(cmd, *menu_dict['args'])
		action.triggered.connect(functools.partial(QApp.cb_then_update,self,cmd))
		for old_action in fileMenu.actions():
			if action.text() == old_action.text():
				fileMenu.insertAction(old_action,action)
				fileMenu.removeAction(old_action)
				return action
		fileMenu.addAction(action)
		return action

	@staticmethod
	def sure():
		if State.getUndoCmd() not in [None, 'save']:
			ok = QtGui.QMessageBox.critical(None, State.appName(), 'Your changes will be lost. Are you sure?',
										QtGui.QMessageBox.Ok | QtGui.QMessageBox.Default, QtGui.QMessageBox.Cancel)
			return ok != QtGui.QMessageBox.StandardButton.Cancel
		return True
		
	def load(self, filename=None):
		if not self.sure(): return
		if filename is None:
			filename, filtr = QtGui.QFileDialog.getOpenFileName(self, 'Choose a file to open', '.', 'SCN (*.scn)')
		if filename == '': return # did not load
		if self.preLoadCB: self.preLoadCB(filename, self)
		State.load(filename)
		self.setFilename(filename)

		if self.loadCB: self.loadCB(filename, self)

	def save(self):
		self.saveAs(self.filename)

	def saveAs(self, filename = None):
		if self.saveCB: self.saveCB(filename)

		if filename is None or filename == '':
			filename, filtr = QtGui.QFileDialog.getSaveFileName(self, 'Choose a file name to save as', '.', 'SCN (*.scn)')
		if filename == '': return # did not save
		State.save(filename)
		self.setFilename(filename)

	def closeEvent(self, event):
		if not self.sure(): event.ignore()
		else:               event.accept()

	def quit(self):
		if not self.sure(): return
		QtGui.qApp.quit()

	def setFields(self, type, attrs):
		global fields
		fields[type] = attrs

	def updateFieldKey(self, key):
		#print 'updateFieldKey',key 
		if not State.hasKey(key): return # TODO maybe need to remove something from UI
		sel = State.getSel()
		if sel is None: return self.qfields.setKeyValue(key, State.getKey(key))
		s = '/%s/attrs/'%sel.strip('/')
		#print 'updateFieldKey',s,key
		if key.startswith(s): self.qfields.setKeyValue(key[len(s):], State.getKey(key))

	#@profile
	def updateMenus(self):
		'''Keeps the GUI menus and the attribute editor in sync with the actual state.'''
		undoCmd = State.getUndoCmd()
		if undoCmd is None: self.undoItem.setText('&Undo')
		else:               self.undoItem.setText('&Undo [%s]' % undoCmd)
		redoCmd = State.getRedoCmd()
		if redoCmd is None: self.redoItem.setText('Re&do')
		else:               self.redoItem.setText('Re&do [%s]' % redoCmd)
		#print undoCmd, redoCmd
		if self.attributeEditor is not None:
			sel = State.getSel()
			if sel is None: self.qfields.setFields('',[],{})
			else:
				st = State.getKey(sel+'/type',None)
				sa = State.getKey(sel+'/attrs',{})
				global fields
				self.qfields.setFields(sel, fields.get(st,[]), sa)
			self.attributeEditor.setWidget(self.qfields)
		self.updateGL()

	def setFieldValueCommand(self, field, value):
		'''Be careful, this is usually called from a qt callback, so we mustn't rebuild qt objects here.'''
		if self.value_is_adjusting: return
		#print 'setFieldValueCommand', repr(field), repr(value)
		#print '/%s/attrs/%s' % (State.getSel(),field)
		State.setKey('/%s/attrs/%s' % (State.getSel(),field),value)
		self.edit_signal.emit(State.getSel(),field,value) # DH
		self.clean_state() # if changing the field has side effects, these should happen before the push (I think)
		State.push('Set %s' % str(field))
		undoCmd = State.getUndoCmd()
		if undoCmd is None: self.undoItem.setText('&Undo')
		else:               self.undoItem.setText('&Undo [%s]' % undoCmd)
		redoCmd = State.getRedoCmd()
		if redoCmd is None: self.redoItem.setText('Re&do')
		else:               self.redoItem.setText('Re&do [%s]' % redoCmd)

	@staticmethod
	def createKeyCommand(key, value):
		key = State.addKey(key,value)
		State.push('Create %s' % State.keyToName(key))
		app.updateMenus()

	@staticmethod
	def select(key):
		#print 'in select',key
		State.setSel(key) # TODO this should be a command, and should link with the UI
		global app
		app.syncOutliner()
		app.updateMenus()
		#print 'finished select'
	
	def syncOutliner(self):
		'''Ensure that the outliner is synchronized with the current selection'''
		try:
			self.blockUpdate = True
			self.qoutliner.sync()
		except Exception as e:
			print 'exception in syncOutliner',e
		finally:
			self.blockUpdate = False
	
	def clearUndo(self):
		State.clearUndoStack()
		self.updateMenus()
	
	def undo(self):
		State.undo()
		self.updateMenus()

	def redo(self):
		State.redo()
		self.updateMenus()
		
	def loadFilename(self, title='Choose an image to open',directory='.',filtr='Image Files (*.jpg *.jpeg *.png *.bmp *.tif)'):
		return QtGui.QFileDialog.getOpenFileName(self, title, directory, filtr)

	def saveFilename(self, title='Choose an image to save to',directory='.',filtr='Image Files (*.jpg *.jpeg *.png *.bmp *.tif)'):
		return QtGui.QFileDialog.getSaveFileName(self, title, directory, filtr)

	def loadImage(self):
		filename, filtr = self.loadFilename('Choose an image to open', '.', 'Image Files (*.jpg *.jpeg *.png *.bmp *.tif)')
		if filename == '': return # did not load
		# TODO, should be an undoable command
		self.qglview.setImage(filename)
		
	def view(self):
		return self.qview.view
	
	def clean_state(self):
		if State.g_dirty:
			self.value_is_adjusting = True
			try:
				dirty = set(State.g_dirty)
				#print 'dirty',dirty
				State.g_dirty.clear()
				for k in dirty: self.updateFieldKey(k)
				if self.dirtyCB is not None: self.dirtyCB(dirty)
			except Exception as e:
				import traceback
				print 'clean_state exception',e,traceback.format_exc()
				pass
			self.value_is_adjusting = False
	
	def updateGL(self):
		#print 'in updateGL'
		self.clean_state()
		self.view().updateGL()
		#print 'finished'

	def refresh(self):
		self.qtimeline.refresh()

	def refreshImageData(self):
		self.view().refreshImageData()

	def new(self):
		del self.view().primitives[:]
		del self.view().primitives2D[:]
		self.view().points = None
		grid = GLGrid()
		self.view().primitives.append(grid)
		self.view().setLayer('grid', grid)
		self.view().camera.reset2D()
		self.view().camera.reset3D()
		self.qtimeline.setRange(1, 100)
		State.new()
		self.updateMenus()

	def addPoints2D(self, points):
		'''
		add 2d points to the viewer.

		:type points: :class:`numpy.array` (Nx2)
		:param points: array of vertices
		'''
		glPoints = GLPoints2D(points)
		self.view().primitives2D.append(glPoints)
		return self.view().primitives2D[-1]

	def addPoints3D(self, points):
		'''
		add 3d points to the viewer.

		:type points: :class:`numpy.array` (Nx3)
		:param points: array of vertices
		'''
		glPoints = GLPoints3D(points)
		self.view().primitives.append(glPoints)
		return self.view().primitives[-1]

	def addCameras(self, mats, camera_ids=None, movies=None):
		if camera_ids is None: camera_ids = range(len(mats))
		if movies is None: movies = [None]*len(mats)
		for mat, cid, md in zip(mats, camera_ids, movies):
			camera = QGLViewer.Camera(cid)
			camera.setP(mat[2], distortion=mat[3], store=True)
			if md is not None: camera.setImageData(md['vbuffer'],md['vheight'],md['vwidth'],3)
			self.view().addCamera(camera)
		cams = GLCameras(camera_ids, mats)
		self.view().primitives.append(cams)
		return cams

	def _allowFileDrop(self, e):
		return e.mimeData().hasFormat('FileName') or e.mimeData().hasFormat('text/uri-list')

	def dragEnterEvent(self, e):
		if self._allowFileDrop(e):
			e.accept()
			return

	def dropEvent(self, e):
		if self._allowFileDrop(e):
			filename = str(e.mimeData().urls()[0].toLocalFile())
			if filename.lower().endswith('.scn'):
				self.load(filename)
			else:
				self.runtime.handleDroppedFile(filename)


if __name__ == '__main__':
	import sys
	a = QtGui.QApplication(sys.argv)
	a.setStyle('plastique')
	fields = {'image filter':[
		('filename', 'File name', 'Full-path to the file on disk. For sequences, choose any file from the sequence.', 'filename', None),
		('issequence',  'Is sequence', 'Is this an image sequence (checked), or a static image.', 'bool', False),
		(None, None, None, 'if', 'issequence'),
		('numdigits',  'Number of digits', 'The number of digits in the number sequence (pad with leading zeros). Choose 0 for no padding.', 'int', 0, {"min":0,"max":None}),
		('inframe',  'Start frame', 'If this is an image sequence, the first frame of the sequence.', 'int', 0, {"min":0,"max":None}),
		('outframe', 'End frame', 'If this is an image sequence, the last frame of the sequence.', 'int', 0, {"min":0,"max":None}),
		(None, None, None, 'endif', 'issequence'),
		('deinterlace', 'Deinterlace mode', 'If the video is interlaced, choose the deinterlace mode that gives continuous motion.', 'select', 'None', {'enum':['None','Odd only','Even only','Odd-Even','Even-Odd']}),
		('fps', 'Frames per second', 'Controls the playback speed. For deinterlaced video, use fields per second.', 'float', 24.0, {"min":0.0,"max":None}),
	], 'test':[]}
	State.addKey('mykey', {'type':'image filter', 'attrs':{'filename':None,'issequence':True,'fps':24.0}})
	State.setSel('mykey')
	win = QApp()
	def testCB(*x):
		print ("x = {}".format(x))
	win.addWidget(QtGui.QListWidget(),'List Widget',{'currentItemChanged':[testCB]})
	win.set_widget['List Widget'].insertItem(0,'hello world')
	State.addKey('hello',{'type':'test','attrs':{}})
	State.addKey('there',{'type':'image filter','attrs':{'filename':'test'}})
	State.clearUndoStack()
	win.show()
	win.updateMenus()
	# print (State.g_state)

	a.connect(a, QtCore.SIGNAL('lastWindowClosed()') , a.quit)
	sys.exit(a.exec_())
