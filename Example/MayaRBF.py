import maya.cmds as cmds
import maya.mel
import maya.mel as mel
import maya.OpenMayaUI as apiUI
from PySide import QtCore, QtGui
import shiboken, functools, os, pickle
from maya.app.general.mayaMixin import MayaQWidgetDockableMixin
from glob import glob
import json
import copy
import numpy as np
import __main__

g_folder_path = ''


def duplicateSlider(slider):
	try:
		restoreSlider(slider)
	except ValueError:
		pass
	cmds.duplicate(slider, n=slider + '_dup')
	cmds.hide(slider + '_dup')
	cmds.cutKey(slider)
	try:
		cmds.pasteKey(slider + '_dup')
	except RuntimeError:
		pass


def restoreSlider(slider):
	cmds.cutKey(slider + '_dup')
	try:
		cmds.pasteKey(slider)
	except RuntimeError:
		pass
	cmds.delete(slider + '_dup')


def getMayaWindow():
	ptr = apiUI.MQtUtil.mainWindow()
	if ptr is not None:
		return shiboken.wrapInstance(long(ptr), QtGui.QWidget)


def getSliderAttributes():
	mySliders = map(str, cmds.ls(sl=1))
	myAttr = []
	for slider in mySliders:
		attr = map(str, cmds.listAttr(slider, k=1, u=1, v=1, w=1))
		for item in attr:
			if item != 'horPairs' and item != 'verPairs':
				myAttr.append(str(slider) + '.' + str(item))
	stripNamespace = lambda x: x.split(':', 1)[-1]
	return map(stripNamespace, myAttr)


def viewportOff(func):
	"""
	Decorator - turn off Maya display while func is running.
	if func will fail, the error will be raised after.
	"""

	@functools.wraps(func)
	def wrap(*args, **kwargs):

		# Turn $gMainPane Off:a
		mel.eval("paneLayout -e -manage false $gMainPane")

		# Decorator will try/except running the function.
		# But it will always turn on the viewport at the end.
		# In case the function failed, it will prevent leaving maya viewport off.
		try:
			return func(*args, **kwargs)
		except Exception:
			raise  # will raise original error
		finally:
			mel.eval("paneLayout -e -manage true $gMainPane")

	return wrap


def computeRBF(x, c, beta, var=None):
	# takes an input point and the centre of the RBF and returns the value of
	# p(||x - c||)
	if not isinstance(x, np.ndarray):
		x = np.array([[x]])
	elif len(x.shape) == 1:
		x.shape = (-1, 1)
	if var is None:
		return np.exp(-(beta * np.linalg.norm(x - c, axis=1) ** 2))
	else:
		return np.exp(-0.5 * (beta * mahalanobisDist(x, c, var) ** 2))


def mahalanobisDist(x, mu, var):
	# Computes the mahalanobis distance (x-mu) cov^-1 (x-mu).T
	# Work in progress
	if len(x.shape) == 1:
		x.shape = (1, -11)
	if len(mu.shape) == 1:
		mu.shape = (1, -1)
	if len(var.shape) == 1:
		var.shape = (1, -11)
	data = (x - mu) ** 2 / var
	dist = np.sqrt(np.sum(data, axis=1))
	return dist


def normalizedRBFN(X, C, Beta, normalise=True):
	eps = 1e-10
	out = np.zeros((X.shape[0], C.shape[0]), dtype=X.dtype)
	var = np.var(C, axis=0)
	for ci, c in enumerate(C):
		out[:, ci] = computeRBF(X, c, Beta[ci], var=None)
	if normalise:
		row_sums = np.sum(out, axis=1)
		out /= (row_sums[:, np.newaxis] + eps)
	return out


def evaluateRBFN(W, C, Beta, input, normalise=True):
	# Evaluates an RBFN for a given set of Weights, Centres on a given input
	num_dims = W.shape[1]
	output = np.zeros((input.shape[0], num_dims), dtype=input.dtype)
	RBFs = normalizedRBFN(input, C, Beta, normalise)
	output[:] = np.dot(RBFs, W[:, :])
	return output, (RBFs / np.sum(RBFs))[0, :]


def rNearestNeighbourBetaCalculation(C, r):
	if len(C.shape) == 1:
		C = C.reshape(-1, 1)
	r = min((C.shape[0] - 1), r)
	Beta = np.zeros((C.shape[0], 1), dtype=np.float32)
	dists = np.zeros((C.shape[0], C.shape[0]))
	neighbours = np.zeros((C.shape[0], r), dtype=np.float32)
	var = np.var(C, axis=0)
	for i in xrange(C.shape[0]):
		for j in xrange(C.shape[0]):
			dists[i, j] = np.linalg.norm(C[i] - C[j])  # mahalanobisDist(C[i], C[j], var)
		neighbours[i, :] = np.sort(dists[i, :])[1:r + 1]
		Beta[i] = (np.sqrt(np.sum(neighbours[i, :] ** 2) / r))
	return Beta


def totalBetaCalc(C):
	if len(C.shape) == 1:
		C = C.reshape(-1, 1)
	d_max = 0
	var = np.var(C, axis=0)
	for i in xrange(C.shape[0]):
		for j in xrange(i + 1, C.shape[0]):
			dist = np.linalg.norm(C[i] - C[j])  # mahalanobisDist(C[i, :], C[j, :], var)
			if dist > d_max:
				d_max = dist
	Beta = np.array([np.sqrt(2 * C.shape[0]) / d_max] * C.shape[0]).reshape(-1, 1)
	return Beta


def trainRBFN(X, Y, normalise=True):
	# Trains a RBFN given a set of known points
	# returns the weights and centres
	if not isinstance(Y, np.ndarray):
		Y = np.array([[Y]])
	elif len(Y.shape) == 1:
		Y.shape = (-1, 1)
	# Beta = np.array([1] * X.shape[0])
	Beta = rNearestNeighbourBetaCalculation(X, 5)
	Beta = (totalBetaCalc(X)) * (1 / Beta)
	x_size, y_dim = X.shape[0], Y.shape[1]
	G = np.zeros((x_size, x_size), dtype=X.dtype)
	G[:, :] = normalizedRBFN(X, X, Beta, normalise)
	W = np.zeros((x_size, y_dim), dtype=X.dtype)
	error = np.zeros(y_dim, dtype=np.float32)
	for dim in xrange(y_dim):
		W[:, dim] = np.linalg.lstsq(G, Y[:, dim])[0]
		# print "W_{}: {}".format(dim, W[:, dim])
		error[dim] = np.linalg.norm(Y[:, dim] - np.dot(G, W[:, dim]).T)
	print "Error: {}".format(np.mean(error))
	return W, X, Beta


def trianglesToEdgeList(triangles, numVerts=None):
	'''Convert a list of triangle indices to an array of up-to-10 neighbouring vertices per vertex (following anticlockwise order).'''
	if numVerts is None: numVerts = np.max(triangles) + 1 if len(triangles) else 1
	if numVerts < 1: numVerts = 1  # avoid empty arrays
	T = [dict() for t in xrange(numVerts)]
	P = [dict() for t in xrange(numVerts)]
	for t0, t1, t2 in triangles:
		T[t0][t1], T[t1][t2], T[t2][t0] = t2, t0, t1
		P[t1][t0], P[t2][t1], P[t0][t2] = t2, t0, t1
	S = np.zeros((numVerts, 10), dtype=np.int32)
	for vi, (Si, es, ps) in enumerate(zip(S, T, P)):
		Si[:] = vi
		if not es: continue
		v = es.keys()[0]
		while v in ps: v = ps.pop(v)
		for li in xrange(10):
			Si[li] = v
			if v not in es: break
			v = es.pop(v, vi)
	return S


def pointsToEdges(points, mapping_list):
	# mapping_list is such that i is mapped to mapping_list[i]
	from scipy.spatial import Delaunay
	tris = Delaunay(points).simplices
	edges = trianglesToEdgeList(tris)
	edgeList = set()
	for i in xrange(edges.shape[0]):
		for j in xrange(edges.shape[1]):
			if edges[i, j] > i:
				edgeList.add((mapping_list[i], mapping_list[edges[i, j]]))
	return np.array(list(edgeList))


class QListWidget(QtGui.QListView):
	item_selected = QtCore.Signal(int)
	focus_changed = QtCore.Signal(bool)
	item_renamed = QtCore.Signal(str, str)
	item_checked = QtCore.Signal(str, bool)

	def __init__(self, items=[], parent=None, renameEnabled=False):
		super(QListWidget, self).__init__(parent)
		self.item_count = 0
		self.renameEnabled = renameEnabled
		self.overrideSelection = None
		self.selectedItem = None
		self.item_list_model = None
		self.item_selection_model = None
		self.setDragEnabled(True)
		self.setDragDropOverwriteMode(False)
		self.setDragDropMode(QtGui.QAbstractItemView.InternalMove)
		self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
		self.createWidgets()
		for item in items:
			self.addItem(item)

	def createWidgets(self):
		self.item_list_model = QtGui.QStandardItemModel(self)
		self.item_list_model.setSortRole(QtCore.Qt.DisplayRole)
		self.item_list_model.dataChanged.connect(self.handleDataChange)
		self.setModel(self.item_list_model)
		self.item_selection_model = self.selectionModel()
		self.item_selection_model.selectionChanged.connect(self.handleItemSelect)
		self.setMinimumHeight(60)
		self.setSizePolicy(QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred))

	def handleDataChange(self, *args):
		print "Data Change: {}".format(args)
		self.overrideSelection = args[0].row()
		newText = self.getItem(args[0].row())
		selectedIndex = args[0].row()
		if selectedIndex == self.selectedItem[0] and newText <> self.selectedItem[1]:
			print "Emit: {} to {}".format(self.selectedItem[1], newText)
			self.item_renamed.emit(self.selectedItem[1], newText)
			self.selectedItem[1] = newText
		else:
			item = self.item_list_model.item(args[0].row())
			self.item_checked.emit(item.data(QtCore.Qt.DisplayRole),
								   True if item.checkState() == QtCore.Qt.Checked else False)
			print item.data(QtCore.Qt.DisplayRole), True if item.checkState() == QtCore.Qt.Checked else False

	def focusInEvent(self, *args):
		self.focus_changed.emit(True)

	def focusOutEvent(self, *args):
		self.focus_changed.emit(False)

	def handleItemSelect(self, *args):
		if self.overrideSelection is not None:
			self.setUserSelection(self.overrideSelection)
			self.overrideSelection = None
			return
		try:
			selectedIndex = self.getSelection()
			self.selectedItem = [selectedIndex, self.getItem(selectedIndex)]
			print "Selected: {}".format(self.selectedItem[1])
			self.item_selected.emit(selectedIndex)
		except AttributeError:
			pass

	def getSelection(self):
		try:
			selection = self.item_selection_model.selection().indexes()[0].row()
		except IndexError:
			selection = -1
		return selection

	def removeItem(self, index):
		self.item_list_model.takeRow(index)
		self.item_count -= 1

	def clear(self):
		while self.item_count:
			self.removeItem(0)

	def addItem(self, mitem, checked=None):
		item = QtGui.QStandardItem()
		item.setData(mitem, QtCore.Qt.DisplayRole)
		item.setEditable(self.renameEnabled)
		item.setDropEnabled(False)
		if checked is not None:
			item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
			item.setCheckState(QtCore.Qt.Unchecked if not checked else QtCore.Qt.Checked)
		# Can be used to store data linked to the name
		# item.setData(customData, QtCore.Qt.UserRole)
		self.item_list_model.appendRow(item)
		self.item_count += 1

	def addItems(self, items, checked=None):
		if checked is None:
			checked = [None] * len(items)
		for item, check in zip(items, checked):
			self.addItem(item, checked=check)

	def setUserSelection(self, index):
		if self.item_count > 0:
			self.setCurrentIndex(self.item_list_model.item(index).index())
			self.selectedItem = self.getItem(index)

	def getItems(self):
		return [self.item_list_model.item(i).data(QtCore.Qt.DisplayRole) for i in xrange(0, self.item_count)]

	def getItem(self, index):
		return self.item_list_model.item(index).data(QtCore.Qt.DisplayRole)


class QfloatWidget(QtGui.QLineEdit):
	''' draggable spin box. ctrl+ left, middle or right button will scrub the values in the spinbox
	by different amounts
	'''
	valueChanged = QtCore.Signal(list)

	def __init__(self, parent=None):
		super(QfloatWidget, self).__init__(parent)
		# self.setDecimals(4)
		# self.setButtonSymbols(QtGui.QAbstractSpinBox.NoButtons)
		# self.setKeyboardTracking(False)  # don't emit 3 times when typing 100
		self.minstep = 0.001
		self._dragging = False
		self.setContextMenuPolicy(QtCore.Qt.PreventContextMenu)  # no copy/paste menu to interfere with dragging
		# catch the mouse events from the lineedit that is a child of the spinbox
		# editor = self.findChild(QtGui.QLineEdit)
		self.installEventFilter(self)
		self.editingFinished.connect(functools.partial(self.handleEditingFinished))
		# Initialise the current value
		self.current_value = None
		# Create a new Validator
		# dblValidator = QtGui.QDoubleValidator(self)
		dbl_validator = floatValidator(self)
		self.setValidator(dbl_validator)
		# Initialise the Range variables to nothing.
		self.setRange()

	def handleEditingFinished(self):
		self.setValue(self.text())

	def setValue(self, v):
		v = float(v)  # ensure it's a float!
		if self.current_value == v: return
		self.current_value = v
		self.setText(str(v))
		# if not self._dragging:
		self.valueChanged.emit(v)

	# Constrains the spin box to between two values
	def setRange(self, min=None, max=None):
		try:
			self.validator().setRange(min, max)
		# print "Valid from {} to {}".format(str(self.validator().bottom()), str(self.validator().top()))
		except:
			print "Inputs to QfloatWidget.setRange() are invalid with values {} and {}".format(min, max)

	# Allows the box to be locked or unlocked
	# Defaults to true so foo.setLocked() would lock "foo"
	def setLocked(self, status=True):
		assert isinstance(status, bool), "Lock value is not a boolean"
		self.setReadOnly(status)

	def value(self):
		return float(self.text())

	def text(self):
		ret = super(QfloatWidget, self).text()
		return ret

	def eventFilter(self, obj, event):
		if event.type() == QtGui.QMouseEvent.MouseButtonPress:
			if not event.modifiers() & QtCore.Qt.ControlModifier:
				return False
			self.gpx, self.gpy = event.globalX(), event.globalY()
			self.startX, self.startY = event.x(), event.y()
			if event.button() & QtCore.Qt.LeftButton:
				self._dragging = self.minstep
			if event.button() & QtCore.Qt.MiddleButton:
				self._dragging = self.minstep * 100
			if event.button() & QtCore.Qt.RightButton:
				self._dragging = self.minstep * 10000
			return True
		elif event.type() == QtGui.QMouseEvent.MouseButtonRelease:
			if self._dragging is not False:
				self._dragging = False
				self.setValue(self.text())
			else:
				self._dragging = False
			return True
		elif event.type() == QtGui.QMouseEvent.MouseMove:
			if self._dragging:
				if not self.isReadOnly():
					newValue = (self.value() + (event.x() - self.startX) * self._dragging)
					if self.validator().bottom() is not None or self.validator().top() is not None:
						newValue = np.clip(newValue, self.validator().bottom(), self.validator().top())
					self.setValue(newValue)
				QtGui.QCursor.setPos(self.gpx, self.gpy)
				return True
		return False


class floatValidator(QtGui.QValidator):
	def __init__(self, parent=None):
		from re import compile as re_compile
		QtGui.QValidator.__init__(self, parent)
		self.parent = parent
		self.min_value = None
		self.max_value = None
		# RegExp for a valid number including scientific notation
		self._re = re_compile("^[-+]?[0-9]*\.?[0-9]*([eE][-+]?[0-9]*)?$")

	def setRange(self, min=None, max=None):
		try:
			self.min_value = None if min is None else float(min)
			self.max_value = None if max is None else float(max)
		except ValueError:
			assert False, "Incorrect value types for floatValidator.setRange()"

	def bottom(self):
		return self.min_value

	def top(self):
		return self.max_value

	def validate(self, text, length):
		if len(text) == 0: return (QtGui.QValidator.Intermediate)
		if self.parent.hasFocus():
			if not self._re.match(text):
				return (QtGui.QValidator.Invalid)
		else:
			try:
				value = float(text)
			except ValueError:
				return (QtGui.QValidator.Invalid)
			if self.min_value is not None and value < self.min_value: return (QtGui.QValidator.Invalid)
			if self.max_value is not None and value > self.max_value: return (QtGui.QValidator.Invalid)
		return (QtGui.QValidator.Acceptable)

	def fixup(self, input):
		if input == "":
			self.parent.setText(str(self.min_value) if self.min_value is not None else 0.0)
		else:
			try:
				value = float(input)
			except ValueError:  # Error is with an incomplete scientific notation
				input = input[:input.find("e")]
				value = float(value)
			if self.min_value is not None or self.max_value is not None:
				value = np.clip(value, self.min_value, self.max_value)
			self.parent.setText(str(value))


class QintWidget(QtGui.QLineEdit):
	''' draggable spin box. ctrl+ left, middle or right button will scrub the values in the spinbox
	by different amounts
	'''
	valueChanged = QtCore.Signal(list)

	def __init__(self, parent=None):
		super(QintWidget, self).__init__(parent)
		# self.setDecimals(4)
		# self.setButtonSymbols(QtGui.QAbstractSpinBox.NoButtons)
		# self.setKeyboardTracking(False)  # don't emit 3 times when typing 100
		self.minstep = 1
		self._dragging = False
		self.setContextMenuPolicy(QtCore.Qt.PreventContextMenu)  # no copy/paste menu to interfere with dragging
		# catch the mouse events from the lineedit that is a child of the spinbox
		# editor = self.findChild(QtGui.QLineEdit)
		self.installEventFilter(self)
		self.editingFinished.connect(functools.partial(self.handleEditingFinished))
		# Add a Validator
		int_validator = intValidator(self)
		self.setValidator(int_validator)
		self.setRange()

	def handleEditingFinished(self):
		self.setValue(self.text())

	def setValue(self, v):
		v = int(v)  # ensure it's an int!
		if self.text() == str(v): return
		self.setText(str(v))
		# if not self._dragging:
		self.valueChanged.emit(v)

	# Constrains the spin box to between two values
	def setRange(self, min=None, max=None):
		try:
			if min is not None: self.validator().setBottom(min)
			if max is not None: self.validator().setTop(max)
		# print "Valid from {} to {}".format(str(self.validator().bottom()), str(self.validator().top()))
		except:
			print "Inputs to QintWidget.setRange() are invalid with values {} and {}".format(min, max)

	# Allows the box to be locked or unlocked
	# Defaults to true so foo.setLocked() would lock "foo"
	def setLocked(self, status=True):
		assert isinstance(status, bool), "Lock value is not a boolean"
		self.setReadOnly(status)

	def value(self):
		return int(self.text())

	def text(self):
		ret = super(QintWidget, self).text()
		return ret

	def eventFilter(self, obj, event):
		if event.type() == QtGui.QMouseEvent.MouseButtonPress:
			if not event.modifiers() & QtCore.Qt.ControlModifier:
				return False
			self.gpx, self.gpy = event.globalX(), event.globalY()
			self.startX, self.startY = event.x(), event.y()
			if event.button() & QtCore.Qt.LeftButton:
				self._dragging = self.minstep
			if event.button() & QtCore.Qt.MiddleButton:
				self._dragging = self.minstep * 100
			if event.button() & QtCore.Qt.RightButton:
				self._dragging = self.minstep * 10000
			return True
		elif event.type() == QtGui.QMouseEvent.MouseButtonRelease:
			if self._dragging is not False:
				self._dragging = False
				self.setValue(self.text())
			else:
				self._dragging = False
			return True
		elif event.type() == QtGui.QMouseEvent.MouseMove:
			if self._dragging:
				if not self.isReadOnly():
					newValue = (self.value() + (event.x() - self.startX) * self._dragging)
					if self.validator().bottom() is not None or self.validator().top() is not None:
						newValue = np.clip(newValue, self.validator().bottom(), self.validator().top())
					self.setValue(newValue)
				QtGui.QCursor.setPos(self.gpx, self.gpy)
				return True
		return False


class intValidator(QtGui.QValidator):
	def __init__(self, parent=None):
		QtGui.QValidator.__init__(self, parent)
		self.parent = parent
		self.min_value = None
		self.max_value = None

	def setRange(self, min=None, max=None):
		try:
			self.min_value = None if min is None else int(min)
			self.max_value = None if max is None else int(max)
		except ValueError:
			assert False, "Incorrect value types for floatValidator.setRange()"

	def bottom(self):
		return self.min_value

	def top(self):
		return self.max_value

	def validate(self, text, length):
		if len(text) == 0 or text == "-": return (QtGui.QValidator.Intermediate)
		if self.parent.hasFocus():
			try:
				value = int(text)
			except ValueError:
				return (QtGui.QValidator.Invalid)
		else:
			try:
				value = int(text)
			except ValueError:
				return (QtGui.QValidator.Invalid)
			value = int(text)
			if self.min_value is not None and value < self.min_value: return (QtGui.QValidator.Invalid)
			if self.max_value is not None and value > self.max_value: return (QtGui.QValidator.Invalid)
		return (QtGui.QValidator.Acceptable)

	def fixup(self, input):
		if input == "" or input == "-":
			self.parent.setText(str(self.min_value) if self.min_value is not None else 0)
		else:
			if self.min_value is not None or self.max_value is not None:
				value = np.clip(int(input), self.min_value, self.max_value)
			self.parent.setText(str(value))


class FaceRBFMainWindow(QtGui.QMainWindow):  # MayaQWidgetDockableMixin,
	def __init__(self, parent=getMayaWindow()):
		super(FaceRBFMainWindow, self).__init__(parent)
		self.setParent(parent)
		self.setWindowTitle('Face RBFN')
		self.setWindowFlags(
			QtCore.Qt.Window | QtCore.Qt.WindowTitleHint | QtCore.Qt.CustomizeWindowHint | QtCore.Qt.WindowMinimizeButtonHint |
			QtCore.Qt.WindowMaximizeButtonHint | QtCore.Qt.WindowCloseButtonHint)
		self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
		self.faceRBFWindow = FaceRBFWindow(self)
		self.setCentralWidget(self.faceRBFWindow)
		self.filePath = None
		menubar = self.menuBar()
		fileMenu = menubar.addMenu('&File')
		self.createFileMenuActions(fileMenu)
		viewMenu = menubar.addMenu('&View')
		toggleImageAction = viewMenu.addAction('&Toggle Image')
		toggleImageAction.triggered.connect(self.toggleImage)
		bakeMenu = menubar.addMenu('&Bake')
		bakeFramesAction = bakeMenu.addAction('Bake &Frames')
		bakeFramesAction.triggered.connect(self.bakeFrames)
		refreshMenu = menubar.addMenu('&Refresh')
		refreshMoCapAction = refreshMenu.addAction('Refresh &MoCap')
		refreshMoCapAction.triggered.connect(self.refreshMoCap)
		refreshDataAction = refreshMenu.addAction('Refresh all &Data')
		refreshDataAction.triggered.connect(self.refreshData)
		refreshImagesAction = refreshMenu.addAction('Refresh Images')
		refreshImagesAction.triggered.connect(self.faceRBFWindow.refreshImages)
		importMenu = menubar.addMenu('&Import')
		importPoseMenu = importMenu.addAction('Import &Pose')
		importPoseMenu.triggered.connect(self.importPoses)
		meshMenu = menubar.addMenu('&Mesh')
		self.toggleMeshAction = meshMenu.addAction('&Show Mesh')
		self.toggleMeshAction.triggered.connect(self.toggleMesh)

	# solverMenu = menubar.addMenu('&Solver')
	# self.selectModeAction = solverMenu.addAction('Use &Mahalanobis')
	# self.selectModeAction.triggered.connect(self.toggleMode)
	# offsetMenu = menubar.addMenu('&Offsets')
	# self.setFromOffsetAction = offsetMenu.addAction('Set &Base Neutral')
	# self.setFromOffsetAction.triggered.connect(self.faceRBFWindow.setFromOffset)
	# self.setToOffsetAction = offsetMenu.addAction('Set &Current Neutral')
	# self.setToOffsetAction.triggered.connect(self.faceRBFWindow.setToOffset)

	def toggleMesh(self):
		if self.toggleMeshAction.text() == '&Show Mesh':
			self.faceRBFWindow.showMesh(True)
			self.toggleMeshAction.setText('&Hide Mesh')
		else:
			self.faceRBFWindow.showMesh(False)
			self.toggleMeshAction.setText('&Show Mesh')

	def resizeEvent(self, event):
		self.faceRBFWindow.doResize()

	def createFileMenuActions(self, menu):
		saveAction = menu.addAction('&Save')
		saveAction.triggered.connect(self.saveGroups)
		saveAsAction = menu.addAction('Save &As')
		saveAsAction.triggered.connect(functools.partial(self.saveGroups, True))
		loadAction = menu.addAction('&Load')
		loadAction.triggered.connect(self.loadGroups)

	def toggleImage(self):
		self.faceRBFWindow.toggleImage()

	def bakeFrames(self):
		self.faceRBFWindow.bakeFrames()

	def refreshMoCap(self):
		self.faceRBFWindow.refreshMoCap()

	def refreshData(self):
		self.faceRBFWindow.refreshData()

	def saveGroups(self, saveAs=False):
		self.faceRBFWindow.setGroupPoseOrder()
		if saveAs or self.filePath is None:
			if self.filePath is not None:
				folderPath = self.filePath.rsplit('.', 1)[0] + '\\'
			else:
				folderPath = ''
			save_name = cmds.fileDialog(dm=folderPath + '*.rbfn', m=1)
		else:
			save_name = self.filePath
		if save_name == '': return False
		print "Saving to: {}".format(save_name)
		try:
			with open(save_name, 'w') as f:
				pickle.dump(self.faceRBFWindow.groups, f)
			self.filePath = save_name
			return True
		except IOError:
			self.filePath = None
			self.faceRBFWindow.errorDialog.showMessage("Unable to Save File: {}".format(save_name))
			return False

	def loadGroups(self):
		if self.filePath is not None:
			folderPath = self.filePath.rsplit('.', 1)[0] + '\\'
		else:
			folderPath = ''
		load_name = cmds.fileDialog(dm=folderPath + '*.rbfn', m=0)
		if not os.path.exists(load_name):
			self.faceRBFWindow.errorDialog.showMessage("Unable to load file: {}".format(load_name))
			return
		try:
			with open(load_name, 'r') as f:
				groups = pickle.load(f)
			self.filePath = load_name
			self.faceRBFWindow.setGroups(groups)
		except IOError:
			self.faceRBFWindow.errorDialog.showMessage("Unable to load file: {}".format(load_name))

	def closeEvent(self, evnt):
		print "Closing Main"
		ret = self.faceRBFWindow.closeHandler()
		if ret == False:
			evnt.ignore()

	def setSettings(self, settings_data):
		self.faceRBFWindow.loadSettings(settings_data)

	def importPoses(self):
		if self.filePath is not None:
			folderPath = self.filePath.rsplit('.', 1)[0] + '\\'
		else:
			folderPath = ''
		load_name = cmds.fileDialog(dm=folderPath + '*.rbfn', m=0)
		if not os.path.exists(load_name):
			self.faceRBFWindow.errorDialog.showMessage("Unable to load file: {}".format(load_name))
			return
		try:
			with open(load_name, 'r') as f:
				groups = pickle.load(f)
			self.filePath = load_name
			self.faceRBFWindow.importPoses(groups)
		except IOError:
			self.faceRBFWindow.errorDialog.showMessage("Unable to load file: {}".format(load_name))


class FaceRBFWindow(QtGui.QWidget):
	def __init__(self, parent):
		super(FaceRBFWindow, self).__init__(parent)
		self.groups = {}
		self.activeGroup = None
		self.setting = False
		self.whole = True
		self.showMarkers = True
		self.expression = None
		self.frameJump = True
		self.pixmap = None
		self.offset = None
		self.global_sliders = ['CTL_R_brow_raiseOut.translateY', 'CTL_R_brow_raiseIn.translateY',
							   'CTL_L_brow_raiseIn.translateY', 'CTL_L_brow_raiseOut.translateY',
							   'CTL_R_brow_down.translateY', 'CTL_L_brow_down.translateY',
							   'CTL_R_brow_lateral.translateY', 'CTL_L_brow_lateral.translateY',
							   'CTL_L_neck_stretch.translateY', 'CTL_L_neck_stretch.mastoidContract',
							   'CTL_R_neck_stretch.translateY', 'CTL_R_neck_stretch.mastoidContract',
							   'CTL_C_tongue.translateX', 'CTL_C_tongue.translateY', 'CTL_C_tongue.press',
							   'CTL_C_tongue.inOut', 'CTL_C_tongue.narrowWide', 'CTL_C_tongueRoll.translateX',
							   'CTL_C_tongueRoll.translateY', 'CTL_L_eye.translateX', 'CTL_L_eye.translateY',
							   'CTL_L_eye.relax', 'CTL_L_eye.squintInner', 'CTL_L_eye.cheekRaise', 'CTL_L_eye.squeeze',
							   'CTL_L_eye.widenBlink', 'CTL_L_eye.pupilNarrowWide', 'CTL_R_eye.translateX',
							   'CTL_R_eye.translateY', 'CTL_R_eye.relax', 'CTL_R_eye.squintInner',
							   'CTL_R_eye.cheekRaise', 'CTL_R_eye.squeeze', 'CTL_R_eye.widenBlink',
							   'CTL_R_eye.pupilNarrowWide', 'CTL_C_eye.translateX', 'CTL_C_eye.translateY',
							   'CTL_C_eye.lookAt', 'CTL_L_mouth_dimple.translateY', 'CTL_L_mouth_dimple.sharpUpper',
							   'CTL_L_mouth_dimple.sharpLower', 'CTL_L_mouth_dimple.lipsPress',
							   'CTL_L_mouth_dimple.suckBlow', 'CTL_L_mouth_dimple.lipsBlow',
							   'CTL_L_mouth_dimple.sticky', 'CTL_L_mouth_cornerDepress.translateY',
							   'CTL_L_mouth_cornerDepress.funnel', 'CTL_L_mouth_cornerDepress.purse',
							   'CTL_L_mouth_cornerDepress.towards', 'CTL_L_mouth_cornerDepress.pushPull',
							   'CTL_L_mouth_cornerDepress.thickness', 'CTL_L_mouth_stretch.translateY',
							   'CTL_L_mouth_stretch.upLipFollow', 'CTL_L_mouth_stretch.mouthPress',
							   'CTL_L_mouth_stretch.lipsTighten', 'CTL_L_mouth_stretch.lipsTogether',
							   'CTL_L_mouth_lowerLipDepress.translateY', 'CTL_L_mouth_lowerLipDepress.chinRaise',
							   'CTL_R_mouth_upperLipRaise.translateY', 'CTL_R_mouth_upperLipRaise.chinRaise',
							   'CTL_R_mouth_sharpCornerPull.translateY', 'CTL_R_mouth_sharpCornerPull.mouthPress',
							   'CTL_R_mouth_sharpCornerPull.lipsTighten', 'CTL_R_mouth_sharpCornerPull.lipsTogether',
							   'CTL_R_mouth_cornerPull.translateY', 'CTL_R_mouth_cornerPull.funnel',
							   'CTL_R_mouth_cornerPull.purse', 'CTL_R_mouth_cornerPull.towards',
							   'CTL_R_mouth_cornerPull.open', 'CTL_R_mouth_cornerPull.pushPull',
							   'CTL_R_mouth_cornerPull.thickness', 'CTL_R_mouth_dimple.translateY',
							   'CTL_R_mouth_dimple.sharpUpper', 'CTL_R_mouth_dimple.sharpLower',
							   'CTL_R_mouth_dimple.lipsPress', 'CTL_R_mouth_dimple.suckBlow',
							   'CTL_R_mouth_dimple.lipsBlow', 'CTL_R_mouth_dimple.sticky',
							   'CTL_R_mouth_cornerDepress.translateY', 'CTL_R_mouth_cornerDepress.funnel',
							   'CTL_R_mouth_cornerDepress.purse', 'CTL_R_mouth_cornerDepress.towards',
							   'CTL_R_mouth_cornerDepress.pushPull', 'CTL_R_mouth_cornerDepress.thickness',
							   'CTL_R_mouth_stretch.translateY', 'CTL_R_mouth_stretch.upLipFollow',
							   'CTL_R_mouth_stretch.mouthPress', 'CTL_R_mouth_stretch.lipsTighten',
							   'CTL_R_mouth_stretch.lipsTogether', 'CTL_R_mouth_lowerLipDepress.translateY',
							   'CTL_R_mouth_lowerLipDepress.chinRaise', 'CTL_L_mouth_cornerPull.translateY',
							   'CTL_L_mouth_cornerPull.funnel', 'CTL_L_mouth_cornerPull.purse',
							   'CTL_L_mouth_cornerPull.towards', 'CTL_L_mouth_cornerPull.open',
							   'CTL_L_mouth_cornerPull.pushPull', 'CTL_L_mouth_cornerPull.thickness',
							   'CTL_L_mouth_sharpCornerPull.translateY', 'CTL_L_mouth_sharpCornerPull.mouthPress',
							   'CTL_L_mouth_sharpCornerPull.lipsTighten', 'CTL_L_mouth_sharpCornerPull.lipsTogether',
							   'CTL_L_mouth_upperLipRaise.translateY', 'CTL_L_mouth_upperLipRaise.chinRaise',
							   'CTL_C_mouth.translateX', 'CTL_C_mouth.translateY', 'CTL_C_mouth.upperLipBiteL',
							   'CTL_C_mouth.upperLipBiteR', 'CTL_C_mouth.lowerLipBiteL', 'CTL_C_mouth.lowerLipBiteR',
							   'CTL_C_mouth.stickyCenter', 'CTL_C_mouth.stickyInnerLeft',
							   'CTL_C_mouth.stickyInnerRight', 'CTL_C_mouth.stickyOuterLeft',
							   'CTL_C_mouth.stickyOuterRight', 'CTL_L_nose.translateX', 'CTL_L_nose.translateY',
							   'CTL_L_nose.nasolabialDeepen', 'CTL_L_nose.nWrinkleTweak', 'CTL_R_nose.translateX',
							   'CTL_R_nose.translateY', 'CTL_R_nose.nasolabialDeepen', 'CTL_R_nose.nWrinkleTweak',
							   'CTL_C_jaw.translateX', 'CTL_C_jaw.translateY', 'CTL_C_jaw.clench',
							   'CTL_C_jaw.chinCompressL', 'CTL_C_jaw.chinCompressR', 'CTL_C_jaw.backFwd',
							   'CTL_C_jaw.throatDownUp', 'CTL_C_jaw.digastricDownUp', 'CTL_C_jaw.exhaleInhale',
							   'CTL_C_jaw.swallow', 'CTL_L_ear_up.translateY', 'CTL_R_ear_up.translateY']
		seen = set()
		seen_add = seen.add
		self.global_slider_names = [x.rsplit('.', 1)[0] for x in self.global_sliders]
		self.global_slider_names = [x for x in self.global_slider_names if not (x in seen or seen_add(x))]
		self.global_markers = ['lOuterLowerLip', 'rOuterLowerLip', 'lJaw', 'lDimple', 'rJaw', 'rDimple', 'UpperLip',
							   'rTemple', 'lTemple', 'rOuterEyeBrow', 'rEyeBrow', 'rInnerEyeBrow', 'lInnerEyeBrow',
							   'lEyeBrow', 'lOuterEyeBrow', 'Stab', 'rStab', 'lStab', 'rNose', 'rSneer', 'lNose',
							   'lSneer', 'rCheek', 'lCheek', 'UpperLipRoll', 'Chin', 'rCornerLip', 'rUpperLip',
							   'lUpperLip', 'lCornerLip', 'lLowerLip', 'rLowerLip', 'Jaw', 'rEyeDxyz',
							   'rLowerInnerLipDxyz', 'LowerInnerLipDxyz', 'lLowerInnerLipDxyz', 'lCornerInnerLipDxyz',
							   'lUpperInnerLipDxyz', 'UpperInnerLipDxyz', 'rUpperInnerLipDxyz', 'rCornerInnerLipDxyz',
							   'rOuterLowerLipDxyz', 'rInnerLowerLipDxyz', 'LowerLipDxyz', 'lInnerLowerLipDxyz',
							   'lOuterLowerLipDxyz', 'lCornerLipDxyz', 'lOuterUpperLipDxyz', 'lInnerUpperLipDxyz',
							   'UpperLipDxyz', 'rInnerUpperLipDxyz', 'rOuterUpperLipDxyz', 'rCornerLipDxyz',
							   'lInnerEyeSackDxyz', 'lOuterEyeSackDxyz', 'lOuterEyeDxyz', 'lOuterEyeLidDxyz',
							   'lInnerEyeLidDxyz', 'lInnerEyeDxyz', 'rOuterEyeSackDxyz', 'rInnerEyeSackDxyz',
							   'rInnerEyeDxyz', 'rInnerEyeLidDxyz', 'rOuterEyeLidDxyz', 'rOuterEyeDxyz', 'lEyeDxyz']
		self.errorDialog = QtGui.QErrorMessage(parent=self)
		self.createWidgets()
		self.createConnections()
		name = cmds.ls('*:CTL_R_brow_raiseOut')
		if len(name):
			split_name = name[0].split(':')
			self.namespaceEdit.setText(split_name[0] if len(split_name) > 1 else '')

	def closeHandler(self):
		ret = cmds.confirmDialog(title='Closing...', message='Do you wish to save before closing?',
								 button=['Save', 'Don\'t Save', 'Cancel'], defaultButton='Save', cancelButton='Cancel',
								 dismissString='Cancel')
		if ret == 'Cancel':
			return False
		elif ret == 'Save':
			save_state = self.parent().saveGroups(saveAs=True)
			if save_state:
				return True
			else:
				return False
		elif ret == 'Don\'t Save':
			return True

	def loadSettings(self, settingsData):
		print "load"
		self.settingsData = settingsData
		numItems = self.settingsCombo.count()
		for i in range(numItems):
			self.settingsCombo.removeItem(0)
		self.settingsCombo.addItems(self.settingsData.keys()[::-1])
		data = self.settingsData[self.settingsCombo.currentText()]
		self.setSettings(sliders=data['Sliders'], markers=data['Markers'])

	def setSettings(self, sliders, markers):
		self.global_sliders = sliders
		self.global_markers = markers
		seen = set()
		seen_add = seen.add
		self.global_slider_names = [x.rsplit('.', 1)[0] for x in self.global_sliders]
		self.global_slider_names = [x for x in self.global_slider_names if not (x in seen or seen_add(x))]

	def createWidgets(self):
		globalHLayout = QtGui.QHBoxLayout()
		self.surroundingScroll = QtGui.QScrollArea()
		self.imagePanel = QtGui.QLabel(self)
		self.imagePanel.setSizePolicy(QtGui.QSizePolicy.Ignored, QtGui.QSizePolicy.Ignored)
		self.imagePanel.setScaledContents(True)
		self.surroundingScroll.setWidget(self.imagePanel)
		globalHLayout.addWidget(self.surroundingScroll)
		vLayout = QtGui.QVBoxLayout()

		fLayout = QtGui.QFormLayout()

		self.settingsCombo = QtGui.QComboBox(self)
		fLayout.addRow('Current Config: ', self.settingsCombo)

		self.fileShortEdit = QtGui.QLineEdit(self)
		fLayout.addRow('File Short Name: ', self.fileShortEdit)

		self.namespaceEdit = QtGui.QLineEdit(self)
		fLayout.addRow('Namespace: ', self.namespaceEdit)

		self.setGroupEnable = QtGui.QCheckBox(self)
		fLayout.addRow('Enable: ', self.setGroupEnable)

		vLayout.addLayout(fLayout)

		groupLabel = QtGui.QLabel(self)
		groupLabel.setText('Groups:')
		self.groupList = QListWidget(parent=self, renameEnabled=True)
		self.addGroupButton = QtGui.QPushButton(self)
		self.addGroupButton.setText('Add Group')
		self.removeGroupButton = QtGui.QPushButton(self)
		self.removeGroupButton.setText('Remove Group')

		vLayout.addWidget(groupLabel)
		vLayout.addWidget(self.groupList)
		vLayout.addWidget(self.addGroupButton)
		vLayout.addWidget(self.removeGroupButton)

		poseLabel = QtGui.QLabel(self)
		poseLabel.setText('Poses:')
		self.poseList = QListWidget(parent=self, renameEnabled=True)
		self.addPoseButton = QtGui.QPushButton(self)
		self.addPoseButton.setText('Add Pose')
		# self.addPosesButton = QtGui.QPushButton(self)
		# self.addPosesButton.setText('Add Poses from list')
		self.updatePoseButton = QtGui.QPushButton(self)
		self.updatePoseButton.setText('Update Pose')
		self.updateMoCapButton = QtGui.QPushButton(self)
		self.updateMoCapButton.setText('Update MoCap')
		self.removePoseButton = QtGui.QPushButton(self)
		self.removePoseButton.setText('Remove Pose')

		hLayout = QtGui.QHBoxLayout()

		vL1 = QtGui.QVBoxLayout()
		markerLabel = QtGui.QLabel(self)
		markerLabel.setText('Markers')
		self.markerList = QListWidget(parent=self)
		self.addMarkerButton = QtGui.QPushButton(self)
		self.removeMarkerButton = QtGui.QPushButton(self)
		self.setMarkersButton = QtGui.QPushButton(self)
		self.addMarkerButton.setText('Add Marker')
		self.removeMarkerButton.setText('Remove Marker')
		self.setMarkersButton.setText('Set Markers')
		vL1.addWidget(markerLabel)
		vL1.addWidget(self.markerList)
		vL1.addWidget(self.addMarkerButton)
		vL1.addWidget(self.removeMarkerButton)
		vL1.addWidget(self.setMarkersButton)

		vL2 = QtGui.QVBoxLayout()
		sliderLabel = QtGui.QLabel(self)
		sliderLabel.setText('Sliders')
		self.sliderList = QListWidget(parent=self)
		self.addSliderButton = QtGui.QPushButton(self)
		self.removeSliderButton = QtGui.QPushButton(self)
		self.setSlidersButton = QtGui.QPushButton(self)
		self.addSliderButton.setText('Add Slider')
		self.removeSliderButton.setText('Remove Slider')
		self.setSlidersButton.setText('Set Sliders')
		vL2.addWidget(sliderLabel)
		vL2.addWidget(self.sliderList)
		vL2.addWidget(self.addSliderButton)
		vL2.addWidget(self.removeSliderButton)
		vL2.addWidget(self.setSlidersButton)

		hLayout.addLayout(vL1)
		hLayout.addLayout(vL2)
		vLayout.addLayout(hLayout)

		h_f_layout = QtGui.QHBoxLayout()
		f2Layout = QtGui.QFormLayout()
		self.holdPoseCheck = QtGui.QCheckBox(self)
		f2Layout.addRow('Hold Pose: ', self.holdPoseCheck)
		self.wholeEnable = QtGui.QCheckBox(self)
		self.wholeEnable.setChecked(True)
		f2Layout.addRow('Whole Pose: ', self.wholeEnable)

		f3Layout = QtGui.QFormLayout()
		self.markerEnable = QtGui.QCheckBox(self)
		self.markerEnable.setChecked(True)
		f3Layout.addRow('Show Markers: ', self.markerEnable)
		self.frameJumpEnable = QtGui.QCheckBox(self)
		self.frameJumpEnable.setChecked(True)
		f3Layout.addRow('Jump to frame: ', self.frameJumpEnable)

		h_f_layout.addLayout(f2Layout)
		h_f_layout.addLayout(f3Layout)
		vLayout.addLayout(h_f_layout)
		vLayout.addWidget(poseLabel)
		vLayout.addWidget(self.poseList)
		vLayout.addWidget(self.addPoseButton)
		# vLayout.addWidget(self.addPosesButton)
		h3 = QtGui.QHBoxLayout()
		h3.addWidget(self.updatePoseButton)
		h3.addWidget(self.updateMoCapButton)
		vLayout.addLayout(h3)
		vLayout.addWidget(self.removePoseButton)
		# self.resetPoseButton = QtGui.QPushButton(self)
		# self.resetPoseButton.setText('Reset Pose')
		# vLayout.addWidget(self.resetPoseButton)

		self.trainRBFNButton = QtGui.QPushButton(self)
		self.trainRBFNButton.setText('Train RBFN')
		self.updateSliderLinkButton = QtGui.QPushButton(self)
		self.updateSliderLinkButton.setText('Update Slider Link')
		self.removeSliderLinkButton = QtGui.QPushButton(self)
		self.removeSliderLinkButton.setText('Remove Slider Link')

		for btn in [self.trainRBFNButton, self.updateSliderLinkButton, self.removeSliderLinkButton]:
			vLayout.addWidget(btn)

		globalHLayout.addLayout(vLayout)
		self.setLayout(globalHLayout)

	def createConnections(self):
		self.groupList.item_selected.connect(self.groupSelectionChange)
		self.poseList.item_selected.connect(self.poseSelectionChange)
		self.poseList.focus_changed.connect(self.poseFocusChange)
		self.addGroupButton.pressed.connect(self.addGroup)
		self.removeGroupButton.pressed.connect(self.removeGroup)
		self.setMarkersButton.pressed.connect(self.setMarkers)
		self.setSlidersButton.pressed.connect(self.setSliders)
		self.addPoseButton.pressed.connect(self.addPose)
		# self.addPosesButton.pressed.connect(self.addPoses)
		self.poseList.item_checked.connect(self.togglePose)
		self.removePoseButton.pressed.connect(self.removePose)
		self.updatePoseButton.pressed.connect(self.updatePose)
		self.trainRBFNButton.pressed.connect(self.trainRBFN)
		self.updateSliderLinkButton.pressed.connect(self.setSliderLink)
		self.removeSliderLinkButton.pressed.connect(self.delSliderLink)
		self.setGroupEnable.stateChanged.connect(self.toggleGroupEnable)
		self.addMarkerButton.pressed.connect(self.addMarkers)
		self.removeMarkerButton.pressed.connect(self.removeMarker)
		self.addSliderButton.pressed.connect(self.addSliders)
		self.removeSliderButton.pressed.connect(self.removeSlider)
		self.poseList.item_renamed.connect(self.changePoseName)
		self.groupList.item_renamed.connect(self.changeGroupName)
		# self.resetPoseButton.pressed.connect(self.resetPose)
		self.updateMoCapButton.pressed.connect(functools.partial(self.updatePose, True))
		self.wholeEnable.stateChanged.connect(self.toggleWholeEnable)
		self.markerEnable.stateChanged.connect(self.toggleMarkerEnable)
		self.frameJumpEnable.stateChanged.connect(self.toggleFrameJump)
		self.settingsCombo.currentIndexChanged.connect(self.settingsChanged)

	def settingsChanged(self, index):
		config_name = self.settingsCombo.itemText(index)
		print "Setting Config: {}".format(config_name)
		config = self.settingsData[config_name]
		self.setSettings(sliders=config['Sliders'], markers=config['Markers'])

	def resetPose(self):
		cmds.currentTime(cmds.currentTime(q=1))

	def setGroups(self, groups, start=None, end=None):
		self.groups = {}
		self.groups = groups
		self.refresh()

	def createMesh(self, edge_list, marker_names):
		for ei in xrange(edge_list.shape[0]):
			edge_pair = edge_list[ei, :]
			marker_1 = marker_names[edge_pair[0]]
			marker_2 = marker_names[edge_pair[1]]
			curve = str(cmds.curve(d=1, p=[(0, 0, 0), (0, 0, 0)]))
			if 'Dxyz' in marker_1:
				cmds.parent(curve, 'Dxyz')
			else:
				cmds.parent(curve, 'face_markers')
			cmds.connectAttr(marker_1 + '.t', curve + '.cv[0]', f=True)
			cmds.connectAttr(marker_2 + '.t', curve + '.cv[1]', f=True)
			cmds.setAttr(curve + '.t', 0, 0, 0)
			cmds.setAttr(curve + '.r', 0, 0, 0)
			cmds.setAttr(curve + '.scale', 1, 1, 1)

	def showMesh(self, visible=True):
		try:
			cmds.delete('curve*')
		except ValueError:
			print "No RBFN Mesh Edges to delete"
		if not visible:
			return
		for group in self.groups.values():
			if isinstance(group, list) or not group.get('enabled', True): continue
			group['edge_list'], _ = self.createEdgeList(group)
			self.createMesh(group['edge_list'], group['marker_names'])

	def togglePose(self, pose_name, check_state):
		print "Toggle pose: {}".format(pose_name)
		group = self.groups[self.activeGroup]
		if 'disabled' not in group:
			group['disabled'] = []
		if check_state == True and pose_name in group['disabled']:
			group['disabled'].remove(pose_name)
		elif check_state == False and pose_name not in group['disabled']:
			group['disabled'].append(pose_name)

	def doResize(self):
		if self.pixmap is not None:
			self.imagePanel.setPixmap(self.pixmap.scaled(self.surroundingScroll.size()))
		self.imagePanel.adjustSize()

	def toggleImage(self):
		self.surroundingScroll.setVisible(not self.surroundingScroll.isVisible())

	def toggleFrameJump(self):
		self.frameJump = self.frameJumpEnable.isChecked()
		selection = self.poseList.getSelection()
		if selection == -1: return
		self.poseSelectionChange(selection)

	def refresh(self):
		self.setting = True
		self.groupList.clear()
		if '__group_order' in self.groups and len(self.groups['__group_order']):
			group_names = self.groups['__group_order']
		else:
			group_names = [x for x in self.groups.keys() if not isinstance(self.groups[x], list)]
		self.groupList.addItems(group_names)
		self.markerList.clear()
		self.sliderList.clear()
		self.setGroupEnable.setChecked(False)
		self.setting = False
		if len(self.groups.keys()):
			self.groupList.setUserSelection(0)
			self.setGroup(group_names[0])

	def bakeFrames(self):
		result = cmds.promptDialog(
			title='Face Retarget',
			message='Start Frame:',
			button=['OK', 'Cancel'],
			defaultButton='OK',
			cancelButton='Cancel',
			dismissString='Cancel')
		if result <> 'OK': return
		try:
			start_frame = int(cmds.promptDialog(query=True, text=True))
		except:
			print "Invalid Frame"
			return
		result = cmds.promptDialog(
			title='Face Retarget',
			message='End Frame:',
			button=['OK', 'Cancel'],
			defaultButton='OK',
			cancelButton='Cancel',
			dismissString='Cancel')
		if result <> 'OK': return
		try:
			end_frame = int(cmds.promptDialog(query=True, text=True))
		except:
			print "Invalid Frame"
			return
		self.delSliderLink()

		self.bakeRange(start_frame, end_frame)

	def refreshData(self):
		self.delSliderLink()
		gMainProgressBar = maya.mel.eval('$tmp = $gMainProgressBar');
		numPoses = sum(
			[len(group['slider_data'].keys()) for group in self.groups.values() if not isinstance(group, list)])
		print "Num Poses: {}".format(numPoses)
		if not numPoses: return
		cmds.progressBar(gMainProgressBar,
						 edit=True,
						 beginProgress=True,
						 isInterruptable=True,
						 status='Refreshing All Data ...',
						 maxValue=numPoses)

		for group in self.groups.values():
			if isinstance(group, list) or not group.get('enabled', True): continue
			for pose_name in group['slider_data'].keys():
				if cmds.progressBar(gMainProgressBar, query=True, isCancelled=True):
					break
				file, frame = pose_name.split('_')[0], pose_name.split('_')[-1]
				if file == self.fileShortEdit.text():
					cmds.currentTime(int(frame))
					try:
						marker_data, slider_data = self.getPoseData(moCap=False)
					except:
						continue
					group['marker_data'][pose_name] = marker_data
					group['slider_data'][pose_name] = slider_data
					if 'images' not in self.groups[self.activeGroup]:
						group['images'] = {}
					image_ret = self.getImage()
					if image_ret is not None:
						group['images'][pose_name] = image_ret
				cmds.progressBar(gMainProgressBar, edit=True, step=1)
		cmds.progressBar(gMainProgressBar, edit=True, endProgress=True)

	def refreshMoCap(self):
		self.delSliderLink()
		gMainProgressBar = maya.mel.eval('$tmp = $gMainProgressBar');
		numPoses = sum(
			[len(group['slider_data'].keys()) for group in self.groups.values() if not isinstance(group, list)])
		print "Num Poses: {}".format(numPoses)
		if not numPoses: return
		cmds.progressBar(gMainProgressBar,
						 edit=True,
						 beginProgress=True,
						 isInterruptable=True,
						 status='Refreshing MoCap Data ...',
						 maxValue=numPoses)

		for group in self.groups.values():
			if isinstance(group, list): continue
			for pose_name in group['slider_data'].keys():
				if cmds.progressBar(gMainProgressBar, query=True, isCancelled=True):
					break
				file, frame = pose_name.split('_')[0], pose_name.split('_')[-1]
				if file == self.fileShortEdit.text():
					cmds.currentTime(int(frame))
					try:
						marker_data, slider_data = self.getPoseData(moCap=True)
					except:
						continue
					group['marker_data'][pose_name] = marker_data
				cmds.progressBar(gMainProgressBar, edit=True, step=1)
		cmds.progressBar(gMainProgressBar, edit=True, endProgress=True)

	def refreshImages(self):
		gMainProgressBar = maya.mel.eval('$tmp = $gMainProgressBar');
		numPoses = sum(
			[len(group['slider_data'].keys()) for group in self.groups.values() if not isinstance(group, list)])
		print "Num Poses: {}".format(numPoses)
		if not numPoses: return
		cmds.progressBar(gMainProgressBar,
						 edit=True,
						 beginProgress=True,
						 isInterruptable=True,
						 status='Refreshing MoCap Data ...',
						 maxValue=numPoses)

		for group in self.groups.values():
			if isinstance(group, list): continue
			for pose_name in group['slider_data'].keys():
				if cmds.progressBar(gMainProgressBar, query=True, isCancelled=True):
					break
				file, frame = pose_name.split('_')[0], pose_name.split('_')[-1]
				if file == self.fileShortEdit.text():
					cmds.currentTime(int(frame))
					if 'images' not in self.groups[self.activeGroup]:
						group['images'] = {}
					image_ret = self.getImage()
					if image_ret is not None:
						group['images'][pose_name] = image_ret
				cmds.progressBar(gMainProgressBar, edit=True, step=1)
		cmds.progressBar(gMainProgressBar, edit=True, endProgress=True)

	@viewportOff
	def bakeRange(self, start, stop):
		gMainProgressBar = maya.mel.eval('$tmp = $gMainProgressBar');

		cmds.progressBar(gMainProgressBar,
						 edit=True,
						 beginProgress=True,
						 isInterruptable=True,
						 status='Baking Animation ...',
						 maxValue=(stop + 1 - start) * len(self.groups.keys()))
		try:
			marker_names = []
			slider_names = []
			slider_max, slider_min = [], []
			translateLimitFuncs = {
				'translateX': lambda slider: cmds.transformLimits(
					self.namespaceEdit.text() + ':' + slider.split('.', 1)[0], tx=True, q=True),
				'translateY': lambda slider: cmds.transformLimits(
					self.namespaceEdit.text() + ':' + slider.split('.', 1)[0], ty=True, q=True),
				'translateZ': lambda slider: cmds.transformLimits(
					self.namespaceEdit.text() + ':' + slider.split('.', 1)[0], tz=True, q=True)
			}
			for group_name, group in self.groups.items():
				if isinstance(group, list) or not group.get('enabled', True): continue
				marker_names.extend([x for x in group['marker_names'] if x not in marker_names])
				new_slider_names = [x for x in group['slider_names'] if x not in slider_names]
				slider_names.extend(new_slider_names)
				slider_max.extend([cmds.attributeQuery(slider.split('.', 1)[1],
													   node=self.namespaceEdit.text() + ':' + slider.split('.', 1)[0],
													   max=True)[0]
								   if cmds.attributeQuery(slider.split('.', 1)[1],
														  node=self.namespaceEdit.text() + ':' + slider.split('.', 1)[
															  0], maxExists=True) else
								   None if slider.split('.', 1)[1] not in translateLimitFuncs else
								   translateLimitFuncs[slider.split('.', 1)[1]](slider)[1]
								   for slider in new_slider_names])
				slider_min.extend([cmds.attributeQuery(slider.split('.', 1)[1],
													   node=self.namespaceEdit.text() + ':' + slider.split('.', 1)[0],
													   min=True)[0]
								   if cmds.attributeQuery(slider.split('.', 1)[1],
														  node=self.namespaceEdit.text() + ':' + slider.split('.', 1)[
															  0], minExists=True) else None if slider.split('.', 1)[
																								   1] not in translateLimitFuncs else
				translateLimitFuncs[slider.split('.', 1)[1]](slider)[0]
								   for slider in new_slider_names])
			for frame in xrange(start, stop + 1):
				for group in self.groups.values():
					if cmds.progressBar(gMainProgressBar, query=True, isCancelled=True):
						break
					cmds.progressBar(gMainProgressBar, edit=True, step=1)
					if isinstance(group, list) or not group.get('enabled', True): continue
					if group['weights'] is None: continue
					RBFN = lambda x: evaluateRBFN(group['weights'], group['centres'], group['betas'], x)
					cur_markers = np.zeros((1, len(group['marker_names']) * 3), dtype=np.float32)
					for mi, myMarker in enumerate(group['marker_names']):
						cur_markers[0, (mi) * 3:(mi + 1) * 3] = cmds.getAttr(myMarker + '.translate', t=frame)[0]
					marker_pos = cur_markers.reshape(-1, 3)
					indices = group['edge_list']
					data = np.linalg.norm(marker_pos[indices[:, 0]] - marker_pos[indices[:, 1]], axis=1).reshape(1, -1)
					slider_vals, _ = RBFN(data)
					for si, slider in enumerate(group['slider_names']):
						slider_name, slider_attr = slider.split('.', 1)
						value = float(slider_vals[0, si])
						slider_index = slider_names.index(slider)
						min, max = slider_min[slider_index], slider_max[slider_index]
						if min is not None and value < min:
							value = min
						elif max is not None and value > max:
							value = max
						cmds.setKeyframe(self.namespaceEdit.text() + ':' + slider_name, attribute=slider_attr, v=value,
										 t=frame)
		except:
			self.errorDialog.showMessage(
				'Unable to set KeyFrames, perhaps you have a node locking the sliders or an expression running?')
		cmds.progressBar(gMainProgressBar, edit=True, endProgress=True)

	def addGroup(self):
		result = cmds.promptDialog(
			title='Face Retarget',
			message='Enter the name of the group:',
			button=['OK', 'Cancel'],
			defaultButton='OK',
			cancelButton='Cancel',
			dismissString='Cancel')

		if result == 'OK':
			group_name = cmds.promptDialog(query=True, text=True)
			if group_name not in self.groups:
				self.groups[group_name] = self.newGroup()
				self.groupList.addItem(group_name)
				self.groupList.setUserSelection(self.groupList.item_count - 1)
				self.setGroup(group_name)

	def toggleMarkerEnable(self):
		self.showMarkers = self.markerEnable.isChecked()
		selection = self.poseList.getSelection()
		if selection == -1: return
		self.poseSelectionChange(selection)

	def toggleWholeEnable(self):
		self.whole = self.wholeEnable.isChecked()
		selection = self.poseList.getSelection()
		if selection == -1: return
		self.poseSelectionChange(selection)

	def toggleGroupEnable(self):
		if self.activeGroup is None: return
		self.groups[self.activeGroup]['enabled'] = self.setGroupEnable.isChecked()
		if self.expression is not None:
			self.setSliderLink()
		# if self.expression is not None: self.setSliderLink()
		# if not self.setGroupEnable.isChecked(): cmds.currentTime(cmds.currentTime(q=True))

	def setGroupPoseOrder(self):
		self.groups['__group_order'] = self.groupList.getItems()
		if self.activeGroup is not None and self.activeGroup in self.groups:
			self.groups[self.activeGroup]['pose_order'] = self.poseList.getItems()
			self.groups[self.activeGroup]['marker_order'] = self.markerList.getItems()
			self.groups[self.activeGroup]['slider_order'] = self.sliderList.getItems()

	def newGroup(self):
		return {'marker_names': [], 'slider_names': [], 'key_poses': [], 'training_data': [], 'weights': None,
				'centres': None, 'enabled': True, 'marker_data': {}, 'slider_data': {}}

	def setGroup(self, group_name):
		# print "Setting group {}".format(group_name)
		if group_name not in self.groups: return
		self.setting = True
		self.setGroupPoseOrder()
		self.activeGroup = group_name
		self.markerList.clear()
		if 'marker_order' in self.groups[group_name] and len(self.groups[group_name]['marker_order']):
			marker_names = self.groups[group_name]['marker_order']
		else:
			marker_names = self.groups[group_name]['marker_names']
		self.markerList.addItems(marker_names)
		self.sliderList.clear()
		if 'slider_order' in self.groups[group_name] and len(self.groups[group_name]['slider_order']):
			slider_names = self.groups[group_name]['slider_order']
		else:
			slider_names = self.groups[group_name]['slider_names']
		self.sliderList.addItems(slider_names)
		self.poseList.clear()
		if 'disabled' not in self.groups[group_name]:
			self.groups[group_name]['disabled'] = []
		if 'pose_order' in self.groups[group_name] and len(self.groups[group_name]['pose_order']):
			if np.all([x in self.groups[group_name]['marker_data'].keys() for x in
					   self.groups[group_name]['pose_order']]):
				pose_names = self.groups[group_name]['pose_order']
			else:
				self.groups[group_name]['pose_order'] = []
				pose_names = self.groups[group_name]['marker_data'].keys()
		else:
			pose_names = self.groups[group_name]['marker_data'].keys()
		checked = [True if pose_name not in self.groups[group_name]['disabled'] else False for pose_name in pose_names]
		self.poseList.addItems(pose_names, checked=checked)
		self.setGroupEnable.setChecked(self.groups[group_name].get('enabled', True))
		self.imagePanel.setPixmap(QtGui.QPixmap())
		self.imagePanel.adjustSize()
		self.setting = False

	def removeGroup(self):
		selection = self.groupList.getSelection()
		if selection == -1: return
		group_name = self.groupList.getItem(selection)
		self.setting = True
		self.groups.pop(group_name)
		# print "Removing: {}".format(group_name)
		self.groupList.removeItem(selection)
		self.setting = False
		selection = self.groupList.getSelection()
		if selection == -1: return
		self.groupSelectionChange(self.groupList.getSelection())

	def poseSelectionChange(self, index):
		if self.setting or self.expression is not None: return
		pose_name = self.poseList.getItem(index)
		print "Selected Pose: {}".format(pose_name)
		group = self.groups[self.activeGroup]
		if not self.holdPoseCheck.isChecked():
			if self.frameJump and pose_name.split('_')[0] == self.fileShortEdit.text():
				cmds.currentTime(int(pose_name.split('_')[-1]))
			if self.showMarkers:
				self.setMarkerValues(self.global_markers, group['marker_data'][pose_name])
			if self.whole:
				self.setSliderValues(self.global_sliders, group['slider_data'][pose_name])
			else:
				slider_indices = [self.global_sliders.index(x) for x in group['slider_names']]
				non_slider_names = [self.global_sliders[x] for x in xrange(len(self.global_sliders)) if
									x not in slider_indices]
				self.setSliderValues(group['slider_names'], group['slider_data'][pose_name][slider_indices])
				self.setSliderValues(non_slider_names, np.zeros(len(non_slider_names), dtype=np.float32))
		self.pixmap = QtGui.QPixmap()
		try:
			myImage = group['images'][pose_name]
			self.pixmap.loadFromData(myImage)
		except:
			pass
		self.imagePanel.setPixmap(self.pixmap.scaled(self.surroundingScroll.size()))
		self.imagePanel.adjustSize()

	def changePoseName(self, poseFrom, poseTo):
		print "Pose Change from: {} to: {}".format(poseFrom, poseTo)
		group = self.groups[self.activeGroup]
		group['slider_data'][poseTo] = group['slider_data'].pop(poseFrom)
		group['marker_data'][poseTo] = group['marker_data'].pop(poseFrom)
		if 'images' in group:
			group['images'][poseTo] = group['images'].pop(poseFrom)

	def changeGroupName(self, groupFrom, groupTo):
		# print "Group Change from: {} to: {}".format(groupFrom, groupTo)
		self.groups[groupTo] = self.groups.pop(groupFrom)
		self.activeGroup = groupTo

	@staticmethod
	def setMarkerValues(marker_names, marker_data):
		for mi, marker_name in enumerate(marker_names):
			for ai, axis in enumerate(['X', 'Y', 'Z']):
				cmds.setAttr(marker_name + '.translate' + axis, float(marker_data[mi * 3 + ai]), clamp=True)

	def setSliderValues(self, slider_names, slider_data):
		for si, slider_name in enumerate(slider_names):
			cmds.setAttr(self.namespaceEdit.text() + ':' + slider_name, float(slider_data[si]), clamp=True)

	def groupSelectionChange(self, index):
		if self.setting: return
		# print "{} Selected!".format(index)
		group = self.groupList.getItem(index)
		self.setGroup(group)

	def poseFocusChange(self, hasFocus):
		pass

	def getDxyzEdges(self, marker_pos, dxyz_indices, marker_names):
		print marker_names
		print 'lEyeDxyz' in marker_names
		if 'lInnerEyeDxyz' in marker_names:
			left_indices = [x for mi, x in enumerate(dxyz_indices) if 'l' == marker_names[x][0]]
			right_indices = [x for x in dxyz_indices if x not in left_indices]
			left_map = [dxyz_indices[i] for i in left_indices]
			right_map = [dxyz_indices[i] for i in right_indices]
			if len(left_indices):
				if len(right_indices):
					left_edges = pointsToEdges(marker_pos[left_indices, :], left_map)
					right_edges = pointsToEdges(marker_pos[right_indices, :], right_map)
					edge_list = np.vstack((left_edges, right_edges))
				else:
					edge_list = pointsToEdges(marker_pos[left_indices, :], left_indices)
			else:
				edge_list = pointsToEdges(marker_pos[right_indices, :], right_map)
			return edge_list
		else:
			return pointsToEdges(marker_pos[dxyz_indices, :], dxyz_indices)

	def createEdgeList(self, group):
		# print group.keys()
		active_poses = [x for x in group['marker_data'].keys() if x not in group.get('disabled', [])]
		marker_indices = []
		global_indices = [self.global_markers.index(x) for x in group['marker_names']]
		dxyz_indices = [mi for mi, i in enumerate(global_indices) if 'Dxyz' in self.global_markers[i]]
		cara_indices = [i for i in range(len(global_indices)) if i not in dxyz_indices]
		for i in global_indices:
			marker_indices.extend([3 * i, 3 * i + 1, 3 * i + 2])
		base_pose = active_poses[0]
		marker_pos = group['marker_data'][base_pose][marker_indices].reshape(-1, 3)
		marker_dimensions = np.argsort(np.var(marker_pos, axis=0))[1:]
		marker_pos = marker_pos[:, marker_dimensions]
		if len(cara_indices):
			if len(dxyz_indices):
				cara_edges = pointsToEdges(marker_pos[cara_indices, :], cara_indices)
				dxyz_edges = self.getDxyzEdges(marker_pos, dxyz_indices, group['marker_names'])
				edge_list = np.vstack((cara_edges, dxyz_edges))
			else:
				edge_list = pointsToEdges(marker_pos[cara_indices, :], cara_indices)
		else:
			edge_list = self.getDxyzEdges(marker_pos, dxyz_indices, group['marker_names'])
		return edge_list, marker_indices

	def getDataForGroup(self, group):
		# print group.get('disabled',[])
		active_poses = [x for x in group['marker_data'].keys() if x not in group.get('disabled', [])]
		group['edge_list'], marker_indices = self.createEdgeList(group)
		marker_data = np.zeros((len(active_poses), group['edge_list'].shape[0]), dtype=np.float32)
		slider_data = np.zeros((len(active_poses), len(group['slider_names'])), dtype=np.float32)
		for pi, pose_name in enumerate(active_poses):
			marker_pos = group['marker_data'][pose_name][marker_indices].reshape(-1, 3)
			data = np.linalg.norm(marker_pos[group['edge_list'][:, 0]] - marker_pos[group['edge_list'][:, 1]], axis=1)
			slider_indices = [self.global_sliders.index(x) for x in group['slider_names']]
			marker_data[pi, :] = data
			slider_data[pi, :] = group['slider_data'][pose_name][slider_indices]
		return marker_data, slider_data

	def trainRBFN(self):
		if self.expression is not None:
			try:
				cmds.delete(self.expression)
			except:
				pass
		# try:
		for group_name in self.groups:
			print "Getting data for group: {}".format(group_name)
			group = self.groups[group_name]
			if isinstance(group, list) or not len(group['marker_data'].keys()) or not group['enabled']:
				print "Ignoring Group"
				continue
			marker_data, slider_data = self.getDataForGroup(group)
			# print "{}: {}".format(group_name, marker_data.shape)
			try:
				group['weights'], group['centres'], group['betas'] = trainRBFN(marker_data, slider_data)
			except np.linalg.linalg.LinAlgError:
				msgBox = QtGui.QMessageBox()
				msgBox.setText("The following group failed to train and has been disabled: {}".format(group_name))
				msgBox.exec_()
				group['enabled'] = False
				self.setGroupEnable.setChecked(False)
			# except:
			# import sys
			# self.errorDialog.showMessage("Unexpected error: {}".format(sys.exc_info()[0]))

	def setFromOffset(self):
		marker_data, _ = self.getPoseData(moCap=True)
		self.groups['__From_Offset'] = [marker_data]
		print "Setting Base Offset"

	def setToOffset(self):
		marker_data, _ = self.getPoseData(moCap=True)
		self.groups['__To_Offset'] = [marker_data]
		print "Setting Current Offset"

	def delSliderLink(self, update=True):
		if self.expression is None: return
		try:
			cmds.delete(self.expression)
		except:
			self.expression = None
			pass
		self.expression = None
		slider_old_vals = {}
		if update:
			for slider_name in self.global_sliders:
				slider_old_vals[self.namespaceEdit.text() + ':' + slider_name] = cmds.getAttr(
					self.namespaceEdit.text() + ':' + slider_name)
		for slider in self.global_slider_names:
			restoreSlider(self.namespaceEdit.text() + ':' + slider)
		if update:
			for slider_name, value in slider_old_vals.items():
				cmds.setAttr(slider_name, value)
			# if update and not self.holdPoseCheck.isChecked(): cmds.currentTime(cmds.currentTime(q=True))

	def setSliderLink(self):
		self.delSliderLink(update=False)
		__main__.groups = copy.deepcopy(self.groups)
		if '__From_Offset' in self.groups and '__To_Offset' in self.groups:
			offsets = (self.groups['__From_Offset'][0] - self.groups['__To_Offset'][0]).reshape(-1, 3)
			inds = [i for i in range(len(self.global_markers)) if 'Dxyz' in self.global_markers[i]]
			# print "Setting the following to zero: {}".format([self.global_markers[i] for i in inds])
			offsets[inds, :] = 0
			__main__.offsets = offsets
		else:
			__main__.offsets = np.zeros((len(self.global_markers), 3), dtype=np.float32)
		marker_names = []
		slider_names = []
		slider_max, slider_min = [], []
		translateLimitFuncs = {
			'translateX': lambda slider: cmds.transformLimits(self.namespaceEdit.text() + ':' + slider.split('.', 1)[0],
															  tx=True, q=True),
			'translateY': lambda slider: cmds.transformLimits(self.namespaceEdit.text() + ':' + slider.split('.', 1)[0],
															  ty=True, q=True),
			'translateZ': lambda slider: cmds.transformLimits(self.namespaceEdit.text() + ':' + slider.split('.', 1)[0],
															  tz=True, q=True)
		}
		for group_name, group in self.groups.items():
			if isinstance(group, list) or not group.get('enabled', True): continue
			marker_names.extend([x for x in group['marker_names'] if x not in marker_names])
			new_slider_names = [x for x in group['slider_names'] if x not in slider_names]
			slider_names.extend(new_slider_names)
			slider_max.extend([cmds.attributeQuery(slider.split('.', 1)[1],
												   node=self.namespaceEdit.text() + ':' + slider.split('.', 1)[0],
												   max=True)[0]
							   if cmds.attributeQuery(slider.split('.', 1)[1],
													  node=self.namespaceEdit.text() + ':' + slider.split('.', 1)[0],
													  maxExists=True) else
							   None if slider.split('.', 1)[1] not in translateLimitFuncs else
							   translateLimitFuncs[slider.split('.', 1)[1]](slider)[1]
							   for slider in new_slider_names])
			slider_min.extend([cmds.attributeQuery(slider.split('.', 1)[1],
												   node=self.namespaceEdit.text() + ':' + slider.split('.', 1)[0],
												   min=True)[0]
							   if cmds.attributeQuery(slider.split('.', 1)[1],
													  node=self.namespaceEdit.text() + ':' + slider.split('.', 1)[0],
													  minExists=True) else None if slider.split('.', 1)[
																					   1] not in translateLimitFuncs else
			translateLimitFuncs[slider.split('.', 1)[1]](slider)[0]
							   for slider in new_slider_names])
		for slider in self.global_slider_names:
			duplicateSlider(self.namespaceEdit.text() + ':' + slider)
		__main__.names = copy.deepcopy([marker_names, slider_names, slider_max, slider_min])
		expressionString = 'global float $marker_data[{}];\n'.format(len(marker_names) * 3)
		for mi, marker_name in enumerate(marker_names):
			for ai, axis in enumerate(['X', 'Y', 'Z']):
				expressionString += '$marker_data[{}] = {}.translate{};\n'.format(3 * mi + ai, marker_name, axis)

		expressionString += '\nfloat $slider_data[] = `python(\"import maya.mel as mel; marker_data = mel.eval(\\\"float $temp_marker[] = $marker_data;\\\"); expressionFunction(marker_data)\")`;\n\n'

		for si, slider_name in enumerate(slider_names):
			expressionString += '{} = $slider_data[{}];\n'.format(self.namespaceEdit.text() + ':' + slider_name, si)

		self.expression = cmds.expression(n='RBFN Expression', s=expressionString)

	def getPoseData(self, frame=None, moCap=False):
		marker_data = np.zeros(len(self.global_markers) * 3, dtype=np.float32)
		slider_data = np.zeros(len(self.global_sliders), dtype=np.float32) if not moCap else None
		try:
			for mi, marker_name in enumerate(self.global_markers):
				if frame is not None:
					print "none None frame!"
					marker_datum = cmds.getAttr(marker_name + '.translate', t=frame)[0]
				else:
					marker_datum = cmds.getAttr(marker_name + '.translate')[0]
				marker_data[mi * 3:(mi + 1) * 3] = cmds.getAttr(marker_name + '.translate')[0]
			if not moCap:
				for si, slider_name in enumerate(self.global_sliders):
					if frame is not None:
						print "none None frame!"
						slider_datum = cmds.getAttr(self.namespaceEdit.text() + ':' + slider_name, t=frame)
					else:
						slider_datum = cmds.getAttr(self.namespaceEdit.text() + ':' + slider_name)
					slider_data[si] = slider_datum
		except ValueError as detail:
			self.errorDialog.showMessage("\n".join(detail))
			return
		return marker_data, slider_data

	def addPose(self):
		if self.activeGroup is None: return
		if not self.markerList.item_count or not self.sliderList.item_count:
			self.errorDialog.showMessage('Please add Sliders and Markers before you add a pose')
			return
		result = cmds.promptDialog(
			title='Face Retarget',
			message='Enter the name of the group:',
			button=['OK', 'Cancel'],
			defaultButton='OK',
			cancelButton='Cancel',
			dismissString='Cancel')

		if result == 'OK':
			pose_name = str(cmds.promptDialog(query=True, text=True))
			pose_name = '_'.join((self.fileShortEdit.text(), pose_name, str(int(cmds.currentTime(q=True)))))
			if pose_name not in self.groups[self.activeGroup]['marker_data']:
				try:
					marker_data, slider_data = self.getPoseData()
				except:
					return
				if 'images' not in self.groups[self.activeGroup]:
					self.groups[self.activeGroup]['images'] = {}
				image_ret = self.getImage()
				if image_ret is not None:
					self.groups[self.activeGroup]['images'][pose_name] = self.getImage()
				self.groups[self.activeGroup]['marker_data'][pose_name] = marker_data
				self.groups[self.activeGroup]['slider_data'][pose_name] = slider_data
				self.poseList.addItem(pose_name, checked=True)
				self.poseList.setUserSelection(self.poseList.item_count - 1)
				self.poseSelectionChange(self.poseList.getSelection())

	def removePose(self):
		if self.activeGroup is None: return
		selection = self.poseList.getSelection()
		if selection == -1: return
		self.setting = True
		self.groups[self.activeGroup]['marker_data'].pop(self.poseList.getItem(selection))
		self.groups[self.activeGroup]['slider_data'].pop(self.poseList.getItem(selection))
		self.poseList.removeItem(selection)
		self.setting = False
		self.poseSelectionChange(self.poseList.getSelection())

	def addPoses(self):
		if self.activeGroup is None: return
		if not self.markerList.item_count or not self.sliderList.item_count:
			self.errorDialog.showMessage('Please add Sliders and Markers before you add a pose')
			return
		result = cmds.promptDialog(
			title='Face Retarget',
			message='Enter the poses separated by commas:',
			button=['OK', 'Cancel'],
			defaultButton='OK',
			cancelButton='Cancel',
			dismissString='Cancel')

		if result == 'OK':
			pose_list = cmds.promptDialog(query=True, text=True)
			pose_list = map(int, pose_list.split(','))
			for pose_time in pose_list:
				pose_name = "Frame {}".format(pose_time)
				try:
					marker_data, slider_data = self.getPoseData(frame=pose_time)
				except:
					return
				self.groups[self.activeGroup]['marker_data'][pose_name] = marker_data
				self.groups[self.activeGroup]['slider_data'][pose_name] = slider_data
				self.poseList.addItem(pose_name)

	def updatePose(self, moCap=False):
		selection = self.poseList.getSelection()
		if selection == -1: return
		pose_name = self.poseList.getItem(selection)
		try:
			marker_data, slider_data = self.getPoseData(moCap=moCap)
		except:
			return
		if not moCap:
			self.groups[self.activeGroup]['slider_data'][pose_name] = slider_data
		else:
			self.groups[self.activeGroup]['marker_data'][pose_name] = marker_data
			if 'images' not in self.groups[self.activeGroup]:
				self.groups[self.activeGroup]['images'] = {}
			image_ret = self.getImage()
			if image_ret is not None:
				self.groups[self.activeGroup]['images'][pose_name] = image_ret
			pose_core = pose_name.split('_')[1]
			new_frame = str(int(cmds.currentTime(q=True)))
			self.poseList.item_list_model.item(selection).setData(
				'_'.join((self.fileShortEdit.text(), pose_core, new_frame)), 0)
		# self.changePoseName(pose_name, '_'.join((self.fileShortEdit.text(), pose_core, new_frame)))
		self.poseSelectionChange(selection)

	def renamePose(self, poseFrom, poseTo):
		self.groups[self.activeGroup]['marker_data'][poseTo] = self.groups[self.activeGroup]['marker_data'].pop(poseTo)
		self.groups[self.activeGroup]['slider_data'][poseTo] = self.groups[self.activeGroup]['slider_data'].pop(poseTo)
		try:
			ind = self.groups[self.activeGroup]['pose_order'].index(poseFrom)
			self.groups[self.activeGroup]['pose_order'][ind] = poseTo
		except ValueError or KeyError:
			pass

	def clearPoses(self, group_name):
		group = self.groups[group_name]
		self.poseList.clear()
		group['slider_data'] = {}
		group['marker_data'] = {}
		self.imagePanel.setPixmap(QtGui.QPixmap())
		self.pixmap = None
		self.imagePanel.adjustSize()

	def addSliders(self):
		if self.activeGroup is None: return
		# if self.poseList.item_count:
		# msgBox = QtGui.QMessageBox()
		# msgBox.setText('Adding sliders will invalidate your pose list. Continue?')
		# msgBox.setStandardButtons(QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
		# ret = msgBox.exec_()
		# print ret
		# if ret == QtGui.QMessageBox.No:
		# return
		# else:
		# self.clearPoses(self.activeGroup)

		new_attrs = [x for x in getSliderAttributes() if x in self.global_sliders]
		cur_sliders = []
		for group_name in self.groups:
			if not isinstance(self.groups[group_name], list):
				cur_sliders.extend(self.groups[group_name]['slider_names'])
		error_sliders = [x for x in new_attrs if x in cur_sliders]
		new_attrs = [x for x in new_attrs if x not in cur_sliders]
		self.groups[self.activeGroup]['slider_names'].extend(new_attrs)
		self.sliderList.addItems(new_attrs)
		if len(new_attrs):
			print "Added the following Markers: {}".format(', '.join(new_attrs))
		if len(error_sliders):
			msgBox = QtGui.QMessageBox()
			msgBox.setText(
				"The following sliders are already in use and have been ignored:\n{}".format(", ".join(error_sliders)))
			msgBox.exec_()

	def removeSlider(self):
		if self.activeGroup is None: return
		selection = self.sliderList.getSelection()
		if selection == -1: return
		self.groups[self.activeGroup]['slider_names'].remove(self.sliderList.getItem(selection))
		self.sliderList.removeItem(selection)

	def setSliders(self):
		if self.activeGroup is None: return
		# if self.poseList.item_count:
		# msgBox = QtGui.QMessageBox()
		# msgBox.setText('Setting new sliders will invalidate your pose list. Continue?')
		# msgBox.setStandardButtons(QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
		# ret = msgBox.exec_()
		# print ret
		# if ret == QtGui.QMessageBox.No:
		# return
		# else:
		# self.clearPoses(self.activeGroup)
		new_sliders = getSliderAttributes()
		cur_sliders = []
		for group_name in self.groups:
			if group_name <> self.activeGroup:
				cur_sliders.extend(self.groups[group_name]['slider_names'])
		error_sliders = [x for x in new_sliders if x in cur_sliders]
		new_sliders = [x for x in new_sliders if x not in cur_sliders]
		self.groups[self.activeGroup]['slider_names'] = new_sliders
		self.sliderList.clear()
		self.sliderList.addItems(self.groups[self.activeGroup]['slider_names'])
		if len(error_sliders):
			self.errorDialog.showMessage(
				"The following sliders are already in use: {}".format(", ".join(error_sliders)))

	def addMarkers(self):
		if self.activeGroup is None: return
		# if self.poseList.item_count:
		# msgBox = QtGui.QMessageBox()
		# msgBox.setText('Adding markers will invalidate your pose list. Continue?')
		# msgBox.setStandardButtons(QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
		# ret = msgBox.exec_()
		# print ret
		# if ret == QtGui.QMessageBox.No:
		# return
		# else:
		# self.clearPoses(self.activeGroup)
		try:
			cmds.select('*Sphere', d=1)
			cmds.select('*Shape', d=1)
		except:
			pass
		selection = map(str, cmds.ls(sl=1))
		stripNamespace = lambda x: x.split(':', 1)[-1]
		selection = map(stripNamespace, selection)
		selection = [x for x in selection if x not in self.groups[self.activeGroup]['marker_names']]
		self.groups[self.activeGroup]['marker_names'].extend(selection)
		self.markerList.addItems(selection)
		if len(selection):
			print "Added the following Markers: {}".format(', '.join(selection))

	def removeMarker(self):
		if self.activeGroup is None: return
		selection = self.markerList.getSelection()
		if selection == -1: return
		self.groups[self.activeGroup]['marker_names'].remove(self.markerList.getItem(selection))
		self.markerList.removeItem(selection)

	def setMarkers(self):
		if self.activeGroup is None: return
		# if self.poseList.item_count:
		# msgBox = QtGui.QMessageBox()
		# msgBox.setText('Setting markers will invalidate your pose list. Continue?')
		# msgBox.setStandardButtons(QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
		# ret = msgBox.exec_()
		# print ret
		# if ret == QtGui.QMessageBox.No:
		# return
		# else:
		# self.clearPoses(self.activeGroup)
		try:
			cmds.select('*Sphere', d=1)
			cmds.select('*Shape', d=1)
		except ValueError:
			pass
		selection = map(str, cmds.ls(sl=1))
		stripNamespace = lambda x: x.split(':', 1)[-1]
		selection = map(stripNamespace, selection)
		self.groups[self.activeGroup]['marker_names'] = selection
		self.markerList.clear()
		self.markerList.addItems(self.groups[self.activeGroup]['marker_names'])

	def getImage(self):
		imPlane = findImagePlane()
		if imPlane is None:
			print "No image plane found"
			return None
		else:
			return getImageData(imPlane)

	def importPoses(self, groups):
		for group_name, group in groups.items():
			print group_name
			if isinstance(group, list): continue
			if group_name in self.groups:
				for pose_name in group['slider_data'].keys():
					if pose_name not in self.groups[group_name]['slider_data'].keys():
						self.groups[group_name]['slider_data'][pose_name] = group['slider_data'][pose_name]
						self.groups[group_name]['marker_data'][pose_name] = group['marker_data'][pose_name]
						self.groups[group_name]['images'][pose_name] = group['images'][pose_name]
						print "ADDED POSE: {}".format(pose_name)
					else:
						print "Pose, {}, already exists. Ignored.".format(pose_name)
			else:
				print "No group, {}, exists. Ignored".format(group_name)


def getImageData(ImagePlane, Frame=None):
	if Frame is None:
		Frame = cmds.getAttr(ImagePlane + '.frameExtension')
	Frame += cmds.getAttr(ImagePlane + '.frameOffset')
	base_img = cmds.getAttr(ImagePlane + '.imageName')
	extensionless_img, extension = base_img.rsplit('.', 1)
	numNums = 0
	while True:
		try:
			int(extensionless_img[-(numNums + 1)])
		except:
			break
		numNums += 1
	frameNo = str(int(Frame))
	paddingLen = numNums - len(frameNo)
	padding = '0' * paddingLen
	myImage = extensionless_img[:-numNums] + padding + frameNo + '.' + extension
	try:
		data = open(myImage, "rb").read()
	except:
		data = None
	return data


def findImagePlane():
	try:
		vals = map(str, cmds.ls('*_IP*')) + map(str, cmds.ls('*imagePlane*'))
		for val in vals:
			try:
				cmds.getAttr(val + '.frameExtension')
				return val
			except ValueError:
				continue
			return None
	except:
		return None


def expressionFunction(marker_data):
	import __main__, itertools
	# from time import time
	try:
		groups = __main__.groups
		names = __main__.names
		marker_offsets = __main__.offsets
	except AttributeError:
		return
	marker_names, slider_names, slider_max, slider_min = names
	slider_data = [0.0] * len(slider_names)
	for group_name in groups:
		group = groups[group_name]
		if isinstance(group, list) or not group.get('enabled', True): continue
		if group['weights'] is None: continue
		RBFN = lambda x: evaluateRBFN(group['weights'], group['centres'], group['betas'], x)
		# if cmds.objExists(group_name + 'Error')==False:
		# cmds.polyCube(h=1, n=group_name + 'Error')[1]
		# error_cube = group_name + 'Error'
		cur_markers = np.zeros((len(group['marker_names']), 3), dtype=np.float32)
		cur_marker_offsets = np.zeros_like(cur_markers)
		for mi, myMarker in enumerate(group['marker_names']):
			marker_ind = marker_names.index(myMarker)
			cur_markers[mi, :] = marker_data[3 * marker_ind:3 * (marker_ind + 1)]
			cur_marker_offsets[mi, :] = marker_offsets[marker_ind, :]
		# marker_pos = cur_markers.reshape(-1,3)
		indices = group['edge_list']
		# + cur_marker_offsets[indices[:,0]]
		data = np.linalg.norm(cur_markers[indices[:, 0]] - cur_markers[indices[:, 1]], axis=1).reshape(1, -1)

		# resid_error = np.min(np.linalg.norm(cur_markers - group['centres'], axis = 1))
		# cmds.setAttr( error_cube + '.scaleY', float(resid_error) )
		# t1 = time()
		slider_vals, contribution = RBFN(data)
		# print "RBFN Evaluation took: {}s".format(time() - t1)
		if 0:
			print "\n{}:\n".format(group_name)
			order = np.argsort(contribution)[::-1]
			pose_names = [x for x in group['marker_data'].keys() if x not in group.get('disabled', [])]
			for pi in order:
				print "{0}: {1:2.2f}%".format(pose_names[pi], 100 * contribution[pi])
		# print slider_vals
		for si, slider in enumerate(group['slider_names']):
			slider_name, slider_attr = slider.split('.', 1)
			value = float(slider_vals[0, si])
			slider_index = slider_names.index(slider)
			min, max = slider_min[slider_index], slider_max[slider_index]
			if min is not None and value < min:
				value = min
			elif max is not None and value > max:
				value = max
			slider_data[slider_index] = value
		# cmds.setAttr( slider, value, clamp=True )
	return slider_data


def loadSettings(filename):
	with open(filename, 'rb') as f:
		data = json.load(f)
	return data


def main():
	global g_folder_path
	g_folder_path = os.path.dirname(os.path.realpath(__file__))
	print g_folder_path
	settings_files = glob(os.path.join(g_folder_path, '*.RBF_Settings'))
	print settings_files
	settings_data = None
	for settings_file in settings_files:
		settings_data = loadSettings(settings_file)
		for setting_name in settings_data:
			print setting_name
	np.set_printoptions(threshold=np.nan)
	__main__.expressionFunction = expressionFunction
	myWindow = FaceRBFMainWindow()
	if settings_data is not None:
		myWindow.setSettings(settings_data)
	myWindow.show()


if __name__ == '__main__':
	main()