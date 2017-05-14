import functools
import os
from uuid import uuid4

from PySide import QtGui, QtCore
from PySide.QtCore import QModelIndex, Qt
from PySide.QtGui import QVBoxLayout, QItemSelectionModel
import numpy as np

from UI import QOutliner, createAction, errorDialog
from GViewer import GPanel
from UI.QCore import QfloatWidget, QmatrixWidget
from GCore  import State
from UI import COLOURS
import GRIP
from IO import load
from UI import *

PASS_LIST_TYPE = 'passlist'
OUTLINER_TYPE = 'outliner'
# TODO - Allow multiple retargets

def rgb_to_hex(rgb):
	return '#%02x%02x%02x' % tuple([x*255 for x in rgb[:3]])

class QRetargetPanel(GPanel):
	edited = QtCore.Signal(object)
	hilight = QtCore.Signal(object, list)
	# not necessarily edited, but requests a redraw (after enabled toggle etc)
	updateRequest = QtCore.Signal()

	def __init__(self, parent, model):
		super(QRetargetPanel, self).__init__(parent)
		self.outlinerModel = model
		self.passWidgets = []
		self.parent = parent
		self.rtg_key = None
		self.block_update = False
		self.createMenus()
		self.createWidgets()
		self.createConnections()
		self.clear()

	def setRtg(self, rtg_key):
		self.clear()
		self.block_update = True
		self.rtg_key = rtg_key
		if rtg_key is None: return
		self.srcButton.setText(GRIP.getSourceObjectName(self.rtg_key))
		self.tarButton.setText(GRIP.getTargetObjectName(self.rtg_key))
		self.srcButton.setChecked(True)
		self.tarButton.setChecked(True)
		self.srcButton.setEnabled(True)
		self.tarButton.setEnabled(True)
		self.passList.setRtg(rtg_key)
		ikPasses = GRIP.getIKPassesAndOrder(self.rtg_key)[0]
		for key in ikPasses:
			self.createIkPass(ikPass=ikPasses[key])
		rtg_copy_pass = GRIP.getCopyPass(self.rtg_key)
		if rtg_copy_pass:
			self.createCopyPass(copyPass=rtg_copy_pass, load=True)
		self.block_update = False
		self.edited.emit(self.rtg_key)

	def clear(self):
		for w in self.passWidgets:
			w.deleteLater()
		self.rtg_key = None
		self.srcButton.setText('No Source')
		self.tarButton.setText('No Target')
		self.srcButton.setChecked(False)
		self.tarButton.setChecked(False)
		del self.passWidgets[:]
		self.passList.clear()

	def createMenus(self):
		self.rtgMenu = QtGui.QMenu("&Retarget")
		self.passMenu = QtGui.QMenu("&Pass")
		self.menuBar.addMenu(self.rtgMenu)
		self.menuBar.addMenu(self.passMenu)
		createPassMenu = QtGui.QMenu("&Create")
		self.passMenu.addMenu(createPassMenu)
		deletePassAction = createAction('Delete Pass', self, [self.deletePass], 'Delete the selected motion pass')
		deletePassAction.setEnabled(False)
		self.passMenu.addAction(deletePassAction)
		createPassMenu.addAction(createAction('Copy Pass', self, [self.createCopyPass], 'Create a copy motion pass'))
		createPassMenu.addAction(createAction('IK Pass', self, [self.createIkPass], 'Create a copy motion pass'))
		commitAction = createAction('&Commit', self, [self.commitRetarget], 'Bake the retargeting to the target skeleton')
		commitAction.setEnabled(False)
		cancelRetarget = createAction('&Delete Retargeter', self, [self.deleteRetarget], 'Cancel, deleting this retargeting setup')
		cancelRetarget.setEnabled(False)
		self.rtgMenu.addAction(commitAction)
		self.rtgMenu.addAction(cancelRetarget)
		self.enabledAction = createAction('&Toggle Enable', self, [self.toggleEnabled], 'Enable / Disable this retargeter', checkable=True, checked=True)
		self.rootAction = createAction('&Set Root Retarget', self, [self.toggleRoot], checkable=True, checked=False)
		self.rtgMenu.addAction(self.enabledAction)
		self.rtgMenu.addAction(self.rootAction)

	def createWidgets(self):
		self.sa = QtGui.QScrollArea(self)
		self.sa.setWidgetResizable(True)
		#self.sa.setVerticalScrollBarPolicy( QtCore.Qt.ScrollBarAlwaysOn )
		self.mainWidget = QtGui.QWidget(self)
		self.sa.setWidget(self.mainWidget)

		self.srcButton = QtGui.QPushButton(self)
		self.srcButton.setObjectName("source")
		self.srcButton.setEnabled(False)
		arrow = QtGui.QLabel(self)
		arrow.setPixmap(QtGui.QPixmap(os.path.join(os.path.dirname(__file__),'img/rightArrow.png')))
		self.tarButton = QtGui.QPushButton(self)
		self.tarButton.setEnabled(False)
		self.tarButton.setObjectName("target")
		ss = """ QPushButton {
     border: 2px solid #000000;
     border-radius: 12px;
     min-width: 80px;
     min-height: 25px;
 }

 QPushButton#source {
     background-color: %s;
 }
 QPushButton#target {
     background-color: %s;
 }
 QPushButton#source:checked {
     background-color: %s;
 }
 QPushButton#target:checked {
     background-color: %s;
 }
 QPushButton:pressed {
     background-color: %s;
 }

 QPushButton:flat {
     border: none; /* no border for a flat push button */
 }

 QPushButton:default {
     border-color: navy; /* make the default button prominent */
 }""" % (rgb_to_hex(COLOURS['Background']),
		rgb_to_hex(COLOURS['Background']),
		rgb_to_hex(COLOURS['Bone']),
		rgb_to_hex(COLOURS['Active']),
		rgb_to_hex(COLOURS['Hilighted']))
		self.srcButton.setStyleSheet(ss)
		self.tarButton.setStyleSheet(ss)
		self.srcButton.setCheckable(True)
		self.tarButton.setCheckable(True)
		self.srcButton.setChecked(False)
		self.tarButton.setChecked(False)

		#self.rootOffsetSpinBox = QfloatWidget(self)
		self.passList = QPassListWidget(self)

		labelLayout = QtGui.QHBoxLayout()
		labelLayout.setContentsMargins(0,0,0,0)
		labelLayout.addStretch()
		labelLayout.addWidget(self.srcButton)
		labelLayout.addWidget(arrow)
		labelLayout.addWidget(self.tarButton)
		labelLayout.addStretch()

		#formLayout = QtGui.QFormLayout()
		#formLayout.addRow("Source root offset Y", self.rootOffsetSpinBox)

		self.passLayout = QVBoxLayout()
		self.passLayout.setContentsMargins(0,0,0,0)

		saVlayout = QtGui.QVBoxLayout()

		saVlayout.addLayout(labelLayout)
		#saVlayout.addLayout(formLayout)
		saVlayout.addWidget(self.passList)
		saVlayout.addLayout(self.passLayout)
		saVlayout.setStretch(2,0)
		saVlayout.setStretch(3,1)
		saVlayout.addStretch()
		self.mainWidget.setLayout(saVlayout)
		self.layout().addWidget(self.sa)

	def createConnections(self):
		self.passList.passChanged.connect(self.switchPass)
		self.passList.passMoved.connect(lambda:self.edited.emit(self.rtg_key))
		self.passList.passRenamed.connect(self.changePassName)
		self.srcButton.toggled.connect(self.setSourceVisible)
		self.tarButton.toggled.connect(self.setTargetVisible)

	def commitRetarget(self):
		print "not yet"

	def deleteRetarget(self):
		print "not yet"

	def toggleRoot(self):
		if not self.rtg_key: return
		GRIP.setToState(self.rtg_key, 'attrs/isRoot', self.rootAction.isChecked())
		self.updateRequest.emit()

	def toggleEnabled(self):
		if not self.rtg_key: return
		GRIP.setEnabled(self, self.rtg_key, self.enabledAction.isChecked())
		self.updateRequest.emit()

	def setSourceVisible(self, visible):
		# TODO Make this work with State
		if not self.rtg_key:return
		GRIP.getSourceObject(self.rtg_key)['data']['visible'] = visible
		if not self.block_update: self.updateRequest.emit()

	def setTargetVisible(self, visible):
		# TODO Make this work with State
		if not self.rtg_key:return
		GRIP.getTargetObject(self.rtg_key)['data']['visible'] = visible
		if not self.block_update: self.updateRequest.emit()

	def createCopyPass(self, copyPass=None, load=False):
		if not self.rtg_key or (load == False and GRIP.hasCopyPass(self.rtg_key)): return
		GRIP.addCopyPassToRetargetState(self.rtg_key,copyPass or GRIP.newCopyPassDict())
		for w in self.passWidgets:
			if isinstance(w, QCopyPassWidget):
				w.setRtg(self.rtg_key)
				return
			else:
				w.setVisible(False)
		widget = QCopyPassWidget(self, self.outlinerModel, self.parent)
		widget.setRtg(self.rtg_key)
		widget.passEdited.connect(lambda:self.edited.emit(self.rtg_key))
		#widget.passEdited.connect(lambda:self.refreshJointHilights(self.rtg_key.copyPass.name))
		widget.hilight.connect(self.hilight.emit)
		widget.hilight.connect(self.updateRequest)
		self.passWidgets.insert(0, widget)
		self.passLayout.addWidget(widget)
		self.passList.refresh()
		State.push('Create Copy Pass')

	def createIkPass(self, ikPass=None):
		if not self.rtg_key: return
		cpw = QIKPassWidget(self)
		cpw.setRtg(self.rtg_key)
		if not ikPass:
			ikPass = GRIP.newIKPassDict()
			# ensure name is unique
			index = 1
			ikPasses = GRIP.getIKPassesAndOrder(self.rtg_key)[0]
			while ikPass['name'] in ikPasses:
				ikPass['name'] = "IK Pass %d" % index
				index+=1
			GRIP.addIKPassToRetargetState(self.rtg_key,ikPass)
		cpw.name = ikPass['name']
		cpw.passEdited.connect(lambda:self.refreshJointHilights(ikPass['name']))
		cpw.passEdited.connect(lambda:self.edited.emit(self.rtg_key))
		source_skeleton = GRIP.getFromState(self.rtg_key,'attrs/sourceSkeleton')
		target_skeleton = GRIP.getFromState(self.rtg_key,'attrs/targetSkeleton')
		cpw.setSourceBoneList(source_skeleton['jointNames'])
		cpw.setTargetBoneList(target_skeleton['jointNames'])
		cpw.setIKPass(ikPass)
		for x in self.passWidgets:
			x.setVisible(False)
		self.passWidgets.append(cpw)
		self.passLayout.addWidget(cpw)
		self.passList.refresh()
		State.push('Create IK pass')

	def deletePass(self):
		print "not yet"

	def switchPass(self, passName):
		for w in self.passWidgets:
			w.setVisible(False)
			if w.pass_name == passName:
				w.setTitle(passName) # hacky
				w.setVisible(True)
				self.refreshJointHilights(passName)
		self.updateRequest.emit()

	def changePassName(self, old_name, new_name):
		for w in self.passWidgets:
			if w.pass_name == old_name:
				w.pass_name = new_name

	def refreshJointHilights(self, passName): pass
		# bis = self.rtg.affectedTargetBoneIndexes(passName)
		# self.hilight.emit(self.rtg.sourceObject, None)
		# self.hilight.emit(self.rtg.targetObject, bis)

	def refresh(self):
		if self.rtg_key is None: return
		self.srcButton.setText(GRIP.getSourceObjectName(self.rtg_key))
		self.tarButton.setText(GRIP.getTargetObjectName(self.rtg_key))
		self.srcButton.setChecked(True)
		self.tarButton.setChecked(True)
		self.srcButton.setEnabled(True)
		self.tarButton.setEnabled(True)
		self.passList.stateChangeRefresh()
		for widget in self.passWidgets:
			widget.refresh()
		self.updateRequest.emit()

class QPassListWidget(QtGui.QGroupBox):
	passChanged = QtCore.Signal(str)
	passMoved = QtCore.Signal() # name, direction
	passRenamed = QtCore.Signal(str,str)
	def __init__(self, parent):
		super(QPassListWidget, self).__init__(parent)
		self.createWidgets()
		self.createMenus()
		self.setTitle("Passes")
		self.rtg_key=None
		self._setting = False
		state_key = uuid4().hex
		self._state_key = State.addKey(state_key,{'type':PASS_LIST_TYPE,
												  'attrs':{'selection':0}})
		# TODO
		self.selection = None
		#
		# this list is weird in that I restrict the movement of the copy pass (it sticks at the top)
		# it's useful to keep a count of ikPasses and total passes in order to make that work
		self.passCount = 0
		self.ikPassCount = 0

		# temp hack to handle shonky use of QStandardItemModel
		self.blockUpdate = False

	def createWidgets(self):
		self._passList = QtGui.QListView(self)
		self.passListModel = QtGui.QStandardItemModel(self)
		self.passListModel.setSortRole(QtCore.Qt.UserRole+1)
		self._passList.setModel(self.passListModel)
		self.passListModel.dataChanged.connect(self.handleDataChange)
		plsm = self._passList.selectionModel()
		plsm.selectionChanged.connect(self._handlePassSelect)
		self._passList.setMinimumHeight(60)

		self.toolBar = QtGui.QToolBar(self)
		self.toolBar.setOrientation(QtCore.Qt.Vertical)

		self._passList.setSizePolicy(QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred))#.MinimumExpanding))
		layout = QtGui.QHBoxLayout()
		layout.setContentsMargins(0,0,0,0)
		layout.addWidget(self._passList)
		layout.addWidget(self.toolBar)
		self.setLayout(layout)

	def handleDataChange(self, topLeft, bottomRight):
		# this is a stupid hack. need to write a model that handles this. this model dereferences
		# the pass objects
		if self.blockUpdate:
			return
		item = self.passListModel.itemFromIndex(topLeft)
		new_name = str(item.data(QtCore.Qt.DisplayRole))
		modelPass = item.data(QtCore.Qt.UserRole)
		mpass, pass_type = GRIP.getPass(self.rtg_key, modelPass)
		if new_name != mpass['name']:
			# Update the pass
			old_name = mpass['name']
			mpass['name'] = new_name
			GRIP.setPass(self.rtg_key,pass_type,modelPass,mpass)
			# Update the IKOrder entry
			rtg_ik_order = GRIP.getIKOrder(self.rtg_key)
			# TODO Hacky, find better solution
			for key, ik_pass_name in rtg_ik_order.items():
				if ik_pass_name == old_name:
					rtg_ik_order[key] = new_name
					break
			GRIP.setIKOrder(self.rtg_key,rtg_ik_order)
			# modelPass = str(new_name)
			self.passRenamed.emit(old_name,new_name)
			self.passChanged.emit(new_name)
		self.blockUpdate = True # the next line will cause an update that will trigger another call to here. block it
		try:
			item.setData(mpass['name'], QtCore.Qt.UserRole)
		except:
			print 'exception in GRetargetUI.handleDataChange'
		self.blockUpdate = False

	def move(self, di=1):
		""" move the selected pass up (di=-1) or down (di=1).  Updates the model(ui) and the
		ik pass order """
		sm = self._passList.selectionModel()
		try:
			selectedIndex = sm.selection().indexes()[0]
		except IndexError: # nothing selected at all
			return

		order = selectedIndex.data(QtCore.Qt.UserRole+1)

		# if it's a copy pass or it will be moved out of list bounds then skip
		if order==-1 or (order+di)<0 or (order+di) >= GRIP.getNumberOfIKPasses(self.rtg_key): return

		# swap the actual passes
		GRIP.swapIKPasses(self.rtg_key,order,order+di)
		# print(GRIP.getFromState(self.rtg_key,'attrs/passes/ik/order')) # View Pass Order
		# swap the two items in the list model. the difference in passCount and ikPass count stops
		# the copy pass from being re-ordered. (the difference is the number of non-reorderable
		# passes)
		self.passListModel.item(order+(self.passCount-self.ikPassCount)).setData(order+di, QtCore.Qt.UserRole+1)
		self.passListModel.item(order+di+(self.passCount-self.ikPassCount)).setData(order, QtCore.Qt.UserRole+1)

		# re-sort and notify
		self.passListModel.sort(0)
		self.passMoved.emit()

	def setSelectionToState(self, model_index):
		selected_row = model_index.row()
		selected_column = model_index.column()
		# print (selected_row,selected_column)
		key = '/'.join([self._state_key,'attrs/selection'])
		State.setKey(key,(selected_row,selected_column))

	def setSelectionFromState(self):
		key = '/'.join([self._state_key,'attrs/selection'])
		new_selection = State.getKey(key)
		if isinstance(new_selection,int):
			model_index = self.passListModel.index(0,0)
		else:
			row, column = new_selection
			model_index = self.passListModel.index(row,column)
		self._passList.setCurrentIndex(model_index)

	def _handlePassSelect(self, selected, deselected):
		try:

			selected_index = selected.indexes()[0]
			passName = self.passListModel.data(selected_index, QtCore.Qt.DisplayRole)
			if not self._setting:
				self.setSelectionToState(selected_index)
		except IndexError:
			return
		self.passChanged.emit(passName)
		State.push('Select Pass')

	def createMenus(self):
		up = createAction('Up', self, [functools.partial(self.move,-1)], 'Move pass up',icon=QtGui.QIcon.fromTheme("go-up"))
		down = createAction('Down', self, [functools.partial(self.move,1)], 'Move pass down',icon=QtGui.QIcon.fromTheme("go-down"))
		self.toolBar.addAction(up)
		self.toolBar.addAction(down)

	def setRtg(self, rtg):
		self.rtg_key = rtg
		self.refresh()
		self.setSelectionToState(self._passList.model().index(0,0))
		self.setSelectionFromState()

	def clear(self):
		self.setRtg(None)

	def stateChangeRefresh(self):
		self._setting = True
		self.refresh()
		self.setSelectionFromState()
		self._setting = False

	def refresh(self):
		''' this is called too often on startup. fix '''
		self.passListModel.clear()
		if not self.rtg_key:
			self.ikPassCount = 0
			self.passCount = 0
			return

		rtg_copyPass = GRIP.getCopyPass(self.rtg_key)
		if rtg_copyPass: self.addPass(rtg_copyPass, -1)

		rtg_ik_passes, rtg_ik_order = GRIP.getIKPassesAndOrder(self.rtg_key)
		for n, o in GRIP.getOrderedIKPasses(rtg_ik_passes, rtg_ik_order):
			self.addPass(n,o)
		self.ikPassCount = GRIP.getNumberOfIKPasses(self.rtg_key)
		self.passListModel.sort(0)

	def addPass(self, mpass, order):
		item = QtGui.QStandardItem()
		item.setData(mpass['name'], QtCore.Qt.DisplayRole)
		item.setData(mpass['name'], QtCore.Qt.UserRole)
		item.setData(order, QtCore.Qt.UserRole+1)
		self.passListModel.appendRow(item)
		self.passCount = self.passListModel.rowCount(QtCore.QModelIndex())

class QIKPassWidget(QtGui.QGroupBox):
	passEdited = QtCore.Signal()

	def __init__(self, parent):
		super(QIKPassWidget, self).__init__(parent)
		self.block_update = False
		self.pass_name = None
		self.parent = parent
		self.rtg_key = None
		self._setting = False
		self.mpass = None
		self.pertubations = {'Amplitude':np.zeros(3, dtype=np.float32), 'Frequency':np.zeros(3, dtype=np.float32)}
		self.lat_file = None
		self.createWidgets()
		self.createConnections()
		self.clear()

	def setRtg(self,rtg_key):
		self.rtg_key = rtg_key

	def setIKPass(self, ikPass):
		self.block_update = True
		self._setting = True
		self.mpass = ikPass
		self.pass_name = ikPass['name']
		self.ena.setChecked(ikPass['enabled'])
		self.etn.setCurrentIndex(self.targetBoneModel.stringList().index(ikPass['effectorTargetName']))
		self.ejn.setCurrentIndex(self.sourceBoneModel.stringList().index(ikPass['effectorJointName']))
		self.jco.setCurrentIndex(self.targetBoneModel.stringList().index(ikPass['jointCutoffName']))
		self.eto.setValue(ikPass['effectorTargetOffset'])
		self.wpo.setValue(ikPass['weightPosition'])
		self.wor.setValue(ikPass['weightOrientation'])
		pert_amp = ikPass.get('perturbations', {}).get('Amplitude', np.zeros(3))
		self.perturbation_amplitude.setValue(pert_amp)
		pert_freq = ikPass.get('perturbations', {}).get('Frequency', np.zeros(3))
		self.perturbation_frequency.setValue(pert_freq)
		self.pertubations = {'Amplitude': pert_amp, 'Frequency': pert_freq}
		self.setTitle(ikPass['name'])
		self._setting = False
		self.block_update = False


	def setSourceBoneList(self, bones):
		# YUCK.. need a better way of making the ikpasswidget aware of bones.
		self.sourceBoneModel.setStringList([GRIP.JOINT_NO]+bones)

	def setTargetBoneList(self, bones):
		# YUCK
		self.targetBoneModel.setStringList([GRIP.JOINT_NO]+bones)

	def clear(self):
		self.sourceBoneModel.setStringList([])
		self.targetBoneModel.setStringList([])
		self.etn.clear()
		self.ejn.clear()
		self.jco.clear()
		self.eto.setValue(np.eye(3,4))
		self.wpo.setValue(0)
		self.wor.setValue(0)
		self.perturbation_amplitude.setValue(np.zeros(3))
		self.perturbation_frequency.setValue(np.zeros(3))

	def createWidgets(self):
		self.sourceBoneModel = QtGui.QStringListModel(self)
		self.targetBoneModel = QtGui.QStringListModel(self)
		self.ena = QtGui.QCheckBox(self)
		self.ena.setStatusTip("Select an effector goal, effector cutoff and effector target before enabling")

		self.lat_ena = QtGui.QCheckBox(self)
		self.lat_ena.setStatusTip("Use Lattice?")
		self.lat_file_button = QtGui.QPushButton(self)
		self.lat_file_button.setText('Open')
		self.lat_file_label = QtGui.QLabel(self)
		self.lat_file_label.setText('Not Set')
		self.lat_reload_button = QtGui.QPushButton(self)
		self.lat_reload_button.setText('Reload')
		self.lat_button_layout = QtGui.QHBoxLayout()
		self.lat_button_layout.addWidget(self.lat_file_label)
		self.lat_button_layout.addWidget(self.lat_file_button)
		self.lat_layout = QtGui.QFormLayout()
		self.lat_layout.addRow('Lattice Enabled', self.lat_ena)
		self.lat_layout.addRow(self.lat_file_button, self.lat_file_label)
		self.lat_layout.addWidget(self.lat_reload_button)



		self.etn = QtGui.QComboBox(self)
		self.ejn = QtGui.QComboBox(self)
		self.jco = QtGui.QComboBox(self)
		self.etn.setModel(self.targetBoneModel)
		self.ejn.setModel(self.sourceBoneModel)
		self.jco.setModel(self.targetBoneModel)
		self.eto = QmatrixWidget(3,4, self)
		self.wpo = QfloatWidget(self)
		self.wor = QfloatWidget(self)
		self.perturbation_amplitude = QmatrixWidget(1,3, self)
		self.perturbation_frequency = QmatrixWidget(1,3, self)

		self.autoPushButton = QtGui.QPushButton(self)
		self.clearPushButton = QtGui.QPushButton(self)
		self.autoPushButton.setText('Auto Swizzle')
		self.clearPushButton.setText('Clear Swizzle')

		self.label = QtGui.QLabel(self)
		self.label.setText("Swizzle")
		self.swizCombo = QtGui.QComboBox(self)
		self.swizModel = QtGui.QStringListModel(self)
		self.swizModel.setStringList(GRIP.DEFAULT_SWIZZLES.keys()+['Custom'])
		self.swizCombo.setModel(self.swizModel)

		buttonLayout = QtGui.QHBoxLayout()
		buttonLayout.addWidget(self.autoPushButton)
		buttonLayout.addWidget(self.clearPushButton)
		flayout = QtGui.QFormLayout()
		flayout.addRow(self.label, self.swizCombo)

		self.wpo.setRange(0.0,1.0)
		self.wor.setRange(0.0, 100.0)
		self.jointStiffness = None
		self.latGB = QtGui.QGroupBox(self)
		self.latGB.setTitle('Lattice')
		self.latGB.setLayout(self.lat_layout)
		self.petGB = QtGui.QGroupBox(self)
		self.petGB.setTitle("perturbations")
		self.goalGB = QtGui.QGroupBox(self)
		self.goalGB.setTitle('Goal Type: Source Joint')
		self.targetGB = QtGui.QGroupBox(self)
		layout = QtGui.QVBoxLayout()
		enLayout = QtGui.QFormLayout()

		playout = QtGui.QFormLayout()
		playout.addRow("Amplitude: ", self.perturbation_amplitude)
		playout.addRow("Frequency: ", self.perturbation_frequency)
		self.petGB.setLayout(playout)

		enLayout.addRow('Pass Enabled', self.ena)
		layout.addLayout(enLayout)
		layout.addWidget(self.latGB)
		layout.addWidget(self.petGB)


		glayout = QtGui.QFormLayout()
		glayout.addRow('Effector Goal', self.ejn)
		glayout.addRow(buttonLayout)
		glayout.addRow(flayout)
		glayout.addRow('Effector Offset', self.eto)
		self.goalGB.setLayout(glayout)
		tlayout = QtGui.QFormLayout()
		tlayout.addRow('Effector Cutoff Name', self.jco)
		tlayout.addRow('Effector Target Name', self.etn)
		tlayout.addRow('Position Weight', self.wpo)
		tlayout.addRow('Orientation Weight', self.wor)
		self.targetGB.setLayout(tlayout)
		layout.addWidget(self.goalGB )
		layout.addWidget(self.targetGB)
		layout.addStretch()
		self.setLayout(layout)

	def createConnections(self):
		self.ena.toggled.connect(self.toggleEnabled)
		self.lat_ena.toggled.connect(self.toggleLatEnabled)
		self.wpo.valueChanged.connect(self.editWpo)
		self.wor.valueChanged.connect(self.editWor)
		self.etn.activated.connect(self.editEtn)
		self.ejn.activated.connect(self.editEjn)
		self.jco.activated.connect(self.editJco)
		self.eto.valueChanged.connect(self.editEto)
		self.perturbation_amplitude.valueChanged.connect(functools.partial(self.editperturbation, key='Amplitude'))
		self.perturbation_frequency.valueChanged.connect(functools.partial(self.editperturbation, key='Frequency'))
		self.swizCombo.activated.connect(self.handleSwizEdit, Qt.DirectConnection)
		self.autoPushButton.pressed.connect(self.setAutoSwizzle)
		self.clearPushButton.pressed.connect(self.clearSwizzle)
		self.lat_file_button.pressed.connect(self.openLattice)
		self.lat_reload_button.pressed.connect(self.reloadLattice)

	def setAutoSwizzle(self):
		source_target = self.ejn.currentText()
		target_target = self.etn.currentText()
		print source_target, target_target
		if source_target == "Not Set" or target_target == "Not Set":
			return
		baseSwizzle = GRIP.getAutoSwizzle(self.rtg_key,(source_target,target_target))
		swizName = self._swizName(baseSwizzle)
		new_val = self.eto.matrix.copy()
		new_val[:,:3] = baseSwizzle
		self.eto.setValue(new_val)
		self.editEto(new_val)
		self.swizCombo.setCurrentIndex(self.swizModel.stringList().index(swizName))
		# self.swizzleEdited.emit(self.s, baseSwizzle)
		self.swizCombo.setStyleSheet("background-color:#00AAFF;")


	def handleSwizEdit(self, index):
		swizName = self.swizModel.stringList()[index]
		if swizName == GRIP.SWIZZLE_NO:
			new_val = self.eto.matrix.copy()
			new_val[:,:3] = np.eye(3)
			self.eto.setValue(new_val)
			self.editEto(new_val)
			self.swizCombo.setStyleSheet("")
			return
		elif swizName == GRIP.SWIZZLE_CUSTOM:
			pass
		else:
			self.swizCombo.setStyleSheet("background-color:#00FF00;")
			new_val = self.eto.matrix.copy()
			new_val[:,:3] = GRIP.DEFAULT_SWIZZLES[swizName]
			self.eto.setValue(new_val)
			self.editEto(new_val)

	def clearSwizzle(self):
		self.swizCombo.setCurrentIndex(self.swizModel.stringList().index(GRIP.SWIZZLE_NO))
		new_val = self.eto.matrix.copy()
		new_val[:,:3] = np.eye(3)
		self.eto.setValue(new_val)
		self.editEto(new_val)
		self.swizCombo.setStyleSheet("")
		# self.swizzleEdited.emit(self.s, None)

	@staticmethod
	def _swizName(swizzle):
		for k,v in GRIP.DEFAULT_SWIZZLES.iteritems():
			if np.allclose(swizzle, v):
				return k
		return GRIP.SWIZZLE_CUSTOM

	def reloadLattice(self):
		if not os.path.exists(self.lat_file):
			errorDialog('File no longer accessible', 'The given file no longer exists or is unable to be accessed')
			return
		mappings = load(self.lat_file)[1]['mappings']
		self.setLatticeMappings(mappings)

	def openLattice(self):
		open_dir = '.' if self.lat_file is None else self.lat_file.rsplit('/',1)[0]
		filename, filtr = QtGui.QFileDialog.getOpenFileName(None, 'Choose a file to open', open_dir, 'DEF (*.def)')
		if filename == '': return
		mappings = load(filename)[1]['mappings']
		self.lat_file_label.setText(filename)
		self.lat_file = filename
		self.setLatticeMappings(mappings)


	def setLatticeMappings(self, mappings):
		GRIP.setPassField(self.rtg_key, self.pass_name, 'mappings', mappings)
		State.push('Set Lattice')

	def canEnable(self):
		if self.rtg_key is None: return
		enable = not any([self.etn.currentText()==GRIP.JOINT_NO,self.ejn.currentText()==GRIP.JOINT_NO,self.jco.currentText()==GRIP.JOINT_NO ])
		if not enable:
			if not self._setting:
				GRIP.setPassField(self.rtg_key,self.pass_name,'enabled',False)
			self.ena.setChecked(False)
		self.ena.setEnabled(enable)

	def toggleEnabled(self, value):
		if self.rtg_key is None: return
		if not self._setting:
			GRIP.setPassField(self.rtg_key,self.pass_name,'enabled',value)
			State.push('Set Enable')
		if not self.block_update: self.passEdited.emit()

	def toggleLatEnabled(self, value):
		if self.rtg_key is None: return
		if not self._setting:
			GRIP.setPassField(self.rtg_key,self.pass_name,'lat_enabled',value)
		if not self.block_update: self.passEdited.emit()
		State.push('Set Lattice Enable')

	def editWpo(self, value):
		if self.rtg_key is None: return
		if not self._setting:
			self.mpass = GRIP.setPassField(self.rtg_key,self.pass_name,'weightPosition',value)
		if self.mpass['enabled'] and not self.block_update:
			self.passEdited.emit()
		State.push('Set Weight Position')

	def editWor(self, value):
		if self.rtg_key is None: return
		if not self._setting:
			GRIP.setPassField(self.rtg_key,self.pass_name,'weightOrientation',value)
		self.canEnable()
		self.mpass = GRIP.getPass(self.rtg_key,self.pass_name)[0]
		if self.mpass['enabled'] and not self.block_update:
			self.passEdited.emit()
		State.push('Set Weight Orientation')

	def editEtn(self, index):
		if self.rtg_key is None: return
		if not self._setting:
			GRIP.setPassField(	self.rtg_key,self.pass_name,'effectorTargetName',
									self.targetBoneModel.stringList()[index])
		self.canEnable()
		self.mpass = GRIP.getPass(self.rtg_key,self.pass_name)[0]
		if self.mpass['enabled'] and not self.block_update:
			self.passEdited.emit()
		State.push('Set Effector Target Name')

	def editJco(self, index):
		if self.rtg_key is None: return
		if not self._setting:
			GRIP.setPassField(self.rtg_key,self.pass_name,'jointCutoffName',
								   self.targetBoneModel.stringList()[index])
		self.canEnable()
		self.mpass = GRIP.getPass(self.rtg_key,self.pass_name)[0]
		if self.mpass['enabled']  and not self.block_update:
			self.passEdited.emit()
		State.push('Set Joint Cutoff Name')

	def editEjn(self, index):
		if self.rtg_key is None: return
		joint = self.sourceBoneModel.stringList()[index]
		if not self._setting:
			GRIP.setPassField(self.rtg_key,self.pass_name,'effectorJointName',joint)
		self.canEnable()
		self.mpass = GRIP.getPass(self.rtg_key,self.pass_name)[0]
		if self.mpass['enabled'] and not self.block_update:
			self.passEdited.emit()
		State.push('Set Effector Joint Name')

	def editEto(self, value):
		print "eto edited"
		if self.rtg_key is None: return
		if self.pass_name is None: return
		if not self._setting:
			self.mpass = GRIP.setPassField(self.rtg_key,self.pass_name,'effectorTargetOffset',value)
		if self.mpass['enabled'] and not self.block_update:
			self.passEdited.emit()
		State.push('Set Effector Target Offset')

	def editperturbation(self, value, key):
		if self.rtg_key is None: return
		if self.pass_name is None: return
		self.pertubations[key] = value
		if not self._setting:
			self.mpass = GRIP.setPassField(self.rtg_key,self.pass_name,'perturbations',self.pertubations)
		if self.mpass['enabled'] and not self.block_update:
			self.passEdited.emit()
		State.push('Set Perturbation')
			
	def refresh(self):
		ik_pass, _ = GRIP.getPass(self.rtg_key,self.pass_name)
		source_skeleton = GRIP.getFromState(self.rtg_key,'attrs/sourceSkeleton')
		target_skeleton = GRIP.getFromState(self.rtg_key,'attrs/targetSkeleton')
		self.setSourceBoneList(source_skeleton['jointNames'])
		self.setTargetBoneList(target_skeleton['jointNames'])
		self.setIKPass(ik_pass)

class QCopyPassWidget(QtGui.QGroupBox):
	''' widget for configuring/editing a copy motion pass '''
	passEdited = QtCore.Signal()
	hilight = QtCore.Signal(object, list)

	def __init__(self, parent, model, GRetargeter_obj):
		super(QCopyPassWidget, self).__init__(parent)
		self.model = model
		# TODO
		self.unmapped_source_index = None
		self.unmapped_target_index = None
		self.mapped_index = None
		# TODO
		self.unmappedFilterModel = QJointFilterModel(self)
		self.unmappedFilterModel.setSourceModel(self.model)
		self.mappedFilterModel = QJointFilterModel(self)
		self.mappedFilterModel.showConnected = True
		self.mappedFilterModel.setSourceModel(self.model)
		self.GRetargeter_obj = GRetargeter_obj
		self.rtg_key = None
		self.mpass = None
		# is the scrolling of the bone tree views linked together?
		self._viewsLinked = False
		# temp flags to allow clocking selection connections to avoid recursion
		self.srcSelectionBlocked=False
		self.tarSelectionBlocked=False
		self.pass_name = None
		self._setting = False
		self.createWidgets()
		self.createConnections()

	def updatePass(self):
		if self.rtg_key is None: return
		self.mpass = GRIP.getFromState(self.rtg_key,'attrs/passes/cp')

	def refresh(self):
		self.updatePass()
		self.update()
		self.swizWidget.refresh()
		self.copyOffsetWidget.refresh()

	def updateSelection(self):
		if (self.unmapped_source_index is None or
			self.unmapped_target_index is None or
			self.mapped_index is None):
			self.setDefaultSelection()
		self.srcTree.selectionModel().select(self.unmapped_source_index,QItemSelectionModel.Select)
		self.tarTree.selectionModel().select(self.unmapped_target_index,QItemSelectionModel.Select)
		self.msrcTree.selectionModel().select(self.mapped_index,QItemSelectionModel.Select)

	def update(self):
		''' make the mapped targets green '''
		# clear the model
		for x in self.model.iterIndexes(self.unmappedFilterModel.mapToSource(self.tarTree.rootIndex())):
			self.model.setData(x, QtCore.Qt.UserRole + 3, False) # FIX
		for x in self.model.iterIndexes(self.unmappedFilterModel.mapToSource(self.srcTree.rootIndex())):
			self.model.setData(x, QtCore.Qt.UserRole + 3, False) # FIX
		# set the right ones true
		if not self.mpass: return
		self._setting = True
		self.setTitle(self.mpass['name'])
		self.ena.setChecked(self.mpass['enabled'])
		self.canEnable()
		for s,t in self.mpass['jointPairs']:
			tarIndex = self.model.find(self.unmappedFilterModel.mapToSource(self.tarTree.rootIndex()), t)
			srcIndex = self.model.find(self.unmappedFilterModel.mapToSource(self.srcTree.rootIndex()), s)
			self.model.setData(tarIndex, QtCore.Qt.UserRole+3, True)
			self.model.setData(srcIndex, QtCore.Qt.UserRole+3, True)
		self.unmappedFilterModel.invalidate()
		self.mappedFilterModel.invalidate()
		self._setting = False

	def canEnable(self):
		enable = bool(self.mpass['jointPairs'])
		if not enable and not self._setting:
			self.mpass = GRIP.setPassField(self.rtg_key,self.mpass['name'],'enabled',False)
		self.ena.setEnabled(enable)

	def _autoMatch(self):
		GRIP.copyAutoMatch(self.rtg_key)
		self.updatePass()
		self.update()
		self.passEdited.emit()
		State.push('Auto Match')

	def _autoSwizzle(self):
		GRIP.autoSwizzles(self.rtg_key)
		self.updatePass()
		self.swizWidget.refresh()
		self.update()
		self.passEdited.emit()
		State.push('Auto Swizzles')

	def _clearAllSwizzles(self):
		GRIP.clearSwizzles(self.rtg_key)
		self.updatePass()
		self.swizWidget.refresh()
		self.update()
		self.passEdited.emit()
		State.push('Clear Swizzles')

	def _autoCopyOffsets(self):
		GRIP.autoCopyOffsets(self.rtg_key)
		self.updatePass()
		self.copyOffsetWidget.refresh()
		self.update()
		self.passEdited.emit()
		State.push('Auto Offsets')

	def _autoPositionOffsets(self):
		GRIP.autoPositionOffsets(self.rtg_key)
		self.updatePass()
		self.copyOffsetWidget.refresh()
		self.update()
		self.passEdited.emit()
		State.push('Auto Positions')

	def _clearAllPositionOffsets(self):
		GRIP.clearPositionOffsets(self.rtg_key)
		self.updatePass()
		self.copyOffsetWidget.refresh()
		self.update()
		self.passEdited.emit()
		State.push('Clear Positions')

	def _clearAllCopyOffsets(self):
		GRIP.clearCopyOffsets(self.rtg_key)
		self.updatePass()
		self.copyOffsetWidget.refresh()
		self.update()
		self.passEdited.emit()
		State.push('Clear Offsets')

	def _mapJoints(self):
		''' get the selected source joint and target joint and link them together in the copy pass
		if the source or target are already linked then disconnect both '''
		srcJoint, tarJoint = str(self.getSelectedSourceJoint()), str(self.getSelectedTargetJoint())
		if not all([srcJoint, tarJoint]): return
		for e,(s,t) in enumerate(self.mpass['jointPairs'][:]):
			if s == srcJoint or t==tarJoint:
				del(self.mpass['jointPairs'][e])

		self.mpass['jointPairs'].append((srcJoint,tarJoint))
		GRIP.setPass(self.rtg_key,'cp','',self.mpass)
		self.update() # update the tree
		self.tarTree.selectionModel().clear()
		self.swizWidget.setCopy((srcJoint,tarJoint))
		self.copyOffsetWidget.setCopyOffset((srcJoint,tarJoint))
		self.passEdited.emit()
		State.push('Set Joint Pairs')

	def _unmapJoints(self):
		''' get the selected source joint(s) and target joint(s) and unlink them in the copy pass '''
		srcJoint, tarJoint = self.getSelectedJointPair()

		for e,(s,t) in enumerate(self.mpass['jointPairs'][:]):
			if s == srcJoint:
				del(self.mpass['jointPairs'][e])
				try:
					del self.mpass['copySwizzles'][srcJoint]
				except KeyError:
					pass
				try:
					del self.mpass['copyOffsets'][srcJoint]
				except KeyError:
					pass
			elif t==tarJoint:
				del(self.mpass['jointPairs'][e])
		GRIP.setPass(self.rtg_key,'cp','',self.mpass)
		self.update() # update the tree
		self.mtarTree.selectionModel().clear()
		self.msrcTree.selectionModel().clear()
		self.swizWidget.setCopy(None)
		self.copyOffsetWidget.setCopyOffset(None)
		self.passEdited.emit()
		State.push('Set Joint Pairs')

	def setSwizzle(self):
		self.refresh()
		if not self._setting:
			self.passEdited.emit()

	def setCopyOffset(self):
		self.refresh()
		if not self._setting:
			self.passEdited.emit()

	def setRtg(self, rtg_key):
		self.rtg_key = rtg_key
		self.mpass = GRIP.getFromState(self.rtg_key,'attrs/passes/cp')
		self.pass_name = self.mpass['name']
		self.setSource(GRIP.getSourceObject(self.rtg_key))
		self.setTarget(GRIP.getTargetObject(self.rtg_key))
		self.swizWidget.setRtg(rtg_key)
		self.copyOffsetWidget.setRtg(rtg_key)
		self.update()
		# self.updateSelection()

	def setDefaultSelection(self):
		# TODO Horrible solution, look for better way
		found_mapped, found_unmapped = False, False
		source_index = self.mappedFilterModel.mapToSource(self.msrcTree.rootIndex())
		for index in self.model.iterIndexes(source_index):
			bool_mapped = index.data(Qt.UserRole+3)
			if bool_mapped is True and not found_mapped:
				found_mapped = True
				self.mapped_index = self.mappedFilterModel.mapFromSource(index)
				print "mapped"
				print index.data(Qt.DisplayRole)
			elif bool_mapped is False and not found_unmapped:
				found_unmapped = True
				self.unmapped_source_index = self.unmappedFilterModel.mapFromSource(index)
				print "unmapped"
				print index.data(Qt.DisplayRole)
			if found_mapped and found_unmapped:
				break
		target_index = self.unmappedFilterModel.mapToSource(self.tarTree.rootIndex())
		for index in self.model.iterIndexes(target_index):
			bool_unmapped = index.data(Qt.UserRole+3)
			if bool_unmapped is False:
				self.unmapped_target_index = self.unmappedFilterModel.mapFromSource(index)
				break

	def setSource(self, src):
		if not src:return
		self.src = src
		i = self.unmappedFilterModel.indexOf(self.src)
		j = self.mappedFilterModel.indexOf(self.src)
		if not i:return
		self.srcTree.setRootIndex(i)
		self.msrcTree.setRootIndex(j)
		self.tarTree.expandAll()
		self.mtarTree.expandAll()

	def setTarget(self, tar):
		if not tar:return
		self.tar = tar
		i = self.unmappedFilterModel.indexOf(self.tar)
		j = self.mappedFilterModel.indexOf(self.tar)
		if not i:return
		self.tarTree.setRootIndex(i)
		self.mtarTree.setRootIndex(j)
		self.srcTree.expandAll()
		self.msrcTree.expandAll()

	def createWidgets(self):

		self.toolBar = QtGui.QToolBar(self)
		self.unmappedToolbar  = QtGui.QToolBar(self)
		self.mappedToolbar  = QtGui.QToolBar(self)

		self.toolBar.addAction(createAction('Clear Copy Pass', self.parent(), [self.reset], tip='Clear all copy pass settings'))
		self.toolBar.addAction(createAction('>>passes', self.parent(), [self.printPasses], tip='debug passes'))

		self.unmappedToolbar.addAction(createAction('Auto Match All', self.parent(), [self._autoMatch], tip='Auto match joints from source to target based on joint name'))
		self.unmappedToolbar.addAction(createAction('Map Joint Pair', self.parent(), [self._mapJoints], tip='Link selected joints together in the copy pass'))
		self.unmappedToolbar.addAction(createAction('Link Views', self.parent(), [self._toggleViewLink], tip='Link view scrolling', checkable=True, checked=False))

		self.mappedToolbar.addAction(createAction('Auto Swizzles', self.parent(), [self._autoSwizzle], tip='Compute swizzles for all mapped joints'))
		self.mappedToolbar.addAction(createAction('Auto Offsets', self.parent(), [self._autoCopyOffsets], tip='Compute rotation offsets to account for base pose differences'))
		self.mappedToolbar.addAction(createAction('Auto Positions', self.parent(), [self._autoPositionOffsets], tip='Compute rotation offsets to account for base pose differences'))
		self.mappedToolbar.addAction(createAction('Clear All Swizzles', self.parent(), [self._clearAllSwizzles], tip='Compute swizzles for all mapped joints'))
		self.mappedToolbar.addAction(createAction('Clear All Offsets', self.parent(), [self._clearAllCopyOffsets], tip='Clear all copy offsets'))
		self.mappedToolbar.addAction(createAction('Clear All Positions', self.parent(), [self._clearAllPositionOffsets], tip='Clear all position offsets'))
		self.mappedToolbar.addAction(createAction('UnMap Joint Pair', self.parent(), [self._unmapJoints], tip='Unlink selected joints'))

		self.srcTree = QtGui.QTreeView(self)
		self.tarTree = QtGui.QTreeView(self)
		self.srcTree.setModel(self.unmappedFilterModel)
		self.tarTree.setModel(self.unmappedFilterModel)

		self.srcTree.setHeaderHidden(True)
		self.srcTree.setUniformRowHeights(True) # faster
# 		self.srcTree.setIndentation(12)
		#self.srcTree.setSortingEnabled(True)
		self.tarTree.setHeaderHidden(True)
		self.tarTree.setUniformRowHeights(True)
# 		self.tarTree.setIndentation(12)
		#self.tarTree.setSortingEnabled(True)

		self.msrcTree = QtGui.QTreeView(self)
		self.mtarTree = QtGui.QTreeView(self)
		self.msrcTree.setModel(self.mappedFilterModel)
		self.mtarTree.setModel(self.mappedFilterModel)
		self.tarSm = QtGui.QItemSelectionModel(self.mappedFilterModel)
		self.mtarTree.setSelectionModel(self.tarSm)
		self.srcSm = QtGui.QItemSelectionModel(self.mappedFilterModel)
		self.msrcTree.setSelectionModel(self.srcSm)
		#self.mtarTree.setSortingEnabled(True)
		#self.msrcTree.setSortingEnabled(True)

		self.msrcTree.setHeaderHidden(True)
		self.msrcTree.setUniformRowHeights(True) # faster
# 		self.msrcTree.setIndentation(12)
		self.mtarTree.setHeaderHidden(True)
		self.mtarTree.setUniformRowHeights(True)
# 		self.mtarTree.setIndentation(12)

		self.swizWidget = QSwizzleWidget(self,self.GRetargeter_obj)
		self.copyOffsetWidget = QCopyOffsetWidget(self)

		self.ena = QtGui.QCheckBox(self)

		self.splitter = QtGui.QSplitter(self)
		self.splitter.setOrientation(QtCore.Qt.Vertical)
		self.splitter.setSizePolicy(QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding))

		topWidget = QtGui.QGroupBox(self.splitter)
		topWidget.setTitle("Unmapped Joints")
		bottomWidget = QtGui.QGroupBox(self.splitter)
		bottomWidget.setTitle("Mapped Joints")

		form = QtGui.QFormLayout()
		form.addRow('Pass Enabled', self.ena)

		umLayout= QtGui.QVBoxLayout()
		umLayout.addWidget(self.unmappedToolbar)
		umLayout.setContentsMargins(0,0,0,0)
		leftLayout = QtGui.QHBoxLayout()
		leftLayout.setContentsMargins(0,0,0,0)
		leftLayout.addWidget(self.srcTree)
		leftLayout.addWidget(self.tarTree)
		umLayout.addLayout(leftLayout)
		topWidget.setLayout(umLayout)

		mLayout= QtGui.QVBoxLayout()
		mLayout.addWidget(self.mappedToolbar)
		mLayout.setContentsMargins(0,0,0,0)
		rightLayout = QtGui.QHBoxLayout()
		rightLayout.setContentsMargins(0,0,0,0)
		rightLayout.addWidget(self.msrcTree)
		rightLayout.addWidget(self.mtarTree)
		mLayout.addLayout(rightLayout)

		bottomWidget.setLayout(mLayout)

		self.splitter.addWidget(topWidget)
		self.splitter.addWidget(bottomWidget)

		self.layout = QtGui.QVBoxLayout()
		self.layout.addLayout(form)
		self.layout.addWidget(self.toolBar)

		self.layout.addWidget(self.splitter)
		self.layout.addWidget(self.swizWidget)
		self.layout.addWidget(self.copyOffsetWidget)
		self.layout.setStretch(2,1)
		self.setLayout(self.layout)

	def createConnections(self):
		ssm = self.msrcTree.selectionModel()
		ssm.selectionChanged.connect(self.handleMappedSrcSelectionChanged)
		tsm = self.mtarTree.selectionModel()
		tsm.selectionChanged.connect(self.handleMappedTarSelectionChanged)
		self.swizWidget.swizzleEdited.connect(self.setSwizzle, Qt.DirectConnection)
		self.copyOffsetWidget.copyOffsetEdited.connect(self.setCopyOffset, Qt.DirectConnection)
		self.ena.toggled.connect(self.toggleEnabled)

	def toggleEnabled(self, value):
		self.mpass['enabled'] = value
		if not self._setting:
			GRIP.setPass(self.rtg_key,'cp','',self.mpass)
			self.passEdited.emit()
		State.push('Set Enable')

	def _toggleViewLink(self):
		self._viewsLinked = not self._viewsLinked
		# link the scolling together
		if self._viewsLinked:
			self.srcTree.verticalScrollBar().valueChanged.connect(self.tarTree.verticalScrollBar().setValue)
			self.tarTree.verticalScrollBar().valueChanged.connect(self.srcTree.verticalScrollBar().setValue)
		else:
			self.srcTree.verticalScrollBar().valueChanged.disconnect(self.tarTree.verticalScrollBar().setValue)
			self.tarTree.verticalScrollBar().valueChanged.disconnect(self.srcTree.verticalScrollBar().setValue)

	def handleMappedSrcSelectionChanged(self, selected, deselected):
		''' select the corresponding joint in the target column, or clear the selection if there
		isn't one)'''
		# TODO make selection part of State
		if self.srcSelectionBlocked: return
		matchingTarget = None
		#for i in selected.indexes(): # will be one
		try:
			selectedIndex =  selected.indexes()[0]
		except IndexError:
			joint = None
		else:
			joint = str(selectedIndex.data(QtCore.Qt.DisplayRole))
		print joint
		try:
			matchingTarget = dict(self.mpass['jointPairs'])[joint] # probably slow
		except KeyError:
			pass
		rtg_S = GRIP.getFromState(self.rtg_key,'attrs/sourceSkeleton')
		self.hilight.emit(self.src, [rtg_S['jointIndex'][joint]])
		if not matchingTarget:
			self.swizWidget.setCopy(None)
			self.copyOffsetWidget.setCopyOffset(None)
			self.hilight.emit(self.tar, None)
			return
		rtg_T = GRIP.getFromState(self.rtg_key,'attrs/targetSkeleton')
		self.hilight.emit(self.tar,[rtg_T['jointIndex'][matchingTarget]])

		# find it under the index of the target
		# and select it..
		tarIndex = self.mappedFilterModel.find(self.mtarTree.rootIndex(), matchingTarget)
		if tarIndex:
			self.tarSelectionBlocked=True
			sm = self.mtarTree.selectionModel()
			sm.clear()
			sm.select(tarIndex, QtGui.QItemSelectionModel.Select)
			self.swizWidget.setCopy((joint, matchingTarget))
			self.copyOffsetWidget.setCopyOffset((joint, matchingTarget))
			self.tarSelectionBlocked=False
			if not self._viewsLinked: self.mtarTree.scrollTo(tarIndex, QtGui.QAbstractItemView.PositionAtCenter)

	def handleMappedTarSelectionChanged(self, selected, deselected):
		''' select the corresponding joint in the target column, or clear the selection if there
		isn't one)'''
		# TODO make selection part of State
		if self.tarSelectionBlocked: return
		matchingSource = None
		for i in selected.indexes(): # will be one
			joint = i.data(QtCore.Qt.DisplayRole)
			for x, v in self.mpass['jointPairs']:
				if v == joint:
					matchingSource = x
					break
		rtg_T = GRIP.getFromState(self.rtg_key,'attrs/targetSkeleton')
		self.hilight.emit(self.tar, [rtg_T['jointIndex'][joint]])
		if not matchingSource:
			self.swizWidget.setCopy(None)
			self.copyOffsetWidget.setCopyOffset(None)
			self.hilight.emit(self.tar, None)
			return
		rtg_S = GRIP.getFromState(self.rtg_key,'attrs/sourceSkeleton')
		self.hilight.emit(self.src,[rtg_S['jointIndex'][matchingSource]])

		# find it under the index of the target
		# and select it..
		srcIndex = self.mappedFilterModel.find(self.msrcTree.rootIndex(), matchingSource)
		if srcIndex:
			self.srcSelectionBlocked=True
			sm = self.msrcTree.selectionModel()
			sm.clear()
			sm.select(srcIndex, QtGui.QItemSelectionModel.Select)
			self.swizWidget.setCopy((matchingSource, joint))
			self.copyOffsetWidget.setCopyOffset((matchingSource, joint))
			self.srcSelectionBlocked=False
			if not self._viewsLinked: self.msrcTree.scrollTo(srcIndex, QtGui.QAbstractItemView.PositionAtCenter)

	def reset(self):
		if not self.mpass: return
		del self.mpass['jointPairs'][:]
		self.mpass['copyOffsets'] = {}
		self.mpass['copySwizzles'] = {}
		GRIP.setPass(self.rtg_key,'cp','',self.mpass)
		self.update()
		self.passEdited.emit()
		State.push('Clear Copy Pass')

	def getSelectedSourceJoint(self):
		return self.getSelectedUnmappedJoint(self.srcTree)

	def getSelectedTargetJoint(self):
		return self.getSelectedUnmappedJoint(self.tarTree)

	def getSelectedJointPair(self):
		return self.getSelectedMappedJoint(self.msrcTree), self.getSelectedMappedJoint(self.mtarTree)

	def getSelectedMappedJoint(self, widget):
		try:
			return self.mappedFilterModel.data(widget.selectedIndexes()[0], QtCore.Qt.DisplayRole)
		except IndexError:
			return None

	def getSelectedUnmappedJoint(self, widget):
		try:
			return self.unmappedFilterModel.data(widget.selectedIndexes()[0], QtCore.Qt.DisplayRole)
		except IndexError:
			return None

	def printPasses(self):
		GRIP.printPasses(self.rtg_key)

class QSwizzleWidget(QtGui.QGroupBox):
	swizzleEdited = QtCore.Signal()
	def __init__(self, parent, GRetargeter_obj):
		super(QSwizzleWidget, self).__init__(parent)
		self.GRetargeter_obj = GRetargeter_obj
		self.createWidgets()
		self.createConnections()
		self.rtg_key = None
		self.s, self.t = None, None

	def setRtg(self, rtg_key):
		self.rtg_key = rtg_key
		self.setCopy(None)

	def setCopy(self, cp):
		if cp:
			self.s, self.t = cp
			self.setTitle("Swizzle :"+self.s+' > '+self.t)
		else:
			self.s, self.t = None, None
			self.setTitle("Swizzle")
			return

		self.refresh()

	def refresh(self):
		if self.rtg_key is None: return
		try:
			copy_pass = GRIP.getCopyPass(self.rtg_key)
			currentSwizzle = copy_pass['copySwizzles'][self.s]
		except KeyError:
			swizName = GRIP.SWIZZLE_NO
			swizz_value = np.eye(3)
			self.matrixWidget.setValue(swizz_value)
			self.swizCombo.setStyleSheet("")
			copy_pass['copySwizzles'][self.s] = swizz_value
		else:
			swizName = self._swizName(currentSwizzle)
			self.matrixWidget.setValue(currentSwizzle)
			baseSwizzle = GRIP.getAutoSwizzle(self.rtg_key, (self.s,self.t))
			if not np.allclose(currentSwizzle, baseSwizzle): # (x,+)
				# swizzle has been changed from base
				self.swizCombo.setStyleSheet("background-color:#00FF00;")
			else:
				self.swizCombo.setStyleSheet("background-color:#00AAFF;")

		self.swizCombo.setCurrentIndex(self.swizModel.stringList().index(swizName))

	@staticmethod
	def _swizName(swizzle):
		for k,v in GRIP.DEFAULT_SWIZZLES.iteritems():
			if np.allclose(swizzle, v):
				return k
		return GRIP.SWIZZLE_CUSTOM

	def createWidgets(self):
		self.autoPushButton = QtGui.QPushButton(self)
		self.clearPushButton = QtGui.QPushButton(self)
		self.autoPushButton.setText('Auto Swizzle')
		self.clearPushButton.setText('Clear Swizzle')
		# self.alignButton = QtGui.QPushButton(self)
		# self.alignButton.setText('Coerce Alignment')
		# self.alignButton.setEnabled(False)

		self.label = QtGui.QLabel(self)
		self.label.setText("Swizzle")
		self.swizCombo = QtGui.QComboBox(self)
		self.swizModel = QtGui.QStringListModel(self)
		self.swizModel.setStringList(GRIP.DEFAULT_SWIZZLES.keys()+['Custom'])
		self.swizCombo.setModel(self.swizModel)
		self.matrixWidget = QmatrixWidget(3,3,self)

		buttonLayout = QtGui.QHBoxLayout()
		buttonLayout.addWidget(self.autoPushButton)
		buttonLayout.addWidget(self.clearPushButton)
		layout = QtGui.QVBoxLayout()
		layout.addLayout(buttonLayout)
		flayout = QtGui.QFormLayout()
		flayout.addRow(self.label, self.swizCombo)
		flayout.addRow('Swizzle Matrix', self.matrixWidget)
		# flayout.addRow('Align Swizzle', self.alignButton)
		layout.addLayout(flayout)
		self.setLayout(layout)

	def createConnections(self):
		self.swizCombo.activated.connect(self.handleSwizEdit, Qt.DirectConnection)
		self.matrixWidget.valueChanged.connect(self.handleMatrixEdit, Qt.DirectConnection)
		self.autoPushButton.pressed.connect(self.setAutoSwizzle)
		self.clearPushButton.pressed.connect(self.clearSwizzle)

	def setAutoSwizzle(self):
		if not self.s or not  self.t:
			return
		baseSwizzle = GRIP.getAutoSwizzle(self.rtg_key,(self.s,self.t))
		self.matrixWidget.setValue(baseSwizzle)
		self.handleMatrixEdit(baseSwizzle)

	def clearSwizzle(self):
		self.swizCombo.setCurrentIndex(self.swizModel.stringList().index(GRIP.SWIZZLE_NO))
		self.matrixWidget.setValue(np.eye(3))
		self.handleMatrixEdit(np.eye(3))

	def handleSwizEdit(self, index):
		swizName = self.swizModel.stringList()[index]
		if swizName == GRIP.SWIZZLE_NO:
			val = np.eye(3)
		elif swizName == GRIP.SWIZZLE_CUSTOM:
			pass
		else:
			val = GRIP.DEFAULT_SWIZZLES[swizName]
		self.matrixWidget.setValue(val)
		self.handleMatrixEdit(val)

	def handleMatrixEdit(self, m):
		swizName = self._swizName(m)
		if swizName == GRIP.SWIZZLE_NO:
			m = np.eye(3)
			self.swizCombo.setStyleSheet("")
		else:
			self.swizCombo.setStyleSheet("background-color:#00FF00;")
		self.swizCombo.setCurrentIndex(self.swizModel.stringList().index(swizName))
		GRIP.setToState(self.rtg_key,'attrs/passes/cp/copySwizzles/' + self.s, m.copy())
		self.swizzleEdited.emit()
		State.push('Swizzle Edit')

class QCopyOffsetWidget(QtGui.QGroupBox):
	''' incomplete.  rotations should be represented the same way as swizzles, or, better as
	rx,ry,rz in some frame in some order'''
	copyOffsetEdited = QtCore.Signal()
	def __init__(self, parent):
		super(QCopyOffsetWidget, self).__init__(parent)
		self.createWidgets()
		self.createConnections()
		self.rtg_key = None
		self.value = np.eye(3,4)
		self.posValue = np.zeros((2,3))
		self.s, self.t = None, None
		self.chanIndices = None

	def setRtg(self, rtg_key):
		self.rtg_key = rtg_key
		self.setCopyOffset(None)

	def setCopyOffset(self, cp):
		if cp:
			self.s, self.t = cp
			self.setTitle("Copy Offset :"+self.s+' > '+self.t)
		else:
			self.s, self.t = None, None
			self.setTitle("Copy Offset")
		self.matrixPositionWidget.setValue(np.zeros((2,3)))
		self.refresh()

	def refresh(self):
		if self.rtg_key is None: return
		try:
			rtg_copy_pass = GRIP.getCopyPass(self.rtg_key)
			self.getPositionOffsets()
			self.matrixWidget.setValue(rtg_copy_pass['copyOffsets'][self.s].copy())
		except KeyError:
			self.matrixWidget.setValue(np.eye(3,4))


	def _swizName(swizzle):
		for k,v in GRIP.DEFAULT_SWIZZLES.iteritems():
			if np.allclose(swizzle, v):
				return k
		return GRIP.SWIZZLE_CUSTOM

	def createWidgets(self):
		self.autoPushButton = QtGui.QPushButton(self)
		self.clearPushButton = QtGui.QPushButton(self)
		self.autoPushButton.setText('Auto Copy Offset')
		self.clearPushButton.setText('Clear Copy Offset')

		self.flipXButton = QtGui.QPushButton(self)
		self.flipYButton = QtGui.QPushButton(self)
		self.flipZButton = QtGui.QPushButton(self)
		self.flipXButton.setText('Flip X')
		self.flipYButton.setText('Flip Y')
		self.flipZButton.setText('Flip Z')

		self.autoPositionButton = QtGui.QPushButton(self)
		self.clearPositionButton = QtGui.QPushButton(self)
		self.autoPositionButton.setText('Auto Position Offset')
		self.clearPositionButton.setText('Clear Position Offset')

		self.matrixWidget = QmatrixWidget(3,4,self)
		buttonLayout = QtGui.QHBoxLayout()
		buttonLayout.addWidget(self.autoPushButton)
		buttonLayout.addWidget(self.clearPushButton)
		flipButtonLayout = QtGui.QHBoxLayout()
		flipButtonLayout.addWidget(self.flipXButton)
		flipButtonLayout.addWidget(self.flipYButton)
		flipButtonLayout.addWidget(self.flipZButton)
		layout = QtGui.QVBoxLayout()
		layout.addLayout(buttonLayout)
		layout.addLayout(flipButtonLayout)
		flayout = QtGui.QFormLayout()
		flayout.addRow('Copy Offset Matrix', self.matrixWidget)
		layout.addLayout(flayout)

		self.matrixPositionWidget = QmatrixWidget(2,3,self)

		posLayout = QtGui.QFormLayout()

		buttonLayout2 = QtGui.QHBoxLayout()
		buttonLayout2.addWidget(self.autoPositionButton)
		buttonLayout2.addWidget(self.clearPositionButton)

		posLayout.addRow('Position Offset Matrix',self.matrixPositionWidget)

		layout.addLayout(buttonLayout2)
		layout.addLayout(posLayout)

		self.setLayout(layout)

	def getPositionOffsets(self):
		skel = GRIP.getFromState(self.rtg_key,'attrs/targetSkeleton')
		ti = skel['jointIndex'][self.t]
		tran_from,rot_from,rot_end = skel['jointChanSplits'][2*ti:2*ti+3]
		self.chanIndices = np.ones(6, dtype=int) * -1
		for i in range(tran_from, rot_end):
			self.chanIndices[skel['jointChans'][i]] = i
		where = np.where(self.chanIndices > -1)
		copy_pass = GRIP.getCopyPass(self.rtg_key)
		values = np.zeros(6, dtype=np.float32)
		if 'positionOffsets' in copy_pass:
			values[where] = copy_pass['positionOffsets'][self.chanIndices[where]]
		self.matrixPositionWidget.setValue(values.reshape(2,3))

	def flipAxis(self, axis):
		self.matrixWidget.matrix[axis,:3] = -self.matrixWidget.matrix[axis,:3]
		self.handleMatrixEdit(self.matrixWidget.matrix)

	def createConnections(self):
		self.matrixWidget.valueChanged.connect(self.handleMatrixEdit, Qt.DirectConnection)
		self.autoPushButton.pressed.connect(self.setAutoCopyOffset)
		self.clearPushButton.pressed.connect(self.clearCopyOffset)
		self.matrixPositionWidget.valueChanged.connect(self.handlePositionEdit, Qt.DirectConnection)
		self.autoPositionButton.pressed.connect(self.setAutoPositionOffset)
		self.clearPositionButton.pressed.connect(self.clearPositionOffset)
		self.flipXButton.pressed.connect(functools.partial(self.flipAxis,0))
		self.flipYButton.pressed.connect(functools.partial(self.flipAxis,1))
		self.flipZButton.pressed.connect(functools.partial(self.flipAxis,2))

	def setAutoPositionOffset(self):
		GRIP.getAutoPositionOffset(self.rtg_key, (self.s,self.t))
		self.getPositionOffsets()
		self.copyOffsetEdited.emit()

	def setAutoCopyOffset(self):
		baseCopyOffset = GRIP.getAutoCopyOffset(self.rtg_key, (self.s,self.t))
		if baseCopyOffset is None: return
		self.matrixWidget.setValue(baseCopyOffset)
		self.handleMatrixEdit(self.matrixWidget.matrix)
		self.copyOffsetEdited.emit()

	def clearCopyOffset(self):
		self.matrixWidget.setValue(np.eye(3,4))
		self.handleMatrixEdit(self.matrixWidget.matrix)
		self.copyOffsetEdited.emit()

	def clearPositionOffset(self):
		self.matrixPositionWidget.setValue(np.zeros((2,3)))
		self.handlePositionEdit(self.matrixPositionWidget.matrix)
		self.copyOffsetEdited.emit()

	def handleMatrixEdit(self, m):
		if self.s is None: return
		GRIP.setToState(self.rtg_key,'attrs/passes/cp/copyOffsets/' + self.s, m.copy())
		self.copyOffsetEdited.emit()
		State.push('Copy Offset Edit')

	def handlePositionEdit(self, m):
		if self.s is None: return
		copy_pass = GRIP.getCopyPass(self.rtg_key)
		where = np.where(self.chanIndices > -1)
		if 'positionOffsets' not in copy_pass:
			rtg_T = GRIP.getFromState(self.rtg_key, 'attrs/targetSkeleton')
			copy_pass['positionOffsets'] = np.zeros(rtg_T['numChans'], dtype=np.float32)
		copy_pass['positionOffsets'][self.chanIndices[where]] = m.reshape(-1)[where]
		GRIP.setToState(self.rtg_key,'attrs/passes/cp/positionOffsets/', copy_pass['positionOffsets'])
		self.copyOffsetEdited.emit()
		State.push('Position Edit')

class QJointFilterModel(QtGui.QSortFilterProxyModel):
	def __init__(self, parent):
		super(QJointFilterModel, self).__init__(parent=parent)
		self.showConnected = False

	def find(self, parentIndex, name, role=QtCore.Qt.DisplayRole):
		''' search the model under parentIndex for the item called
		"name". return the model index of that item
		searches recursively, completing the search of each row before searching child items'''
		#pi = self.mapToSource(parentIndex)
		for index in self.iterIndexes(parentIndex):
			#i = self.mapFromSource(index)
			if self.data(index, role) == name:
				return index

	def indexOf(self, obj):
		''' will search the model recursively. returns the index of the item that represents the
		provided object '''
		if not obj: return QtCore.QModelIndex()
		index = self.sourceModel().find(QtCore.QModelIndex(), obj, role=QtCore.Qt.UserRole)
		return self.mapFromSource(index)

	def filterAcceptsRow(self, row, parentIndex):
		index = self.sourceModel().index(row, 0, parentIndex)
		filterValue = index.data(QtCore.Qt.UserRole+3)
		return filterValue is self.showConnected or filterValue is None

	def iterIndexes(self, parentIndex):
		''' yield all child indexes, recursively under the provided parent index '''
		for row in xrange(0, self.rowCount(parentIndex)):
			index = self.index(row, 0, parentIndex)
			yield index
		for row in xrange(0, self.rowCount(parentIndex)):
			index = self.index(row, 0, parentIndex)
			for x in self.iterIndexes(index):
				yield x

class GTreeView(QtGui.QTreeView):
	def mouseMoveEvent(self, event):
		#self.setState(self.state() ^ QtGui.QTreeView.State.DragSelectingState)
		super(GTreeView, self).mouseMoveEvent(event)

class GOutlinerPanel(GPanel):
	selectionChanged = QtCore.Signal(list, list)

	def __init__(self, parent=None):
		super(GOutlinerPanel, self).__init__(parent=parent)

		self.tree = GTreeView(self)
		self.tree.setHeaderHidden(True)
		self.tree.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
		self.tree.setUniformRowHeights(True) # faster
		self.tree.setIndentation(12)

		self.model = QOutliner.QOutlinerModel(self)
		self.tree.setModel(self.model)

		sm = self.tree.selectionModel()
		sm.selectionChanged.connect(self.handleUserSelection)

		self.layout().addWidget(self.tree)
		self.layout().setStretch(1,1)

		state_key = uuid4().hex
		self._state_key = State.addKey(state_key,{'type':OUTLINER_TYPE,
												  'attrs':{'expansion':None}})

	def handleUserSelection(self, selected, deselected):
		''' user changes the selection by clicking, emit what changed '''
		# emit 2 lists of objects, those selected and those deselected
		# nasty, temp
		if len(selected.indexes()) == 1:
			rtg_key = self.model.data(selected.indexes()[0], QOutliner.RTG_ROLE)
		else:
			rtg_key = None
		selectedObjects = []
		deselectedObjects = []
		for x in selected.indexes():
			o = self.model.data(x, QtCore.Qt.UserRole)
			if not o:
				continue
			selectedObjects += self.selectObject(o)

		for x in deselected.indexes():
			o = self.model.data(x, QtCore.Qt.UserRole)
			if not o:
				continue
			deselectedObjects += self.selectObject(o)

		deselectedObjects = [o for o in deselectedObjects if o not in selectedObjects]
		# deselectedObjects = deselectedObjects.difference(selectedObjects)
		self.selectionChanged.emit(selectedObjects, deselectedObjects)

	def selectObject(self, obj):
		obj_type = obj.get('objecttype')
		if obj_type == GRIP.RETARGET_TYPE:
			return [obj]
		elif obj_type == QOutliner.TYPE_TYPE:
			selection = []
			for child_obj in obj['children']:
				selection += self.selectObject(child_obj)
			return selection
		else:
			return [obj]

	@QtCore.Slot()
	def updateSelection(self, selected, deselected):
		''' use to update the selection from somewhere else - like when GL view selections are made
		'''
		sm = self.tree.selectionModel()

		# disconnect the selection changed or you'll get feedback
		sm.selectionChanged.disconnect(self.handleUserSelection)

		# add to seleection
		indexes = [self.model.indexOf(obj) for obj in selected] # not recursive yet
		for idx in indexes: # this could be quicker using selection ranges, but at the moment it's only ever passed single items
			sm.select(idx, QtGui.QItemSelectionModel.Select | QtGui.QItemSelectionModel.Rows)

		# remove from selection
		indexes = [self.model.indexOf(obj) for obj in deselected]
		for idx in indexes:
			sm.select(idx, QtGui.QItemSelectionModel.Deselect | QtGui.QItemSelectionModel.Rows)

		# restore the signal
		sm.selectionChanged.connect(self.handleUserSelection)

	def remove(self, obj):
		''' remove the provided object from the outliner '''
		sm = self.tree.selectionModel()
		sm.selectionChanged.disconnect(self.handleUserSelection)
		self.model.remove(obj)
		sm.selectionChanged.connect(self.handleUserSelection)

	def refresh(self):
		self.model.setScene(GRIP.G_All_Things.values())

def picked(*x):
		print "Picked: {}".format(x)

def main():
	from UI import QApp, QGLViewer
	from UI import GLPoints3D
	from PySide import QtCore, QtGui
	import sys
	appIn = QtGui.QApplication(sys.argv)
	appIn.setStyle('plastique')
	win = QApp.QApp()
	# win.outliner.setVisible(False)
	# win.qfields.setVisible(False)
	outliner = GOutlinerPanel(parent=win)
	retarget_panel = QRetargetPanel(win, outliner.model)
	outliner_connections = {'selectionChanged':[functools.partial(GRIP.updateSelection,retarget_panel),'self.updateGL']}
	win.addWidget(outliner,'Outliner',outliner_connections)
	retarget_connections = {'edited':[functools.partial(GRIP.updateMapping,win)],'updateRequest':[GRIP.updateRTGDict,'self.refresh'],
				   			'hilight':[GRIP.hilightPrimitive]}
	win.addWidget(retarget_panel,'Retarget',retarget_connections,QtCore.Qt.RightDockWidgetArea)
	retarget_panel = win.set_widget['Retarget']
	outliner = win.set_widget['Outliner']
	outliner.model.setScene(GRIP.G_All_Things.values())
	win.addMenuItem({'menu':'&File','item':'&Import',
					 'tip':'Import a file',
					 'cmd':GRIP.fileImportDialog,'args':[win,win.qtimeline,outliner]})
	win.addMenuItem({'menu':'&Test','item':'100 Motions',
					 'tip':'Import 100 Motions',
					 'cmd':GRIP.load100Motions,'args':[win,outliner,win.qtimeline]})
	win.addMenuItem({'menu':'&Edit','item':'Delete',
					 'tip':'Delete the selected objects',
					 'cmd':GRIP.deleteSelected,'args':[win,outliner,retarget_panel]})
	win.addMenuItem({'menu':'&Retarget','item':'Retarget Selected',
					 'tip':'retarget selected motion to selected skeleton',
					 'cmd':GRIP.retargetSelected,'args':[outliner,retarget_panel]})
	win.addMenuItem({'menu':'&File','item':'&Open',
					 'tip':'Open File',
					 'cmd':GRIP.retargeterLoad,'args':[win,outliner,retarget_panel,win.qtimeline]})
	win.addMenuItem({'menu':'&File','item':'&Save',
					 'tip':'Save the current state',
					 'cmd':GRIP.retargeterSave,'args':[win]})
	win.addMenuItem({'menu':'&File','item':'Save &As',
					 'tip':'Save state as',
					 'cmd':GRIP.retargeterSaveAs,'args':[win]})
	win.addMenuItem({'menu':'&Edit','item':'&Undo','shortcut':'Ctrl+Z',
					 'tip':'Undo State',
					 'cmd':GRIP.undo,'args':[win,retarget_panel,outliner,win.qtimeline]})
	win.addMenuItem({'menu':'&Edit','item':'Re&do','shortcut':'Ctrl+Shift+Z',
					 'tip':'Redo State',
					 'cmd':GRIP.redo,'args':[win,retarget_panel,outliner,win.qtimeline]})
	win.addMenuItem({'menu':'&File','item':'&New','shortcut':'Ctrl+N',
					 'tip':'New Scene',
					 'cmd':GRIP.clearAll,'args':[win,outliner,retarget_panel]})
	win.addMenuItem({'menu':'&Scale','item':'Scale &Character',
					 'tip':'Scale a Character',
					 'cmd':GRIP.scaleSelected,'args':[win]})
	win.layout().removeWidget(win.outliner)
	win.outliner.deleteLater()
	win.outliner = None
	win.layout().removeWidget(win.attributeEditor)
	win.attributeEditor.deleteLater()
	win.attributeEditor = None
	appIn.connect(appIn, QtCore.SIGNAL('lastWindowClosed()') , appIn.quit)
	State.clearUndoStack()
	target_point = GLPoints3D(np.zeros(3), colour=[1,0,0,1])
	GRIP.G_Target_Point = target_point
	QGLViewer.makeViewer(primitives=[target_point], callback=functools.partial(GRIP.frameCallback, win) ,pickCallback=picked,appIn=appIn,win=win)

if __name__ == '__main__':
	main()