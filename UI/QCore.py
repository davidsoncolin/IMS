#!/usr/bin/env python
import functools
import numpy as np
from PySide import QtCore, QtGui
from GCore import State
from UI import createAction
import weakref


class QListWidget(QtGui.QListView):
    item_selected = QtCore.Signal(int)
    focus_changed = QtCore.Signal(bool)
    item_renamed = QtCore.Signal(str, str)
    data_changed = QtCore.Signal(dict)

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
        self.createWidgets()
        for item in items:
            self.addItem(item)

    def count(self):
        return self.item_count

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
        print ("Data Change: {}".format(args))
        self.overrideSelection = args[0].row()
        newText = self.getItem(args[0].row(), QtCore.Qt.DisplayRole)
        if newText != self.selectedItem:
            self.item_renamed.emit(self.selectedItem, newText)
            self.selectedItem = newText
        else:
            self.data_changed.emit({})

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
            self.selectedItem = self.getItem(self.getSelection(), QtCore.Qt.DisplayRole)
            print ("Selected: {}".format(self.selectedItem))
            self.item_selected.emit(self.getSelection())
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

    def addItem(self, mitem, data='', index=None):
        item = QtGui.QStandardItem()
        item.setData(mitem, QtCore.Qt.DisplayRole)
        item.setData(data, QtCore.Qt.UserRole)
        item.setEditable(self.renameEnabled)
        item.setDropEnabled(False)
        # Can be used to store data linked to the name
        # item.setData(customData, QtCore.Qt.UserRole)
        if index is None:
            self.item_list_model.appendRow(item)
        else:
            self.item_list_model.insertRow(index, item)
        self.item_count += 1

    def addItems(self, items):
        for item in items:
            self.addItem(item)

    def setUserSelection(self, index):
        if self.item_count > 0:
            self.setCurrentIndex(self.item_list_model.item(index).index())
            self.selectedItem = self.getItem(index, QtCore.Qt.DisplayRole)

    def getItems(self, role=None):
        if role is None:
            return [self.item_list_model.item(i) for i in xrange(0, self.item_count)]
        else:
            return [self.item_list_model.item(i).data(role) for i in xrange(0, self.item_count)]

    def getItem(self, index, role=None):
        if role is None:
            return self.item_list_model.item(index)
        else:
            return self.item_list_model.item(index).data(role)


class QNodeWidget(QListWidget):
    def __init__(self, parent):
        super(QNodeWidget, self).__init__(parent=parent)
        self.cookFrom = -1

        self.connect(self, QtCore.SIGNAL("doubleClicked(QModelIndex)"), self, QtCore.SLOT("ItemDoubleClicked(QModelIndex)"))

    def addItem(self, mitem, data='', index=None):
        super(QNodeWidget, self).addItem(mitem, data, index)

    def getNodes(self):
        items = self.getItems(QtCore.Qt.DisplayRole)
        if self.cookFrom == -1: return items
        evaluate = items[:self.cookFrom + 1]
        return evaluate

    def ItemDoubleClicked(self, index):
        self.changeCookIndex(self.getSelection(), False)

    def changeCookIndex(self, index, allowDeselect=False, flush=True):
        selectedItem = self.getItem(index)
        if index == self.cookFrom and allowDeselect:
            self.cookFrom = -1
            selectedItem.setBackground(QtGui.QColor(255, 255, 255))
        else:
            prevCookIndex = self.cookFrom
            self.cookFrom = index

            if prevCookIndex != -1:
                self.getItem(prevCookIndex).setBackground(QtGui.QColor(255, 255, 255))
            selectedItem.setBackground(QtGui.QColor(50, 0, 180, 150))

        self.data_changed.emit({'flush': flush})

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_P:
            self.changeCookIndex(self.getSelection())


class QOrderedListWidget(QtGui.QGroupBox):
    ''' A list widget where the order of items is important and can be
    changed by the user '''
    item_edit = QtCore.Signal(int, list)

    def __init__(self, items=[], parent=None):
        super(QOrderedListWidget, self).__init__(parent)
        self.item_count = 0
        self.createWidgets()
        self.createMenus()
        self.setTitle("Items")
        for item in items:
            self.addItem(item)

    def createWidgets(self):
        self._itemList = QtGui.QListView(self)
        self.item_list_model = QtGui.QStandardItemModel(self)
        self.item_list_model.setSortRole(QtCore.Qt.UserRole + 1)
        self._itemList.setModel(self.item_list_model)
        self.item_list_model.dataChanged.connect(self.handleDataChange)
        plsm = self._itemList.selectionModel()
        plsm.selectionChanged.connect(self._handleItemSelect)
        self._itemList.setMinimumHeight(60)

        self.toolBar = QtGui.QToolBar(self)
        self.toolBar.setOrientation(QtCore.Qt.Vertical)

        self._itemList.setSizePolicy(QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred))  # .MinimumExpanding))
        layout = QtGui.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._itemList)
        layout.addWidget(self.toolBar)
        self.setLayout(layout)

    def move(self, di=1):
        """ move the selected item up (di=-1) or down (di=1).  Updates the model(ui) and the
        ik item order """
        sm = self._itemList.selectionModel()
        try:
            selectedIndex = sm.selection().indexes()[0]
        except IndexError:  # nothing selected at all
            return

        order = selectedIndex.data(QtCore.Qt.UserRole + 1)

        # if it will be moved out of list bounds then skip
        if (order + di) < 0 or (order + di) >= self.item_count: return

        # swap the two items in the list model.
        self.item_list_model.item(order).setData(order + di, QtCore.Qt.UserRole + 1)
        self.item_list_model.item(order + di).setData(order, QtCore.Qt.UserRole + 1)

        # re-sort and notify
        self.item_list_model.sort(0)
        try:
            selection = sm.selection().indexes()[0]
        except IndexError:
            selection = -1
        self.item_edit.emit(selection, self.getItems())

    def handleDataChange(self):
        pass

    def _handleItemSelect(self, selected, deselected):
        try:
            selection = self._itemList.selectionModel().selection().indexes()[0]
        except IndexError:
            selection = -1
        self.item_edit.emit(selection, self.getItems())

    def setUserSelection(self, index):
        if self.item_count > 0: self._itemList.setCurrentIndex(self.item_list_model.item(index).index())

    def createMenus(self):
        # http://standards.freedesktop.org/icon-naming-spec/icon-naming-spec-latest.html
        up = createAction('Up', self, [functools.partial(self.move, -1)], 'Move item up', icon=QtGui.QIcon.fromTheme("go-up"))
        down = createAction('Down', self, [functools.partial(self.move, 1)], 'Move item down', icon=QtGui.QIcon.fromTheme("go-down"))
        remove = createAction('Remove', self, [functools.partial(self.removeItem)], 'Remove item', icon=QtGui.QIcon.fromTheme("edit-delete"))
        self.toolBar.addAction(up)
        self.toolBar.addAction(down)
        self.toolBar.addAction(remove)

    def removeItem(self):
        sm = self._itemList.selectionModel()
        try:
            selected_item = sm.selection().indexes()[0]
        except IndexError:  # nothing selected at all
            return
        selected_index = selected_item.data(QtCore.Qt.UserRole + 1)
        removed_row = self.item_list_model.takeRow(selected_index)
        self.item_count = self.item_count - 1
        for i in xrange(selected_index, self.item_count):
            self.item_list_model.item(i).setData(i, QtCore.Qt.UserRole + 1)
        # re-sort and notify
        self.item_list_model.sort(0)
        try:
            selection = self._itemList.selectionModel().selection().indexes()[0]
        except IndexError:
            selection = -1
        self.item_edit.emit(selection, self.getItems())

    def addItem(self, mitem, ignore=False):
        item = QtGui.QStandardItem()
        item.setData(mitem, QtCore.Qt.DisplayRole)
        # Can be used to store data linked to the name
        # item.setData(customData, QtCore.Qt.UserRole)
        item.setData(self.item_count, QtCore.Qt.UserRole + 1)
        self.item_list_model.appendRow(item)
        self.item_count = self.item_count + 1
        if not ignore:
            try:
                selection = self._itemList.selectionModel().selection().indexes()[0]
            except IndexError:
                selection = -1
            self.item_edit.emit(selection, self.getItems())

    def getItems(self):
        return [self.item_list_model.item(i).data(QtCore.Qt.DisplayRole) for i in xrange(0, self.item_count)]


class Qselect(QtGui.QComboBox):
    '''Qselect is like a QComboBox, but has correct mouse wheel behaviour (only responds to wheel when it has focus).'''

    def __init__(self, parent=None, options=None, default=None, cb=None):
        QtGui.QComboBox.__init__(self, parent)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        if options != None:
            for item in options: self.addItem(item)
            if default != None:
                self.setCurrentIndex(options.index(default))
        self.cb = cb
        self.connect(self, QtCore.SIGNAL('currentIndexChanged(int)'), self.callback)

    def callback(self, val):
        if self.cb != None: self.cb(self, val)

    def wheelEvent(self, e):
        if self.hasFocus():
            QtGui.QComboBox.wheelEvent(self, e)
        else:
            e.ignore()

    def focusInEvent(self, e):
        e.accept()
        self.setFocusPolicy(QtCore.Qt.WheelFocus)
        QtGui.QComboBox.focusInEvent(self, e)

    def focusOutEvent(self, e):
        e.accept()
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        QtGui.QComboBox.focusOutEvent(self, e)


class Qslide(QtGui.QSlider):
    '''Qslide is like a QSlider, but has correct mouse wheel behaviour (only responds to wheel when it has focus).'''

    def __init__(self, orient, parent=None):
        QtGui.QSlider.__init__(self, orient, parent)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    def wheelEvent(self, e):
        if self.hasFocus():
            QtGui.QSlider.wheelEvent(self, e)
        else:
            e.ignore()

    def focusInEvent(self, e):
        e.accept()
        self.setFocusPolicy(QtCore.Qt.WheelFocus)
        QtGui.QSlider.focusInEvent(self, e)

    def focusOutEvent(self, e):
        e.accept()
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        QtGui.QSlider.focusOutEvent(self, e)


class QslideLimitControl(QtGui.QGroupBox):
    ''' A control that contains a slider and a textbox useful for easy embedding in an app '''
    # TODO handle scrollwheel and keyboard behaviour better, currently it scrolls by the slider units
    # which can be very small
    value_changed = QtCore.Signal(float)

    def __init__(self, loval=0, hival=100, parent=None):
        QtGui.QGroupBox.__init__(self, parent)
        self.parent = parent
        self.limits = [loval, hival]
        self.digits = 2
        self.valueIsAdjusting = False
        self.createWidgets(loval, hival)
        self.createLayout()
        self.setStyleSheet("border:0;")

    def createWidgets(self, loval, hival):
        self.slider = Qslide(QtCore.Qt.Horizontal, self.parent)
        self.value = loval
        self.slider.unit = 1e-4
        self.slider.setRange(min(max(-1e9, round(self.value / self.slider.unit)), 1e9), min(max(-1e9, round(hival / self.slider.unit)), 1e9))
        self.slider.setValue(round(loval / self.slider.unit))
        self.slider.valueChanged[int].connect(self.sliderSet)
        self.display = QtGui.QLineEdit()
        # self.display.setFont(QtGui.QFont('size=8em'))
        self.display.setMaxLength(10)
        unit = 1.0  # float(np.radians(1.0)) ### TODO
        self.display.unit = unit
        self.setDisplayText(self.value / unit)
        self.display.editingFinished.connect(self.displaySet)  # this folds the values of self and di into the callback

    def createLayout(self):
        layout = QtGui.QGridLayout()
        layout.setColumnStretch(0, 5)
        layout.setColumnStretch(1, 2)
        layout.addWidget(self.slider)
        layout.addWidget(self.display)
        self.setLayout(layout)

    def sync(self, value):
        '''Update the gui to match the value; don't invoke the callback.'''
        self.value = value
        block = self.slider.blockSignals(True)  # temporarily avoid callbacks
        self.slider.setValue(round(value / self.slider.unit))
        self.slider.blockSignals(block)
        self.setDisplayText(self.slider.value() * self.slider.unit / self.display.unit)

    # ought to update the lo/hi text boxes too?

    def setValue(self, x, unit):
        '''Set the value: clamp and run the callback. Don't update the gui.'''
        self.value = x * unit
        mn, mx = self.limits
        self.value = max(mn, self.value)
        self.value = min(mx, self.value)
        self.value_changed.emit(self.value)
        return self.value

    def setLo(self, value):
        self.limits[0] = value
        self.slider.setMinimum(min(max(-1e9, round(value / self.slider.unit)), 1e9))  # actually, this might modify hi and value...
        self.setDisplayText(self.slider.value() * self.slider.unit / self.display.unit)
        self.limits[1] = self.slider.maximum() * self.slider.unit / self.display.unit
        return value

    def setHi(self, value):
        self.limits[1] = value
        self.slider.setMaximum(min(max(-1e9, round(value / self.slider.unit)), 1e9))  # actually, this might modify lo and value...
        self.setDisplayText(self.slider.value() * self.slider.unit / self.display.unit)
        self.limits[0] = self.slider.minimum() * self.slider.unit / self.display.unit
        return value

    def sliderSet(self, x):
        if self.valueIsAdjusting: return
        self.valueIsAdjusting = True
        try:
            self.setValue(self.slider.value(), self.slider.unit)
            self.slider.setValue(round(self.value / self.slider.unit))
            self.setDisplayText(self.value / self.display.unit)
        except:
            pass
        self.valueIsAdjusting = False

    def displaySet(self):
        if self.valueIsAdjusting: return
        self.valueIsAdjusting = True
        try:
            v = float(self.display.text())
            self.setValue(v, self.display.unit)
            self.slider.setValue(round(self.value / self.slider.unit))
            self.setDisplayText(self.value / self.display.unit)
        except:
            self.setDisplayText(self.slider.value() * self.slider.unit / self.display.unit)
        self.valueIsAdjusting = False

    def setDisplayText(self, value):
        self.display.setText(str(round(value, self.digits)))


# POTENTIALLY DEPRECATED
class QslideLimitValue(QtGui.QGridLayout):
    '''An object that wraps the layout and gui elements for a floating point value control with limits.'''

    def __init__(self, name, value, loval, hival, cb, cbActor, parent=None):
        QtGui.QGridLayout.__init__(self)
        self.setColumnStretch(0, 1)
        self.setColumnStretch(1, 5)
        self.setColumnStretch(2, 2)
        self.setColumnStretch(3, 1)
        self.setColumnStretch(4, 1)
        self.slider = Qslide(QtCore.Qt.Horizontal, parent)
        self.value = value
        self.slider.unit = 1e-4
        self.slider.setRange(min(max(-1e9, round(loval / self.slider.unit)), 1e9), min(max(-1e9, round(hival / self.slider.unit)), 1e9))
        self.slider.setValue(round(value / self.slider.unit))
        self.slider.valueChanged[int].connect(self.sliderSet)
        self.display = QtGui.QLineEdit()
        # self.display.setFont(QtGui.QFont('size=8em'))
        self.display.setMaxLength(10)
        unit = 1.0  # float(np.radians(1.0)) ### TODO
        self.display.unit = unit
        self.display.setText(str(value / unit))
        self.display.editingFinished.connect(self.displaySet)  # this folds the values of self and di into the callback
        self.limits = [loval, hival]
        self.lo = QtGui.QLineEdit()
        self.lo.setMaxLength(10)
        self.lo.unit = unit
        self.lo.setText(str(loval / unit))
        self.lo.editingFinished.connect(self.loSet)  # this folds the values of self and di into the callback
        self.hi = QtGui.QLineEdit()
        self.hi.setMaxLength(10)
        self.hi.unit = unit
        self.hi.setText(str(hival / unit))
        self.hi.editingFinished.connect(self.hiSet)  # this folds the values of self and di into the callback
        self.name = name
        self.label = QtGui.QLabel('<font size=8em>%s</font>' % name)
        self.addWidget(self.label)
        self.addWidget(self.slider)
        self.addWidget(self.display)
        self.addWidget(self.lo)
        self.addWidget(self.hi)
        self.cb = cb
        self.cbActor = cbActor
        self.valueIsAdjusting = False

    def sync(self, value):
        '''Update the gui to match the value; don't invoke the callback.'''
        self.value = value
        block = self.slider.blockSignals(True)  # temporarily avoid callbacks
        self.slider.setValue(round(value / self.slider.unit))
        self.slider.blockSignals(block)
        self.display.setText(str(self.slider.value() * self.slider.unit / self.display.unit))

    # ought to update the lo/hi text boxes too?

    def setValue(self, x, unit):
        '''Set the value: clamp and run the callback. Don't update the gui.'''
        self.value = x * unit
        mn, mx = self.limits
        self.value = max(mn, self.value)
        self.value = min(mx, self.value)
        print ("setValue")
        self.cb(self.cbActor, self.name, self.value)
        return self.value

    def setLo(self, x, unit):
        # do validation
        value = float(x) * unit
        self.limits[0] = value
        self.slider.setMinimum(min(max(-1e9, round(value / self.slider.unit)), 1e9))  # actually, this might modify hi and value...
        self.display.setText(str(self.slider.value() * self.slider.unit / self.display.unit))
        self.hi.setText(str(self.slider.maximum() * self.slider.unit / self.hi.unit))
        self.limits[1] = self.slider.maximum() * self.slider.unit / self.hi.unit
        return value

    def setHi(self, x, unit):
        # do validation
        value = float(x) * unit
        self.limits[1] = value
        self.slider.setMaximum(min(max(-1e9, round(value / self.slider.unit)), 1e9))  # actually, this might modify lo and value...
        self.display.setText(str(self.slider.value() * self.slider.unit / self.display.unit))
        self.lo.setText(str(self.slider.minimum() * self.slider.unit / self.lo.unit))
        self.limits[0] = self.slider.minimum() * self.slider.unit / self.lo.unit
        return value

    def sliderSet(self, x):
        if self.valueIsAdjusting: return
        self.valueIsAdjusting = True
        try:
            self.setValue(self.slider.value(), self.slider.unit)
            self.slider.setValue(round(self.value / self.slider.unit))
            self.display.setText(str(self.value / self.display.unit))
        except:
            pass
        self.valueIsAdjusting = False

    def displaySet(self):
        if self.valueIsAdjusting: return
        self.valueIsAdjusting = True
        try:
            v = float(self.display.text())
            self.setValue(v, self.display.unit)
            self.slider.setValue(round(self.value / self.slider.unit))
            self.display.setText(str(self.value / self.display.unit))
        except:
            self.display.setText(str(self.slider.value() * self.slider.unit / self.display.unit))
        self.valueIsAdjusting = False

    def loSet(self):
        if self.valueIsAdjusting: return
        self.valueIsAdjusting = True
        try:
            v = float(self.lo.text())
            value = self.setLo(v, self.lo.unit)
            self.lo.setText(str(value / self.lo.unit))
        except:
            self.lo.setText(str(self.limits[0] / self.lo.unit))
        self.valueIsAdjusting = False

    def hiSet(self):
        if self.valueIsAdjusting: return
        self.valueIsAdjusting = True
        try:
            v = float(self.hi.text())
            value = self.setHi(self.hi.text(), self.hi.unit)
            self.hi.setText(str(value / self.hi.unit))
        except:
            self.hi.setText(str(self.limits[1] / self.hi.unit))
        self.valueIsAdjusting = False


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
        self.current_value = None
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
        if self.current_value == v: return
        self.current_value = v
        self.setText(str(v))
        # if not self._dragging:
        self.valueChanged.emit(v)

    # Constrains the spin box to between two values
    def setRange(self, min=None, max=None):
        try:
            self.validator().setRange(min,max)
        # print "Valid from {} to {}".format(str(self.validator().bottom()), str(self.validator().top()))
        except:
            print ("Inputs to QintWidget.setRange() are invalid with values {} and {}".format(min, max))

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


class QLineWidget(QtGui.QLineEdit):
    valueChanged = QtCore.Signal(list)

    def __init__(self, parent=None):
        super(QLineWidget, self).__init__(parent)
        self._dragging = False
        self.current_value = None
        self.setContextMenuPolicy(QtCore.Qt.PreventContextMenu)
        self.installEventFilter(self)
        self.editingFinished.connect(functools.partial(self.handleEditingFinished))

    def handleEditingFinished(self):
        self.setValue(self.text())

    def setValue(self, v):
        v = str(v)
        if self.current_value == v: return
        self.current_value = v
        self.setText(v)
        self.valueChanged.emit(v)

    def setLocked(self, status=True):
        assert isinstance(status, bool), "Lock value is not a boolean"
        self.setReadOnly(status)

    def value(self):
        return self.text()

    def text(self):
        ret = super(QLineWidget, self).text()
        return ret


class QTextWidget(QtGui.QTextEdit):
    valueChanged = QtCore.Signal(list)

    def __init__(self, parent=None):
        super(QTextWidget, self).__init__(parent)
        self.setTabChangesFocus(True)
        self.current_value = None
        self.setFont(QtGui.QFont('Courier New', 8, QtGui.QFont.Normal, 0))
        self.resultHighlighter = PythonHighlighter(self)

    def focusOutEvent(self, event):
        super(QTextWidget, self).focusOutEvent(event)
        self.setValue(self.toPlainText())

    def setValue(self, v):
        v = str(v)
        if self.current_value == v: return
        self.current_value = v
        self.setText(v)
        self.valueChanged.emit(v)

    def value(self):
        return self.value()


class QCommandEntryWidget(QtGui.QTextEdit):
    def __init__(self, *args):
        QtGui.QTextEdit.__init__(self, *args)
        self.setAcceptRichText(False)

    def keyPressEvent(self, keyEvent):
        if (
                    (keyEvent.key() == QtCore.Qt.Key_Enter) or
                    (keyEvent.key() == QtCore.Qt.Key_Return and
                             keyEvent.modifiers() & QtCore.Qt.ControlModifier)):
            self.emit(QtCore.SIGNAL('enterPressed()'))
        elif keyEvent.key() == QtCore.Qt.Key_Tab:
            keyEvent.accept()
            self.emit(QtCore.SIGNAL('tabPressed()'))
        else:
            QtGui.QTextEdit.keyPressEvent(self, keyEvent)

class HighlightingRule:
    def __init__(self, pattern, format):
        self.pattern = pattern
        self.format = format


class PythonHighlighter(QtGui.QSyntaxHighlighter):
    """
    Python Highlighter code borrowed from
    http://wiki.python.org/moin/PyQt/Python syntax highlighting
    """
    def __init__(self, document):
        QtGui.QSyntaxHighlighter.__init__(self, document)
        self.document = document
        self.highlightingRules = []

        STYLES = {
            'keyword': self.format('blue'),
            'operator': self.format('black'),
            'brace': self.format('brown'),
            'defclass': self.format('darkBlue', 'bold'),
            'string': self.format('magenta'),
            'string2': self.format('darkMagenta'),
            'comment': self.format('darkGreen', 'italic'),
            'self': self.format('black', 'italic'),
            'numbers': self.format('purple'),
        }

        # Python keywords
        keywords = [
            'and', 'assert', 'break', 'class', 'continue', 'def',
            'del', 'elif', 'else', 'except', 'exec', 'finally',
            'for', 'from', 'global', 'if', 'import', 'in',
            'is', 'lambda', 'not', 'or', 'pass', 'print',
            'raise', 'return', 'try', 'while', 'yield',
            'None', 'True', 'False',
        ]

        # Python operators
        operators = [
            '=',
            # Comparison
            '==', '!=', '<', '<=', '>', '>=',
            # Arithmetic
            '\+', '-', '\*', '/', '//', '\%', '\*\*',
            # In-place
            '\+=', '-=', '\*=', '/=', '\%=',
            # Bitwise
            '\^', '\|', '\&', '\~', '>>', '<<',
        ]

        # Python braces
        braces = [
            '\{', '\}', '\(', '\)', '\[', '\]',
        ]

        self.tri_single = (QtCore.QRegExp("'''"), 1, STYLES['string2'])
        self.tri_double = (QtCore.QRegExp('"""'), 2, STYLES['string2'])

        rules = []

        # Keyword, operator, and brace rules
        rules += [(r'\b%s\b' % w, 0, STYLES['keyword'])
                  for w in keywords]
        rules += [(r'%s' % o, 0, STYLES['operator'])
                  for o in operators]
        rules += [(r'%s' % b, 0, STYLES['brace'])
                  for b in braces]

        # All other rules
        rules += [
            # 'self'
            (r'\bself\b', 0, STYLES['self']),

            # Double-quoted string, possibly containing escape sequences
            (r'"[^"\\]*(\\.[^"\\]*)*"', 0, STYLES['string']),
            # Single-quoted string, possibly containing escape sequences
            (r"'[^'\\]*(\\.[^'\\]*)*'", 0, STYLES['string']),

            # 'def' followed by an identifier
            (r'\bdef\b\s*(\w+)', 1, STYLES['defclass']),
            # 'class' followed by an identifier
            (r'\bclass\b\s*(\w+)', 1, STYLES['defclass']),

            # From '#' until a newline
            (r'#[^\n]*', 0, STYLES['comment']),

            # Numeric literals
            (r'\b[+-]?[0-9]+[lL]?\b', 0, STYLES['numbers']),
            (r'\b[+-]?0[xX][0-9A-Fa-f]+[lL]?\b', 0, STYLES['numbers']),
            (r'\b[+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\b', 0, STYLES['numbers']),
        ]

        # Build a QRegExp for each pattern
        self.rules = [(QtCore.QRegExp(pat), index, fmt) for (pat, index, fmt) in rules]

    def format(self, color, style=''):
        _color = QtGui.QColor()
        _color.setNamedColor(color)

        _format = QtGui.QTextCharFormat()
        _format.setForeground(_color)
        if 'bold' in style:
            _format.setFontWeight(QtGui.QFont.Bold)

        if 'italic' in style:
            _format.setFontItalic(True)

        return _format

    def highlightBlock(self, text):
        # Do other syntax formatting
        for expression, nth, format in self.rules:
            index = expression.indexIn(text, 0)

            while index >= 0:
                # We actually want the index of the nth match
                index = expression.pos(nth)
                length = len(str(expression.cap(nth)))
                self.setFormat(index, length, format)
                index = expression.indexIn(text, index + length)

            self.setCurrentBlockState(0)

        # Do multi-line strings
        in_multiline = self.match_multiline(text, *self.tri_single)
        if not in_multiline:
            in_multiline = self.match_multiline(text, *self.tri_double)

    def match_multiline(self, text, delimiter, in_state, style):
        # If inside triple-single quotes, start at 0
        if self.previousBlockState() == in_state:
            start = 0
            add = 0
        # Otherwise, look for the delimiter on this line
        else:
            start = delimiter.indexIn(text)
            # Move past this match
            add = delimiter.matchedLength()

        # As long as there's a delimiter match on this line...
        while start >= 0:
            # Look for the ending delimiter
            end = delimiter.indexIn(text, start + add)
            # Ending delimiter on this line?
            if end >= add:
                length = end - start + add + delimiter.matchedLength()
                self.setCurrentBlockState(0)
            # No; multi-line string
            else:
                self.setCurrentBlockState(in_state)
                length = len(str(text)) - start + add

            # Apply formatting
            self.setFormat(start, length, style)
            # Look for the next match
            start = delimiter.indexIn(text, start + length)

class PythonConsole(QtGui.QFrame):
    import re
    findIdentifier = re.compile(r'([a-zA-Z0-9.]*)$')

    def __init__(self, *args):
        QtGui.QFrame.__init__(self, *args)

        # Set layout and split the results field (top) and the command field (bottom)
        self.layout = QtGui.QVBoxLayout(self)
        self.splitter = QtGui.QSplitter(QtCore.Qt.Vertical, self)
        self.splitter.setOpaqueResize(1)
        self.layout.addWidget(self.splitter)

        # Initialise environment
        self.environment = {}

        # Build result widget
        self.resultWidget = QtGui.QTextEdit(self.splitter)
        self.resultWidget.setReadOnly(True)
        self.resultWidget.setFont(QtGui.QFont('Courier New', 8, QtGui.QFont.Normal, 0))
        self.resultWidget.setMinimumHeight(50)
        self.resultWidget.setTabStopWidth(20)
        self.resultHighlighter = PythonHighlighter(self.resultWidget)

        # Insert a welcome message to results
        import sys
        welcomeMsg = 'Welcome to Python Earthling\n' + sys.version + '\n\n'
        self.resultWidget.setText(welcomeMsg)

        # Build command widget
        self.commandWidget = QCommandEntryWidget(self.splitter)
        self.commandWidget.setFont(QtGui.QFont('Courier New', 8, QtGui.QFont.Normal, 0))
        self.commandWidget.setMinimumHeight(28)
        self.commandWidget.setTabStopWidth(20)
        self.commandHighlighter = PythonHighlighter(self.commandWidget)
        self.connect(self.commandWidget, QtCore.SIGNAL('enterPressed()'), self.enterCommand)
        self.connect(self.commandWidget, QtCore.SIGNAL('tabPressed()'), self.tabCommand)

        # Define text formats
        self.normalTextFormat = QtGui.QTextCharFormat()
        self.normalTextFormat.setFontWeight(QtGui.QFont.Normal)
        self.resultTextFormat = QtGui.QTextCharFormat()
        self.resultTextFormat.setForeground(QtGui.QColor(40, 40, 200))
        self.resultTextFormat.setFontWeight(QtGui.QFont.Normal)
        self.suggestionTextFormat = QtGui.QTextCharFormat()
        self.suggestionTextFormat.setForeground(QtGui.QColor(20, 160, 20))
        self.suggestionTextFormat.setFontWeight(QtGui.QFont.Normal)
        self.errorTextFormat = QtGui.QTextCharFormat()
        self.errorTextFormat.setForeground(QtGui.QColor(200, 40, 40))
        self.errorTextFormat.setFontWeight(QtGui.QFont.Normal)

        # Initialise history and set actions to scroll up and down through the history
        self.history = []
        self.historyPosition = 0
        self.previousHistoryAction = QtGui.QAction('Previous History', self)
        self.previousHistoryAction.setShortcut(QtGui.QKeySequence('Alt+Up'))
        self.nextHistoryAction = QtGui.QAction('Previous History', self)
        self.nextHistoryAction.setShortcut(QtGui.QKeySequence('Alt+Down'))
        self.previousHistoryAction.triggered.connect(self.previousHistory)
        self.nextHistoryAction.triggered.connect(self.nextHistory)
        self.commandWidget.addAction(self.previousHistoryAction)
        self.commandWidget.addAction(self.nextHistoryAction)

        self.buildMenuBar()

        # IO redirection
        self.stdout = self._Stdout(self.resultWidget)
        self.stderr = self._Stderr(self.resultWidget)

        self.runCommand('from Ops import Runtime')
        self.runCommand('runtime = Runtime.getInstance()')
        self.runCommand('interface = runtime.interface')
        self.clearHistory()

    def buildMenuBar(self):
        # Set actions and shortcuts
        cutShortcut = QtGui.QKeySequence(QtGui.QKeySequence.Cut).toString()
        copyShortcut = QtGui.QKeySequence(QtGui.QKeySequence.Copy).toString()
        pasteShortcut = QtGui.QKeySequence(QtGui.QKeySequence.Paste).toString()

        self.scriptSaveAction = QtGui.QAction('Save Script...', self)
        self.scriptLoadAction = QtGui.QAction('Load Script...', self)
        self.scriptSaveHistoryAction = QtGui.QAction('Save History...', self)
        self.scriptFetchHistoryAction = QtGui.QAction('Fetch History', self)
        self.scriptClearHistoryAction = QtGui.QAction('Clear History', self)
        self.scriptSaveAction.triggered.connect(self.saveScript)
        self.scriptLoadAction.triggered.connect(self.loadScript)
        self.scriptSaveHistoryAction.triggered.connect(self.saveHistory)
        self.scriptFetchHistoryAction.triggered.connect(self.fetchHistory)
        self.scriptClearHistoryAction.triggered.connect(self.clearHistory)

        self.editClearAction = QtGui.QAction('Clear', self)
        self.editCutAction = QtGui.QAction('Cut\t%s' % cutShortcut, self)
        self.editCopyAction = QtGui.QAction('Copy\t%s' % copyShortcut, self)
        self.editPasteAction = QtGui.QAction('Paste\t%s' % pasteShortcut, self)
        self.editClearAction.triggered.connect(self.clear)
        self.editCutAction.triggered.connect(self.cut)
        self.editCopyAction.triggered.connect(self.copy)
        self.editPasteAction.triggered.connect(self.paste)

        # Create menus
        self.menuBar = QtGui.QMenuBar(self)
        self.layout.setMenuBar(self.menuBar)

        self.scriptMenu = QtGui.QMenu('Script')
        self.menuBar.addMenu(self.scriptMenu)
        self.scriptMenu.addAction(self.scriptSaveAction)
        self.scriptMenu.addAction(self.scriptLoadAction)
        self.scriptMenu.addSeparator()
        self.scriptMenu.addAction(self.scriptSaveHistoryAction)
        self.scriptMenu.addAction(self.scriptFetchHistoryAction)
        self.scriptMenu.addAction(self.scriptClearHistoryAction)

        self.editMenu = QtGui.QMenu('Edit')
        self.menuBar.addMenu(self.editMenu)
        self.editMenu.addAction(self.editClearAction)
        self.editMenu.addAction(self.editCutAction)
        self.editMenu.addAction(self.editCopyAction)
        self.editMenu.addAction(self.editPasteAction)

    def saveScript(self):
        filename = QtGui.QFileDialog.getSaveFileName(self, 'Save Script', selectedFilter='*.py')
        if filename and filename[0]:
            filename = str(filename[0])
            file(filename, 'wt').write(self.commandWidget.toPlainText())

    def loadScript(self):
        filename = QtGui.QFileDialog.getOpenFileName(self, 'Load Script', selectedFilter='*.py')
        if filename and filename[0]:
            filename = str(filename[0])
            commands = file(filename, 'rt').read()
            self.commandWidget.clear()
            self.commandWidget.setText(commands)
            self.commandWidget.moveCursor(QtGui.QTextCursor.End, QtGui.QTextCursor.MoveAnchor)
            self.commandWidget.setFocus()

    def historyToString(self):
        return '\n'.join(self.history)

    def saveHistory(self):
        filename = QtGui.QFileDialog.getSaveFileName(self, 'Save History', selectedFilter='*.py')
        if filename and filename[0]:
            filename = str(filename[0])
            file(filename, 'wt').write(self.historyToString())

    def fetchHistory(self):
        self.commandWidget.clear()
        self.commandWidget.setText(self.historyToString())
        self.commandWidget.moveCursor(QtGui.QTextCursor.End, QtGui.QTextCursor.MoveAnchor)
        self.commandWidget.setFocus()

    def clearHistory(self):
        self.history = []

    def clear(self):
        self.resultWidget.clear()

    def cut(self):
        if (len(str(self.commandWidget.textCursor().selectedText()))):
            self.commandWidget.cut()
        else:
            self.resultWidget.cut()

    def copy(self):
        if (len(str(self.commandWidget.textCursor().selectedText()))):
            self.commandWidget.copy()
        else:
            self.resultWidget.copy()

    def paste(self):
        self.commandWidget.paste()

    def previousHistory(self):
        # Triggered using Alt+Up
        # Find the previous (decremented position) in the history and insert it in the right
        # place in the command field if available
        self.historyPosition = min(self.historyPosition + 1, len(self.history))
        if not self.historyPosition:
            self.commandWidget.clear()
        else:
            self.commandWidget.setText(self.history[-self.historyPosition])
            self.commandWidget.moveCursor(QtGui.QTextCursor.End, QtGui.QTextCursor.MoveAnchor)

    def nextHistory(self):
        # Triggered using Alt+Down
        # Find the next (incremented position) in the history and insert it in the right
        # place in the command field if available
        self.historyPosition = max(self.historyPosition - 1, 0)
        if not self.historyPosition:
            self.commandWidget.clear()
        else:
            self.commandWidget.setText(self.history[-self.historyPosition])
            self.commandWidget.moveCursor(QtGui.QTextCursor.End, QtGui.QTextCursor.MoveAnchor)

    def echoCommand(self, command, format=None):
        # Print the command to the result field
        # Set a default text format if it hasn't been supplied
        if format is None: format = self.normalTextFormat

        # Split the lines
        lines = command.splitlines()
        if lines and not lines[-1].strip():
            del lines[-1]

        self.resultWidget.moveCursor(QtGui.QTextCursor.End, QtGui.QTextCursor.MoveAnchor)
        for line in lines:
            textCursor = self.resultWidget.textCursor()
            # textCursor.insertText(">> ", format)
            textCursor.insertText("%s\n" % line, format)

    def enterCommand(self):
        # Execute the command as the user just pressed Ctrl-Enter or Ctrl-Return
        # Get the position of the text cursor and get the command from the command field
        cursor = self.commandWidget.textCursor()
        command = str(self.commandWidget.toPlainText())

        # Maya behaviour:
        # If the user has selected a particular bit of command text we keep it, otherwise
        # we clear the command field
        if cursor.hasSelection():
            start, end = cursor.selectionStart(), cursor.selectionEnd()
            command = command[start:end]
        else:
            self.commandWidget.setText('')
            self.commandWidget.textCursor().setPosition(0)

        # Echo the command to the result field and execute the command
        self.echoCommand(command, format=self.resultTextFormat)
        self.runCommand(command)

    def tabCommand(self):
        # Print command completion if the user presses the tab key
        # Create a completer
        import rlcompleter, os
        completer = rlcompleter.Completer(self.environment)

        # Get the text we just wrote and look for the nearest identifier
        index = self.commandWidget.textCursor().position()
        if index == 0:
            text = ''
        else:
            text = str(self.commandWidget.toPlainText())[:index]
            match = self.findIdentifier.search(text)
            if match: text = match.group(1)

        # Remember the length of the text we wrote for later when we want to
        # add to it
        textOriginalLength = len(text)

        # Try to find all the states (suggestions) available for the command text
        # Collect the available options to a list and build a cache to avoid repetitions
        options = []
        cache = {}
        try:
            currentState = 0
            while True:
                result = completer.complete(text, currentState)
                currentState += 1

                if result is None: break
                if cache.has_key(result): continue

                cache[result] = True
                options.append(result)

        except TypeError as e:
            print (str(e))

        if len(options) == 0: return

        # Check it's not the same as what we just wrote
        if len(options) == 1 and options[0] != text:
            self.commandWidget.insertPlainText(options[0][textOriginalLength:])
        else:
            commonPrefix = os.path.commonprefix(options)
            if len(commonPrefix) > textOriginalLength:
                self.commandWidget.insertPlainText(commonPrefix[textOriginalLength:])

            self.resultWidget.textCursor().insertText(' '.join(options) + '\n', self.suggestionTextFormat)

        self.resultWidget.moveCursor(QtGui.QTextCursor.End, QtGui.QTextCursor.MoveAnchor)

    def runCommand(self, command):
        # Add the command to history (even if it fails) and only store the last 100 entries
        self.history.append(command)
        self.history = self.history[-1000:]
        self.historyPosition = 0

        # Standard streams
        import sys
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = self.stdout
        sys.stderr = self.stderr

        # Evaluate/execute command and report results
        try:
            result = None
            try:
                self.resultWidget.moveCursor(QtGui.QTextCursor.End, QtGui.QTextCursor.MoveAnchor)
                result = eval(command, self.environment, self.environment)
            except SyntaxError:
                exec (command, self.environment)

            # Check if the evaluation was successful and if so report it in the results field
            # Add the results to the environment
            if result is not None:
                message = str(result)
                self.environment['_'] = message
                self.echoCommand(message)

        except:
            # Get the traceback information and add the formatted output to the results field
            import traceback, sys
            exceptionType, exception, tb = sys.exc_info()
            entries = traceback.extract_tb(tb)
            entries.pop(0)

            # Build and print a list containing the error report
            lines = []
            if entries:
                lines += traceback.format_list(entries)

            lines += traceback.format_exception_only(exceptionType, exception)
            for line in lines:
                self.echoCommand(line, format=self.errorTextFormat)

        finally:
            # Restore streams
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            self.resultWidget.moveCursor(QtGui.QTextCursor.End, QtGui.QTextCursor.MoveAnchor)

    class _Stdout:
        def __init__(self, resultWidget):
            self.resultWidgetRef = weakref.ref(resultWidget)
            self.stdoutTextFormat = QtGui.QTextCharFormat()
            self.stdoutTextFormat.setFontWeight(QtGui.QFont.Normal)

        def write(self, msg):
            widget = self.resultWidgetRef()
            if not widget: return

            widget.textCursor().insertText(msg, self.stdoutTextFormat)
            widget.textCursor().movePosition(QtGui.QTextCursor.End)

        def flush(self):
            pass

    class _Stderr:
        def __init__(self, resultWidget):
            self.resultWidgetRef = weakref.ref(resultWidget)
            self.errorTextFormat = QtGui.QTextCharFormat()
            self.errorTextFormat.setForeground(QtGui.QColor(200, 40, 40))
            self.errorTextFormat.setFontWeight(QtGui.QFont.Normal)

        def write(self, msg):
            widget = self.resultWidgetRef()
            if not widget: return

            widget.textCursor().insertText(msg, self.errorTextFormat)
            widget.textCursor().movePosition(QtGui.QTextCursor.End)

        def flush(self):
            pass


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
        # print ("Valid from {} to {}".format(str(self.validator().bottom()), str(self.validator().top())))
        except:
            print ("Inputs to QfloatWidget.setRange() are invalid with values {} and {}".format(min, max))

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


class QvectorWidget(QtGui.QWidget):
    valueChanged = QtCore.Signal(list)

    def __init__(self, size, parent=None):
        super(QvectorWidget, self).__init__(parent)
        self.vector = np.zeros(size, dtype=np.float32)
        layout = QtGui.QHBoxLayout()
        for vi in range(size):
            w = QfloatWidget(self)
            layout.addWidget(w)
            w.valueChanged.connect(functools.partial(self.handleValueChanged, vi), QtCore.Qt.DirectConnection)
        layout.setContentsMargins(0, 0, 0, 0)
        self.blockSignals = False
        self.setLayout(layout)

    def handleValueChanged(self, vi, v):
        self.vector[vi] = v
        if not self.blockSignals:
            self.valueChanged.emit(self.vector)

    def setValue(self, v):
        self.blockSignals = True
        self.vector[:] = v
        for vi, v in enumerate(self.vector):
            self.layout().itemAt(vi).widget().setValue(v)
        self.blockSignals = False


class QmatrixWidget(QtGui.QWidget):
    valueChanged = QtCore.Signal(list)
    '''
    this should be replaced with qdatawidget mappers and a proper qt model of the retargetting
    data structure
    '''

    def __init__(self, rows, cols, parent=None):
        super(QmatrixWidget, self).__init__(parent)
        self.rows = rows
        self.cols = cols
        self.matrix = np.zeros((rows, cols), dtype=np.float32)
        self.blockSignals = False

        layout = QtGui.QVBoxLayout()
        for ri in range(rows):
            row = QvectorWidget(cols, self)
            row.valueChanged.connect(functools.partial(self.handleValueChanged, ri), QtCore.Qt.DirectConnection)
            layout.addWidget(row)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def handleValueChanged(self, ri, v):
        self.matrix[ri, :] = v
        if not self.blockSignals:
            self.valueChanged.emit(self.matrix)

    def setValue(self, v):
        self.blockSignals = True
        self.matrix[:, :] = v.copy()
        for ri, rv in enumerate(self.matrix):
            self.layout().itemAt(ri).widget().setValue(rv)
        self.blockSignals = False


class QKeySequenceEdit(QtGui.QLineEdit):
    ''' line edit for capturing key sequences. use in a keyboard shortcut editor (although probably
    better as labels on a button rather than a line edit'''

    def __init__(self, *args):
        super(QKeySequenceEdit, self).__init__(*args)
        self.keySequence = None

    def setKeySequence(self, keySequence):
        self.keySequence = keySequence
        self.setText(self.keySequence.toString(QtGui.QKeySequence.NativeText))

    def keyPressEvent(self, event):
        if event.type() == QtCore.QEvent.KeyPress:
            key = event.key()

            if key == QtCore.Qt.Key_unknown:
                return

            # just a modifier? Ctrl, Shift, Alt, Meta.
            if key in [QtCore.Qt.Key_Control, QtCore.Qt.Key_Shift, QtCore.Qt.Key_Alt, QtCore.Qt.Key_Meta]:
                # print("Single click of special key: Ctrl, Shift, Alt or Meta")
                # print("New KeySequence:", QtGui.QKeySequence(key).toString(QtGui.QKeySequence.NativeText))
                return

            # check for a combination of user clicks
            modifiers = event.modifiers()
            if modifiers & QtCore.Qt.ShiftModifier: key += QtCore.Qt.SHIFT
            if modifiers & QtCore.Qt.ControlModifier: key += QtCore.Qt.CTRL
            if modifiers & QtCore.Qt.AltModifier: key += QtCore.Qt.ALT
            if modifiers & QtCore.Qt.MetaModifier: key += QtCore.Qt.META

            self.setKeySequence(QtGui.QKeySequence(key))
            event.accept()


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


if __name__ == '__main__':
    import sys
    # from UI import QAppCtx
    app = QtGui.QApplication(sys.argv)
    app.setStyle('plastique')

    # with QAppCtx():
    # dialog = QmatrixWidget(3,4,None)
    # def p(*x): print (x)
    # dialog.valueChanged.connect(p)
    # dialog.setValue(np.eye(3,4))
    # dialog.show()
    # def modeSelectCB(mode,val):
    # print (mode,val)
    # options = ['orig','proj','proj_freeze','synth','diff']
    # win = Qselect(options = options, default = 'diff', cb = modeSelectCB)
    # win.show()

    listWidget = QOrderedListWidget(['Hello', 'World', 'this', 'is', 'a', 'test'])
    listWidget.setStyleSheet("border:0;")
    listWidget.show()


    def testCB(*x): print ("Value is: {}".format(x))


    slideWidgetHolder = QtGui.QGroupBox()
    slideWidget = QslideLimitValue("Test Slider", 0, -180, 180, testCB, "Slider")
    # layout = QtGui.QHBoxLayout()
    # layout.setContentsMargins(0,0,0,0)
    # layout.addWidget(slideWidget.slider)
    # layout.addWidget(slideWidget.display)
    slideWidgetHolder.setLayout(slideWidget)
    slideWidgetHolder.show()

    slideWidget2 = QslideLimitControl()
    slideWidget2.show()

    app.connect(app, QtCore.SIGNAL('lastWindowClosed()'), app.quit)
    sys.exit(app.exec_())
