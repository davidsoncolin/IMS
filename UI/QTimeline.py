#!/usr/bin/env python
import functools
from PySide import QtCore, QtGui
import GCore
from UI import createAction
from . import QApp
#: time in milliseconds between each frame step (ie the playback rate). Default is 59.94fps,
#: or (1001/60) milliseconds per frame

TIME_STEP = 1001 / 60.
DEFAULT_FRAME_STEP = 1
FRAME_STEPS = [1,2,3,4,8,16]
DEFAULT_RATE = 59.94
RATES = [1.,6.,11.988,12.,14.985,15.,23.976,24.,24.975,25.,29.97,30.,48.,49.95,50.,59.94,60.,95.9,96.,99.9,100.,119.88,120.,1000.] # over 60 may not work because of qt refresh rate

MIN_SAMPLE = -2e9  # a few hours at >100 samples per second
MAX_SAMPLE = 2e9


class QTimeline(QtGui.QWidget):
	'''time control widget.  features:

	* draggable time slider
	* playback toggle button
	* editable current time field

	TODO:: marked sub-range (queryable)
	TODO:: loop playback or play once option.

	* option to keep to playback rate (and jump some frames)
	* option to play every frame no matter the speed'''

	#: Qt Signal, emits the new frame number (int) when current frame changes
	playing = QtCore.Signal(int)

	#: Qt signal emitted when the rate changes. the value is the time step in milliseconds
	rateChanged = QtCore.Signal(float)

	#: Qt Signal emitted when frame changed, but not due to playback
	frameChanged = QtCore.Signal()

	def __init__(self, parent=None):
		super(QTimeline, self).__init__(parent)

		self._playing = False
		self._lo = self._hi = None

		# not 'public' because setting the frame requires ui updates etc (use :property:`frame`)
		self._frame = 0

		#: frames per second
		self._fps = DEFAULT_RATE
		self._lastFewTimes = []

		#: During playback, frame will be incremented by this number of frames each :data:`TIME_STEP`
		#: this has no effect on stepping the time slider or current frame field with the keyboard,
		#: which always step by 1 frame
		self._frameStep = DEFAULT_FRAME_STEP

		#: callback function to be executed every time frame changes
		self.cb = None

		# widgets
		# editable display for start sample
		self.startSpinBox = QtGui.QSpinBox(self)
		self.startSpinBox.setStatusTip("Set the start sample")
		self.startSpinBox.setMinimum(MIN_SAMPLE)
		self.startSpinBox.setMaximum(MAX_SAMPLE)
		self.startSpinBox.setButtonSymbols(QtGui.QAbstractSpinBox.NoButtons)
		self.startSpinBox.setKeyboardTracking(False)  # don't emit 3 times when typing 100

		# editable display for end sample
		self.endSpinBox = QtGui.QSpinBox(self)
		self.endSpinBox.setStatusTip("Set the end sample")
		self.endSpinBox.setMinimum(MIN_SAMPLE)
		self.endSpinBox.setMaximum(MAX_SAMPLE)
		self.endSpinBox.setButtonSymbols(QtGui.QAbstractSpinBox.NoButtons)
		self.endSpinBox.setKeyboardTracking(False)

		# editable display for current frame number
		self.currentSpinBox = QtGui.QSpinBox(self)
		self.currentSpinBox.setWrapping(True)
		self.currentSpinBox.setKeyboardTracking(False)
		self.currentSpinBox.setMinimum(MIN_SAMPLE)
		self.currentSpinBox.setMaximum(MAX_SAMPLE)
		self.currentSpinBox.setStatusTip("The current sample")

		self.frameSlider = QtGui.QSlider(self)
		self.frameSlider.setOrientation(QtCore.Qt.Horizontal)

		playIcon = QtGui.QIcon()
		playIcon.addPixmap(self.style().standardIcon(QtGui.QStyle.SP_MediaPlay).pixmap(32, 32), state=QtGui.QIcon.State.Off)
		playIcon.addPixmap(self.style().standardIcon(QtGui.QStyle.SP_MediaStop).pixmap(32, 32), state=QtGui.QIcon.State.On)
		play = createAction('Play', self, [self.playToggle], 'Start/stop playback', checkable=True, checked=False, icon=playIcon, globalContext=True)
		nextFrame = createAction('Step Forward', self, [self.gotoNextFrame,self.frameChanged.emit], 'Step forward one timestep', icon=self.style().standardIcon(QtGui.QStyle.SP_MediaSkipForward), globalContext=True)
		previousFrame = createAction('Step Backward', self, [functools.partial(self.gotoNextFrame, -1),self.frameChanged.emit], 'Step backward one timestep', icon=self.style().standardIcon(QtGui.QStyle.SP_MediaSkipBackward), globalContext=True)
		forwardSec = createAction('Forward one second', self, [self.gotoNextSecond,self.frameChanged.emit], 'Step forward one timestep', icon=self.style().standardIcon(QtGui.QStyle.SP_MediaSkipForward), globalContext=True)
		backwardSec = createAction('Backward one second', self, [functools.partial(self.gotoNextSecond, -1),self.frameChanged.emit], 'Step backward one timestep', icon=self.style().standardIcon(QtGui.QStyle.SP_MediaSkipBackward), globalContext=True)
		goStart = createAction('Go to range start', self, [lambda: self.__setattr__('frame', self._lo),self.frameChanged.emit], 'Jump to start of playback range', icon=self.style().standardIcon(QtGui.QStyle.SP_MediaSeekBackward), globalContext=True)
		goEnd = createAction('Go to range end', self, [lambda: self.__setattr__('frame', self._hi),self.frameChanged.emit], 'Jump to end of playback range', icon=self.style().standardIcon(QtGui.QStyle.SP_MediaSeekForward), globalContext=True)
		playControls = QtGui.QToolBar(self)
		playControls.addAction(goStart)
		playControls.addAction(previousFrame)
		playControls.addAction(play)
		playControls.addAction(nextFrame)
		playControls.addAction(goEnd)

		# rate selector
		self.rateComboBox = QtGui.QComboBox(self)
		self.rateComboBox.setStatusTip("Set the playback rate")
		self.rateModel = QtGui.QStringListModel(sorted(['%g fps'%r for r in RATES], key=GCore.humanKey), self)
		self.rateComboBox.setModel(self.rateModel)
		self.rateComboBox.setCurrentIndex(self.rateModel.stringList().index('%g fps' % DEFAULT_RATE))

		# step selector
		self.stepComboBox = QtGui.QComboBox(self)
		self.stepComboBox.setStatusTip("Set the frame step")
		self.stepModel = QtGui.QStringListModel(sorted(['x%d' % x for x in FRAME_STEPS], key=GCore.humanKey), self)
		self.stepComboBox.setModel(self.stepModel)
		self.stepComboBox.setCurrentIndex(self.stepModel.stringList().index('x%d'%DEFAULT_FRAME_STEP))

		# buttons turned off because they're too fiddly & we'll have separate buttons to do that
		self.currentSpinBox.setButtonSymbols(QtGui.QAbstractSpinBox.NoButtons)

		self.timer = QtCore.QTimer(self)

		# initialise default ranges / values
		self.setRange(0, 100)

		# layout
		layout = QtGui.QHBoxLayout()
		layout.addWidget(self.startSpinBox)
		layout.addWidget(self.frameSlider)
		layout.addWidget(self.endSpinBox)
		layout.addWidget(self.currentSpinBox)
		layout.addWidget(playControls)
		layout.addWidget(self.rateComboBox)
		layout.addWidget(self.stepComboBox)
		layout.setContentsMargins(0, 0, 0, 0)
		self.setLayout(layout)

		# no need for this to expand vertically, and this also means there't no resize handle when
		# used in a dock widget
		self.setSizePolicy(QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Fixed))

		# connections
		self.timer.timeout.connect(self.timerTimeout)
		self.timer.setSingleShot(True)
		self.frameSlider.valueChanged.connect(self._seekFrame)
		self.frameSlider.valueChanged.connect(self.currentSpinBox.setValue)
		self.frameSlider.sliderReleased.connect(self.frameChanged.emit)
		self.currentSpinBox.valueChanged.connect(lambda: self.setRange(min(self._lo, self.currentSpinBox.value()), max(self._hi, self.currentSpinBox.value())))
		self.currentSpinBox.valueChanged.connect(self.frameSlider.setValue)
		self.currentSpinBox.editingFinished.connect(self.frameChanged.emit)
		self.startSpinBox.valueChanged.connect(lambda: self.setRange(self.startSpinBox.value(), self._hi))
		self.endSpinBox.valueChanged.connect(lambda: self.setRange(self._lo, self.endSpinBox.value()))
		self.stepComboBox.currentIndexChanged.connect(self._setStepFromCombo)
		self.rateComboBox.currentIndexChanged.connect(self._setRateFromCombo)

	@property
	def isPlaying(self):
		return self._playing

	@property
	def frame(self):
		''' yields the current frame

		if you set the current frame using this property:

		* the ui will  update as expected
		* if the frameNumber is out of the current lo/hi range then the frame number will be clipped

		.. note::
			* currentFrame signal is NOT emitted if the current frame is unchanged
			* frame callback function is NOT executed if the current frame is unchanged

		:param int frameNumber: the desired go-to frame.'''

		# small speed hit by having a this as a property, but it allows the use of a setter, which
		# is required in order to update ui components if you want to call QTimeline.frame = X and
		# have it behave correctly
		return self._frame

	@frame.setter
	def frame(self, frameNumber):
		''' set the current frame

		:param int frameNumber: the desired go-to frame.'''
		# everything else is handled from the frameSlider.valueChanged signal (assuming that
		# frameNumber != self._frame)
		self.frameSlider.setValue(frameNumber)

	@property
	def lo(self):
		''' yields the start frame. setting this property will adjust the range of the time
		slider.  the end frame number will be preserved unless it would be less than the new min
		in which case it is set to min+1'''
		return self._lo

	@lo.setter
	def lo(self, frameNumber):
		''' set the minimum time slider frame '''
		self.setRange(frameNumber, min(frameNumber + 1, self._hi))

	@property
	def hi(self):
		''' yields the end frame. setting this property will adjust the range of the time
		slider.  the start frame number will be preserved unless it would be greater than the new
		end value in which case it is set to end-1'''
		return self._hi

	@hi.setter
	def hi(self, frameNumber):
		''' set the minimum time slider frame '''
		self.setRange(min(frameNumber - 1, self._lo), frameNumber)

	@property
	def frameStep(self):
		''' yields the current frameStep.  setting this will update the frame step combo and add the
		value to the combo list if it isn't in the defaults '''
		return self._frameStep

	@frameStep.setter
	def frameStep(self, value):
		''' set the current frame step '''
		if self._frameStep==value or value < 1: return
		self._frameStep = value
		self.frameSlider.setSingleStep(self._frameStep)
		if value not in FRAME_STEPS:
			FRAME_STEPS.append(value)
			FRAME_STEPS.sort()
			self.stepModel.setStringList(sorted(['x%d' % x for x in FRAME_STEPS], key=GCore.humanKey))
		self.stepComboBox.setCurrentIndex(self.stepModel.stringList().index('x%s' % value))

	@property
	def fps(self):
		''' yields the current fps.'''
		return self._fps

	@fps.setter
	def fps(self, value):
		''' set the current fps. update the fps combo and add the value to the combo list if it isn't in the defaults'''
		if self._fps==value: return
		self._fps = value
		self.frameSlider.setSingleStep(self._fps)
		if value not in RATES:
			RATES.append(value)
			RATES.sort()
			self.rateModel.setStringList(sorted(['%g fps' % x for x in RATES], key=GCore.humanKey))
		self.rateComboBox.setCurrentIndex(self.rateModel.stringList().index('%g fps' % value))


	def setRange(self, lo, hi, frameStep=None, fps=None):
		''' set the playback min/max range

		:param int lo: the range start frame
		:param int hi: the range end frame'''
		self._lo, self._hi = lo, hi
		self.frameSlider.setRange(self._lo, self._hi)
		self.startSpinBox.setValue(self._lo)
		self.endSpinBox.setValue(self._hi)
		self._frame = min(max(self._frame, self._lo), self._hi)
		if frameStep is not None: self.frameStep = frameStep
		if fps is not None: self.fps = fps

	@QtCore.Slot(int)
	def _seekFrame(self, frame):
		''' go to a frame (slot called when frameslider changes)  don't call this directly or the
		time slider will not update

		:param int frame: the frame to set to'''
		self.playing.emit(frame)
		self._frame = frame
		# run the callback if installed
		if self.cb: self.cb(self._frame) # CBD currently the callback is responsible for redrawing the GL view (this is not ideal)
		else: QApp.app.updateGL() # CBD if no callback is installed then we have to update manually

	@QtCore.Slot()
	def timerTimeout(self):
		import time
		t0 = time.time()
		self._lastFewTimes.append(t0)
		self.gotoNextFrame(direction=1)
		t1 = time.time()
		sleep_time = max(0,min((self._lastFewTimes[0] + len(self._lastFewTimes) / float(self._fps) - t1)*1000.0, 1000./self._fps+10))
		if len(self._lastFewTimes) > 5: self._lastFewTimes.pop(0)
		self.timer.start(int(sleep_time+0.5))

	@QtCore.Slot()
	def gotoNextFrame(self, direction=1):
		''' increment frame by frameStep and set the slider value to new frame.  set the frame
		number back to the start if it exceeds the max. this is called by the timer during
		playback'''
		frame = self._frame + direction * self._frameStep
		if frame > self._hi:
			frame = self._lo
		elif frame < self._lo:
			frame = self._hi
		# seekFrame will be called (self.frame will be set) via the connection from the frameSlider
		self.frameSlider.setValue(frame)

	@QtCore.Slot()
	def gotoNextSecond(self, direction=1):
		''' increment frame by frameStep and set the slider value to new frame.  set the frame
		number back to the start if it exceeds the max. this is called by the timer during
		playback'''
		frame = self._frame + self._frameStep * direction * int(self._fps + 0.5)
		if frame > self._hi:
			frame = self._lo
		elif frame < self._lo:
			frame = self._hi
		# seekFrame will be called (self.frame will be set) via the connection from the frameSlider
		self.frameSlider.setValue(frame)

	@QtCore.Slot()
	def _setRateFromCombo(self, index):
		''' sets the time step when the rate combo is changed. looks up the actual time step value
		from the :data:`RATES` dict based on the name of the selected rate in the combo '''
		self._fps = RATES[index]
		self._lastFewTimes = []
		if self._playing:
			self.timer.stop()
			self.timer.start(int(1000./self._fps))
		self.rateChanged.emit(1000./self._fps)

	@QtCore.Slot()
	def _setStepFromCombo(self, index):
		''' sets the frame step when the rate combo is changed. looks up the actual frame step
		value from the :data:`FRAME_STEPS` dict based on the name of the selected rate in the
		combo
		# TODO: discuss.. should the current frame be changed to the nearest multiple of the
		step? i think that gives more predictable playback.  it's complicated by the time slider
		being draggable to values that are impossible to get to by single stepping.  the slider
		would need modifying to support dragging with snapping.
		'''
		self._frameStep = FRAME_STEPS[index]
		self.frameSlider.setSingleStep(self._frameStep)

	@QtCore.Slot()
	def playToggle(self):
		''' toggles playback state '''
		self._playing = not self._playing
		if self._playing:
			self.timer.start(int(1000./self._fps))
			self._lastFewTimes = []
		else:
			self.timer.stop()
			self.frameChanged.emit()

	def refresh(self):
		self._seekFrame(self.frame)

def printFrame(frame):
	print (frame)

if __name__ == '__main__':
	import sys
	app = QtGui.QApplication(sys.argv)
	app.setStyle('plastique')
	win = QTimeline()
	win.playing.connect(printFrame)

	# win.frame = 110  # this will actually set to 100 because range is only 0-100
	# assert(win.frame == 100)
	# win.frame = 50
	win.show()
	# win.setRange(0, 10000)
	# win.frameStep = 3
	sys.exit(app.exec_())
