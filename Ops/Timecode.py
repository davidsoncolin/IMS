import math
from datetime import datetime, timedelta

from Ops import Op

def splitTc(tc):
	hrs, mins, secs, frames = tc.split(":")
	return hrs, mins, secs, frames

def TCFtoInt(tc, fps):
	hrs, mins, secs, frames = splitTc(tc)
	fps = math.ceil(float(fps))
	if hrs != "" and mins != "" and secs != "" and frames != "" and fps != "":
		mins = (int(hrs) * 60) + int(mins)
		secs = (int(mins) * 60) + int(secs)
		frames = (int(secs) * (int(fps))) + int(frames)
		return frames

	return None

def TCSub(tc1, tc2, fps):
	""" tc1 minus tc2 == Result """
	tc1hr, tc1min, tc1sec, tc1frame = splitTc(tc1)
	tc2hr, tc2min, tc2sec, tc2frame = splitTc(tc2)

	tc1Delta = timedelta(hours=int(tc1hr), minutes=int(tc1min), seconds=int(tc1sec))
	tc2Delta = timedelta(hours=int(tc2hr), minutes=int(tc2min), seconds=int(tc2sec))

	tcDate = datetime.fromtimestamp(int(tc1Delta.total_seconds()) - int(tc2Delta.total_seconds()))

	totalFrames = int(tc1frame) - int(tc2frame)

	if totalFrames < 0:
		totalFrames += fps
		tcDate = tcDate - timedelta(seconds=1)

	return "%s:%02d" % (tcDate.strftime("%H:%M:%S"), int(totalFrames))


class SetFrameRange(Op.Op):
	def __init__(self, name='/SetFrameRange', locations=''):
		fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'locations', 'Location containing a frame range', 'string', locations, {}),
		]

		super(self.__class__, self).__init__(name, fields)

	def cook(self, location, interface, attrs):
		if not interface.opParamsDirty(): return

		# Get frame range from location and set if we find any
		frameRange = interface.attr('frameRange')
		if not frameRange: return
		if len(frameRange) != 2: return

		interface.setFrameRange(frameRange[0], frameRange[1])
		interface.updateTimeline()
		self.logger.info('Set range to [%d, %d]' % (frameRange[0], frameRange[1]))


# Register Ops
import Registry
Registry.registerOp('Set Frame Range', SetFrameRange)

