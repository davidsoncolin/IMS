import Op
from IO import PyTISStream

class StreamSkeleton(Op.Op):
	def __init__(self, name='/StreamSkeleton', source='', skelName='', scale=1.0):
		self.fields = [
			('name', 'name', 'name', 'string', name, {}),
			('Source', 'Source', 'Source', 'string', source, {}),
			('SkelName', 'SkelName', 'SkelName', 'string', skelName, {}),
			('Scale', 'Scale', 'Scale', 'float', scale, {})
		]

		super(self.__class__, self).__init__(name, self.fields)

		print "Init!"
		self.client = None

	def cook(self, location, interface, attrs):
		if self.client is None:
			self.client = PyTISStream.initServer()

		sourceLocation = attrs['Source']
		if not interface.hasAttr("skelDict", atLocation=sourceLocation):
			return

		skelDict = interface.attr("skelDict", atLocation=sourceLocation)
		if not self.client.IsConnected():
			self.client.Start('', 6500)
		self.client.WriteAll(PyTISStream.serialiseSkeleton(attrs['SkelName'], skelDict, scale=attrs['Scale']))


# Register Ops
import Registry
Registry.registerOp('Stream Skeleton', StreamSkeleton)
