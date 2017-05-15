import zmq
import IO

"""
Functionality for client connecting to ReframeServer
Actions are subscribe and request
"""


class ReframeClient:
	def __init__(self, host='localhost', portReq=None, portSub=None, subscriptionFilter='TIMS'):
		"""
		:string hostName: machine name on which server is running
		:int port: port number on server to connect to
		"""
		self.host, self.portSub, self.portReq = host, portSub, portReq

		if self.portSub:
			self.client = zmq.Context().socket(zmq.SUB)
			if isinstance(portSub, list):
				for p in portSub:
					print "Connecting to port:", p
					self.client.connect('tcp://%s:%d' % (self.host, p))
			else:
				self.client.connect('tcp://%s:%d' % (self.host, portSub))

			self.client.setsockopt(zmq.SUBSCRIBE, subscriptionFilter)

		if self.portReq:
			self.req = zmq.Context().socket(zmq.REQ)
			if isinstance(portReq, list):
				for p in portReq:
					print "Connecting to port:", p
					self.req.connect('tcp://%s:%d' % (self.host, p))
			else:
				self.req.connect('tcp://%s:%d' % (self.host, self.portReq))

		self.state = None
		self.stateHash = None

	def getSocket(self):
		return self.client

	def request(self, s):
		if not self.req: return

		self.req.send(s)
		return self.req.recv()

	def subscribe(self):
		if not self.client: return

		data = None
		# while self.client.poll(timeout=0):
		data = self.client.recv() # trash the stream until the last frame

		frame = None
		if data != None:
			frame, _ = IO.unwrap(data)
			assert (_ == '')

		#if type(frame) == dict:
			#frameHash = frame['hash']
			#if frameHash != self.stateHash:
				#print 'rehashing!'
				#data = self.request('self.state[%d]' % frameHash)
				#if data == 'fail':
					#print 'for some reason, couldn\'t get state for hash ', frameHash
					#self.state = None
					#self.stateHash = None
				#else:
					#frameState, _ = IO.unwrap(data)
					#self.state = frameState
					#self.stateHash = frameState['hash']

		return frame
