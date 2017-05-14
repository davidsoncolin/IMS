import time
import zmq
from zmq import Poller
import IO

from multiprocessing import Process, Pool


# Note: Add push/pull to distribute processing to workers (e.g. within Ops)
tcpPrefix = 'tcp://'
ipcPrefix = 'ipc://'

# Address handling functions such as tcp, ipc, pgm, etc.
def tcp(host, port):
	return '%s%s:%d' % (tcpPrefix, host, port)

def tcpBind(port):
	return '%s%s:%d' % (tcpPrefix, '*', port)

def ipc(paths):
	return ipcPrefix + '/'.join(paths)

def splitAddress(address):
	if address.startswith(tcpPrefix):
		tcpSplit = address.split(tcpPrefix)
		tcpAddress = tcpSplit[1]
		if tcpAddress.rindex(':'):
			host, port = tcpAddress.split(':')
			return host, int(port)

		return tcpAddress, None

	elif address.startswith(ipcPrefix):
		ipcSplit = address.split(ipcPrefix)
		return ipcSplit

	return address


class Publisher:
	def __init__(self, address, bind=True):
		self.socket = zmq.Context().socket(zmq.PUB)
		self.socket.set_hwm(1)

		if bind:
			self.socket.bind(address)
		else:
			self.socket.connect(address)

	def publish(self, message, header='default'):
		if message is None: return

		try:
			message = IO.wrap(message)
			publishMessage = '%s %s' % (header, message)
			self.socket.send(publishMessage)
		except Exception as e:
			print 'Could not publish data:', e


class PeriodicPublisher:
	def __init__(self, address, publishCb, topic, fps=24., bind=True):
		self.publisher = Publisher(address, bind)
		self.publishCb = publishCb
		self.topic = topic
		self.fps = fps
		self.dt = 1 / self.fps
		self.publishTime = time.time()

	def publish(self):
		if time.time() > self.publishTime + self.dt:
			message = self.publishCb()
			if message is not None:
				self.publisher.publish(message, self.topic)

			self.publishTime = time.time()

	def getTimeout(self):
		return self.dt


class Requester:
	def __init__(self, timeout=-1, poolSize=4):
		self.timeout = timeout
		self.socket = None
		self.numConnections = 0
		self.poller = Poller()

	def addServer(self, address, bind=False, linger=-1):
		if self.socket is None:
			self.socket = zmq.Context().socket(zmq.REQ)
			# self.socket.setsockopt(zmq.LINGER, linger)
			self.poller.register(self.socket, zmq.POLLIN)

		if bind:
			self.socket.bind(address)
		else:
			self.socket.connect(address)

		self.numConnections += 1

	def numConnections(self):
		return self.numConnections

	def request(self, message):
		if message is None or self.socket is None: return None
		message = IO.wrap(message)

		self.socket.send(message)
		socks = dict(self.poller.poll(self.timeout))
		if socks:
			replyMessage = self.socket.recv()

			reply, _ = IO.unwrap(replyMessage)
			return reply

		return None

	def requestAllServers(self, message):
		replies = []
		for n in range(self.numConnections):
			reply = self.request(message)
			replies.append(reply)

		return replies

	def requestAllServersCb(self, message, replyCb):
		for n in range(self.numConnections):
			reply = self.request(message)
			replyCb(reply)

	def requestAllServersPool(self, message, poolSize=0):
		if poolSize < 1: poolSize = self.numConnections
		pool = Pool(processes=poolSize)
		replies = pool.map(self.request, [message for n in range(self.numConnections)])
		return replies


class Listener:
	def __init__(self, timeout=-1):
		self.controllerSocket = None
		self.timecodeSocket = None
		self.requestSocket = None
		self.subscribers = {}
		self.subscriberCallbacks = {}
		self.subscriberTopicFilters = {}

		self.periodicPublisher = None

		self.commandCb = None
		self.requestCb = None

		self.timeout = timeout
		self.poller = Poller()
		self.running = False
		self.pollCounter = 0

	def connectController(self, address, commandCb=None):
		self.controllerSocket = zmq.Context().socket(zmq.PAIR)
		self.controllerSocket.bind(address)
		self.notifyConnected(address)
		self.poller.register(self.controllerSocket, zmq.POLLIN)
		self.commandCb = commandCb

	def sendController(self, message):
		wrappedMsg = IO.wrap(message)
		self.controllerSocket.send(wrappedMsg)

	def notifyConnected(self, address):
		msg = {'status': 'connected', 'address': address}
		self.sendController(msg)

	def notifyReady(self, address):
		msg = {'status': 'ready', 'address': address}
		self.sendController(msg)

	def listenToRequests(self, address, requestCb):
		self.requestCb = requestCb
		self.requestSocket = zmq.Context().socket(zmq.REP)
		self.requestSocket.bind(address)
		self.poller.register(self.requestSocket, zmq.POLLIN)

	def subscribeTimecode(self, address, timecodeCb):
		self.addSubscription(address, timecodeCb, label='_TC_', filters=['TIMECODE'])

	def addSubscription(self, address, subCb, label='default', filters=['default']):
		if label not in self.subscribers:
			self.subscribers[label] = zmq.Context().socket(zmq.SUB)
			self.subscribers[label].set_hwm(1)
			#self.subscribers[label].setsockopt(zmq.RCVHWM, 1)
			self.subscriberCallbacks[label] = subCb
			self.subscriberTopicFilters[label] = filters
			self.poller.register(self.subscribers[label], zmq.POLLIN)

		print "Subscribing", label, "to:", address
		self.subscribers[label].connect(address)

		if filters:
			for filter in filters:
				self.subscribers[label].setsockopt(zmq.SUBSCRIBE, filter)
		else:
			self.subscribers[label].setsockopt(zmq.SUBSCRIBE, '')

	def setPeriodicPublisher(self, publisher, overrideTimeout=True):
		self.periodicPublisher = publisher
		if overrideTimeout: self.timeout = publisher.getTimeout()

	def start(self):
		self.running = True
		self._main()

	def stop(self):
		self.running = False

	def pollRegistered(self, expected=-1, timeout=-1):
		if expected == -1: expected = len(self.poller.sockets)
		self.pollCounter = expected
		while self.pollCounter > 0:
			self.poll(timeout)

	def poll(self, timeout=-1):
		if timeout == -1: timeout = self.timeout
		socks = dict(self.poller.poll(timeout))

		if self.controllerSocket in socks and socks[self.controllerSocket] == zmq.POLLIN:
			command = self.controllerSocket.recv()
			command, _ = IO.unwrap(command)
			self.pollCounter -= 1

			# Allow the user of this module to react to this command and to modify its contents
			if self.commandCb:
				modifiedCommand = self.commandCb(command)
				if modifiedCommand is not None: command = modifiedCommand

			if 'action' not in command: return
			action = command['action']
			if action == 'exit':
				self.running = False

		if self.requestSocket in socks and socks[self.requestSocket] == zmq.POLLIN:
			self.pollCounter -= 1
			message = self.requestSocket.recv()
			message, _ = IO.unwrap(message)
			replyMessage = self.requestCb(message)
			if replyMessage is None: return
			wrappedReply = IO.wrap(replyMessage)
			self.requestSocket.send(wrappedReply)

		if socks:
			for label in self.subscribers.keys():
				if self.subscribers[label] in socks and socks[self.subscribers[label]] == zmq.POLLIN:
					self.pollCounter -= 1
					data = self.subscribers[label].recv()

					# In case there is not space in the data, use find. Find returns -1 when the string isn't found
					sepIndex = data.find(' ')
					if sepIndex > -1 and not data[:4] == 'TIMS' and data[0:sepIndex] in self.subscriberTopicFilters[label]:
						topic, content = data[0:sepIndex], data[sepIndex + 1:]
					else:
						topic, content = '', data

					content, _ = IO.unwrap(content)
					if content is not None:
						self.subscriberCallbacks[label](content)

		if self.periodicPublisher:
			self.periodicPublisher.publish()

	def _main(self):
		while self.running:
			self.poll()


class ControllerServer:
	def __init__(self):
		self.pairSockets = []
		self.poller = Poller()

	def connect(self, address):
		socket = zmq.Context().socket(zmq.PAIR)
		socket.connect(address)
		self.poller.register(socket, zmq.POLLIN)
		self.pairSockets.append(socket)
		return socket

	def send(self, message, socket=None):
		message = IO.wrap(message)
		if socket:
			socket.send(message)
		else:
			for socket in self.pairSockets:
				socket.send(message)

	def sendAction(self, action, socket=None):
		message = {'action': action}
		self.send(message, socket)

	def receive(self, receiveCb, timeout=-1):
		socks = dict(self.poller.poll(timeout))
		if socks:
			for si, socket in enumerate(self.pairSockets):
				if self.pairSockets[si] in socks and socks[self.pairSockets[si]] == zmq.POLLIN:
					msg = socket.recv()
					msg, _ = IO.unwrap(msg)
					receiveCb(si, msg)
