from threading import Thread, Lock
from time import sleep, time
import socket

class SocketServer(object):
	def __init__(self):
		self._thread = None
		self._server_socket = None
		self._last_msg = (None,None)
		self._connections = {}
		self._connections_lock = Lock()

	def __del__(self):
		print "Deleting!"
		self.Stop()
		
	def Start(self, address, port):
		'''
		Start the thread to handle the connections
		'''

		if self._server_socket is not None or self._thread is not None:
			self.Stop()

		try:
			print "Creating Server on {}:{}".format(address, port)
			self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			self._server_socket.setblocking(0)  # Non Blocking causes a send to error if it can't offload the data straight away
			self._server_socket.bind((address, port))
			self._server_socket.listen(1)
		except socket.error:
			print "Failed to create Server"
			if self._server_socket is not None:
				self._server_socket.close()
				self._server_socket = None
			return False
		self._last_msg = (time(),'')
		self._thread = Thread(target=self._Thread_Loop)
		self._thread.start()
		return True

	def _Thread_Loop(self):
		try:
			while self._server_socket is not None:
				try:
					client_socket,addr = self._server_socket.accept()
					client_socket.setblocking(0)
					print "New Connection from {}".format(addr)
					with self._connections_lock: self._connections[addr] = client_socket
				except socket.error:
					pass
				tod,data = self._last_msg
				if tod is not None:
					if time() > tod + 5.0: break # timeout after 5 seconds
					# close any sockets that can't be transmitted on
					deads = []
					with self._connections_lock:
						for addr,client_socket in self._connections.items():
							try:
								client_socket.sendall(data)
							except socket.error as er:
								try:
									print "Removing {address}".format(address=addr)
									print er.message
									# Shutdown and remove the failed connection from the list of connections.
									client_socket = self._connections.pop(addr)
									client_socket.close()
								except:
									print '_Thread_Loop Exception'
				sleep(0.01) # maximum fps = 100
		except Exception as er:
			print er.message
		finally:
			print "Finally"
			with self._connections_lock: 
				while self._connections:
					addr,client_socket = self._connections.popitem()
					try:
						print "Removing {address}".format(address=addr)
						client_socket.close()
					except: 
						print '_Thread_Loop Exception 2'
						pass
			ss = self._server_socket
			self._server_socket = None
			if ss is not None: ss.close()
					

	def WriteAll(self, data):
		'''
		Loop through the current connections and try to send data to them.
		'''
		with self._connections_lock: clients = self._connections.keys()
		if self._server_socket is not None:
			self._last_msg = (time(),data)
			return True,clients
		return False,clients

	def Stop(self):
		ss,th = self._server_socket,self._thread
		self._server_socket,self._thread = None,None
		if ss is not None: ss.close()	
		if th is not None: th.join()

	def IsConnected(self):
		return self._server_socket is not None


def EXAMPLE_SERVER(hostname=socket.gethostname(), port=8877, its=100, sleepTime=0.1):

	print "EXAMPLE SERVER: Starting the connection Manager"
	myServer = SocketServer()
	myServer.Start(hostname, port)
	for i in xrange(its):
		print "EXAMPLE SERVER: Writing some Data"
		myServer.WriteAll("hello")
		sleep(sleepTime)
	print "EXAMPLE SERVER: Stopping the connection Manager"
	myServer.Stop()


def EXAMPLE_CLIENT(hostname=socket.gethostname(), port=8877, its=100, sleepTime=0.1):
	'''
	Randomly start and destroy clients to the server to check how the server handles connects and disconnects
	'''

	from random import random, randrange

	clients = []
	for i in xrange(its):
		if random() < 0.5:
			print "NEW CLIENT"
			newClient = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			try:
				newClient.connect((hostname, port))
				# Connect succeeded to make a note of it
				clients.append(newClient)
			except socket.error:
				# Failed to connect so kill it
				newClient.shutdown(socket.SHUT_RDWR)
				newClient.close()
		else:
			if len(clients):
				idx = randrange(len(clients))
				print "CLOSE CLIENT {}".format(idx)
				clients[idx].shutdown(socket.SHUT_RDWR)
				clients[idx].close()
				del clients[idx]
				while len(clients) and random() < 0.10:
					# 10% chance to kill an additional client
					idx = randrange(len(clients))
					print "CLOSE ANOTHER CLIENT {}".format(idx)
					clients[idx].shutdown(socket.SHUT_RDWR)
					clients[idx].close()
					del clients[idx]
		sleep(sleepTime)

	# Close all remaining connections on end
	for client in clients:
		client.shutdown(socket.SHUT_RDWR)
		client.close()

	print "EXAMPLE CLIENT: Finished."


if __name__ == "__main__":
	svr = Thread(target=EXAMPLE_SERVER, kwargs={'its': 5000})
	cli = Thread(target=EXAMPLE_CLIENT, kwargs={'its': 5000})

	svr.start()
	cli.start()

	svr.join()
	cli.join()

	print "Fin."
