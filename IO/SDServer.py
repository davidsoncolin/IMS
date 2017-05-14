#!/usr/bin/env python

import IO
import zmq
import time
import sys
import socket

def camName_to_camId(camName):
	if type(camName) == int: return int(camName)
	cameraId = int(camName.split('.')[3]) - 100
	return cameraId

def camId_to_camName(camId):
	return '10.10.6.%d' % (camId+100)

def updateCameraState(key, camId, value):
	g_state['Thresholds'][camId] = value
	g_state['Hash'] += 1
	
def processData(data, address):
	'''data is the raw data, address is the camera/sync box address.'''
	# TODO, this properly
	cameraId = camName_to_camId(address)
	isSyncBox = (cameraId == 0)
	
	#print 'yo mama', len(data), data[:10], '...', data[-32:]
	# decode the data
	(frame, flags, etc) = struct.unpack('>HIHIHIHI', data[:16])
	if flags & 0x10000000: # this is an image (?)
		# decode the image,,,
		frameNumber,packetNum,numPackets = []# TODO
		ret = {}
		if not g_images.has_key(camName):
			if packetNum != 1:
				# TODO we need to push out the partial image!?
				[]
			ret['PARTIAL_'+camName] = g_images.pop(camName)
			g_images[camName] = [[],np.zeros(numPackets,dtype=np.bool)]
		g_images[camName][0].extend(imageData)
		g_images[camName][1][packetNum] = True
		if np.all(g_images[camName][1]):
			ret[camName] = g_image.pop(camName)
		if ret == {}: return None
		return {'Type':'Image', 'Image':ret}
	if flags & 0x20000000: # this is a command response
		if not cameraTicketIds.has_key(cameraId):
			print 'WARNING!!! This was unexpected!!!'
			return None
		if commandType == 'Threshold':
			updateCameraState('Thresholds', camId, value)
		return {'Type':'Response','RawData':data,'TicketId':cameraTicketIds.pop(cameraId), 'Time':time.time() }
	if flags & 0x40000000: # this is a centroid for a particular camera
		# decode centroids,,..
		frameNumber,packetNum,numPackets = []# TODO
		cds = [] # TODO
		ret = None
		if not g_frame.has_key(camName):
			if packetNum != 1: # we need to push out the old frame!
				ret = g_frame
				g_frame = {}
			g_frame[camName] = [[],np.zeros(numPackets,dtype=np.bool)]
		g_frame[camName][0].extend(cds)
		g_frame[camName][1][packetNum] = True
		if np.all(g_frame[camName][1]):
			# TODO you could test here to see whether all cameras have all data and push out a complete frame instead of waiting for the first packet of the next frame to push it out
			pass
		return ret
	return {'Type':'Centroids','Flags':flags,'RawData':data, 'Time':time.time()}

def setThreshold(g_socket, (camName, threshold), id):
	threshold = int(threshold)
	assert(threshold >= 0 and threshold <= 255)
	str = '*'+chr(255)+chr(threshold)
	g_socket.sendto(str, (camName,1234))
	camId = camName_to_camId(camName)
	cameraTicketIds[camId] = (id, time.time())

if __name__ == '__main__' :

	# TODO figure out the number of cameras by interrogation of the IPs
	HostPort = ('10.10.6.10',1235)
	g_state = { 'NumCams':3, 'CameraNames': ['Syncbox','Cam1','Cam2','Cam3'], 'Fps':60, 'Thresholds':[0,100,200,300], 'Hash': 0 }
	
	if (len(sys.argv) > 1):
		HostPort = sys.argv[1]
		print 'Setting hostname to ',HostPort

	serverSocket = zmq.Context().socket(zmq.PUB)
	serverSocket.bind('tcp://*:20667')
	commandSocket = zmq.Context().socket(zmq.REP)
	commandSocket.bind('tcp://*:20666')
		
	g_socket = socket.socket(2,2)
	g_socket.settimeout(0.0)
	g_socket.bind(HostPort)
	commands = [] # ((command to send, camera to send it to), ticket id)
	cameraTicketIds = {}
	nextId = 0
	validCommands = { 'SET_THRESHOLD': setThreshold }
	g_frame = {}
	g_images = {}

	# ('SET_THRESHOLD', ('10.10.6.104',200))
	
	while True:
		data = g_socket.recvfrom(10240)
		if data != Null:
			publish = processData(*data)
			serverSocket.send(IO.wrap(publish))
		# see if any commands are coming in and add them to the list
		if commandSocket.poll(timeout=0):
			cmd,_ = IO.unwrap(commandSocket.recv()) # (command to send, camera to send it to) for example, in the simplest and most insecure case
			assert(_ == '')
			if cmd == 'GET STATE':
				commandSocket.send(IO.wrap(g_state))
			else:
				commandSocket.send(IO.wrap({'TicketId':str(nextId)}))
				if len(cmd) > 0 and validCommands.has_key(cmd[0]):
					commands.append([cmd,nextId])
					nextId += 1
				else:
					print 'Unknown command', cmd
		# go through all of the commands in order looking for cameras who haven't been given a command
		for ci,c in enumerate(commands):
			if cameraTicketIds.has_key(c[1]): continue
			cmd,id = commands.pop(ci)
			validCommands[cmd[0]](g_socket, cmd[1], id)

	# Booting client would look like this:
	allCameras = {'10.10.6.101':{'Threshold':175}, '10.10.6.102':{'Threshold':200}, '10.10.6.104': {'Threshold':185} }
	for camName, params in allCameras.iteritems():
		for param,value in params.iteritems():
			req = zmq.Context().socket(zmq.REQ)
			req.connect('tcp://localhost:20666')
			if param == 'Threshold':
				req.send(IO.wrap(('SET_THRESHOLD', (camName,value))))
				ticket = req.recv()
			
	# Client would look like this:
	if False:
		context = zmq.Context()
		client = context.socket(zmq.SUB)
		client.connect('tcp://localhost:20667')
		client.setsockopt(zmq.SUBSCRIBE, 'TIMS') # only accept packets with the correct header
		while True: # or in a timeout callback from qt
			while client.poll(timeout=0): data = client.recv() # trash the stream until the last frame
			frame = None
			if data != None:
				frame,_ = IO.unwrap(data)
				print 'Got frame', frame
				assert(_ == '')
			if clickedOnTheSetThresholdButton:
				req = zmq.Context().socket(zmq.REQ)
				req.connect('tcp://localhost:20666')
				req.send(IO.wrap(('SET_THRESHOLD', ('10.10.6.104',200))))
				ticket = req.recv()
				print 'Got ticket', ticket
