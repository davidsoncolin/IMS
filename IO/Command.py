#!/usr/bin/env python

import ViconConnect
import IO
import zmq
import time
import sys

if __name__ == '__main__' :

	print sys.argv[1]
	client = zmq.Context().socket(zmq.REQ)
	client.connect('tcp://localhost:18666')
	client.send(sys.argv[1])
	print IO.unwrap(client.recv())

	#for example:
	#python Command.py current_frame=-1
