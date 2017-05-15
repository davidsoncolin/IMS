#!/usr/bin/env python
from ReframeClient import ReframeClient

"""
A minimal reframe client app which just reads motion
capture data from a reframe server.
"""


def main():
	global client
	client = ReframeClient('localhost', 18667)
	while True:
		frame = client.subscribe()
		if frame is not None: print frame['frame_number']


if __name__ == '__main__':
	main()
