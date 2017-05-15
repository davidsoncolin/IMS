#!/usr/bin/env python
import os
import IO
from IO.ReframeServer import ReframeServer

		
import ViconConnect
def hashState(frame): # TODO this is a bit slow and rubbish, the hash only has to change (eg randint); BUT Vicon doesn't notify us of any change, so we have to test everything anyway
	s = str(frame['num_subjects'])
	for subject in frame['subjects']:
		s += subject['name'] + '_'.join(subject['bone_names']) + '_'.join(subject['bone_parents']) + '_'.join(subject['marker_names'])
	return hash(s)

def animDict(frame, hash):
	regular_implementation = False
	if regular_implementation:
		frame['hash'] = hash
		return frame
	else:
		subjects = []
		for subject in frame['subjects']:
			subjects.append({'name':subject['name'],'bone_Ts':subject['bone_Ts'],'bone_Rs':subject['bone_Rs'],'marker_Ts':subject['marker_Ts'],'marker_occludeds':subject['marker_occludeds']})
		return {'subjects':subjects,'tc':frame['tc'],'frame_number':frame['frame_number'],'latency':frame['latency'],'unlabelled_markers':frame['unlabelled_markers'],'hash':hash}

frame_number = 0
def main():
	global frame_number
	# Program options
	TransmitMulticast = False
	HostName = "localhost:801"
	import sys
	if (len(sys.argv) > 1):
		HostName = sys.argv[1]
		print 'Setting hostname to ',HostName

	viconClient = ViconConnect.viconConnect(HostName, TransmitMulticast)

	def getViconFrame():
		return ViconConnect.viconParseFrame(viconClient)

	def getRecordViconFrame():
		global frame_number
		frame_number = (frame_number + 1) % 1000
		frame = getViconFrame()
		prefix =  os.path.join(os.environ['GRIP_TEMP'],'saved_frame_')
		file_name = prefix + str(frame_number) + ".data"
		IO.save(file_name,frame)
		return frame

	def getRecordedFrame():
		global frame_number
		frame_number = frame_number + 1
		prefix =  os.path.join(os.environ['GRIP_TEMP'],'saved_frame_')
		file_name = prefix + str(frame_number) + ".data"
		if ( not os.path.isfile(file_name) ):
			frame_number = 1
		file_name = prefix + str(frame_number) + ".data"
		header,data = IO.load(file_name)
		return data

	get_frame = getViconFrame
	#get_frame = getRecordedFrame
	#get_frame = getRecordViconFrame
	server = ReframeServer(get_frame, hashState, '*', port = 18667)
	server.main_loop()

if __name__ == '__main__' :
	main()
