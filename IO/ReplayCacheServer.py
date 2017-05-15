#!/usr/bin/env python
import os
import IO
from ReframeServer import ReframeServer
from ReframeData import dummy_data_provider

def hashState(frame): # TODO this is somewhat inefficient, but more advanced methods of data reduction are in the pipelne
	"""
	Note only skeleton descriptor data, etc, are use in this hash function
	translation and rotation information are not

	The hash is used to determine if any actor has exited/entered the scene i.e. the hash will have changed

	:param frame: data frame
	:return: Python hash value
	"""
	s = str(frame['num_subjects'])
	for subject in frame['subjects']:
		s += subject['name'] + '_'.join(subject['bone_names']) + '_'.join(subject['bone_parents']) + '_'.join(subject['marker_names'])
	return hash(s)

def setFrame(frame):
	global current_frame
	current_frame = frame

frame_number = 0
def main():
	"""
	a small applicaton which instantiates a reframe server using mo-cap data stored in files
	"""
	global frame_number

	def getRecordedFrame(): # get frame from a numbered sequence of stored files
		global frame_number
		frame_number = frame_number + 1
		prefix = os.path.join(os.environ['GRIP_TEMP'],'saved_frame_')
		file_name = prefix + str(frame_number) + ".data"
		if ( not os.path.isfile(file_name) ): # cycle back to beginning again when go over the number of stored file
			frame_number = 1
		file_name = prefix + str(frame_number) + ".data"
		print("file name = " + file_name)
		header,data = IO.load(file_name)
		return data

	use_dummy_data = True
	if ( use_dummy_data ):
		get_frame = dummy_data_provider().getFrame
	else:
		get_frame = getRecordedFrame

	server = ReframeServer(get_frame, hashState, '*', port = 18667)
	server.start()

if __name__ == '__main__' :
	main()
