import numpy as np
import re
from GCore import SolveIK, Calibrate
import ISCV

def readAsciiRaw(filepath):
	'''
	Takes a Giant Ascii Raw file and converts the data to a Python representation

	:param filepath: Path to Raw File to be parsed
	:return: A dict containing the Raw Data in an easy to use form
	'''
	rawLines = open(filepath, 'rb').readlines()
	line,frames,TCs,numCams,numFrames = 2,{},{},int(rawLines[0]),int(rawLines[1])
	rawDict = { 'frames':frames, 'TCs':TCs, 'numCams':numCams, 'numFrames':numFrames }
	while line < len(rawLines):
		lineData = rawLines[line].split()
		fi = int(lineData[0])
		TCs[fi] = lineData[1]
		detections = []
		line += 1
		for camera in xrange(numCams):
			numDetections = int(rawLines[line])
			dets = np.float32([l.split()[1:3] for l in rawLines[line+1:line+numDetections+1]])
			line += numDetections+1
			detections.append(dets)
		frames[fi] = detections
	return rawDict

def readCal(filepath):
	calLines = open(filepath, 'rb').readlines()
	calDict = { 'Cameras': [] }
	line = 2
	while line < len(calLines):
		assert "CAMERA" in calLines[line], "Unexpected Formatting"
		cameraDict = {
			'DLT': np.zeros(11, dtype=np.float32),
			'MAT': np.eye(3, 4 , dtype=np.float32),
			'IMAGE_REGION': np.zeros(4, dtype=np.float32),
			'DISTORTION': np.zeros(2, dtype=np.float32)
		}
		line += 1
		while True:
			section = re.split("[ :]+", calLines[line].strip())[0].upper()
			# print "Section: {}".format(section)
			line += 1
			if section == "DLT":
				cameraDict['DLT'] = np.float32(calLines[line:line+11])
				line += 11
				cameraDict['MAT'] = DLTtoMat( cameraDict['DLT'] )
				assert calLines[line].strip() == "}", "Unexpected Formatting"
				line += 1
			elif section == 'IMAGE_REGION':
				for edge in xrange(4):
					cameraDict['IMAGE_REGION'][edge] = np.float32(re.split("[ :]+", calLines[line].strip())[-1])
					line += 1
				assert calLines[line].strip() == "}", "Unexpected Formatting"
				line += 1
			elif section == 'DISTORTION':
				for k in xrange(2):
					cameraDict['DISTORTION'][k] = np.float32(re.split("[ :]+", calLines[line].strip())[-1])
					line += 1
					if calLines[line].strip() == '}':
						break
				assert calLines[line].strip() == "}", "Unexpected Formatting"
				line += 1
			elif section == 'K':
				cameraDict['DISTORTION'][0] = np.float32(re.split("[ :]+", calLines[line - 1].strip())[-1]) * 256 * 256
			else:
				while calLines[line].strip() <> '}':
						line += 1
				line += 1
			if calLines[line].strip() == '}':
				calDict['Cameras'].append(cameraDict)
				line += 1
				break
		while line < len(calLines) and not len(calLines[line].strip()):
			line += 1
	return calDict

def DLTtoMat( DLT ):
	mat = np.ones(12, dtype=np.float32)
	mat[:-1] = DLT
	mat.shape = (3,4)
	mat[:,3] *= np.float32([254, 254, 25.4]) # convert to mm (and strange factors of 10)
	mat[2,:] *= -256.0 # convert from "looking down z" to "looking down -z", and scale pixels to [-1,1]
	mat /= np.linalg.norm(mat[2,:3])
	return mat

def frameCentroidsToDets(frame, mats=None):
	'''Extract the centroids for given cameras and undistorts them. Returns a list of x2ds and splits per camera.'''
	detRawData, splits = SolveIK.list_of_lists_to_splits(frame, dtype=np.float32)
	detRawData = (detRawData - np.float32([256,256])) /  np.float32([256,-256])
	if mats is None: return detRawData[:,:2].copy(), splits
	return Calibrate.undistort_dets(detRawData, splits, mats)

if __name__ == '__main__':
	print "Reading an Ascii Raw:\n"
	import os, sys
	rawFilePath = sys.argv[1]
	rawDict = readAsciiRaw(rawFilePath)
	print rawDict['numCams']
	print rawDict['frames'].keys()
	print "Frame 1:"
	print "\tTimecode:\n\t{}".format(rawDict['TCs'][1])
	print "\tNumber of Detection:\n\t{}".format(map(len,rawDict['frames'][1]))
	print "\tdata, splits:\n\t{}".format(frameCentroidsToDets(rawDict['frames'][1]))
