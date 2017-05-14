import numpy as np
import os
from OpenGL import GL

import Op
import ISCV
import cv2
from UI import QGLViewer
# from UI.QGLViewer import PaintOptions

from PIL import Image, ImageMath
from time import time
from StringIO import StringIO


class SkelToImg(Op.Op):
	def __init__(self, name='/SkelToImg', locations='', calibration='', mesh='', startFrame=1, directory=''):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('locations', 'locations', 'Skeleton locations', 'string', locations, {}),
			('calibration', 'calibration', 'Calibration location', 'string', calibration, {}),
			('mesh', 'mesh', 'Mesh location', 'string', mesh, {}),
			('startFrame', 'frame', 'Start frame', 'int', startFrame, {}),
			('directory', 'directory', 'Target directory', 'string', directory, {})
		]

		super(self.__class__, self).__init__(name, fields)

	def cook(self, location, interface, attrs):
		startFrame = attrs['startFrame']
		if interface.frame() < startFrame: return

		if not attrs['directory']: return

		skelDict = interface.attr('skelDict')
		if not skelDict: return

		calibrationLocation = attrs['calibration']
		if not calibrationLocation: calibrationLocation = interface.root()

		# if not 'mesh' in attrs or not attrs['mesh']: return
		# mesh = attrs['mesh']

		mats = interface.attr('mats', atLocation=calibrationLocation)
		if not mats: return
		Ps = np.array([m[2] / (np.sum(m[2][0, :3] ** 2) ** 0.5) for m in mats], dtype=np.float32)

		from Ops import Interface
		vs, vs_labels = Interface.getWorldSpaceMarkerPos(skelDict)
		normals = interface.attr('normals')
		if normals is None: return

		if self.visibility is None: self.visibility = ISCV.ProjectVisibility.create()
		self.visibility.setNormals(normals)
		vs_labels = np.array(skelDict['markerNames'], dtype=np.int32)
		proj_x2ds, proj_splits, _ = ISCV.project_visibility(vs, vs_labels, Ps, self.visibility)

		win = interface.win()
		# if not win.view().primitives2D:
		#	 win.view().primitives2D = QGLViewer.makePrimitives2D(([], []), ([], []))
		#
		# win.view().primitives2D[1].setData(proj_x2ds, proj_splits)
		# win.view().primitives2D[1].pointSize = 8
		# win.view().primitives2D[1].colour = [1., 1., 1., 1.]

		numCameras = len(proj_splits) - 1
		directory = attrs['directory']

		# Only draw 2D primitives
		# options = PaintOptions()
		# options.setDrawNone()
		# options.drawPrimitives2D = True

		for ci in range(numCameras):
			cameraIndex = ci + 1
			win.view().camera = win.view().cameras[cameraIndex]
			# win.view().paintGL(options)
			win.view().paintToImageFile(ci, interface.frame(), '')

			cameraName = str(cameraIndex)
			filename = os.path.join(directory, 'Camera_' + cameraName + '_Frame_' + str(interface.frame()) + '.jpg')

			buffer = GL.glReadPixels(0, 0, win.view().width, win.view().height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
			cv_image = cv2.cv.CreateImage((win.view().width, win.view().height), cv2.cv.IPL_DEPTH_8U, 3)
			cv2.cv.SetData(cv_image, buffer)
			cv2.cv.Flip(cv_image)
			cv2.cv.SaveImage(filename, cv_image)


def blueDistance(colour):
	# Returns a measure of how similar the blue channel is to the average of the channels
	return colour[2] - (colour[0] + colour[1] + colour[2]) / 3


class LayerToImg(Op.Op):
	def __init__(self, name='/LayerToImg', locations='', startFrame=0, enable=False):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('locations', 'locations', 'Layer locations', 'string', locations, {}),
			('startFrame', 'frame', 'Start frame', 'int', startFrame, {}),
			('enable', 'enable', 'enable', 'bool', enable, {})
		]

		super(self.__class__, self).__init__(name, fields)

	def cook(self, location, interface, attrs):
		if not attrs['enable']: return
		startFrame = attrs['startFrame']
		if interface.frame() < startFrame: return

		cameraIndex = 0
		# win.view().camera = win.view().cameras[cameraIndex]

		win = interface.win()
		w, h = win.view().width, win.view().height
		wp, hp = w / 2, h / 2

		# Paint a particular layer by name
		win.view().paintLayer(location, w, h)

		# Get GL buffer
		buffer = (GL.GLubyte * (4 * w * h))(0)
		# print len(buffer)
		GL.glReadPixels(0, 0, w, h, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, array=buffer)
		# npIm = np.fromstring(buffer, dtype=np.uint8).reshape(h, w, 4)
		# 
		# samples = npIm.tostring()
		# pix = fitz.Pixmap(fitz.csRGB, w, h, samples)
		# png_buffer = pix.getPNGData(savealpha=0)
		image = Image.frombytes(mode="RGBA", size=(w, h), data=buffer)
		image = image.transpose(Image.FLIP_TOP_BOTTOM)

		red, green, blue, alpha = image.split()
		image.putalpha(ImageMath.eval("convert((((t - d((r, g, b))) >> 31) + 1) * a, 'L')", t=10, d=blueDistance, r=red, g=green,
						   b=blue, a=alpha))
		
		png_buffer = StringIO()
		# t1 = time()
		image.save(png_buffer, 'png', compress_level=1)
		# print time() - t1

		interface.setAttr('png_buffer', png_buffer.getvalue(), atLocation='/root/images')

		# # Create image plane
		# img_vs = np.array([[-wp, -hp, 0], [wp, -hp, 0], [wp, hp, 0], [-wp, hp, 0]], dtype=np.float32)
		# img_fs = [[0, 1, 2, 3]]
		# img_ts = np.array([[1, 0, 0, 2000], [0, 1, 0, 1000], [0, 0, 1, -4500]], dtype=np.float32)
		# img_vts = [[0, 1], [1, 1], [1, 0], [0, 0]]
		# imgAttrs = {
		#	 'names': ['plane'],
		#	 'vs': [img_vs],
		#	 'tris': [img_fs],
		#	 'vts': [img_vts],
		#	 'transforms': [img_ts],
		#	 'texture': image,
		#	 'colour': [(0, 0, 1, 1)]
		# }
		# interface.createChild('plane', 'mesh', attrs=imgAttrs)


# Register Ops
import Registry
Registry.registerOp('Dump SkelToImg', SkelToImg)
Registry.registerOp('Dump LayerToImg', LayerToImg)
