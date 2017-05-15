import Op, Interface
import numpy as np
from ISCV import filter_image

from UI import QApp
import cv2
from PIL import Image
import os
from datetime import datetime
import sys

class Blur(Op.Op):
	def __init__(self, name='/Image Blur', locations='', b1=1, b2=10):
		fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'Locations', 'Locations', 'string', locations, {}),
			('b1', 'Small blur radius', 'This is part of the image blur filter which controls the size of smallest detected features.', 'float', b1, {"min" :0.0, "max": 20.0}),
			('b2', 'Large blur radius', 'This is part of the image blur filter which controls the size of largest detected features.', 'float', b2, {"min": 0.0, "max": 20.0}),
			('showresult', 'Show Result', 'Display the result of this Filter', 'bool', True)
		]

		super(self.__class__, self).__init__(name, fields)

	def cook(self, location, interface, attrs):
		# TODO: Should use current location to read and set imgs (not atLocation). Just have to remove and update calling code.
		imgs = interface.attr('imgs')
		if imgs is None: return
		b1 = attrs['b1']
		b2 = attrs['b2']

		for img in imgs:
			data = filter_image(img, int(b1), int(b2))
			#if attrs['showresult']:
			img[:] = data

		interface.setAttr('imgs', imgs)


class BlurPartial(Op.Op):
	def __init__(self, name='/Blur Partial', locations='', b1=1, b2=10):
		fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'Locations', 'Locations', 'string', locations, {}),
			('b1', 'Small blur radius', 'This is part of the image blur filter which controls the size of smallest detected features.', 'float', b1, {"min": 0.0, "max": 20.0}),
			('b2', 'Large blur radius', 'This is part of the image blur filter which controls the size of largest detected features.', 'float', b2, {"min": 0.0, "max": 30.0}),
			('showresult', 'Show Result', 'Display the result of this Filter', 'bool', True),
		]
		super(self.__class__, self).__init__(name, fields)
		self.subtractors = []
		self.trained = False


	def cook(self, location, interface, attrs):
		training = interface.attr('train', default={'train': False, 'reset': False, 'send_plate': False})
		# TODO: Should use current location to read and set imgs (not atLocation). Just have to remove and update calling code.
		imgs = interface.attr('imgs')
		if imgs is None: return
		b1 = attrs['b1']
		b2 = attrs['b2']

		while len(self.subtractors) != len(imgs):
			self.subtractors.append(cv2.createBackgroundSubtractorMOG2())

		print "\r%s | %s                                 " %(datetime.utcnow(), training); sys.stdout.flush()

		blurred_images = [None] * len(imgs)
		for img_index, img in enumerate(imgs):

			if training['reset']:
				cv2.ocl.setUseOpenCL(False)
				self.subtractors[img_index].clear()
				cv2.ocl.setUseOpenCL(True)
				self.trained = False

			if training['train']:
				cv2.ocl.setUseOpenCL(False)
				self.subtractors[img_index].apply(img)
				cv2.ocl.setUseOpenCL(True)
				self.trained = True

			if self.trained:

				trained_image = self.subtractors[img_index].apply(img, learningRate=0) # Input Image - Current
				# Threshold the image
				_, thresh = cv2.threshold(trained_image, 0, 255, cv2.THRESH_BINARY)
				im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
				contours_clean = [contour for contour in contours if cv2.contourArea(contour) > 5000] # Threshold the contours
				plate = np.zeros(shape=img.shape, dtype=np.uint8)

				boxes = []
				box_index = 0

				for ci, contour in enumerate(contours_clean):

					x, y, w, h = cv2.boundingRect(contour)
					min_corners = [(x, y), (x + w, y + h)]
					for b_i, box in enumerate(boxes):
						result = self.boxes_intersect(min_corners, box)
						if not result: continue
						boxes[b_i] = self.box_extended(min_corners, box)
					else:
						boxes.append(min_corners)
						box_index += 1

				bbox_final = boxes[0] if len(boxes) > 0 else ((0,0),(0,0))
				for box in boxes:
					bbox_final = self.box_extended(bbox_final, box)
					(x, y), (x2, y2) = box
					img[:] = filter_image(img, 3, 21, x, y, x2-x, y2-y)
					plate[y:y2, x:x2, :] = img[y:y2, x:x2, :]


				interface.setAttr("bbox", bbox_final)

				blurred_images[img_index] = plate
				img[:] = plate
			else:
				img[:] = filter_image(img, int(b1), int(b2))

		if training['send_plate']:
			interface.setAttr('imgs_blurred', imgs)
		else:
			interface.setAttr('imgs', imgs)

	def boxes_intersect(self, a, b):
		(a_min_x, a_min_y), (a_max_x, a_max_y) = a
		(b_min_x, b_min_y), (b_max_x, b_max_y) = b

		if a_max_x < b_min_x: return False
		if a_min_x > b_max_x: return False
		if a_max_y < b_min_y: return False
		if a_min_y > b_max_y: return False
		return True

	def box_extended(self, a, b):
		(min_x, min_y), (max_x, max_y) = a
		(b_min_x, b_min_y), (b_max_x, b_max_y) = b

		if min_x > b_min_x: min_x = b_min_x
		if min_y > b_min_y: min_y = b_min_y

		if max_x < b_max_x: max_x = b_max_x
		if max_y < b_max_y: max_y = b_max_y

		return (min_x, min_y), (max_x, max_y)


class BackgroundTrainer(Op.Op):
	def __init__(self, name='/Background Trainer', locations='', b1=1, b2=10):
		fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'Locations', 'Locations', 'string', locations, {}),
			('train', 'Train Now', 'Process the current data to train', 'bool', True),
			('reset', 'Reset', 'Reset the training data', 'bool', False),
			('sendPlates', 'Send Plates', 'Send Plates over', 'bool', False)
		]
		super(self.__class__, self).__init__(name, fields)
		self.subtractors = []

	def cook(self, location, interface, attrs):
		interface.setAttr('train', {'train': attrs['train'],
									'reset' : attrs['reset'],
									'send_plate': attrs['sendPlates']})


class ExportImage(Op.Op):
	'''
	This is slow function as it is exporting the frames out to disk - only to be used when debugging image issues.
	'''
	def __init__(self, name='/Image Blur', locations='', savepath='', suffix=''):
		fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'Locations', 'Locations', 'string', locations, {}),
			('savepath', 'SavePath', 'SavePath', 'string', savepath, {}),
			('suffix', 'suffix', 'suffix', 'string', suffix, {}),
		]
		super(self.__class__, self).__init__(name, fields)


	def cook(self, location, interface, attrs):
		# TODO: Should use current location to read and set imgs (not atLocation). Just have to remove and update calling code.
		imgs = interface.attr('imgs')
		if imgs is None: return
		for index, img in enumerate(imgs):
			tc = interface.attr('timecode', default='25:25:25:25').replace(":","")
			image_name = "{tc}_{index}_{suffix}.png".format(index=interface.attr('cId'),tc=tc, suffix=attrs['suffix'])
			image_path = os.path.join(attrs['savepath'], image_name)

			export_img = Image.fromarray(img)
			with open(image_path, 'wb') as f:
				export_img.save(f)


class BlurChangedArea(Op.Op):
	def __init__(self, name='/Image Blur', locations='', b1=1, b2=10):
		fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'Locations', 'Locations', 'string', locations, {}),
			('b1', 'Small blur radius', 'This is part of the image blur filter which controls the size of smallest detected features.', 'float', b1, {"min" :0.0, "max": 20.0}),
			('b2', 'Large blur radius', 'This is part of the image blur filter which controls the size of largest detected features.', 'float', b2, {"min": 0.0, "max": 20.0}),
			('showresult', 'Show Result', 'Display the result of this Filter', 'bool', True)
		]
		super(self.__class__, self).__init__(name, fields)
		self.subtractors = []


	def cook(self, location, interface, attrs):
		# TODO: Should use current location to read and set imgs (not atLocation). Just have to remove and update calling code.
		imgs = interface.attr('imgs')
		if imgs is None: return
		b1 = attrs['b1']
		b2 = attrs['b2']

		training_image = False

		if len(self.subtractors) != len(imgs):
			training_image = True
			self.subtractors = []
			self.subtractors.append(cv2.createBackgroundSubtractorMOG2())

		for img in imgs:
			data = filter_image(img, int(b1), int(b2))
			#if attrs['showresult']:
			img[:] = data

		interface.setAttr('imgs', imgs)


class BlurPython(Op.Op):
	def __init__(self, name='/Image Blur', locations='', b1=1, b2=10):
		fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'Locations', 'Locations', 'string', locations, {}),
			('b1', 'Small blur radius', 'This is part of the image blur filter which controls the size of smallest detected features.', 'float', b1, {"min" :0.0, "max": 20.0}),
			('b2', 'Large blur radius', 'This is part of the image blur filter which controls the size of largest detected features.', 'float', b2, {"min": 0.0, "max": 20.0}),
			('showresult', 'Show Result', 'Display the result of this Filter', 'bool', True)
		]

		self.lookup = np.power((np.arange(256, dtype=np.float32) + 20.0) / 275.0, 0.4545) * 255
		super(self.__class__, self).__init__(name, fields)

	def cook(self, location, interface, attrs):
		# TODO: Should use current location to read and set imgs (not atLocation). Just have to remove and update calling code.
		imgs = interface.attr('imgs')
		if imgs is None: return
		b1 = 2 * attrs['b1'] + 1
		b2 = 2 * attrs['b2'] + 1

		for img in imgs:
			n1 = np.zeros([img.shape[0], img.shape[1], 1], dtype=np.uint8)
			n1[:, :, :] = self.lookup[img[:, :, 1]].reshape(img.shape[0], img.shape[1], 1)

			b1 = cv2.boxFilter(n1, 0, (3, 3)).astype(np.float32)
			b2 = cv2.boxFilter(n1, 0, (21, 21)).astype(np.float32)
			b3 = np.zeros((b2.shape[0], b2.shape[1], 3), dtype=np.uint8)
			b3[:, :, :] = np.clip((((b1[:, :]) / (b2[:, :])) - 0.75) * 512, 0, 255).reshape(b1.shape[0], b1.shape[1], 1)

			#if attrs['showresult']:
			img[:] = b3

		interface.setAttr('imgs', imgs)


class Brick(Op.Op):
	def __init__(self, name='/Image Brick', locations='', brightDotThreshold=180, darkDotThreshold=223):
		fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'Locations', 'Locations', 'string', locations, {}),
			('brightDotThreshold', 'Bright Dot Threshold', 'Brightness threshold of light detections.', 'int', brightDotThreshold, {"min": 0, "max": 254}),
			('darkDotThreshold', 'Dark Dot Threshold', 'Darkness threshold of dark detections.', 'int', darkDotThreshold, {"min": 0, "max": 254}),
			('showresult', 'Show Result', 'Display the result of this Filter', 'bool', True)
		]

		super(self.__class__, self).__init__(name, fields)

	def cook(self, location, interface, attrs):
		imgs = interface.attr('imgs', atLocation=location)
		if imgs is None: return

		attrs = self.getAttrs()
		brightDotThreshold = attrs['brightDotThreshold']
		darkDotThreshold = attrs['darkDotThreshold']

		lookup = np.zeros(256, dtype=np.uint8)
		lookup[brightDotThreshold:] = 255
		lookup[255 - darkDotThreshold:brightDotThreshold] = 128

		for ci, img in enumerate(imgs):
			data = lookup[img]
			if attrs['showresult']:
				img[:] = data

		interface.setAttr('imgs', imgs, atLocation=location)


class RedBrick(Op.Op):
	def __init__(self, name='/Image Red Brick', locations='', brightDotThreshold=180, enableBlur=True, blurSize=5):
		fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'Locations', 'Locations', 'string', locations, {}),
			('brightDotThreshold', 'Bright Dot Threshold', 'Brightness threshold of light detections', 'int', brightDotThreshold, {"min": 0, "max": 254}),
			('enableBlur', 'Enable Blur', 'Enable Blur', 'bool', enableBlur),
			('blurSize', 'Blur Size', 'Gaussian Kernel Size', 'int', blurSize, {"min": 3, "max": 64}),
			('showresult', 'Show Result', 'Display the result of this Filter', 'bool', True)
		]

		super(self.__class__, self).__init__(name, fields)

	def cook(self, location, interface, attrs):
		imgs = interface.attr('imgs', atLocation=location)
		if imgs is None: return

		attrs = self.getAttrs()
		brightDotThreshold = attrs['brightDotThreshold']
		enableBlur = attrs['enableBlur']
		blurSize = attrs['blurSize']

		import cv2
		for ci, img in enumerate(imgs):
			# h, w, c
			data = np.zeros_like(img)
			lookup = np.zeros(256, dtype=np.uint8)
			lookup[brightDotThreshold:] = 255
			lookup[:brightDotThreshold] = 96

			# Push red a little
			data[:, :, 0] = img[:, :, 0]
			data[:, :, 1] = img[:, :, 1] / 3
			data[:, :, 2] = 0#img[:, :, 2] / 3

			# Crush & linearise
			data = lookup[img]
			data = _rgb_3chGrey(data)

			# Blur a little
			if enableBlur:
				data = cv2.GaussianBlur(data, (blurSize, blurSize), 1)

			if attrs['showresult']:
				img[:] = data

		interface.setAttr('imgs', imgs, atLocation=location)


def _rgb_3chGrey(RGBimg):
	"""
	https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale
	convert RGB image to greyscale

	expects RGBimg[h,w,3], returns the same shape
	"""
	ret = np.zeros_like(RGBimg)
	ret[:, :, 0] = RGBimg[:, :, 0] * 0.2989 + RGBimg[:, :, 1] * 0.5870 + RGBimg[:, :, 2] * 0.1140
	ret[:, :, 1] = RGBimg[:, :, 0] * 0.2989 + RGBimg[:, :, 1] * 0.5870 + RGBimg[:, :, 2] * 0.1140
	ret[:, :, 2] = RGBimg[:, :, 0] * 0.2989 + RGBimg[:, :, 1] * 0.5870 + RGBimg[:, :, 2] * 0.1140
	return ret


# Register Ops
import Registry
Registry.registerOp('Image Blur', Blur)
Registry.registerOp('Image Brick', Brick)
Registry.registerOp('Image Red Brick', RedBrick)
