#!/usr/bin/env python

import numpy as np
from StringIO import StringIO
from PIL import Image

def compress(img, quality=90):
	'''compress an ndarray into a jpeg string.'''
	if isinstance(img, np.ndarray):
		b = StringIO()
		Image.fromarray(img).save(b, 'JPEG', quality=quality)
		return b.getvalue()
	return img

def decompress(img):
	'''decompress a jpeg string into an ndarray.'''
	if isinstance(img, str):
		img = np.uint8(Image.open(StringIO(img))).copy() # return a copy so that it is forced to be C-contiguous
	return img
