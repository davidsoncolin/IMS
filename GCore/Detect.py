#!/usr/bin/env python

"""
GCore/Detect.py

Requires:
	numpy

	GRIP
		ISCV(detect_bright_dots)
"""

import numpy as np
import ISCV

def detect_dots(data, pixel_threshold, opts):
	"""
	Extract the dots from data contained in a single frame. The detected dots are then filtered based on
	the min/max dot size, circularity, etc.
	"""
	min_min, max_max = 2,opts['max_dot_size']*2

	if isinstance(pixel_threshold, tuple) and len(pixel_threshold) == 3:
		dots = ISCV.detect_bright_dots(data, pixel_threshold[0], pixel_threshold[1], pixel_threshold[2])
	else:
		dots = ISCV.detect_bright_dots(data, pixel_threshold, pixel_threshold, pixel_threshold)

	min_ds, max_ds, circ = opts['min_dot_size']**4, opts['max_dot_size']**4, opts['circularity_threshold']**2
	filtered_dots = [d
		for d in dots
		if (d.x1 - d.x0 + d.y1 - d.y0) > min_min
		and (d.x1- d.x0 + d.y1 - d.y0) < max_max
		and (d.sxx * d.syy) > min_ds
		and (d.sxx * d.syy) < max_ds
		and d.sxx < circ * d.syy
		and d.syy < circ * d.sxx
	]

	height, width, chans = data.shape
	psc = np.float32([2.0/width, -2.0/width])
	pof = np.float32([-1.0, height/float(width)])

	dotScreenCoords = np.array([[dot.sx, dot.sy] for dot in filtered_dots], dtype=np.float32).reshape(-1, 2) * psc + pof

	return filtered_dots, dotScreenCoords



def detect_dots_with_box(data, pixel_threshold, opts, x1, y1, x2, y2):
	"""
	Extract the dots from data contained in a single frame. The detected dots are then filtered based on
	the min/max dot size, circularity, etc.
	"""
	min_min, max_max = 2,opts['max_dot_size']*2

	if isinstance(pixel_threshold, tuple) and len(pixel_threshold) == 3:
		dots = ISCV.detect_bright_dots(data, pixel_threshold[0], pixel_threshold[1], pixel_threshold[2], x1, y1, x2, y2)
	else:
		dots = ISCV.detect_bright_dots(data, pixel_threshold, pixel_threshold, pixel_threshold, x1, y1, x2, y2)

	min_ds, max_ds, circ = opts['min_dot_size']**4, opts['max_dot_size']**4, opts['circularity_threshold']**2
	filtered_dots = [dot
		for dot in dots
		if (dot.x1 - dot.x0 + dot.y1 - dot.y0) > min_min
		and (dot.x1- dot.x0 + dot.y1 - dot.y0) < max_max
		and (dot.sxx * dot.syy) > min_ds
		and (dot.sxx * dot.syy) < max_ds
		and dot.sxx < circ * dot.syy
		and dot.syy < circ * dot.sxx
	]

	height, width, chans = data.shape
	psc = np.array([2.0/width, -2.0/width], dtype=np.float32)
	pof = np.array([-1.0, height/float(width)], dtype=np.float32)

	dotScreenCoords = np.array([[dot.sx, dot.sy] for dot in filtered_dots], dtype=np.float32).reshape(-1, 2) * psc + pof

	return filtered_dots, dotScreenCoords