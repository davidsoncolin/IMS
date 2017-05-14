#!/usr/bin/env python

import numpy as np

class Timer:    
	def __enter__(self):
		import time
		self.start = time.clock()
		return self

	def __exit__(self, *args):
		self.end = time.clock()
		self.interval = self.end - self.start
		print (self.interval)


class atdict(dict):
	__getattr__ = dict.__getitem__
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__


def list_of_lists_to_splits(list_of_lists, dtype=np.int32):
	'''Convert a list of lists to a flat list with splits.'''
	splits = np.cumsum([0]+map(len,list_of_lists),dtype=np.int32)
	data = np.array([x for y in list_of_lists for x in y],dtype=dtype)
	return data, splits

def humanKey(key):
	''' Use this as the key in sort functions (e.g. list.sort(key=smartKey)) to sort
	alphanumeric strings in a way people expect
	example : ['9','10','99', '100'] rather than ['10','100','9','99'] '''
	import re
	return [int(t) if t.isdigit() else t.upper() for t in re.split('([0-9]+)', key) if t]
