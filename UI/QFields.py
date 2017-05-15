#!/usr/bin/env python

import functools
import numpy as np
from PySide import QtCore, QtGui
import UI.QCore as QCore # Qselect, Qslider, QfloatWidget, QvectorWidget, QmatrixWidget

import UI.QApp

'''QFieldsEditor : a fields editor with callbacks/update hooks
- state is a dict
- layout is generated by a list of (key, text description, help text, type, defaultvalue,[options dict]) tuples
- recognised field types are: select, if/endif, header, bopen, bclose, [icon,] filename, int, int3, float, float3, bool, region (int4), text, ...
'''

# a data conversion routine to correct a data mismatch bug that has crept in
def tuples_to_dicts(fields ):
	fields_dict = []
	for ft in fields:
		if len(ft) != 2:
			key      = ft[0]
			label    = ft[1]
			hover    = ft[2]
			editType = ft[3]
			default  = ft[4]
			mini_dict = dict(ft[5]) if len(ft)==6 else {}
			if ( label is not None ):
				mini_dict["longname"] = label
			if ( hover is not None ):
				mini_dict["tip"] = hover
			if ( editType is not None ):
				mini_dict["type"] = editType
			if (  default is not None ):
				mini_dict["default"] = default
			ft = (key,mini_dict)
		fields_dict.append( ft )
	return fields_dict


class QFieldsEditor(QtGui.QScrollArea):
	def __init__(self, parent=None):
		self.parent = parent
		self.all_fields = {}
		QtGui.QScrollArea.__init__(self, parent)
	
	def setKeyValue(self, key, value):
		if key in self.keyToQedit:
			qedit = self.keyToQedit[key]
			if isinstance(qedit,QtGui.QCheckBox): qedit.setChecked(value in [True,'yes','on',1, 2])
			elif isinstance(value,str) or isinstance(value,unicode): qedit.setText(value)
			else: qedit.setValue(value)
	
	def setFields(self, title, fields, values):
		# TODO, show the title somewhere
		self.all_fields = fields
		if self.layout() is not None:
			layouts = [self.layout()]
			while len(layouts):
				if not layouts[-1].count(): layouts.pop(); continue
				child = layouts[-1].takeAt(0)
				wid = child.widget()
				lay = child.layout()
				wid.deleteLater()
				if lay is not None and lay.count(): layouts.append(lay)
		else:
			self.setWidgetResizable(True)
			area = QtGui.QWidget(self)
			self.setWidget(area)
			self.gridLayout = QtGui.QGridLayout(self)
			area.setLayout(self.gridLayout)
		
			#layout.deleteLater()
		self.keyToQedit = {}
		testif,ifvalue = None,False

		fields =  tuples_to_dicts(fields)
		for key,field in fields:
			qedit = None
			if field.has_key('vec'): continue # ignore vecs
			if not values.has_key(key): continue # and field.has_key('explicit')
			label = field.get('longname',key)
			hover = field.get('tip','Tooltip')
			editType = field['type']
			if editType is None: continue
			default = field.get('default',None)
			value = values.get(key,default)

			if editType is None or editType == "alias":
				continue

			if editType == 'string' or editType == 'filename': 
				# TODO button for file browser
				qedit = QCore.QLineWidget(self)
				qedit.setText(str(value))
				qedit.valueChanged.connect(functools.partial(self.valueChanged_string, key))
			elif editType == 'text':
				qedit = QCore.QTextWidget(self)
				qedit.setText(str(value))
				qedit.valueChanged.connect(functools.partial(self.valueChanged_string, key))
			elif editType == 'if':
				testif = default
				ifvalue = (values[testif] == True)
				continue
			elif editType == 'endif': 
				testif = None
				continue
			elif editType == 'bool': 
				qedit = QtGui.QCheckBox(self)
				trues = [True,'yes','on',1, 2]# 2 is more true than 1
				falses = [False,'no','off',0]
				if value not in trues+falses: print ('WARNING: unexpected bool',value)
				qedit.setChecked(value in trues)
				qedit.stateChanged.connect( functools.partial(self.valueChanged_bool, key) )
			elif editType == 'short' or editType == 'int' or editType == 'long':
				qedit = QCore.QintWidget(self)
				qedit.setRange(field.get("min",None),field.get("max",None))
				qedit.setLocked(field.get("locked",False))
				if value is not None:
					assert str(value).isdigit(), 'WARNING bad field'+repr(field)+repr(value)
					qedit.setValue(value)
				qedit.valueChanged.connect(functools.partial(self.valueChanged_int,key))
			elif editType == 'float' or editType == 'double':
				qedit = QCore.QfloatWidget(self)
				qedit.setRange(field.get("min",None),field.get("max",None))
				qedit.setLocked(field.get("locked",False))
				if value is not None:
					qedit.setValue(value)
				qedit.valueChanged.connect(functools.partial(self.valueChanged_float,key))
			elif editType == 'float2' or editType == 'double2':
				qedit = QCore.QvectorWidget(2,self)
				if value is not None:
					qedit.setValue(np.array(value,dtype=np.float32).reshape(2))
				qedit.valueChanged.connect(functools.partial(self.valueChanged_float2,key))
			elif editType == 'float3' or editType == 'double3':
				qedit = QCore.QvectorWidget(3,self)
				if value is not None:
					qedit.setValue(np.array(value,dtype=np.float32).reshape(3))
				qedit.valueChanged.connect(functools.partial(self.valueChanged_float3,key))
			elif editType == 'float4' or editType == 'double4':
				qedit = QCore.QvectorWidget(4,self)
				if value is not None:
					qedit.setValue(np.array(value,dtype=np.float32).reshape(4))
				qedit.valueChanged.connect(functools.partial(self.valueChanged_float3,key))
			elif editType == 'matrix': # TODO Matrices should be able to be of arbitrary size
				qedit = QCore.QmatrixWidget(3,4,self)
				if value is not None:
					qedit.setValue(np.array((value),dtype=np.float32).reshape(3,4))
				qedit.valueChanged.connect(functools.partial(self.valueChanged_matrix,key))
			elif editType == 'select' or editType == 'enum':
				qedit = QCore.Qselect(self)
				enum_options = field.get('enum',[])
				for item in enum_options: qedit.addItem(item)
				if value is not None: qedit.setCurrentIndex(value if isinstance(value,int) else enum_options.index(value))
				qedit.currentIndexChanged.connect(functools.partial(self.valueChanged_enum,key))
			elif editType.startswith('vec_'):
				continue # don't show vecs for now
			else:
				print ('unknown type',editType,key)
				qedit = QtGui.QLineEdit(self)
				qedit.setText(str(value))
			qlabel = QtGui.QLabel(label)
			qlabel.setToolTip(hover)
			fi = self.gridLayout.rowCount()
			self.gridLayout.addWidget(qlabel,fi,0)
			if qedit != None:
				self.gridLayout.addWidget(qedit,fi,1)
				self.keyToQedit[key] = qedit
				if testif != None:
					self.connect(self.keyToQedit[testif], QtCore.SIGNAL("stateChanged(int)"), qedit.setEnabled)
					if not ifvalue: qedit.setEnabled(False)
		self.repaint()

	def valueChanged_bool(self, key, value):
		# Normalise Truth of tri-state bool to python True or False
		self.parent.setFieldValueCommand( key, (value>0) )
		
	def valueChanged_enum(self, key, value):
		self.parent.setFieldValueCommand(key, int(value))

	def valueChanged_int(self, key, value):
		self.parent.setFieldValueCommand(key, int(value))

	def valueChanged_float(self, key, value):
		self.parent.setFieldValueCommand(key, float(value))

	def valueChanged_float2(self, key, value):
		self.parent.setFieldValueCommand(key, value)

	def valueChanged_float3(self, key, value):
		self.parent.setFieldValueCommand(key, value)

	def valueChanged_matrix(self, key, value):
		self.parent.setFieldValueCommand(key, value)

	def valueChanged_string(self, key, value):
		self.parent.setFieldValueCommand(key, value)


if __name__ == '__main__':
	import sys
	app = QtGui.QApplication(sys.argv)

	win = QFieldsEditor()
	win.show()
	
	fields = [
		('filename', 'File name', 'Full-path to the file on disk. For sequences, choose any file from the sequence.', 'filename', None),
		('issequence',  'Is sequence', 'Is this an image sequence (checked), or a static image.', 'bool', False),
		(None, None, None, 'if', 'issequence'),
		('numdigits',  'Number of digits', 'The number of digits in the number sequence (pad with leading zeros). Choose 0 for no padding.', 'int', 0, {"min":0,"max":None}),
		('inframe',  'Start frame', 'If this is an image sequence, the first frame of the sequence.', 'int', 0, {"min":0,"max":None}),
		('outframe', 'End frame', 'If this is an image sequence, the last frame of the sequence.', 'int', 0, {"min":0,"max":None}),
		(None, None, None, 'endif', 'issequence'),
		('deinterlace', 'Deinterlace mode', 'If the video is interlaced, choose the deinterlace mode that gives continuous motion.', 'select', 'None', {'enum':['None','Odd only','Even only','Odd-Even','Even-Odd']}),
		('fps', 'Frames per second', 'Controls the playback speed. For deinterlaced video, use fields per second.', 'float', 24.0, {"min":0.0,"max":None}),
	]
	win.setFields('Image Settings', fields, {'fps':60.0, 'filename':'test.png','deinterlace':'Even-Odd', 'issequence':True, 'inframe':100,'outframe':200,'numdigits':4})
	app.connect(app, QtCore.SIGNAL('lastWindowClosed()') , app.quit)
	sys.exit(app.exec_())