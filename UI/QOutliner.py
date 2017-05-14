from Example import GRIP
from PySide import QtCore, QtGui

import logging
LOG = logging.getLogger(__name__)

from GCore import *
from UI import T_SKELETON, T_CAMERAS, T_MOTION, K_NAME

# the individual component of the path this represents. eg like, a, file, path
PATH_ROLE = QtCore.Qt.UserRole + 1
# the complete path to this item eg /like/a/file/path
FULLPATH_ROLE = QtCore.Qt.UserRole + 2

OK_ROLE = QtCore.Qt.UserRole + 3

RTG_ROLE = QtCore.Qt.UserRole + 4

TYPE_TYPE = 'TYPE'

def createItem(parent, obj):
	if obj['objecttype'] == T_CAMERAS:
		return CameraGrpItem(parent, obj)
	elif obj['objecttype'] in [T_MOTION, T_SKELETON]:
		return SkelItem(parent, obj)
	elif obj['objecttype'] in ['retarget_type']:
		return RetargetItem(parent,obj)
	else:
		return NullItem(parent, obj)


class BaseItem(object):
	'''
	the base item is the invisible root item and a base class for all the other item types.'''
	def __init__(self, parent, obj):
		self.parent = parent
		self.name = '/root'
		self.obj = obj
		self.children = []
		self.childCounter = 0
		self.queried = False
		self.ok = None

	def child(self, row):
		return self.children[row]

	def row(self):
		if self.parent:
			return self.parent.children.index(self)
		return 0

	def removeRow(self, row):
		del self.children[row]
		self.childCounter -= 1

	def path(self):
		return self.name

	def fullpath(self):
		if self.parent and self.parent.fullpath():
			return self.parent.fullpath() + "/" + self.path()
		return  "/" + self.path()

	def tooltip(self):
		return self.fullpath()

	def rowCount(self):
		return self.childCounter

	def addChildren(self,children):
		self.children += [createItem(self,child) for child in children]
		self.childCounter = len(self.children)
		self.queried = True

	def hasChildren(self):
		return bool(self.childCounter)

class RetargetItem(BaseItem):
	def __init__(self,parent,obj):
		super(RetargetItem,self).__init__(parent,obj)
		self.name = obj['data']['name']
		self.rtg_key = obj['data']['rtg_key']

	def rowCount(self):
		# Retarget Operation currently does not have any children
		if not self.queried:
			self.children = []
			self.childCounter = 0
			self.queried = True
		return self.childCounter

class TypeItem(BaseItem):
	def __init__(self, parent, obj):
		super(TypeItem,self).__init__(parent,obj)
		self.name = obj['data']['name']
		self.contains_types = obj['data']['contains']

	def rowCount(self):
		if not self.obj:
			self.queried = True
		if not self.queried:
			self.children = [createItem(self, child) for child in self.obj['children']]
			self.childCounter = len(self.children)
			self.queried = True
		return self.childCounter

	def hasChildren(self):
		return bool(self.childCounter)


class NullItem(BaseItem):
	def __init__(self, parent, obj):
		super(NullItem, self).__init__(parent, obj)
		self.name = obj['data'][K_NAME]

	def rowCount(self):
		if not self.obj:
			self.queried = True
		if not self.queried:
			self.children = [createItem(self, child) for child in self.obj['children']]
			self.childCounter = len(self.children)
			self.queried = True
		return self.childCounter

	def hasChildren(self):
		return bool(self.childCounter)


class SkelItem(BaseItem):
	def __init__(self, parent, obj):
		super(SkelItem, self).__init__(parent, obj)
		self.name = obj['data'][K_NAME]
		self.hi=False

	def rowCount(self):
		if not self.queried:
			self.children=[]
			if self.hi:
				#jntd = {x:None for x in range(48)}
				jntd={}
				#for k,v in self.obj.skelDict['jointIndex'].iteritems():
				#for i, name in enumerate(self.obj.skelDict['jointNames'], self.obj.skelDict['jointParents'])):
				for v,x in enumerate(self.obj['skelDict']['jointParents']):
					try:
						parent = jntd[x]
					except KeyError:
						parent=self
					jntd[v] = HiJointPrimitiveItem(parent,self.obj['skelDict']['jointNames'][v],v)
					jntd[v].root = self
					parent.children.append(jntd[v])
					'''
					if x == -1:
						self.children.append(jntd[v])
						jntd[v].parent=self
					else:
						jntd[x].children.append(jntd[v])
						jntd[v].parent=jntd[x]
					#jnt[index HiJointPrimitiveItem(name,index)
					'''
					
				#self.children = [HiJointPrimitiveItem(self, name, index) for name, index in zip(self.obj.skelDict['jointNames'], self.obj.skelDict['jointParents']) if index == -1]
			else:
				self.children = [JointPrimitiveItem(self, name, index) for index, name in enumerate(self.obj['skelDict']['jointNames'])]

			self.childCounter = len(self.children)
			self.queried = True
		return self.childCounter

	def hasChildren(self):
		return True


class HiJointPrimitiveItem(BaseItem):
	def __init__(self, parent, name, index):
		super(HiJointPrimitiveItem, self).__init__(parent, name)
		# hack to find root item directly
		self.root = parent
		self.index = index
		self.name = name
		self.children=[]
		# temp hack so that bones can look up their skel object
		self.root = None

	def rowCount(self):
		return len(self.children)

	def hasChildren(self):
		return bool(len(self.children))

class JointPrimitiveItem(BaseItem):
	def __init__(self, parent, name, index):
		super(JointPrimitiveItem, self).__init__(parent, name)
		self.index = index
		self.name = name

	def rowCount(self):
		return 0

	def hasChildren(self):
		return False


class CameraPrimitiveItem(BaseItem):
	def __init__(self, parent, primitive, index):
		super(CameraPrimitiveItem, self).__init__(parent, primitive)
		self.index = index
		self.name = primitive.names[index]

	def rowCount(self):
		return 0

	def hasChildren(self):
		return False

class CameraGrpItem(NullItem):
	def __init__(self, parent, obj):
		super(CameraGrpItem, self).__init__(parent, obj)

	def query(self):
		if not self.obj:
			self.queried = True
		if not self.queried:
			self.children = [CameraPrimitiveItem(self, self.obj.primitive, i) for i in range(len(self.obj.primitive))]

			self.childCounter = len(self.children)
			self.queried = True

	def hasChildren(self):
		return True

class QOutlinerModel(QtCore.QAbstractItemModel):
	def __init__(self, parent):
		super(QOutlinerModel, self).__init__(parent)
		self.rootNodes = []

	def setScene(self, topItems):
		# temp, until adding is done properly
		self.topItems = topItems
		self.refresh()

	def refresh(self):
		self.beginResetModel()
		skel_Type = TypeItem(None,{'objecttype':TYPE_TYPE,'data':{'name':'Skeletons','contains':[T_SKELETON]},'children':[]})
		motions_Type = TypeItem(None,{'objecttype':TYPE_TYPE,'data':{'name':'Motions','contains':[T_MOTION]},'children':[]})
		retargets_Type = TypeItem(None,{'objecttype':TYPE_TYPE,'data':{'name':'Retargets','contains':[GRIP.RETARGET_TYPE]},'children':[]})
		self.rootNodes=[]
		self.rootNodes.append(skel_Type)
		self.rootNodes.append(motions_Type)
		self.rootNodes.append(retargets_Type)
		for obj in self.topItems:
			type_found = False
			for type_container in self.rootNodes:
				if obj['objecttype'] in type_container.contains_types:
					type_container.addChildren([obj])
					type_found = True
					break
			if not type_found:
				self.rootNodes.append(createItem(None,obj))
		index = QtCore.QModelIndex()
		self.propagateRows(index)
		self.endResetModel()
		# self.rootNodes = [createItem(None, obj) for obj in self.topItems]

	def propagateRows(self,parentIndex=None):
		# Recursively generate rows in the outliner for the data
		parentIndex = QtCore.QModelIndex() if parentIndex is None else parentIndex
		row_count = self.rowCount(parentIndex)
		for row in xrange(0, row_count):
			index = self.index(row, 0, parentIndex)
			self.propagateRows(index)

	def add(self, obj):
		# assume sure obj isn't already in the model (would be slow to check every time)
		# rely on the object's parent being in the model
		for type_item in self.rootNodes:
			if obj['objecttype'] in type_item.contains_types:
				parentIndex = self.indexOf(type_item.obj)
				self.beginInsertRows(parentIndex, type_item.rowCount(), type_item.rowCount())
				type_item.addChildren([obj])
				self.endInsertRows()
				return
		self.beginInsertRows(QtCore.QModelIndex(), len(self.rootNodes), len(self.rootNodes))
		insert_item = createItem(None,obj)
		self.rootNodes.append(insert_item)
		self.endInsertRows()

	def remove(self, obj):
		# what row is the obj in under it's parent?
		objIndex = self.indexOf(obj)
		if objIndex is None: # Not in Model
			return
		item = objIndex.internalPointer()
		parentItem = item.parent
		if parentItem:
			row = item.parent.children.index(item)
			pindex = objIndex.parent()
			self.beginRemoveRows(pindex,row,row)
			parentItem.removeRow(row)
		else:
			row = self.rootNodes.index(item)
			pindex = QtCore.QModelIndex()
			self.beginRemoveRows(pindex,row,row)
			del self.rootNodes[row]
			
		self.endRemoveRows()

	def removeRow(self, row, parentObject):
		parentIndex = self.indexOf(parentObject)
		self.beginRemoveRows(parentIndex, row, row)
		if not parentIndex.isValid():
			del self.rootNodes[row]
		else:
			item = parentIndex.internalPointer()
			item.removeRow(row)
		self.endRemoveRows()

	def iterIndexes(self, parentIndex):
		''' yield all child indexes, recursively under the provided parent index '''
		item =  parentIndex.internalPointer()
		# if item: print "Searching Node: %s" % item.name
		row_count = self.rowCount(parentIndex)
		for row in xrange(0, row_count):
			index = self.index(row, 0, parentIndex)
			yield index
		for row in xrange(0, row_count):
			index = self.index(row, 0, parentIndex)
			for x in self.iterIndexes(index):
				yield x


	def find(self, parentIndex, name, role=QtCore.Qt.DisplayRole):
		''' search the model under parentIndex for the item called
		"name". return the model index of that item 
		searches recursively, completing the search of each row before searching child items'''
		for index in self.iterIndexes(parentIndex):
			test_obj = self.data(index, role)
			try:
				if test_obj == name:
					return index
			except:
				continue
		'''
		for row in xrange(0, self.rowCount(parentIndex)):
			index = self.index(row, 0, parentIndex)
		for row in xrange(0, self.rowCount(parentIndex)):
			index = self.index(row, 0, parentIndex)
			i =  self.find(index, name, role)
			if i: return i
		return None
		'''

	def indexOf(self, obj):
		''' will search the model recursively. returns the index of the item that represents the
		provided object '''
		if not obj: return QtCore.QModelIndex()
		return self.find(QtCore.QModelIndex(), obj, role=QtCore.Qt.UserRole)

	####################### QAbtractItemModel methods
	def data(self, index, role):
		if not index.isValid():
			return None

		item = index.internalPointer()
		if role == QtCore.Qt.DisplayRole:
			return item.name
		elif role == QtCore.Qt.UserRole:
			# hack until primitive selection resolved
			return item.obj if 'data' in item.obj else item.parent.obj
		elif role == QtCore.Qt.ToolTipRole:
			return item.tooltip()
		elif role == PATH_ROLE:
			return item.path()
		elif role == FULLPATH_ROLE:
			return item.fullpath()
		elif role == OK_ROLE:
			return item.ok
		elif role == RTG_ROLE:
			return item.rtg_key if hasattr(item,'rtg_key') else None
		#elif role == QtCore.Qt.BackgroundColorRole:
		#	return [None, QtGui.QColor(0,255,0)][item.ok]
		return None

	def setData(self, index, role, value):
		if role != OK_ROLE:
			return
		if not index.isValid(): return
		
		item = index.internalPointer()
		if not isinstance(item, JointPrimitiveItem): # hack to make sure that only joints get this filter property.
			return
		item.ok = value
		self.dataChanged.emit(index, index)

	def columnCount(self, index):
		return 1

	def index(self, row, column, parent):
		if not parent.isValid():
			return self.createIndex(row, column, self.rootNodes[row])
		parentNode = parent.internalPointer()
		return self.createIndex(row, column, parentNode.children[row])

	def parent(self, index):
		if not index.isValid():
			return QtCore.QModelIndex()
		node = index.internalPointer()
		if node.parent is None:
			return QtCore.QModelIndex()
		else:
			return self.createIndex(node.parent.row(), 0, node.parent)

	def rowCount(self, parent):
		if not parent.isValid():
			return len(self.rootNodes)
		node = parent.internalPointer()
		return node.rowCount()

	def hasChildren(self, index):
		''' should the row at index be expandable? '''
		if not index.isValid():
			return bool(self.rootNodes)
		item = index.internalPointer()
		return item.hasChildren()


if (__name__ == '__main__'):
	app = QtGui.QApplication([])
	# just show the menu model in a default TreeView

	from GCore.base import atdict
	d = [atdict({'data': {'name':'1parent'}, 'children':[], 'objecttype':T_SKELETON}), 
		 atdict({'data': {'name':'2parent'}, 'children':[atdict({'data':{'name':'2.1'}, 'children':[], 'objecttype':None}), atdict({'data':{'name':'2.2'}, 'children':[], 'objecttype':None})], 'objecttype':T_SKELETON})]
	# here's the column widget, standalone
	widget = QtGui.QTreeView()

	model = QOutlinerModel(widget)
	model.setScene(d*10)
	widget.setModel(model)
	widget.show()
	
	app.exec_()