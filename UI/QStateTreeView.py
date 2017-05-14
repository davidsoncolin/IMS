from PySide import QtCore, QtGui
from GCore import State

'''the idea here is to make a treeview that reflects (part of) the State tree.
TODO it should update when the State changes
TODO it should synchronize with the State selection
'''
class QStateTreeView(QtGui.QTreeView):
	def __init__(self, parent=None, root=''):
		QtGui.QTreeView.__init__(self, parent)
		self.setModel(QtGui.QStandardItemModel(1,2))
		self.setSelectionModel(QtGui.QItemSelectionModel(self.model()))
		self.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
		self.setUniformRowHeights(True)
		self.setSortingEnabled(True)
		self.sortByColumn(0, QtCore.Qt.AscendingOrder)
		self.set_root(root)
		#self.selectionModel().currentChanged.connect(self.print_sel)

	def index_to_item(self, index):
		return self.model().itemFromIndex(index)

	def item_to_index(self, item):
		return self.model().indexFromItem(item)

	#def print_sel(self, index):
		#print self.index_to_item(index).data()

	def set_root(self, root):
		self.root = root
		self.model().setHorizontalHeaderLabels(['key','val'])
		self.generate_item_children(self.model())
		#self.expanded.connect(self.expanded_callback) # expand stubs when clicked on

	def expanded_callback(self, index):
		self.generate_item_children(self.index_to_item(index))

	def generate_item_children(self, parent):
		root = self.root if parent is self.model() else str(parent.data())
		parent.setRowCount(0) # remove any pre-existing stubs or rows
		elements = State.subkeys(root)
		if elements and isinstance(elements[0], str):
			for text in sorted(elements): # TODO order should match sort mode
				key = '%s/%s'%(root,text)
				sk = State.subkeys(key)
				item = QtGui.QStandardItem(self.tr(text))
				item.setData(key)
				if isinstance(sk, list):
					#item.appendRow(QtGui.QStandardItem(self.tr(''))) # make a stub for subkeys
					self.generate_item_children(item) # TODO the whole point is to evaluate this lazily; 
					# but it causes crashes of course when synchronizing the treeview to match the
					# state selection (which changed by the user clicking on a geo).
					parent.appendRow(item)
				else:
					v = State.g_state[key]
					v = v[0] if v[0] in '<{[(' else str(State.thaw(v))
					item2 = QtGui.QStandardItem(self.tr(v))
					parent.appendRow([item,item2])


	def generate_path_to(self, key):
		'''follow a path to find the item for a key. expand all stubs until the key exists.'''
		def find_item(parent, p):
			for i in xrange(parent.rowCount()):
				child = parent.child(i)
				if child.text() == p: return child
			return None
		if key.startswith(self.root): key = key[len(self.root):]
		parent = self.model()
		path = key.strip('/').split('/')
		if path == []: return parent
		if parent.rowCount()==1 and parent.item(0).text()=='': self.generate_item_children(parent)
		parent = parent.findItems(path.pop(0))
		if parent == []: return None
		parent = parent[0]
		for p in path:
			if parent.rowCount()==1 and parent.child(0).text() == '': self.generate_item_children(parent) # stub
			parent = find_item(parent, p)
			if parent is None: return parent
		return parent

	def sync(self):
		'''synchronize the selection with the State selection'''
		index = self.selectionModel().currentIndex()
		key = State.getSel()
		if key is None: return
		if index.isValid():
			item = self.index_to_item(index)
			if item.data() == key: return
		item = self.generate_path_to(key)
		self.selectionModel().setCurrentIndex(self.item_to_index(item), QtGui.QItemSelectionModel.ClearAndSelect)

class QSTVWidget(QtGui.QWidget):
	def __init__(self, parent=None, root=''):
		QtGui.QWidget.__init__(self, parent)
		self.layout = QtGui.QVBoxLayout()
		self.treeView = QStateTreeView(self, root)
		self.layout.addWidget(self.treeView)
		self.setLayout(self.layout)

if __name__ == "__main__":
	import sys
	data = {'Alice':{'Keys':[0,1,2],'Purse':{'Cellphone':'nope'}},'Bob':{'Wallet':{'Credit card':53,'Money':[[0,1],[1,2]]}}}
	State.setKey('/doc',data)
	app = QtGui.QApplication([])
	window = QSTVWidget(root='/doc')
	window.show()
	sys.exit(app.exec_())
