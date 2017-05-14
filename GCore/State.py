#!/usr/bin/env python

import IO

def freeze(v): return str(IO.encode(v))
def thaw(s): ret,e = IO.decode(s); assert(e==len(s)); return ret

g_state = {}
g_dirty = set() # keys that have been modified

from re import compile as re_compile
g_reLastNum = re_compile(r'(?:[^\d]*(\d+)[^\d]*)+')
g_undos,g_redos,g_undo = [],[],{}

primarySelection = None # TODO, sort this out properly!

# State is a registry of key/value pairs
# Leaf keys hold a value which is frozen as a string
# Other ('FieldDict') keys hold a dict of key/value pairs where the key is a field and the value is the key for that data

# g_state is a dict of key/values. the values are strings, dicts or lists
# values are encoded by being frozen to strings
# dicts have their values encoded as keys
# lists have their elements encoded as keys
# there is a root dict at key ''; other keys start with '/'

# # How to:
# addKey(whatever) # commands ...
# setKey(whatever)
# delKey(whatever)
# push('user description')
# # the added commands form a fragment until the next push; then the user description appears in the undo stack
# # if the same keys are set multiple times, only the last value is stored (to prevent huge accumulations)
# # if the same command is repeated, it is concatenated with the previous command (so that undo undoes both)

## Utility Function ##

# TODO rethink this; we could store only a <set> or <list nub> and generate the dict/list from the keys
# after State.setKey('/images',[0])
# currently: {'':{'images':'/images'}, '/images':['/images/0'], '/images/0':0}
# future: {'':set('images'), '/images':list, '/images/0':0}
def keyType(v):
	if isinstance(v,dict): return 1
	if isinstance(v,list): return 2
	return 0

def getSel(): global primarySelection; return primarySelection
def setSel(sel): global primarySelection; primarySelection = sel
def version(): return getKey('/version/version')
def appName(): return getKey('/version/app')
def uniqueKeys(): global g_state; return g_state[''].viewkeys() # the root key, contains all the root objects
def hasKey(k): global g_state; return k in g_state
def allKeys(): global g_state; return g_state.viewkeys()

def refKey(path, create=False):
	'''return reference information for a particular path'''
	global g_state, g_undo
	path = path.strip('/').split('/')
	key = '' # NB the root key is the empty string
	for f in path:
		parent_key = key
		if create and parent_key not in g_state:
			g_state[parent_key] = {}
			g_undo.setdefault('del',set()).add(parent_key)
			g_undo.setdefault('add',{}).pop(parent_key,None) # in one or the other
		parent = g_state[parent_key]
		if isinstance(parent,list):
			f = int(f)
			key = parent_key+'/'+str(f)
			if create and f == len(parent):
				parent.append(key) # allow appends to lists!
				g_undo.setdefault('del',set()).add(key)
				g_undo.setdefault('add',{}).pop(key,None) # in one or the other
		elif isinstance(parent,dict):
			key = parent_key+'/'+f
			if create and f not in parent:
				parent[f] = key
				g_undo.setdefault('del',set()).add(key)
				g_undo.setdefault('add',{}).pop(key,None) # in one or the other
		else:
			print 'State.refKey what?',path,create,f,type(parent)
			assert False
	return key, parent, f, parent_key

def getKeys(l):
	return [getKey(k) for k in l]

def getKey(k, default=Exception, depth=0):
	if depth == 10: return default
	if k is None: return None
	global g_state
	k = '/'+k.strip('/')
	if k == '/': k = '' # allow the root key
	v = g_state.get(k,None)
	if v is None:
		assert default is not Exception,'missing key '+k
		return default
	t = keyType(v)
	if t == 0: return thaw(v)
	if t == 1: return dict([(k,getKey(vv,default=default,depth=depth+1)) for k,vv in v.iteritems()])
	if t == 2: return [getKey(vv,default=default,depth=depth+1) for vv in v]

def subkeys(k, default=Exception):
	'''return the subkeys for a particular key. returns None if the key is a value.'''
	if k is None: return None
	global g_state
	k = '/'+k.strip('/')
	if k == '/': k = '' # allow the root key
	v = g_state.get(k,None)
	if v is None:
		assert default is not Exception,'missing key '+k
		return default
	t = keyType(v)
	if t == 0: return None
	if t == 1: return v.keys()
	if t == 2: return range(len(v))

def nextKey(k):
	'''Generate the next numbered key'''
	global g_state, g_reLastNum
	if not g_state.has_key(k) and not g_state.has_key('/' + k): return k
	ki,kj,kv = len(k),len(k),1
	match = g_reLastNum.search(k)
	if match:
		ki, kj = match.span(1)
		kv = int(k[ki:kj])
		ki = max(ki, kj - len(str(kv))) # in case of leading zeros
	ks = k[:ki]+'%d'+k[kj:]
	while True:
		nk = (ks % kv)
		if not g_state.has_key(nk): return nk
		kv += 1


## Undo Functions

def getUndoCmd():
	global g_undos
	if g_undo: return g_undo.get('cmd','fragment')
	if not g_undos: return None
	return g_undos[-1].get('cmd','whoops')

def push(cmd):
	'''Name the command and push onto the undo stack.'''
	global g_undo, g_undos, g_redos
	#print 'pushing',cmd
	#if not g_undo: print 'warning: nothing to push'
	if g_undo:
		g_undo['cmd'] = cmd
		# test for concatentation
		# concatenate repeated commands that only modify the same keys (eg dragging a slider)
		if g_undos:
			tmp = g_undos[-1]
			# sometimes commands set other keys...
			if set(tmp.viewkeys()) == set(['cmd','set']) and tmp.viewkeys() == g_undo.viewkeys() and tmp['cmd'] == cmd:
				g_undo['set'].update(tmp['set'])
				g_undos.pop()
		g_undos.append(g_undo)
	g_undo = {}
	g_redos = []


def undo():
	global g_undo, g_undos, g_redos
	if not g_undo and g_undos: g_undo = g_undos.pop()
	if g_undo:
		g_redos.append(repeat(g_undo))
		g_undo = {}
	
def getRedoCmd():
	'''peek at the redo cmd; return None if there is none'''
	global g_redos
	if not g_redos: return None
	return g_redos[-1].get('cmd','whoops')

def redo():
	global g_undo, g_undos, g_redos
	if g_undo:
		print 'state.redo warning'
		g_undos.append(g_undo)
		g_undo = {}
	if not g_redos: return None
	g_undos.append(repeat(g_redos.pop()))

def repeat(undo):
	'''redo a command while generating the undo command'''
	global g_state, g_dirty
	redo = {'cmd':undo.get('cmd','fragment')}
	g_dirty.update(undo.get('del',[]))
	dels = sorted(undo.get('del',[]))
	for k in dels[::-1]:
		if k in g_state:
			redo.setdefault('add',{})[k] = _delKey(k)
	g_dirty.update(undo.get('add',{}).viewkeys())
	adds = undo.get('add',{})
	for k in sorted(adds.viewkeys()):
		redo.setdefault('del',set()).add(k)
		_setKey(k, adds[k], do_freeze=False)
	g_dirty.update(undo.get('set',{}).viewkeys())
	for k,v in undo.get('set',{}).iteritems():
		#print 'set',k, k in g_state
		redo.setdefault('set',{})[k] = _delKey(k,False)
		_setKey(k, v, do_freeze=False)
		#g_state[k] = v
	return redo

## Key Creation and Editing

def addKey(k,v):
	'''Add a key/value pair to the dictionary. NOTE the actual key is returned, which may be different from the requested one.'''
	return setKey(nextKey(k),v)

def setKey(k,v):
	'''Update the value for a given key, or add a new key if it doesn't exist.'''
	assert isinstance(k,str),'k should be a str not '+str(type(k))
	global g_state, g_undo
	k = '/'+k.strip('/')
	if k == '/': k = '' # allow modifying the root key, for experts only!
	has_k = hasKey(k)
	k = refKey(k, create=True)[0]
	if has_k: g_undo.setdefault('set',{}).setdefault(k,_delKey(k, False))
	else:
		g_undo.setdefault('add',{}).pop(k,None)
		g_undo.setdefault('del',set()).add(k)
	_setKey(k, v)
	return k

def _setKey(k, v, do_freeze=True):
	# an internal function that doesn't touch the undo stack. set a key
	global g_state, g_dirty
	t = keyType(v)
	k, parent, f, parent_key = refKey(k)
	if isinstance(parent,list) and f == len(parent): parent.append(k)
	if isinstance(parent,dict) and f not in parent: parent[f] = k
	   
	# only dicts where all keys are strings are deep-indexed
	if t == 1 and not all(isinstance(vk,str) for vk in v.iterkeys()):
		#print 'WARNING: %s is a dict with keys of type %s' % (k,type(v.keys()[0]))
		t = 0

	if t == 1:
		dct = {}
		g_state[k] = dct
		for vk,vv in v.items():
			assert isinstance(vk,str),'should be a str, not '+str(type(vk))
			kk = k+'/'+vk
			dct[vk] = kk
			_setKey(kk,vv,do_freeze)
	elif t == 2:
		dct = []
		g_state[k] = dct
		for vk,vv in enumerate(v):
			kk = k+'/'+str(vk)
			dct.append(kk)
			_setKey(kk,vv,do_freeze)
	else:
		g_state[k] = freeze(v) if do_freeze else v
	g_dirty.add(k)

## Key Deletion

def delKey(k):
	global g_undo
	k = '/'+k.strip('/')
	g_undo.setdefault('add',{})[k] = _delKey(k)
	g_undo.setdefault('del',set()).discard(k) # in one or the other

def _delKey(key, unlink_parent=True):
	'''
	:param key: The Key in the State dict to remove
	:param unlink_parent: Whether to remove the link from the parent (delete, must not be mid-list) or not
	:return: the value of the key

	A private function that removes a key from the dict but does not add anything to the undo stack. Doesn't unlink the parent.
	'''
	global g_state, g_dirty
	if unlink_parent:
		key, parent, field, parent_key = refKey(key)
		g_dirty.add(parent_key)
		pt = keyType(parent)
		if pt == 1: parent.pop(field,None)
		elif pt == 2:
			field = int(field)
			# TODO this could go wrong if you extended a list by multiple keys in a single command
			assert field == len(parent)-1,'can only remove the last key in a list: '+repr(field)+"!="+repr(len(parent)-1)+': '+key
			parent.pop()
	v = g_state.pop(key)
	g_dirty.add(key)
	t = keyType(v)
	if t == 0: return v # NB not thawed here
	if t == 1: return dict([(k,_delKey(vv, False)) for k,vv in v.iteritems()])
	if t == 2: return [_delKey(vv, False) for vv in v]

## Save and Load


def save(filename):
	print 'saving',filename
	global g_state
	IO.save(filename, g_state)
	push('save')

def load(filename):
	print 'loading',filename
	tmp = IO.load(filename)[1]
	load_version = thaw(tmp['/version/version'])
	load_app = thaw(tmp['/version/app'])
	# don't load bad version or app
	assert(load_version == version()),'file has a bad version:'+str(load_version)
	assert(load_app == appName()),'file has a bad app name:'+str(load_app)
	new(tmp)

def clearUndoStack():
	global g_undos, g_redos,g_undo
	g_undos,g_redos,g_undo = [],[],{}

def new(state=None):
	global g_state,g_dirty
	g_dirty.update(g_state.viewkeys()) # all the keys have changed
	g_state = {} if state is None else state
	g_dirty.update(g_state.viewkeys()) # all the keys have changed
	setKey('/version',{'type':'note','version':'0.0', 'app':'Imaginarium App'})
	clearUndoStack()
	setSel(None)

new()

if __name__ == "__main__":
	test_range = 10 # Set the number of keys to test
	verbose = True

	print '-- test undo/redo --'
	state = dict([('key'+str(v),{'v':[1,2,3,'value'+str(v)]}) for v in range(10)])
	print state
	setKey('/test',state)
	push('set keys')
	assert getKey('/test') == state
	setKey('/test/key5/v/2', 'yoyo')
	state['key5']['v'][2] = 'yoyo'
	push('set key5')
	assert getKey('/test') == state
	delKey('/test/key5')
	key5 = state.pop('key5')
	push('del key 5')
	assert getKey('/test') == state
	undo()
	state['key5'] = key5
	assert getKey('/test') == state
	redo()
	key5 = state.pop('key5')
	assert getKey('/test') == state
	undo()
	state['key5'] = key5
	undo()
	state['key5']['v'][2] = 3
	assert getKey('/test') == state
	new()
	
	## Test Insertion ##

	print ("-- Testing insertion --")
	test_keys = []
	for i in xrange(test_range):
		test_key = 'test/key'+str(i)
		test_value = 'test_value ' + str(i)
		test_keys.append(addKey(test_key,test_value))
		key_found = hasKey(test_keys[i])
		state_value = getKey(test_keys[i])
		if verbose:
			print ("Adding key: {} - With value: {} - Key found: {} - Key value: {}".format(
				test_key, test_value, key_found, state_value
			))
		assert key_found, "Test key {} not found\nFailed Key: {}".format(i, test_keys[i])
		assert state_value == test_value, 'Value {} was not expected value\nExpected: {} - Received: {}'.format(
			i,test_value,state_value
		)
		key, parent, f, parent_key = refKey(test_keys[i])
		has_child_key = parent[f] == test_keys[i]
		print ('Parent: {} - Parent has child key: {}'.format(parent_key, has_child_key))

	print ("\nInsertion completed successfully - No issues found\n")

	## Test Value Updates
	# TODO Try setting different data types

	print ("-- Testing Value Updates --")
	for ki, key in enumerate(test_keys):
		old_val = getKey(key)
		new_val = old_val[::-1]
		setKey(key,new_val)
		state_value = getKey(key)
		if verbose:
			print ("From: {} - To: {} - Received: {}".format(old_val,new_val,state_value))
		assert new_val == state_value, 'Key update {} failed on key: {}'.format(ki, key)

	print ("\nUpdates completed successfully - No issues found\n")

	## Test Key Deletion

	print ("-- Testing Key Deletion --")

	for ki, key in enumerate(test_keys):
		delKey(key)
		key_delete = not hasKey(key)
		if verbose:
			print ("Deleting key: {} - Key deleted: {}".format(key,key_delete))
		assert key_delete, 'Deletion {} failed on key: {}'.format(ki,key)

	print ("\nDeletions completed successfully - No issues found\n")
