
registry = {}
def registerOp(name, type):
	registry[name] = type

def getRegisteredOps():
	return registry
