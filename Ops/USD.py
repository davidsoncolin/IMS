import Op
import numpy as np

from pxr import Usd, UsdGeom

class Stage(Op.Op):
	def __init__(self, name='/UsdStage', locations='/root', filename=''):
		self.fields = [
			('name', 'name', 'name', 'string', name, {}),
			('locations', 'locations', 'locations', 'string', locations, {}),
			('filename', 'USD Filename', 'USD Filename (.usda)', 'string', filename, {})
		]

		super(self.__class__, self).__init__(name, self.fields)

	def cook(self, location, interface, attrs):
		if not attrs['filename']:
			self.logger.error('No filename specified')
			return

		filename = self.resolvePath(attrs['filename'])
		stage = Usd.Stage.Open(filename)
		if stage is None:
			self.logger.error('Error loading USD stage: %s' % filename)
			return

		for path in stage.Traverse():
			childName = path.GetName()
			typeName = path.GetTypeName()
			propertyNames = path.GetPropertyNames()

			childType = typeName
			attrs = {}
			# attrs['visibility'] = path.GetAttribute('visibility') if 'visibility' in propertyNames else True

			xform = interface.attr('xform')
			if xform is None: xform = np.eye(4, 4, dtype=np.float32)
			if 'xformOp:translate' in propertyNames:
				translate = np.float32(path.GetAttribute('xformOp:translate').Get())
				xform[:3, 3] = translate
				attrs['xform'] = xform

			if typeName == 'Xform':
				childType = 'group'

			elif typeName == 'Sphere':
				childType = 'primitive'
				attrs['primitiveType'] = typeName
				attrs['radius'] = float(path.GetAttribute('radius').Get())

			elif typeName == 'Cube':
				childType = 'primitive'
				attrs['primitiveType'] = typeName
				attrs['size'] = float(path.GetAttribute('size').Get())

			# print childName, childType
			# print path.GetPath().pathString
			# print '/root' + path.GetPath().pathString
			interface.createChild(path.GetPath().pathString[1:], childType, attrs=attrs)

		# xformSphere = np.eye(4, 4, dtype=np.float32)
		# xformCube = np.eye(4, 4, dtype=np.float32)
		#
		# xformSphere[:3, 3] = [1000, 0, 0]
		# xformCube[:3, 3] = [-1000, 1000, 500]
		#
		# sphereAttrs = {
		# 	'primitiveType': 'sphere', 'radius': 1000., 'slices': 100, 'stacks': 100,
		# 	'colour': (0, 1, 0, 0.7), 'xform': xformSphere
		# }
		# interface.createChild('sphere', 'primitive', attrs=sphereAttrs)
		#
		# cubeAttrs = {
		# 	'primitiveType': 'cube', 'size': 800.,
		# 	'colour': (0, 0, 1, 0.7), 'xform': xformCube
		# }
		# interface.createChild('cube', 'primitive', attrs=cubeAttrs)


# Register Ops
import Registry
Registry.registerOp('Import USD Stage', Stage)