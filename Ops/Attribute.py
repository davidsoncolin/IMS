import Op, Interface


class SetAttribute(Op.Op):
	def __init__(self, name='/SetAttribute', locations='', attrName='', attrValue='', attrType='float', frameRange=''):
		fields = [
			('name', 'name', 'Op name', 'string', name, {}),
			('locations', 'locations', 'Locations containing the attribute', 'string', locations, {}),
			('attrName', 'Attribute name', 'Attribute target name', 'string', attrName, {}),
			('value', 'Value', 'Attribute value', 'string', attrValue, {}),
			('frameRange', 'Frame range', 'Frame range', 'string', frameRange, {}),
			('type', 'Type', 'Attribute type', 'select', attrType, {'enum': ('float', 'int', 'str')})
		]

		super(self.__class__, self).__init__(name, fields)

	def cook(self, location, interface, attrs):
		if not self.useFrame(interface.frame(), attrs['frameRange']): return

		name = attrs['name']
		self.name = name
		attrName = attrs['attrName']
		value = attrs['value']
		if attrs['type'] == 'float': value = float(value)
		elif attrs['type'] == 'int': value = int(value)

		interface.setAttr(attrName, value)


class SetVisibility(Op.Op):
	def __init__(self, name='/SetVisibility', locations='', visibility=True):
		fields = [
			('name', 'name', 'Op name', 'string', name, {}),
			('locations', 'Locations', 'Locations', 'string', locations, {}),
			('visibility', 'Visibility', 'Visibility', 'bool', visibility, {})
		]

		super(self.__class__, self).__init__(name, fields)

	def cook(self, location, interface, attrs):
		interface.setAttr('visible', attrs['visibility'], forceCreate=False)


class CopyAttributes(Op.Op):
	def __init__(self, name='/CopyAttributes', locations='', sourceLocation='', attributeNames=''):
		fields = [
			('name', 'name', 'Op name', 'string', name, {}),
			('locations', 'Locations', 'Locations', 'string', locations, {}),
			('sourceLocation', 'Source location', 'Source location', 'string', sourceLocation, {}),
			('attributeNames', 'Attribute names', 'Attribute names', 'string', attributeNames, {})
		]

		super(self.__class__, self).__init__(name, fields)

	def cook(self, location, interface, attrs):
		if not attrs['sourceLocation']:
			self.logger.info('No source location specified')
			return

		if not attrs['attributeNames']:
			self.logger.info('No attribute names specified')
			return

		source = attrs['sourceLocation']
		attributeNamesList = self.resolveLocations(attrs['attributeNames'], prefix='')
		for attrName in attributeNamesList:
			attr = interface.attr(attrName, atLocation=source)
			if attr is not None:
				interface.setAttr(attrName, attr)


# Register Ops
import Registry
Registry.registerOp('Attribute Set', SetAttribute)
Registry.registerOp('Visibility Set', SetVisibility)
Registry.registerOp('Copy Attributes', CopyAttributes)
