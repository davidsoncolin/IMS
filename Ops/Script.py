import Op, Registry


class Python(Op.Op):
	def __init__(self, name='/Python Script', locations='/root', script='', frameRange=''):
		fields = [
			('name', 'Name', 'name', 'string', name, {}),
			('locations', 'Locations', 'Locations', 'string', locations, {}),
			('script', 'Script', 'Script', 'text', script, {}),
			('frameRange', 'Frame range', 'Frame range', 'string', frameRange, {})
		]

		super(self.__class__, self).__init__(name, fields)

	def cook(self, location, interface, attrs):
		if not self.useFrame(interface.frame(), attrs['frameRange']): return
		if not attrs['script']: return
		script = attrs['script']

		# Evaluate/execute command and report results
		import sys
		try:
			result = None
			try:
				result = eval(script, globals(), locals())
			except SyntaxError:
				exec(script, globals(), locals())

			# Check if the evaluation was successful and if so report it in the results field
			# Add the results to the environment
			if result is not None:
				message = str(result)
				self.environment['_'] = message
				print(message)

		except:
			# Get the traceback information and add the formatted output to the results field
			import traceback, sys
			exceptionType, exception, tb = sys.exc_info()
			entries = traceback.extract_tb(tb)
			entries.pop(0)

			# Build and print a list containing the error report
			lines = []
			if entries:
				lines += traceback.format_list(entries)

			lines += traceback.format_exception_only(exceptionType, exception)
			for line in lines:
				print line


# Register Ops
Registry.registerOp('Python Script', Python)
