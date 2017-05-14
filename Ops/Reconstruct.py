import numpy as np
import Op, Interface
import ISCV
from GCore import Recon


class PointsFromDetections(Op.Op):
	def __init__(self, name='/Reconstruct 3D from Dets', locations='', calibration='', tiltThreshold=0.0002, x2dThreshold=0.01,
	             x3dThreshold=30.0, minRays=3, seedX3ds='', showContributions=True, pointSize=8.0, colour=(1.0, 0.5, 0.0, 0.7),
	             setLabels=False, mesh='', visibilityLod='', intersection_threshold=100., generateNormals=False, frameRange=''):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('locations', 'Detection locations', 'Detection locations', 'string', locations, {}),
			('calibration', 'Calibration location', 'Calibration location', 'string', calibration, {}),
			('tilt_threshold', 'Tilt threshold', 'Slack factor for tilt pairing', 'float', tiltThreshold, {}),
			('x2d_threshold', 'Detection threshold', 'Detections threshold', 'float', x2dThreshold, {}),
			('x3d_threshold', '3D threshold', '3D threshold', 'float', x3dThreshold, {}),
			('min_rays', 'Min. number of rays', 'Minimum number of rays', 'int', minRays, {}),
			('seed_x3ds', '3D seed location', 'Existing 3D seed location', 'string', seedX3ds, {}),
			('show_contributions', 'Show contributions', 'Show camera contributions', 'bool', showContributions, {}),
			('pointSize', '3D Point size', '3D Point size', 'float', pointSize, {}),
			('colour', '3D Point colour', '3D Point colour', 'string', str(colour), {}),
			('setLabels', 'Set labels', 'Set labels', 'bool', setLabels, {}),
			('mesh', 'Mesh', 'Mesh location', 'string', mesh, {}),
			('visibilityLod', 'Visibility LOD location', 'Visibility LOD location', 'string', visibilityLod, {}),
			('intersection_threshold', 'Intersection threshold', 'Intersection threshold', 'float', intersection_threshold, {}),
			('generateNormals', 'Generate normals', 'Generate normals for visibility checks', 'bool', generateNormals, {}),
			('frameRange', 'Frame range', 'Frame range', 'string', frameRange, {})
		]

		super(self.__class__, self).__init__(name, fields)
		self.visibility = None

	def flush(self):
		self.visibility = None

	def cook(self, location, interface, attrs):
		if not self.useFrame(interface.frame(), attrs['frameRange']): return

		calibrationLocation = attrs['calibration']
		if not calibrationLocation: calibrationLocation = interface.root()

		# Get the mats from the calibration location
		mats = interface.attr('mats', atLocation=calibrationLocation)
		if mats is None:
			self.logger.error('Attribute mats not found at: %s' % calibrationLocation)
			return

		Ps = interface.attr('Ps', atLocation=calibrationLocation)
		if Ps is None:
			Ps = np.array([m[2] / (np.sum(m[2][0, :3] ** 2) ** 0.5) for m in mats], dtype=np.float32)

		# Get the detections from the location we are cooking
		# x2ds = interface.attr('x2ds')
		# x2ds_splits = interface.attr('x2ds_splits')
		x2ds = interface.attr('x2ds')
		x2ds_splits = interface.attr('x2ds_splits')
		x2ds_bright = interface.attr('x2ds', atLocation='/root/cameras/bright')
		x2ds_bright_splits = interface.attr('x2ds_splits', atLocation='/root/cameras/bright')

		if x2ds is None or x2ds_splits is None:
			self.logger.error('Detections not found at: %s' % location)
			return

		# Get configuration parameters
		tilt_threshold = attrs['tilt_threshold']
		x2d_threshold = attrs['x2d_threshold']
		x3d_threshold = attrs['x3d_threshold']
		min_rays = attrs['min_rays']
		seed_x3ds_location = attrs['seed_x3ds']
		seed_x3ds = None

		if min_rays < 2:
			self.logger.error('You need at least 2 rays but you specified the minimum to be: %d' % min_rays)
			return

		if seed_x3ds_location:
			seed_x3ds = interface.attr('x3ds', atLocation=seed_x3ds_location)

		if self.visibility is None: self.visibility = ISCV.ProjectVisibility.create()

		# Check if we have normals
		if attrs['mesh'] and interface.hasAttr('normals', atLocation=attrs['mesh']):
			normals = interface.attr('normals', atLocation=attrs['mesh'])
			self.visibility.setNormals(normals)

		# Check if we have visibility LODs
		if 'visibilityLod' in attrs and attrs['visibilityLod']:
			visibilityLod = interface.location(attrs['visibilityLod'])
			if visibilityLod is not None:
				lodTris = visibilityLod['tris']
				lodVerts = visibilityLod['verts']
				lodNormals = visibilityLod['faceNormals']
				tris = lodVerts[lodTris]
				cameraPositions = np.array([m[4] for m in mats], dtype=np.float32)
				self.visibility.setLods(tris, cameraPositions, np.concatenate((lodNormals)),
				                        attrs['intersection_threshold'], attrs['generateNormals'])

		# Calculate the 3D reconstructions from the detections
		# for n in range(min_rays, min_rays - 1, -1):
		# 	x3ds, labels = Recon.intersect_rays(x2ds, x2ds_splits, Ps, mats, seed_x3ds=seed_x3ds, tilt_threshold=tilt_threshold,
		#                                         x2d_threshold=x2d_threshold, x3d_threshold=x3d_threshold, min_rays=n)
		# 	seed_x3ds = x3ds
		x3ds, labels, _, _ = Recon.intersect_rays(x2ds, x2ds_splits, Ps, mats, seed_x3ds=seed_x3ds, tilt_threshold=tilt_threshold,
	                                        x2d_threshold=x2d_threshold, x3d_threshold=x3d_threshold, min_rays=min_rays,
	                                        numPolishIts=3, forceRayAgreement=True,
		                                    visibility=self.visibility)

		# Recon.intersect_rays(x2ds, splits, self.Ps, self.mats, seed_x3ds=x3ds, tilt_threshold=self.tilt_threshold,
		# 	                                                     x2d_threshold=self.x2d_threshold, x3d_threshold=self.x3d_threshold, min_rays=self.min_rays,
		# 	                                                     numPolishIts=settings.numPolishIts, forceRayAgreement=settings.forceRayAgreement,
		# 	                                                     visibility=settings.visibility)
		#  x3ds, labels = Recon.intersect_rays(x2ds, x2ds_splits, x2ds_bright, x2ds_bright_splits, Ps, mats, seed_x3ds=seed_x3ds, tilt_threshold=tilt_threshold,
	    #                                    x2d_threshold=x2d_threshold, x3d_threshold=x3d_threshold, min_rays=min_rays)

		if not x3ds.any() or not labels.any(): return
		x3ds_labels = np.arange(np.max(labels) + 1)

		if attrs['setLabels']:
			interface.setAttr('labels', labels)
		else:
			interface.setAttr('labels', [])

		# Find which cameras contribute to the 3D reconstructions (optional?)
		cameraPositions = np.array([m[4] for m in mats], dtype=np.float32)
		cameraContributions = {}
		for label3d in x3ds_labels:
			camIds = [interface.findCameraIdFromRayId(rayId, x2ds_splits) for rayId in np.where(labels == label3d)[0]]
			cameraContributions[label3d] = camIds

		# Create 3D points attributes on the cooked location
		pAttrs = {
			'x3ds': x3ds,
			'x3ds_labels': x3ds_labels,
			'x3ds_colour': eval(attrs['colour']),
			'x3ds_pointSize': attrs['pointSize'],
			'cameraContributions': cameraContributions,
			'showCameraContributions': attrs['show_contributions'],
			'cameraPositions': cameraPositions
		}
		interface.createChild('reconstructed', 'points3d', attrs=pAttrs)


class PointsFromDetectionsAll(Op.Op):
	def __init__(self, name='/PointsFromDetectionsAll', locations='', calibration='', tiltThreshold=0.0002,
	             pointSize=8.0, colour=(1.0, 0.5, 0.0, 0.7),  mesh='', visibilityLod='',
	             intersection_threshold=100., generateNormals=True, frameRange=''):
		fields = [
			('name', 'Name', 'Name', 'string', name, {}),
			('locations', 'Detection locations', 'Detection locations', 'string', locations, {}),
			('calibration', 'Calibration location', 'Calibration location', 'string', calibration, {}),
			('tilt_threshold', 'Tilt threshold', 'Slack factor for tilt pairing', 'float', tiltThreshold, {}),
			('pointSize', '3D Point size', '3D Point size', 'float', pointSize, {}),
			('colour', '3D Point colour', '3D Point colour', 'string', str(colour), {}),
			('mesh', 'Mesh', 'Mesh location', 'string', mesh, {}),
			('visibilityLod', 'Visibility LOD location', 'Visibility LOD location', 'string', visibilityLod, {}),
			('intersection_threshold', 'Intersection threshold', 'Intersection threshold', 'float', intersection_threshold, {}),
			('generateNormals', 'Generate normals', 'Generate normals for visibility checks', 'bool', generateNormals, {}),
			('frameRange', 'Frame range', 'Frame range', 'string', frameRange, {})
		]

		super(self.__class__, self).__init__(name, fields)
		self.visibility = None

	def flush(self):
		self.visibility = None

	def calculate3dPointsFromDetections(self, x2ds, splits, mats, Ps=None, tilt_threshold=0.0002):
		import itertools

		Ts = np.array(zip(*mats)[4],dtype=np.float32)

		if Ps is None:
			Ps = np.array([m[2] / (np.sum(m[2][0, :3] ** 2) ** 0.5) for m in mats], dtype=np.float32)

		numCameras = len(splits) - 1
		E = ISCV.compute_E(x2ds, splits, Ps)
		rays = Recon.dets_to_rays(x2ds, splits, mats)
		cameraPositions = np.array([m[4] for m in mats], dtype=np.float32)
		data = []

		def norm(a):
			return a / (np.sum(a ** 2) ** 0.5)

		tilt_axes = np.array([norm(np.dot([-m[0][0, 2], -m[0][1, 2], m[0][0, 0]], m[1][:3, :3])) for m in mats], dtype=np.float32)

		# Create all combinations of ci < cj
		cameraPairCombinations = np.array(list(itertools.combinations(range(numCameras), 2)), dtype=np.int32)

		knownCamPairs = [
			(7, 12), (5, 9), (3, 9), (4, 12), (7, 10), (8, 12), (0, 9), (3, 4), (1, 9), (2, 7), (1, 2), (0, 11),
			(5, 11), (1, 3), (2, 12), (9, 10), (10, 12), (7, 8), (9, 12), (4, 10), (11, 12), (6, 10), (6, 9),
			(8, 10), (3, 6), (0, 7), (4, 9), (1, 7), (0, 5), (2, 4), (1, 10), (5, 7), (3, 12), (4, 6), (2, 11),
			(3, 7), (3, 10), (4, 8), (4, 11), (0, 1), (5, 12), (1, 6), (7, 11), (2, 3), (2, 8), (1, 4), (1, 8),
			(0, 8), (6, 7), (1, 11), (8, 9), (0, 10), (10, 11), (9, 11), (5, 10), (0, 12), (3, 5), (8, 11),
			(0, 3), (5, 8), (7, 9), (6, 11), (6, 12), (1, 5), (6, 8), (3, 8), (0, 6), (2, 5), (0, 4), (5, 6),
			(1, 12), (4, 7), (2, 6), (2, 10), (4, 5), (3, 11), (0, 2), (2, 9)
		]

		# Find valid pairs of camera rays that could intersect and create a 3D reconstruction
		for ci, cj in cameraPairCombinations:
		# for (ci, cj) in knownCamPairs:
			ui, uj = range(splits[ci], splits[ci + 1]), range(splits[cj], splits[cj + 1])
			if len(ui) == 0 or len(uj) == 0: continue
			axis = cameraPositions[cj] - cameraPositions[ci]
			camPairDist = np.linalg.norm(axis)
			if camPairDist > 7000.: continue
			tilt_i = np.dot(map(norm, np.cross(rays[ui], axis)), tilt_axes[ci])
			tilt_j = np.dot(map(norm, np.cross(rays[uj], axis)), tilt_axes[ci])  # NB tilt_axes[ci] not a bug
			io = np.argsort(tilt_i)
			jo = np.argsort(tilt_j)
			for ii, d0 in enumerate(tilt_i[io]):
				for ji, d1 in enumerate(tilt_j[jo]):
					diff = d0 - d1
					if abs(diff) < tilt_threshold:
						d = [int(ui[io[ii]]), int(uj[jo[ji]])]
						cams = [int(ci), int(cj)]
						entry = {'pair': d, 'cameraIds': cams}
						data.append(entry)

		# Create 3D reconstructions from ray pairs
		x3ds = []
		for entry in data:
			d = entry['pair']
			E0, e0 = E[d, :, :3].reshape(-1, 3), E[d, :, 3].reshape(-1)
			x3d = np.linalg.solve(np.dot(E0.T, E0) + np.eye(3) * 1e-7, -np.dot(E0.T, e0))
			ai, aj = x3d - Ts[ci], x3d - Ts[cj]
			angle = np.degrees(np.arccos(np.dot(ai, aj) / (np.linalg.norm(ai) * np.linalg.norm(aj))))
			if angle > 120: continue
			x3ds.append(x3d)

		return x3ds, data, rays, cameraPositions

	def cook(self, location, interface, attrs):
		if not self.useFrame(interface.frame(), attrs['frameRange']): return
		calibrationLocation = attrs['calibration']
		if not calibrationLocation: calibrationLocation = interface.root()

		# Get the mats from the calibration location
		mats = interface.attr('mats', atLocation=calibrationLocation)
		if mats is None: return

		Ps = interface.attr('Ps', atLocation=calibrationLocation)

		# Get the detections from the location we are cooking
		x2ds = interface.attr('x2ds')
		x2ds_splits = interface.attr('x2ds_splits')

		if self.visibility is None: self.visibility = ISCV.ProjectVisibility.create()

		# Check if we have normals
		if attrs['mesh'] and interface.hasAttr('normals', atLocation=attrs['mesh']):
			normals = interface.attr('normals', atLocation=attrs['mesh'])
			self.visibility.setNormals(normals)

		# Check if we have visibility LODs
		if 'visibilityLod' in attrs and attrs['visibilityLod']:
			visibilityLod = interface.location(attrs['visibilityLod'])
			if visibilityLod is not None:
				lodTris = visibilityLod['tris']
				lodVerts = visibilityLod['verts']
				lodNormals = visibilityLod['faceNormals']
				tris = lodVerts[lodTris]
				cameraPositions = np.array([m[4] for m in mats], dtype=np.float32)
				self.visibility.setLods(tris, cameraPositions, np.concatenate((lodNormals)),
				                        attrs['intersection_threshold'], attrs['generateNormals'])

		# Calculate the 3D reconstructions from the detections
		x3ds, reconstructionData, rays, cameraPositions = self.calculate3dPointsFromDetections(x2ds, x2ds_splits, mats, Ps,
		                                                                                       tilt_threshold=attrs['tilt_threshold'])
		x3ds = np.array(x3ds, dtype=np.float32)

		labellingData = LabelData((x2ds, x2ds_splits), reconstructionData, rays, cameraPositions, mats)
		x3ds_means, x3ds_labels, x3ds_normals, _, _ = cleanAndLabelX3ds(labellingData, x3ds, range(5, 3, -1), self.visibility)

		if x3ds_means is None or not x3ds_means.any(): return

		# Create 3D points attributes on the cooked location
		# interface.setAttr('x3ds', x3ds)
		interface.setAttr('x3ds', x3ds_means)
		interface.setAttr('x3ds_colour', eval(attrs['colour']))
		interface.setAttr('x3ds_pointSize', attrs['pointSize'])
		interface.setAttr('x3ds_labels', x3ds_labels)

		interface.setType('points')


# Test
class LabelData:
	def __init__(self, data1, reconstructionData, rays, cameraPositions, mats):
		self.labels = -np.ones(len(rays), dtype=np.int32)
		self.labelPositions = {}
		self.x3d_threshold = 10.0
		self.x2ds, self.splits = data1
		self.reconstructionData = reconstructionData
		self.rays = rays
		self.originalRays = rays
		self.cameraPositions = cameraPositions
		self.mats = mats
		self.labels_temp = -np.ones_like(self.labels)
		self.labelPositions_temp = {}
		self.Ps = np.array([m[2] / (np.sum(m[2][0, :3] ** 2) ** 0.5) for m in mats], dtype=np.float32)

		self.triangles = []
		self.points = np.array([])
		self.pointLabels = None
		self.pointNormals = None

		self.cloud = None
		self.scores, self.matches, self.matches_splits = [], [], []

	def getX3ds(self, tempLabels=False):
		if tempLabels:
			return self.solveLabels_temp()

		self.solveLabels()
		return self.points, self.pointLabels, self.pointNormals

	def getClusterData(self, x3ds):
		if self.cloud is None:
			cloud = ISCV.HashCloud3D(x3ds, self.x3d_threshold)
			self.scores, self.matches, self.matches_splits = cloud.score(x3ds)

		return self.scores, self.matches, self.matches_splits

	def solveLabels(self):
		self.points, self.pointLabels, self.pointNormals, E, x2d_labels = Recon.solve_x3ds_normals(self.x2ds,
																											self.splits,
																											self.labels,
																											self.Ps,
																											self.rays)

		print "solveLabels:", len(self.points), np.min(self.pointLabels), np.max(self.pointLabels), "(#points | min label | max label)"

	def solveLabels_temp(self):
		points, pointLabels, pointNormals, E, x2d_labels = Recon.solve_x3ds_normals(self.x2ds,
																											self.splits,
																											self.labels_temp,
																											self.Ps,
																											self.rays)

		print "solveLabels:", len(points), np.min(pointLabels), np.max(pointLabels), "(#points | min label | max label)"
		return points, pointLabels, pointNormals

	def approveLabels(self, approved_point_labels):
		for label in approved_point_labels:
			self.labels[np.where(self.labels_temp == label)[0]] = label # Surely this can be done better?

		self.labels_temp = np.copy(self.labels)

	def getMeshData(self, solve=False):
		if solve:
			self.getX3ds()

		triangles = np.array(self.triangles, dtype=np.int32)
		if triangles.any():
			verts = self.points
			tris = np.array(self.triangles, dtype=np.int32)
			return tris, verts

		return np.array([]), np.array([])

def findCameraIdFromRayId(rayId, camRaySplits):
	dists = rayId - camRaySplits[:-1]
	dists[np.where(dists < 0)[0]] = np.sum(camRaySplits)
	return dists.argmin()

def cleanAndLabelX3ds(labellingData, x3ds, N, allowStealing=True, pts=np.array([]), visibility=None):
	global cameraContributions, rayInfo

	labels = labellingData.labels_temp
	labelPositions = labellingData.labelPositions
	x3d_threshold = labellingData.x3d_threshold
	x2ds = labellingData.x2ds
	splits = labellingData.splits
	reconstructionData = labellingData.reconstructionData
	rays = labellingData.rays
	cameraPositions = labellingData.cameraPositions

	# We want to get only the points that have N neighbours within 1cm
	# TODO: Cache this as we'll be using it multiple times
	#cloud = ISCV.HashCloud3D(x3ds, x3d_threshold)
	#scores, matches, matches_splits = cloud.score(x3ds)
	scores, matches, matches_splits = labellingData.getClusterData(x3ds)

	#clusterMeanPoints = []
	registry = []
	x3ds_means = []
	x3ds_normals = []
	cameraContributions = []
	# clusterCameraContributions = []
	rawData = None
	rayInfo = []
	labelsAdded = []

	#x2ds, splits = data1
	#Ps = np.array([m[2] / (np.sum(m[2][0, :3] ** 2) ** 0.5) for m in mats], dtype=np.float32)
	Ps = labellingData.Ps

	volatileLabels = []
	goldStandardLabels = []

	for n in N:
		#print ">> Min Rays:", n
		whichMatches = np.where(matches_splits[1:] - matches_splits[:-1] >= n)[0]
		clusterSplitPairs = np.array(zip(matches_splits[:-1], matches_splits[1:]))[whichMatches]

		if n == N: rawData = x3ds[whichMatches]

		clusterCounter = 0
		x3ds_clusters = []
		x3ds_clusterColours = []
		x3ds_clusterMeans = []
		x3ds_clusterMeansColours = []
		x3ds_clusterLabels = []

		for matchFrom, matchTo in clusterSplitPairs:
			# Find the points for this cluster and calculate the mean position
			pointIndices = matches[matchFrom:matchTo]
			numPoints = len(pointIndices)
			assert(numPoints >= n)
			clusterMean = np.mean(x3ds[pointIndices], axis=0)

			if len(np.where(np.linalg.norm(clusterMean - cameraPositions, axis=1) < x3d_threshold * 6.0)[0]) > 0:
				continue

			if pts.any():
				if len(pts.shape) == 1:
					dists = np.linalg.norm(clusterMean - pts)
				else:
					dists = np.linalg.norm(clusterMean - pts, axis=1)

				if len(np.where(dists > x3d_threshold * 10.0)[0]) > 0:
					continue

			cluster = x3ds[pointIndices]
			x3ds_clusters.extend(cluster)
			randomColour = np.concatenate((np.random.rand(3), np.array([0.5], dtype=np.float32)))
			x3ds_clusterColours.extend(np.tile(randomColour, (cluster.shape[0], 1)))
			x3ds_clusterMeans.append(clusterMean)
			x3ds_clusterMeansColours.append(randomColour)
			x3ds_clusterLabels.append(clusterCounter)

			# Get all the rays used to make the points in this cluster. This will be a Nx3 matrix
			rayIndices = np.unique([reconstructionData[pi]['pair'] for pi in pointIndices])
			pointRays = rays[rayIndices]

			# Calculate the dot product for each combination of rays. This will be a NxN matrix
			raysDps = np.dot(pointRays, pointRays.T)

			# Find the ray which has the highest agreement with the others (sum of dot products)
			bestRay = np.sum(raysDps > 0, axis=0).argmax()

			# Find which other rays are in agreement with the best ray (dp > 0)
			goodRays = np.where(raysDps[bestRay] > 0.05)[0]

			# As all the (good) rays in the cluster should be contributing to creating a single point, we will
			# give them a new label that identifies them with the detection/reconstruction for that point
			#currentLabel = len(clusterMeanPoints)
			currentLabel = len(labelPositions)
			labelForPointReconstruction = currentLabel

			# Only continue with rays from a unique set of cameras
			camerasForRays = [findCameraIdFromRayId(rayId, splits) for rayId in rayIndices[goodRays]]
			uniqueRayCams, uniqueRayCamsIdx = np.unique(camerasForRays, return_index=True)
			goodRays = goodRays[uniqueRayCamsIdx]
			rayInfo.append(raysDps[goodRays]) # TODO: Fix.. nonsense

			existingLabelsForRays = labels[rayIndices[goodRays]]
			knownLabelIndices = np.where(existingLabelsForRays != -1)[0]
			rayIdsForKnownLabels = rayIndices[knownLabelIndices]
			camerasForKnownLabels = [findCameraIdFromRayId(rayId, splits) for rayId in rayIdsForKnownLabels]
			uniqueCams, uniqueCamsIdx = np.unique(camerasForKnownLabels, return_index=True)
			knownLabelIndices = knownLabelIndices[uniqueCamsIdx]
			knownLabels = existingLabelsForRays[knownLabelIndices]

			clusterCounter += 1

			# We check if any of the rays have been assigned a label before (i.e. they will contribute to
			# reconstructing a 3D point). If that is the case then we have to make decision whether we
			# want to our rays in this cluster to contribute to the existing label (reconstruction), or
			# if we want to steal the labelled rays so that they now contribute to creating a new label
			# for this cluster
			threshold = x3d_threshold ** 2
			for label in np.unique(knownLabels):
				# The ray has been labelled to create a 3D point. If that point is within threshold distance
				# of the current cluster we give this cluster the same label. In essence we are merging the
				# rays in this cluster with the rays that are already contributing to the label.
				# However, if the reconstructed label and the cluster mean are further away from each other
				# we will relabel it with the new label for this cluster which equates to stealing it.
				#dist = np.linalg.norm(clusterMeanPoints[label] - clusterMean)
				dist = np.linalg.norm(labelPositions[label] - clusterMean)
				if dist < threshold:
					labelForPointReconstruction = label
					break
					# threshold = dist

			_clusterId, _clusterX3dId = len(labelPositions) - 1, len(x3ds_clusterMeans) - 1

			# Label the rays with the new or existing (merged) label
			useNewLabel = False
			unknownLabels = np.where(existingLabelsForRays == -1)[0]
			if labelForPointReconstruction == currentLabel:
				# No merging is going on
				if len(unknownLabels) > 0:
					labels[rayIndices[goodRays][unknownLabels]] = currentLabel
					useNewLabel = True

				if allowStealing:
					for knownLabel in knownLabelIndices:
						rayIdsWithLabel = np.where(labels == existingLabelsForRays[knownLabel])[0]
						numRaysForLabel = len(rayIdsWithLabel)
						# if existingLabelsForRays[knownLabel] not in volatileLabels and numRaysForLabel < 3:
						# if existingLabelsForRays[knownLabel] not in goldStandardLabels and numRaysForLabel < 3:
						# if existingLabelsForRays[knownLabel] not in goldStandardLabels:
						agreement = np.where(np.sum(np.dot(bestRay, rays[rayIdsWithLabel]) > 0, axis=1) > 1)[0]
						if True:
							labels[rayIndices[goodRays][knownLabel]] = currentLabel
							useNewLabel = True

			else:
				# Employ merging strategy
				if allowStealing:
					rayIdsWithLabel = np.where(labels == labelForPointReconstruction)[0]
					agreement = np.where(np.sum(np.dot(bestRay, rays[rayIdsWithLabel]) > 0, axis=1) > 1)[0]

					labels[rayIndices[goodRays][unknownLabels]] = labelForPointReconstruction

					for knownLabel in knownLabelIndices:
						numRaysForLabel = len(np.where(labels == existingLabelsForRays[knownLabel])[0])
						# if existingLabelsForRays[knownLabel] not in goldStandardLabels and numRaysForLabel < 3:
						if existingLabelsForRays[knownLabel] not in goldStandardLabels:
							labels[rayIndices[goodRays][knownLabel]] = currentLabel
							useNewLabel = True
				else:
					labels[rayIndices[goodRays]] = labelForPointReconstruction

			if useNewLabel:
				labelPositions[currentLabel] = clusterMean
				labelsAdded.append(currentLabel)

		goldStandardLabels = np.where(labels != -1)[0]

	if len(np.where(labels != -1)[0]) == 0:
		return np.array([]), np.array([]), np.array([]), rawData, labelsAdded

	# x3ds_means, x3ds_labels, _, _ = Recon.solve_x3ds(x2ds, splits, labels, Ps)
	x3ds_means, x3ds_labels, x3ds_normals, _, _ = Recon.solve_x3ds_normals(x2ds, splits, labels, Ps, rays)

	# x2d_threshold = 30. / 2000.
	# clouds = ISCV.HashCloud2DList(x2ds, splits, x2d_threshold)
	# _, labels, _ = clouds.project_assign_visibility(x3ds_means, None, Ps, x2d_threshold, visibility)

	labellingData.labels = labels
	usedLabels = np.array(np.where(labels != -1)[0], dtype=np.int32)

	return x3ds_means, x3ds_labels, x3ds_normals, rawData, labelsAdded


# Register Ops
import Registry
Registry.registerOp('Reconstruct 3D from Dets', PointsFromDetections)
Registry.registerOp('Reconstruct 3D from Dets (all)', PointsFromDetectionsAll)
