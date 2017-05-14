import numpy as np


scores = {}
processedEdges = {}

def uniqueRows(a):
	a = np.ascontiguousarray(a)
	unique_a = np.unique(a.view([('', a.dtype)] * a.shape[1]))
	return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))


class TriangleCache:
	def __init__(self, x3ds):
		self.x3ds = x3ds
		self.sortedTriangles = []
		self.sortedBlacklistedCandidates = []
		self.neighbourNormals, self.neighbourTriangles = {}, {}

	def out(self):
		print type(self.x3ds), type(self.sortedTriangles), type(self.sortedBlacklistedCandidates), type(self.neighbourNormals), type(self.neighbourTriangles)
		return (self.x3ds.tolist(), self.sortedTriangles, self.sortedBlacklistedCandidates, self.neighbourNormals, self.neighbourTriangles)

	def set(self, data):
		x3ds, sortedTriangles, sortedBL, neighbourNormals, neighbourTriangles = data
		self.x3ds = x3ds
		self.sortedTriangles = sortedTriangles
		self.sortedBlacklistedCandidates = sortedBL
		self.neighbourNormals = neighbourNormals
		self.neighbourTriangles = neighbourTriangles

	def triangleExists(self, triangle, candidateLabels, includeBlacklisted=False):
		t = candidateLabels[np.sort(triangle)].tolist()
		if t in self.sortedTriangles:
			return True

		if includeBlacklisted and t in self.sortedBlacklistedCandidates:
			return True

		return False

	def addTriangle(self, triangle, triangleNormal, triangles, candidateLabels):
		if triangle is None or triangleNormal is None:
			return False

		if self.triangleExists(triangle, candidateLabels):
			return False

		triangles.append(triangle)
		self.sortedTriangles.append(candidateLabels[np.sort(triangle)].tolist())

		labelledTriangle = candidateLabels[triangle].tolist()
		for i in labelledTriangle:
			if i in self.neighbourNormals:
				self.neighbourNormals[i].append(triangleNormal.tolist())
				self.neighbourTriangles[i].append(labelledTriangle)
			else:
				self.neighbourNormals[i] = [triangleNormal.tolist()]
				self.neighbourTriangles[i] = [labelledTriangle]

		return True

	def addBlacklistedCandidate(self, triangleCandidate, candidateLabels):
		sortedCandidate = candidateLabels[np.sort(triangleCandidate)].tolist()
		if sortedCandidate not in self.sortedBlacklistedCandidates:
			self.sortedBlacklistedCandidates.append(sortedCandidate)

	def getIndicesFromLabels(self, labels, candidateLabels):
		labels = np.array(labels, dtype=np.int32)
		if not labels.any(): return np.array([])
		if labels[0][0] == 637 and labels[0][1] == 1403:
			print "..."

		try:
			return np.where(labels.reshape(-1, 1) == candidateLabels)[1].reshape(labels.shape)
		except Exception:
			return np.array([])

	def getTriangleNeighboursForPoint(self, point, candidateLabels):
		point = candidateLabels[point]
		neighbouringNormals = uniqueRows(np.sort(self.neighbourNormals[point]))
		neighbouringTriangles = uniqueRows(np.sort(self.neighbourTriangles[point]))

		neighbouringTriangles = self.getIndicesFromLabels(neighbouringTriangles, candidateLabels)

		return np.array(neighbouringNormals, dtype=np.float32), np.array(neighbouringTriangles, dtype=np.int32)

	def getTriangleNeighboursForTriangle(self, triangle, candidateLabels):
		neighbouringNormals = []
		neighbouringTriangles = []

		for i in candidateLabels[triangle]:
			if i in self.neighbourNormals and i in self.neighbourTriangles:
				neighbouringNormals.extend(self.neighbourNormals[i])
				neighbouringTriangles.extend(self.getIndicesFromLabels(self.neighbourTriangles[i], candidateLabels))

		return np.array(neighbouringNormals, dtype=np.float32), np.array(neighbouringTriangles, dtype=np.int32)


def registerTriangleAction(registry, triangle, labels, text, verbose=True, sortedTriangle=None):
	if not verbose: return
	if sortedTriangle is None:
		sortedTriangle = np.array_str(np.sort(labels[triangle]))

	entry = "Triangle:", sortedTriangle, "|", str(text)
	registry.append(entry)

def pointsInTriangle(cylTriangle, cylPoints):
	import matplotlib.pyplot as plt
	from matplotlib.patches import Polygon
	from mpl_toolkits.mplot3d import Axes3D

	coeffs = np.linalg.lstsq(cylTriangle.T, cylPoints.T)[0]
	if len(np.where(np.sum(coeffs > 0, axis=1) == 3)[0]) > 0:
		#fig = plt.figure()
		#ax = fig.add_subplot(111)
		#ax.autoscale(enable=True)
		#ax.add_patch(plt.Polygon(cylPoints, fill=None, edgecolor='r'))
		#ax.add_patch(plt.Polygon(cylPoints, fill=None, edgecolor='b'))
		#ax.add_patch(plt.Polygon(cylTriangle, fc='y'))
		#plt.plot(cylPoints[:, 0], cylPoints[:, 1], linestyle='None', marker='x')
		#plt.show()
		print coeffs
		return True

	return False

def pointsInTriangleArea(cylTriangle, cylPoints):
	v0 = cylTriangle[2] - cylTriangle[0]
	v1 = cylTriangle[1] - cylTriangle[0]

	for point in cylPoints:
		v2 = point - cylTriangle[0]

		areaP0 = np.cross(v2, v0)
		areaP1 = np.cross(v1, v2)
		areaTri = np.cross(v1, v0)

		if areaTri < 0:
			areaP0, areaP1, areaTri = -areaP0, -areaP1, -areaTri

		inTriangle = areaP0 >= 0 and areaP1 >= 0 and (areaP0 + areaP1) <= areaTri
		if inTriangle:
			return True

	return False

def testIntersects(neighbours, neighbouringNormals, triangleNormal, points, triangle, labels):
	import matplotlib as mpl
	import matplotlib.pyplot as plt
	from matplotlib.patches import Polygon
	from mpl_toolkits.mplot3d import Axes3D
	import sys

	if len(neighbours) == 0: return False

	tm = np.mean(points, axis=0)

	a, b, c, = points[0], points[1], points[2]
	e0 = b - a
	e1 = c - b
	e2 = a - c

	if False:
		uu = np.dot(e0, e0)
		uv = np.dot(e0, e1)
		vv = np.dot(e1, e1)

	# TODO: Do this in a single step (all neighbour normals)

	plot = False

	if False:
		fig = plt.figure()
		ax = fig.gca(projection='3d')

		tri = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
		ax.plot_trisurf(tri[:, 0], tri[:, 1], tri[:, 2], color='g', alpha=0.6)

		# pts = np.array([[0.6, 0.2, 0], [0.2, 0.6, 0], [0.2, 0.2, 0]], dtype=np.float32)
		# pts = np.array([[0.45, 0.1, 0], [0.45, 0.45, 0], [0.1, 0.45, 0]], dtype=np.float32)
		pts = np.array([[0.025, 0.025, 0], [0.95, 0.025, 0], [0.025, 0.95, 0]], dtype=np.float32)

		c = np.linalg.lstsq(tri.T, pts.T)[0]
		print c

		# coeffs = np.array([[0.1, 0.1, 0.8], [0.1, 0.8, 0.1], [0.8, 0.1, 0.1]])
		coeffs = np.array([[0.1, 0.1, 0.8], [0.1, 0.8, 0.1], [0.8, 0.1, 0.1]])
		testOrigins = np.dot(tri.T, coeffs).T

		# coeffs2 = np.array([[0.2, 0.2, 0.6], [0.6, 0.2, 0.2], [0.2, 0.6, 0.2]])
		# coeffs2 = np.array([[0.45, 0.1, 0.45], [0.45, 0.45, 0.1], [0.1, 0.45, 0.45]])
		coeffs2 = np.array([[0.95, 0.025, 0.025], [0.025, 0.95, 0.025], [0.025, 0.025, 0.95]])
		testOrigins2 = np.dot(tri.T, coeffs2.T).T

		# tl = np.array([tm])
		# tb = tm + 60 * triangleNormal
		# tl = np.append(tl, [tb], axis=0)
		# ax.plot(tl[:, 0], tl[:, 1], tl[:, 2], color='g')
		# ax.plot(l2[:, 0], l2[:, 1], l2[:, 2], color='y')
		#
		# ax.plot_trisurf(neighbour[:, 0], neighbour[:, 1], neighbour[:, 2], color='b', alpha=0.6)
		#
		# l = np.array([rayOrigin])
		# l_end = rayOrigin + 60 * normal
		# l = np.append(l, [l_end], axis=0)
		# ax.plot(l[:, 0], l[:, 1], l[:, 2], color='b')
		#
		ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color='b', marker='o')
		ax.plot(testOrigins[:, 0], testOrigins[:, 1], testOrigins[:, 2], color='r', marker='x')
		ax.plot(testOrigins2[:, 0], testOrigins2[:, 1], testOrigins2[:, 2], color='r', marker='x')
		plt.show()

	for neighbour, normal in zip(neighbours, neighbouringNormals):
		if (np.sort(neighbour, axis=0) == np.sort(points, axis=0)).all():
			continue

		rayOrigin = np.mean(neighbour, axis=0)
		rayDirection = normal

		# Check if ray is parallel to the triangle plane
		dp = np.dot(triangleNormal, rayDirection)
		if abs(dp) < 0.0001:
			# print "Ray and triangle plane are parallel:", dp, triangleNormal, normal
			continue

		# Get intersect point of ray with triangle plane
		k = tm - rayOrigin
		t = np.dot(triangleNormal, k) / dp

		if abs(t) > 30.0:
			continue

		l2 = np.array([rayOrigin])
		p = rayOrigin + t * rayDirection
		l2 = np.append(l2, [p], axis=0)

		intersects = False

		coeffs = np.array([
			[0.1, 0.1, 0.8],
			[0.1, 0.8, 0.1],
			[0.8, 0.1, 0.1],
			[0.2, 0.2, 0.6],
			[0.6, 0.2, 0.2],
			[0.2, 0.6, 0.2],
			[0.45, 0.1, 0.45],
			[0.45, 0.45, 0.1],
			[0.1, 0.45, 0.45],
			[0.95, 0.025, 0.025],
			[0.025, 0.95, 0.025],
			[0.025, 0.025, 0.95]
		])
		testOrigins = np.dot(neighbour.T, coeffs.T).T
		testOrigins = np.append(testOrigins, [rayOrigin], axis=0)

		# Check if the point is within the triangle
		for origin in testOrigins:
			k = tm - origin
			t = np.dot(triangleNormal, k) / dp
			l2 = np.array([origin])
			p = origin + t * rayDirection

			dpA = np.dot(np.cross(e0, p - a), triangleNormal)
			dpB = np.dot(np.cross(e1, p - b), triangleNormal)
			dpC = np.dot(np.cross(e2, p - c), triangleNormal)
			if dpA >= 0 and dpB >= 0 and dpC >= 0:
				intersects = True

			dpA = np.dot(np.cross(e0, p - a), -triangleNormal)
			dpB = np.dot(np.cross(e1, p - b), -triangleNormal)
			dpC = np.dot(np.cross(e2, p - c), -triangleNormal)
			if dpA >= 0 and dpB >= 0 and dpC >= 0:
				intersects = True

		if plot:
			fig = plt.figure()
			ax = fig.gca(projection='3d')

			ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], color='g', alpha=0.6)
			tl = np.array([tm])
			tb = tm + 60 * triangleNormal
			tl = np.append(tl, [tb], axis=0)
			ax.plot(tl[:, 0], tl[:, 1], tl[:, 2], color='g')
			ax.plot(l2[:, 0], l2[:, 1], l2[:, 2], color='y')

			ax.plot_trisurf(neighbour[:, 0], neighbour[:, 1], neighbour[:, 2], color='b', alpha=0.6)

			l = np.array([rayOrigin])
			l_end = rayOrigin + 60 * normal
			l = np.append(l, [l_end], axis=0)
			ax.plot(l[:, 0], l[:, 1], l[:, 2], color='b')

			ax.plot(testOrigins[:, 0], testOrigins[:, 1], testOrigins[:, 2], color='b', marker='o')
			plt.show()

		if False:
			intersects = True
			w = p - points[0]
			wu = np.dot(w, e0)
			wv = np.dot(w, e1)
			d = uv * uv - uu * vv
			s = (uv * wv - vv * wu) / d
			t = (uv * wu - uu * wv) / d
			intersects = True
			if (s < 0.0 or s > 1.0) or (t < 0.0 or t > 1.0):
				intersects = False

		if False:
			nn = np.cross(rayDirection, e1)
			det = np.dot(e0, p)
			eps = sys.float_info.epsilon
			if det > -eps and det < eps:
				intersects = False

			inv_det = 1.0 / det
			dist = rayOrigin - points[0]
			u = np.dot(dist, nn) * inv_det
			if u < 0.0 or u > 1.0:
				intersects = False

			q = np.cross(dist, e0)
			v = np.dot(rayDirection, q) * inv_det
			if v < 0.0 or u + v > 1.0:
				intersects = False

			tt = np.dot(e1, q) * inv_det
			if tt > eps:
				intersects = True

		if False:
			intersects = True
			v1 = points[0] - rayOrigin
			v2 = points[1] - rayOrigin
			n1 = np.cross(v2, v1)
			n1 /= np.linalg.norm(n1)
			d1 = np.dot(-rayOrigin, n1)
			if np.dot(p, n1) + d1 < 0:
				intersects = False

		if False:
			intersects = pointsInTriangle(np.array(neighbour), np.array([p]))

		if intersects:
			if plot:
				fig = plt.figure()
				ax = fig.gca(projection='3d')

				ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], color='g', alpha=0.6)
				tl = np.array([tm])
				tb = tm + 60 * triangleNormal
				tl = np.append(tl, [tb], axis=0)
				ax.plot(tl[:, 0], tl[:, 1], tl[:, 2], color='g')
				ax.plot(l2[:, 0], l2[:, 1], l2[:, 2], color='y')

				ax.plot_trisurf(neighbour[:, 0], neighbour[:, 1], neighbour[:, 2], color='b', alpha=0.6)

				for origin in testOrigins:
					l = np.array([origin])
					l_end = origin + 60 * rayDirection
					l_end2 = origin + 60 * -rayDirection
					l = np.append(l, [l_end], axis=0)
					l = np.append(l, [l_end2], axis=0)
					ax.plot(l[:, 0], l[:, 1], l[:, 2], color='b')

				plt.show()

			return True
		#else:
		#	ax.plot(l2[:, 0], l2[:, 1], l2[:, 2], color='r')

	return False

def scoreTriangle(triangle, registry, x3ds, labels, rayNormals, rayNormalsDp, cache, triangleNormal=None,
				  threshold_neighbourNorm=0.4, threshold_rayAgreement=0.3, threshold_normalRayAgreement=0.3,
				  minAngleRels=25, forgiving=False, verbose=True):
	import hashlib
	global scores

	if triangleNormal is None:
		triangleNormal = np.cross(x3ds[triangle[1], :] - x3ds[triangle[2], :], x3ds[triangle[0], :] - x3ds[triangle[2], :])
		norm = np.linalg.norm(triangleNormal)
		if np.isclose(norm, 0):
			return None, None, None

		triangleNormal /= np.linalg.norm(triangleNormal)

	# Calculate the agreement between the triangle normal and the ray normals
	# making up the triangle's reconstructed points
	raysTriangleDps = np.dot(rayNormals, triangleNormal)
	if np.mean(raysTriangleDps) < 0:
		triangle = triangle[::-1]
		triangleNormal *= -1

	sortedTriangle = np.array_str(np.sort(labels[triangle]))
	# numNeighbours = len(cache.neighbourTriangles)

	# scoreIdentifier = "%s:%.2f:%.2f:%.2f:%.0f:%s:%d" % (sortedTriangle, threshold_neighbourNorm, threshold_rayAgreement,
	# 												 threshold_normalRayAgreement, minAngleRels, str(forgiving), numNeighbours)
	# scoreHash = hashlib.sha224(scoreIdentifier).hexdigest()
	# if scoreHash not in scores:
	score = _scoreTriangle(triangle, registry, x3ds, labels, rayNormals, rayNormalsDp, cache, triangleNormal,
						   threshold_neighbourNorm, threshold_rayAgreement, threshold_normalRayAgreement,
						   minAngleRels, forgiving, verbose, sortedTriangle)

	if score is None:
		return None, None, None

	# 	scores[scoreHash] = score
	# else:
	# 	print "Returning hashed score:", scores[scoreHash]

	# return triangle, triangleNormal, scores[scoreHash]
	return triangle, triangleNormal, score

def _scoreTriangle(triangle, registry, x3ds, labels, rayNormals, rayNormalsDp, cache, triangleNormal=None,
				  threshold_neighbourNorm=0.4, threshold_rayAgreement=0.3, threshold_normalRayAgreement=0.3,
				  minAngleRels=25, forgiving=False, verbose=True, sortedTriangle=None):
	import itertools
	global scores

	verts = x3ds[triangle]

	# if triangleNormal is None:
	# 	triangleNormal = np.cross(x3ds[triangle[1], :] - x3ds[triangle[2], :], x3ds[triangle[0], :] - x3ds[triangle[2], :])
	# 	norm = np.linalg.norm(triangleNormal)
	# 	if np.isclose(norm, 0):
	# 		return None, None, None
	#
	# 	triangleNormal /= np.linalg.norm(triangleNormal)

	edgesIt = itertools.combinations(triangle, 2)
	edgeLengths = []
	for e0, e1 in edgesIt:
		edgeLengths.append(np.linalg.norm(x3ds[e0] - x3ds[e1]))

	# verbose = testObserveTriangle(triangle, labels, "Self test")

	edgeLengths = np.array(edgeLengths)
	if min(edgeLengths) < 35.0 or max(edgeLengths) > 100.0 or len(np.where(edgeLengths > 80.0)[0]) > 1:
		registerTriangleAction(registry, triangle, labels, "Edge lengths are not great: %s" % np.array_str(edgeLengths), sortedTriangle)
		return None

	# We want the triangles to be a roughly certain shape so check the angles and reject the odd ones out
	a = np.linalg.norm(verts[0] - verts[1])
	b = np.linalg.norm(verts[1] - verts[2])
	c = np.linalg.norm(verts[0] - verts[2])
	cosC = (a * a + b * b - c * c) / (2 * a * b)
	cosB = (a * a + c * c - b * b) / (2 * a * c)
	if abs(cosB) > 1 or abs(cosC) > 1:
		return None

	if forgiving:
		minAngleRels += 10

	angleC = np.degrees(np.arccos(cosC))
	angleB = np.degrees(np.arccos(cosB))
	angleA = 180 - (angleB + angleC)
	angles = np.sort([angleA, angleB, angleC])

	# We subtract the smallest angle and then check whether at least two of the other angles are within X degrees
	# (the smallest one will be counted being 0 degrees away)
	angleRels = angles - np.min(angles)
	similarAngles = np.where(angleRels < minAngleRels)[0]

	# NOTE: This is prone to errors as e.g. 20, 40, and 110 will have 0.4, 0.18, 0.45 which would pass
	# angleRels = np.array([angles[0] / angles[1], angles[0] / angles[2], angles[1] / angles[2]])
	# similarAngles = np.where((angleRels > minAngleRels) & (angleRels < 1.3))[0]

	numEqualAngles = len(similarAngles)

	# We are looking for triangles where 2 angles are roughly the same (isosceles) and are not too
	# elongated, e.g. largest angle not too large or too small
	# if numEqualAngles == 0:
	if numEqualAngles == 1:
		registerTriangleAction(registry, triangle, labels, "Irregular angles: %s" % np.array_str(angleRels), sortedTriangle)
		return None

	minAngle = 50.0
	maxAngle = 110.0
	if forgiving:
		minAngle = 38.0
		maxAngle = 130.0

	oddAngleRelsMapping = [2, 1, 0]
	oddAngleIndex = oddAngleRelsMapping[similarAngles[0]]
	oddAngle = angles[oddAngleIndex]
	if not minAngle < oddAngle < maxAngle:
		registerTriangleAction(registry, triangle, labels, "Main angle out of range: %s" % np.array_str(angles[oddAngleIndex]), sortedTriangle)
		return None

	# Collect the agreement of ray normals for each pair of points making
	# up the triangle. Triangles made up of points with disjoint ray
	# directions are more likely to be spurious
	rayPointsDps = rayNormalsDp[triangle][:, triangle]
	minRayPointsDp = np.min(rayPointsDps)

	# We want a fairly high level of agreement so we check against
	# a fairly tight value (e.g. 60 degree variation which allows
	# for some variation in terms of ray normals)
	# if minRayPointsDp < threshold_rayAgreement:
	# 	registerTriangleAction(registry, triangle, labels, "Point rays not in agreement: %s" % np.array_str(minRayPointsDp), sortedTriangle)
		# return None

	# # Calculate the agreement between the triangle normal and the ray normals
	# # making up the triangle's reconstructed points
	# raysTriangleDps = np.dot(rayNormals, triangleNormal)
	# if np.mean(raysTriangleDps) < 0:
	# 	triangle = triangle[::-1]
	# 	triangleNormal *= -1
	# 	raysTriangleDps = np.dot(rayNormals, triangleNormal)
	raysTriangleDps = np.dot(rayNormals, triangleNormal)

	minRaysTriangleDp = np.min(raysTriangleDps)

	# Determine the order in which we author the triangle which
	# decides its surface normal
	# if minRayPointsDp < 0:
	# registerTriangleAction(registry, triangle, labels, "Deciding normal: %s" % np.array_str(raysTriangleDps), sortedTriangle)#, existingDp
	# if existingDp:

	# Here we test the absolute value of the rays-triangle dp as
	# the surface normal could face either way which is irrelevant
	# for this test
	# if abs(minRaysTriangleDp) < threshold_normalRayAgreement:
	#if abs(np.mean(minRaysTriangleDp)) < threshold_normalRayAgreement:
	if minRaysTriangleDp < threshold_normalRayAgreement:
		registerTriangleAction(registry, triangle, labels, "Triangle normal and ray normals not in agreement: %s | %s" % (np.array_str(minRaysTriangleDp), np.array_str(raysTriangleDps)), sortedTriangle)
		return None

	# If we have used the vertices to create other triangles then we
	# want to make sure the new triangle roughly lines up with the
	# existing ones
	neighbouringNormals, neighbouringTriangles = cache.getTriangleNeighboursForTriangle(triangle, labels)
	#neighbouringNormals = []
	#neighbouringTriangles = []
	#for i in triangle:
	#	if i in coll_normals and i in coll_triangles:
	#		neighbouringNormals.extend(coll_normals[i])
	#		neighbouringTriangles.extend(coll_triangles[i])

	if forgiving:
		threshold_neighbourNorm -= 0.4

	existingDp = 0
	if neighbouringNormals.any():
		existingDp = np.dot(neighbouringNormals, triangleNormal)
		minExistingDp = abs(np.min(existingDp))
		if minExistingDp < threshold_neighbourNorm:
			registerTriangleAction(registry, triangle, labels, "Triangle normal not in agreement with neighbouring normals: %s" % np.array_str(minExistingDp), sortedTriangle)
			return None

	if neighbouringTriangles.any():
		neighbouringTriangles = uniqueRows(neighbouringTriangles)
		edges = []
		edgesIt = itertools.combinations(triangle, 2)
		for e0, e1 in edgesIt:
			edges.append([e0, e1])
			n0 = np.sum(e0 == neighbouringTriangles, axis=1)
			n1 = np.sum(e1 == neighbouringTriangles, axis=1)
			numTrianglesContainingEdge = len(np.where(n0 + n1 == 2)[0])
			if numTrianglesContainingEdge > 1:
				registerTriangleAction(registry, triangle, labels, "Edge is already shared by two triangles: %s, %s" % (str(labels[e0]), str(labels[e1])), sortedTriangle)
				# if verbose: print " >>", neighbouringTriangles
				return None

		if testIntersects(x3ds[neighbouringTriangles], neighbouringNormals, triangleNormal, verts, triangle, labels):
			registerTriangleAction(registry, triangle, labels, "Intersects with a neighbour", sortedTriangle)
			return None

	score = (angleRels[1], angles, oddAngleIndex, len(neighbouringTriangles), edgeLengths, existingDp, minRayPointsDp, minRaysTriangleDp)

	# return triangle, triangleNormal, score
	return score

def getDistTests(maxDistThreshold, minDistThreshold):
	distTestMax = np.tile(maxDistThreshold, 3)
	distTestMin = np.tile(minDistThreshold, 3)
	return distTestMax, distTestMin

def getCandidatesForPoint(point, x3ds, maxDistThreshold, minDistThreshold=None, usedIdx=None):
	distTestMax, distTestMin = getDistTests(maxDistThreshold, minDistThreshold)
	pointCoords = x3ds[point]

	if usedIdx:
		pointDiffs = abs(pointCoords - x3ds[usedIdx])
	else:
		pointDiffs = abs(pointCoords - x3ds)

	maxTest = np.sum(pointDiffs < distTestMax, axis=1) == 3

	if minDistThreshold:
		minTest = np.sum(pointDiffs >= distTestMin, axis=1) == 2
		candidates = np.where((maxTest) & (minTest))[0]
	else:
		candidates = np.where(maxTest)[0]

	return candidates

def processPoint(point, edges, x3ds, maxDistThreshold, minDistThreshold, triangle=None, usedIdx=None):
	pointCoords = x3ds[point]

	if usedIdx:
		pointDists = np.linalg.norm(x3ds[usedIdx] - pointCoords, axis=1)
	else:
		pointDists = np.linalg.norm(x3ds - pointCoords, axis=1)

	candidates = getCandidatesForPoint(point, x3ds, maxDistThreshold, minDistThreshold, usedIdx=usedIdx)
	for ci in np.argsort(pointDists[candidates]):
		candidate = candidates[ci]
		if usedIdx:
			candidate = usedIdx[candidate]
		if not triangle or (triangle and candidate not in triangle):
			edges.add((point, candidate))

def processEdges(edges, triangles, x3ds, x3ds_labels, maxDistThreshold, minDistThreshold,
				 rayNormals, rayNormalsDp, cache, registry,
				 threshold_neighbourNorm=0.25, threshold_rayAgreement=0.25, threshold_normalRayAgreement=0.25,
				 its=0, usedIdx=None, labellingData=None, forgiving=False, maxIts=10):
	global processedEdges
	for edge in edges:
		edgeKey = np.array_str(np.sort(x3ds_labels[list(edge)]))
		if edgeKey in processedEdges and processedEdges[edgeKey] == 2:
			continue

		processEdge(edge, edgeKey, triangles, x3ds, x3ds_labels, maxDistThreshold, minDistThreshold, rayNormals, rayNormalsDp,
					cache, registry, threshold_neighbourNorm, threshold_rayAgreement, threshold_normalRayAgreement,
					its, usedIdx, labellingData, forgiving, maxIts)

def processEdge(edge, edgeKey, triangles, x3ds, x3ds_labels, maxDistThreshold, minDistThreshold, rayNormals, rayNormalsDp, cache,
				registry, threshold_neighbourNorm=0.25, threshold_rayAgreement=0.25, threshold_normalRayAgreement=0.25,
				its=0, usedIdx=None, labellingData=None, forgiving=False, maxIts=10):
	global processedEdges
	e0, e1 = edge
	# print "Edge:", edgeKey

	e0l = x3ds_labels[e0]
	e1l = x3ds_labels[e1]

	candidates_e0 = getCandidatesForPoint(e0, x3ds, maxDistThreshold, usedIdx=usedIdx)
	candidates_e1 = getCandidatesForPoint(e1, x3ds, maxDistThreshold, usedIdx=usedIdx)
	candidates = np.unique(np.concatenate((candidates_e0, candidates_e1), axis=0))
	if usedIdx:
		candidates = np.unique(np.array(usedIdx)[candidates])

	if not candidates.any():
		return

	triangleCandidates = []
	scoreData = []

	for candidate in candidates:
		if candidate in [e0, e1]: continue

		triangleCandidate = [e0, e1, candidate]
		if cache.triangleExists(triangleCandidate, x3ds_labels, includeBlacklisted=True):
			continue

		rayCand = rayNormals[triangleCandidate] if rayNormals is not None else None
		triangle, triangleNormal, score = scoreTriangle(triangleCandidate, registry, x3ds, x3ds_labels,
														rayCand, rayNormalsDp,
														cache, None, threshold_neighbourNorm,
														threshold_rayAgreement, threshold_normalRayAgreement,
														30.0, forgiving=forgiving)

		if triangle:
			triangleCandidates.append([triangle, triangleNormal])
			scoreData.append(score)

	# Proceed if there are any triangles that can realistically be created from this edge
	if triangleCandidates and scoreData:
		# Examine the triangles on offer and pick the best one - Temp: Pick the best shape
		scores = []
		for cand, data in zip(triangleCandidates, scoreData):
			angleRels = data[0]
			angles = data[1]
			oddAngleIndex = data[2]
			neighbouringTriangles = data[3]
			edgeLengths = data[4]
			neighbourAgreements = data[5]
			oddAngleDev = abs(angles[oddAngleIndex] - 90)
			if neighbouringTriangles == 0:
				neighbourEffect = 1
			else:
				neighbourEffect = neighbouringTriangles

			#score = (angleRels + oddAngleDev) / neighbourEffect - np.mean(neighbourAgreements) * 20
			score = (angleRels + oddAngleDev) * (1 / np.exp(np.mean(neighbourAgreements)))
			scores.append(score)

			# print "> Triangle:", x3ds_labels[cand[0]], score

		# if np.where(x3ds_labels == 1521)[0] in [e0, e1]:# and np.where(x3ds_labels == 1498)[0] in [e0, e1]:
		# 	print ">>", x3ds_labels[[e0, e1]], len(triangleCandidates), x3ds_labels[triangleCandidates], scores

		# Prefer connecting to points that already share triangles (as we assume we've carefully selected them :P) coll_triangles[i]

		minIndex = np.argmin(scores)
		if scores[minIndex] > 80:
			cache.addBlacklistedCandidate(triangleCandidates[minIndex][0], x3ds_labels)
			return

		pickIndex = np.argmin(scores)
		newTriangle, triangleNormal = triangleCandidates[pickIndex]

		if cache.addTriangle(newTriangle, triangleNormal, triangles, x3ds_labels):
			if edgeKey in processedEdges:
				processedEdges[edgeKey] = 2
			else:
				processedEdges[edgeKey] = 1

			registerTriangleAction(registry, newTriangle, x3ds_labels, "Added: %.2f | %s" % (np.min(scores), scoreData[pickIndex]))
			processTriangle(newTriangle, triangles, x3ds, x3ds_labels, rayNormals, rayNormalsDp, cache, registry, its + 1, usedIdx=usedIdx, labellingData=labellingData, forgiving=forgiving, maxIts=maxIts)

def processTriangle(triangle, triangles, x3ds, x3ds_labels, rayNormals, rayNormalsDp, cache, registry, its=0, usedIdx=None, labellingData=None, forgiving=False, maxIts=10):
	import itertools
	maxDistThreshold = 70.0
	minDistThreshold = 25.0

	if its == maxIts: return

	triangleNormal = np.cross(x3ds[triangle[1], :] - x3ds[triangle[2], :], x3ds[triangle[0], :] - x3ds[triangle[2], :])
	triangleNormal /= np.linalg.norm(triangleNormal)

	# minIndex = np.argmin(1 - abs(triangleNormal))
	# distTestMax[minIndex] = maxDistThreshold * (1 - abs(triangleNormal))[minIndex] * 3
	# distTestMin[np.argmin(1 - abs(triangleNormal))] = 0

	processedCandidates = []

	edges = set(((triangle[0], triangle[1]), (triangle[0], triangle[2]), (triangle[1], triangle[2])))
	for point in triangle:
		if True:
			edges = set(((triangle[0], triangle[1]), (triangle[0], triangle[2]), (triangle[1], triangle[2])))
			candidates = []
			_, neighbouringTriangles = cache.getTriangleNeighboursForPoint(point, x3ds_labels)
			for neighbour in neighbouringTriangles:
				edgesIt = itertools.combinations(neighbour, 2)
				for e0, e1 in edgesIt:
					if point not in [e0, e1]: continue
					n0 = np.sum(e0 == neighbouringTriangles, axis=1)
					n1 = np.sum(e1 == neighbouringTriangles, axis=1)
					numTrianglesContainingEdge = len(np.where(n0 + n1 == 2)[0])
					if numTrianglesContainingEdge == 1:
						if e0 != point and e0 not in candidates: candidates.append(e0)
						if e1 != point and e1 not in candidates: candidates.append(e1)

			if candidates:
				edgesIt = itertools.combinations(candidates, 2)
				for e0, e1 in edgesIt:
					triangleCandidate = [point, e0, e1]
					if cache.triangleExists(triangleCandidate, x3ds_labels): continue
					# verbose = testObserveTriangle(candidate, x3ds_labels, "Two edges match-up")

					if np.linalg.norm(x3ds[e0] - x3ds[e1]) > 75.0: continue
					rayCand = rayNormals[triangleCandidate] if rayNormals is not None else None
					newTriangle, triangleNormal, score = scoreTriangle(triangleCandidate, registry, x3ds, x3ds_labels,
																	   rayCand, rayNormalsDp,
																	   cache, None, 0.0, 0.0, 0.0, forgiving=True)
																	   # cache, None, 0.0, -0.25, -0.25, forgiving=True)
					if newTriangle:
						if cache.addTriangle(newTriangle, triangleNormal, triangles, x3ds_labels):
							processedCandidates.append(newTriangle)
							registerTriangleAction(registry, newTriangle, x3ds_labels, "Added from two edges")
							processTriangle(newTriangle, triangles, x3ds, x3ds_labels, rayNormals, rayNormalsDp, cache, registry, its + 1, usedIdx=usedIdx)

			processPoint(point, edges, x3ds, maxDistThreshold, minDistThreshold, triangle, usedIdx=usedIdx)

	if labellingData is None:
		processEdges(edges, triangles, x3ds, x3ds_labels, maxDistThreshold, minDistThreshold, rayNormals, rayNormalsDp,
					 cache, registry, 0.2, -0.16, -0.16, its, usedIdx=usedIdx, forgiving=forgiving, maxIts=maxIts)


def createTriangles(x3ds_means, x3ds_labels, rayNormals, usedIdx=None, forgiving=False, maxIts=10):
	registry = []
	triangles = []

	edges = set()
	# sortedTriangles = []
	# coll_normals, coll_triangles = {}, {}
	rayNormalsDp = np.dot(rayNormals, rayNormals.T)
	#points = np.where(x3ds_labels == 842)[0]
	points = np.where(x3ds_labels != -1)[0]

	cache = TriangleCache(x3ds_means)

	for point in points:
		processPoint(point, edges, x3ds_means, 50.0, 25.0, usedIdx=usedIdx)

	processEdges(edges, triangles, x3ds_means, x3ds_labels, 50.0, 25.0, rayNormals, rayNormalsDp,
				 cache, registry, forgiving=forgiving, maxIts=maxIts)

	return np.int32(triangles)