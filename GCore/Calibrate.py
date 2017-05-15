#!/usr/bin/env python

"""
GCore/Calibrate.py

Requires:
	sys
	numpy
	openCV as cv2
	
	ISCV
	
"""
import sys
import numpy as np

import ISCV
from GCore import Recon

def composeRT(R,tx_ty_tz,interest):
	'''Form the extrinsic matrix. WARNING, this generates a 4x4 matrix.'''
	RT = np.eye(4,dtype=np.float32)
	RT[:3,:3] = R
	RT[:3,3] = -np.dot(R,tx_ty_tz)
	RT[2,3] -= interest
	return RT

def composeR(pan_tilt_roll):
	"""
	Compose a 3x3 matrix representing the 'pan, tilt, roll' rotation passed
	
	Args:
		pan_tilt_roll (float[2]): rotation in Degrees
		
	Returns:
		float[3][3]: rotation matrix
		
	"""
	ps = -np.radians(pan_tilt_roll)
	(s1,s0,s2),(c1,c0,c2) = np.sin(ps),np.cos(ps)
	cc,cs,sc,ss = c1*c2,c1*s2,s1*c2,s1*s2
	return np.array([[ss*s0+cc,s2*c0,cs*s0-sc],[sc*s0-cs,c2*c0,cc*s0+ss],[c0*s1,-s0,c0*c1]],dtype=np.float32)

def composeK(fovX,ox=0,oy=0,square=1,skew=0):
	'''Form the intrinsic matrix. WARNING, this generates a 4x4 matrix'''
	K = np.eye(4,dtype=np.float32)
	f = 1.0/np.tan(np.radians(0.5*fovX))
	K[0,0] = f
	K[1,1] = f*square
	K[0,1],K[0,2],K[1,2] = skew,ox,oy
	return K

def composeKRT(K,RT):
	'''Form a projection matrix.'''
	return np.dot(K, RT)

def composeP_fromData(lens,pan_tilt_roll,tx_ty_tz,interest):
	'''Form a 4x4 projection matrix. WARNING, this generates a 4x4 matrix'''
	return np.dot(composeK(*lens), composeRT(composeR(pan_tilt_roll),tx_ty_tz,interest))

def decomposeR(R, axes='yxz'):
	"""
	Decompose a 3x3 rotation matrix into a vector of 3 radians.
	
	The rotation order is traditional right-to-left 'xyz'= R(z)*R(y)*R(x).
	The returned values will be in the order specified.
	
	Args:
		R (float[3][3]): Rotation matrix to decompose
		axes (string[3]): rotation order of return - note right to left.
		
	Retuns:
		float[3]: "ret" - resulting rotation in degrees
	"""
	i = ord(axes[0])-ord('x')
	if len(axes) == 1:	parity = 1 # single channel
	else:				parity = (ord(axes[1])-ord(axes[0])+3)
	j,k = (i+parity)%3,(i+2*parity)%3
	cj = (R[i,i]**2 + R[j,i]**2)**0.5
	if cj > 1e-30:	ret = np.array([np.arctan2(R[k,j],R[k,k]),np.arctan2(-R[k,i],cj),np.arctan2(R[j,i],R[i,i])],dtype=np.float32)
	else:			ret = np.array([np.arctan2(-R[j,k],R[j,j]),np.arctan2(-R[k,i],cj),0.0],dtype=np.float32)
	if ((parity%3) == 2): ret = -ret
	return np.degrees(ret) #[:len(axes)]

def decomposeRT(RT, interest=None, setInterest=True):
	from math import atan2, sqrt
	R,T = RT[:3,:3],RT[:3,3]
	cj = sqrt(R[1,1]*R[1,1] + R[0,1]*R[0,1])
	if cj > 1e-30:	ret = np.array([atan2(R[2,0],R[2,2]),atan2(-R[2,1],cj),atan2(R[0,1],R[1,1])])
	else:			ret = np.array([atan2(-R[0,2],R[0,0]),atan2(-R[2,1],cj),0.0])
	if setInterest: interest = sqrt(np.dot(T,T)) # choose an interest distance based on the origin
	return -np.degrees(ret), -np.dot(R.T, T + [0,0,interest]), interest

def decomposeK(K):
	from math import atan2
	#assert (np.allclose(K[0,0],K[1,1])), 'Weird non-square %f %f' % (K[0,0],K[1,1])
	return float(np.degrees(atan2(1.0,K[0,0])*2.0)),float(K[0,2]),float(K[1,2]),float(K[1,1]/K[0,0]),float(K[0,1])


def decomposeKRT(P):
	"""
	Form the decomposition P = K_3x3 RT_3x4.
	
	P can be 4x4 or 3x4.
	
	Args:
		P (float[3|4][4]): Projection matrix to decompose.
	
	Returns:
		float[3][3]: "K" - Intrinsic Matrix (skew, focal length (x,y), PP)
		float[3][4]: "RT" - Extrinsic Matrix ( world position & orientation)
		
	"""
	K = np.eye(3,dtype=np.float32)
	RT = np.zeros((3,4),dtype=np.float32)
	norm = np.sum(P[2,:3]**2)**-0.5 # should be 1.0
	P = P * norm # normalize the P-mat
	# K[2,:] = [0,0,1]
	# P[2,:] = RT[2,:]
	RT[2,:] = P[2,:]
	# K[1,:] = [0,fy,oy]
	# P[1,:] = fy * RT[1,:] + oy * RT[2,:]
	# P[1,:3].RT[2,:3] = oy
	K[1,2] = oy = np.dot(P[1,:3],RT[2,:3])
	fRT_1 = P[1,:] - oy * RT[2,:]
	K[1,1] = fy = np.sum(fRT_1[:3]**2)**0.5
	RT[1,:] = fRT_1/fy
	# K[0,:] = [fx,skew,ox,0]
	# P[0,:] = fx * RT[0,:] + skew * RT[1,:] + ox * RT[2,:]
	# P[0,:3].RT[2,:3] = ox
	# P[0,:3].RT[1,:3] = skew
	RT[0,:3] = np.cross(RT[1,:3],RT[2,:3]) # assuming RHCS
	K[0,2] = ox		= np.dot(P[0,:3],RT[2,:3])
	K[0,1] = skew	= np.dot(P[0,:3],RT[1,:3]) # should be 0.0
	K[0,0] = fx		= np.dot(P[0,:3],RT[0,:3])
	RT[0,3] = (P[0,3] - skew * RT[1,3] - ox * RT[2,3])/fx
	return K, RT # hopefully fx == fy and skew == 0.0 ...!

def makeMat(P, ks, width_height):
	"""
	Form a GRIP Camera Mat from the Projection matirx, distortion coeficents and sensor dimentions passed
	
	The return is the canonical 'Camera Mat' we use through GRIP. cf. AZ.
	
	https://en.wikipedia.org/wiki/Camera_matrix
	
	Args:
		P (float[3][4]: Projection matirx.
		(k1,k2) (float[2]): Radial Distortion factors k1, k2.
		width_height (int[2]) Sensor width, height in pixels.
		
	Returns:
		float[3][3]:	K
		float[3][4]:	RT
		float[3][4]:	P
		float[2]:		(k1, k2)
		float[3]:		T
		int[2]:			width_height
	
	Requires:
		decomposeKRT
	"""
	(k1,k2) = ks
	K, RT = decomposeKRT(P)
	T = -np.dot(RT[:,:3].T, RT[:,3])
	return (K,RT,np.dot(K,RT),np.float32([k1,k2]),T,np.int32([width_height[0],width_height[1]]))

def makeUninitialisedMat(ci, width_height, K=None):
	"""
	Form a 'Default' GRIP Camera Mat from the given sensor size and optional Extrinsic Matrix 'K'
	
	Args:
		ci (int): Camera Index - uninitalized cameras are spread along +ve X at a spacing of 100mm.
		width_height (int[2]) Sensor width, height in pixels.
		K (float[3][3]): Optional Extrinsic matrix. Default=None.
		
	Returns: (result of makeMat)
		float[3][3]:	K
		float[3][4]:	RT
		float[3][4]:	P
		float[2]:		(k1, k2)
		float[3]:		T
		int[2]:			width_height
	"""
	if K is None: K = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float32)
	I = np.array([[1,0,0,(ci+1)*-100],[0,1,0,0],[0,0,1,0]],dtype=np.float32) # spread the cameras along the positive x-axis
	return makeMat(np.dot(K,I),(0,0),width_height)

def undistort_points_mat(dets, mat, ret):
	ISCV.undistort_points(dets, float(-mat[0][0,2]), float(-mat[0][1,2]), float(mat[3][0]), float(mat[3][1]), ret)
	
def undistort_dets(dets, splits, mats):
	ret = np.zeros((splits[-1],2), dtype=np.float32)
	for mat,c0,c1 in zip(mats,splits[:-1],splits[1:]):
		undistort_points_mat(dets[c0:c1], mat, ret[c0:c1])
	return ret, splits

def rigid_align_points(A, B, scale=False, out=None):
	"""
	Given Nx3 matrices A and B with coordinates of N corresponding points.
	Solve A RT.T = B for rotation-translation matrix [R;T].
	R (A - mean(A)).T = (B - mean(B)).T for rotation matrix R.
	
	Args:
		A (float[][3]): array of points
		B (float[][3]): array of points
		scale (bool): Default = False.
		out(float[3][4]): Reference to variable to put result into. Default = None.
		
	Returns:
		float[3][4] "RT" - also a reference to Arg "out"
		
	Requires:
		np.linalg.svd
		np.linalg.det
		
	"""
	RT = out
	if RT is None: RT = np.zeros((3,4), dtype=np.float32)
	Bmean, Amean = np.mean(B, axis=0), np.mean(A, axis=0)
	A0,B0 = A-Amean,B-Bmean
	R = np.dot(B0.T, A0)
	S0, S1, S2 = np.linalg.svd(R) # U, S, VT
	np.dot(S0, S2, out=R) # U . VT
	if np.linalg.det(R) < 0: S0[:,2] *= -1; np.dot(S0, S2, out=R)
	if scale:
		sc = (np.sum((B0)**2)/(1e-8+np.sum((A0)**2)))**0.5
		R *= sc
	RT[:,:3] = R
	RT[:,3] = (Bmean-np.dot(R, Amean))
	return RT

def rigid_align_points_inliers(A,B, scale=False,out=None, threshold_ratio=2.0):
	RT = np.eye(3,4,dtype=np.float32) #rigid_align_points(A[inliers],B[inliers], scale=scale,out=out)
	for it in range(5):
		tmp = np.sum((np.dot(A,RT[:3,:3].T) + RT[:,3] - B)**2,axis=1)
		fifth = np.sort(tmp)[5]
		threshold = (fifth + 1.0) * (threshold_ratio**2)
		if fifth > 1e10: threshold = 0
		inliers = np.where(tmp < threshold)[0]
		if len(inliers) < 5: return np.eye(3,4,dtype=np.float32),inliers
		#print (it,len(inliers))
		RT = rigid_align_points(A[inliers],B[inliers], scale=scale,out=RT)
	return RT,inliers
	
def rigid_align_points_RANSAC(A,B, scale=False,out=None):
	# choose 3 points from the two sets; compute the residual of the rigid alignment; pick the one with the lowest residual
	import random
	best = 1e10
	RT = out
	if RT is None: RT = np.zeros((3,4), dtype=np.float32)
	for it in range(100):
		pass
		#TODO

def bundle_adjust(x2ds_data, x2ds_labels, mats):
	'''Given some 2d detections and labels, and some initialised matrices, refine the cameras.'''
	#TODO
	pass

def solve_camera_distortion_from_3d(x3s, x2s, P):
	"""
	Find distortion parameters given assignments
	
	(px,py) -> (1 + k1 r2 + k2 r2*r2) (px-ox,py-oy) + (ox,oy)
	r2 = (px-ox)**2+(py-oy)**2
	s = r + k1 r^3 + k2 r^5
	Solve [r^3, r^5] [k1; k2] = [s - r]
	Solve [r^4, r^6] [k1; k2] = [sr - rr]
	
	Args:
		x3s
		x2s
		P
		
	Returns:
		float[2]: "x" - k1, k2 as solved.
		
	Requires:
		decomposeKRT
		np.linalg.solve
	"""
	numPoints = len(x2s)
	K,RT = decomposeKRT(P)
	ox,oy = float(-K[0,2]), float(-K[1,2])
	sr = np.zeros(numPoints,dtype=np.float32)
	r35 = np.zeros((numPoints,2),dtype=np.float32)
	tmp = np.dot(x3s, P[:,:3].T) + P[:,3].reshape(1,3)
	ss = np.sum((tmp[:,:2]/-tmp[:,2].reshape(-1,1) - [ox,oy])**2, axis=-1)**0.5
	rs = np.sum((x2s - [ox,oy])**2,axis=-1)**0.5
	r35[:,0] = rs**4
	r35[:,1] = rs**6
	sr[:] = (ss-rs)*rs
	x = np.linalg.solve(np.dot(r35.T,r35), np.dot(r35.T,sr))
	return x
	
def cv2_solve_camera_from_3d(x3ds, x2ds, Kin=None, distortion=None, solve_distortion=False, solve_principal_point=False, solve_focal_length=True):
	"""
	Use opencv to solve a camera from corresponding 2d and 3d points.
	
	Because opencv uses a left-handed coordinate system, we do some funky maths.
		Our equation: K RT [x;y;z;1] = a[px;py;-1] (with a > 0)
		OPENCV eqn:   K RT [x;y;z;1] = a[px;py; 1]
	
	Let X3 = diag(1;-1;-1) be a matrix that flips y and z (== rotation 180 degrees about x).
	Importantly, det(X3) = 1, so premultiplying the equations by X3 doesn't flip the determinant. Also X3 X3 = I.
	When applied to [px;py;1], it flips py and reverses the direction the camera looks.
	
		X3 K RT = [X3 K X3] X3 RT [x;y;z;1] = X3 a[px;py;1] = a[px;-py;-1]
		
	[X3 K X3] introduces minus signs in K[0,1] and K[0,2].
	X3 RT affects last two rows of R and T.
	
	If we rotate RT 180 degrees about x AND flip py AND negate K[0,1:], then we convert from an opencv camera to ours.
	Finally, the optical centre is a problem. opencv wants us to work in pixels; whereas we want to put the origin in the middle.
	If we are refining a camera, opencv wants the optical centre to be inside the image. So we create an arbitrary 4x4 pixel image
	and add 2 to the 2d coordinates and to the optical centre.
	
	ADDITIONALLY, we offset the 3d coordinates to be closer to the origin, because opencv seems to want it that way. Seriously?
	
	Args:
		x3ds
		x2ds
		Kin Default = None
		distortion Default = None
		solve_distortion Default = False
		solve_principal_point Default = False
		solve_focal_length Default = True
		
	Returns:
		float[3][4]: "P" - solved P matrix
		float[2]: "ks" - k1, k2
		float: "rms" - error from cv2.calibrateCamera
		
	Requires:
		cv2.calibrateCamera
		cv2.solvePnP
		cv2.Rodrigues
	"""
	import cv2
	
	if Kin is None: Kin = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float32)
	else: Kin = Kin.copy()
	distCoeffs = np.zeros(5,dtype=np.float32)
	x2 = np.array([1,-1],dtype=np.float32)
	x3 = np.array([[1],[-1],[-1]],dtype=np.float32)
	Kin[0,1:] = -Kin[0,1:]
	Kin[0,1] = 0 # zero skew!
	fs = 10
	Kin[:2,2] += fs*0.5
	if distortion is not None: # attempt to initialise the distortion...
		f2 = Kin[0,0]**2
		distCoeffs[0] = -distortion[0]*f2
		distCoeffs[1] = -(distortion[1] -3*(distortion[0]**2))*(f2*f2)
	flags = cv2.CALIB_USE_INTRINSIC_GUESS|cv2.CALIB_FIX_ASPECT_RATIO|cv2.CALIB_FIX_K3|cv2.CALIB_ZERO_TANGENT_DIST
	if not solve_principal_point: flags = flags | cv2.CALIB_FIX_PRINCIPAL_POINT
	if not solve_focal_length: flags = flags | cv2.CALIB_FIX_FOCAL_LENGTH
	if not solve_distortion: flags = flags|cv2.CALIB_FIX_K1|cv2.CALIB_FIX_K2
	offset = np.mean(x3ds,axis=0)
	#print 'cf mean/median',offset,np.median(x3ds,axis=0)
	# K R' (x - offset) + K T' = K R' x + K (T' - R' offset); so T = T' - R offset
	# K R' M (x - offset) + K T' = K (R' M) x + K (T' - R' M offset); so R = R' M and T = T' - R offset
	rms,K,dist,rv,tv = cv2.calibrateCamera([np.array(x3ds-offset,dtype=np.float32)],[np.array(x2ds,dtype=np.float32)*x2 + fs*0.5],(fs,fs),Kin,distCoeffs,flags=flags)
	K0 = K.copy()
	if tv[0][2] < 0:
		#print '(WARNING) inverted camera [shouldn\'t happen], attempting to fix', 'num points %d'%len(x3ds)
		tv[0] *= -1
		cv2.solvePnP(np.array(x3ds-offset,dtype=np.float32),np.array(x2ds,dtype=np.float32)*x2 + fs*0.5, Kin, distCoeffs, rv[0], tv[0], True)
	R = cv2.Rodrigues(np.array(rv[0]))[0]
	T = np.array(tv[0],dtype=np.float32) - np.dot(R,offset.reshape(3,1))
	P = np.zeros((3,4),dtype=np.float32)
	P[:3,:3] = np.dot(K0,R)
	K[:2,2] -= fs*0.5
	K[0,1:] = -K[0,1:]
	P[:3,:3] = np.dot(K, R * x3)
	P[:3,3] = np.dot(K, T * x3).T
	# distortion is not the same model as ours. so we solve it again
	if distortion is not None or solve_distortion:
		ks = solve_camera_distortion_from_3d(x3ds, x2ds, P)
		#if not np.allclose(ks,0,0.5,0.5): print 'WARNING bad ks',ks,[k1,k2] #; ks = np.zeros(2,dtype=np.float32)
	else: ks = np.zeros(2,dtype=np.float32)
	return P, ks, rms

def cv2_solve_camera_from_3d_multi(x3ds, x2ds, Kin=None, distortion=None, solve_distortion=False, solve_principal_point=False, solve_focal_length=True):
	"""
	Use opencv to solve a camera from corresponding 2d and 3d points.
	
	Because opencv uses a left-handed coordinate system, we do some funky maths.
		Our equation: K RT [x;y;z;1] = a[px;py;-1] (with a > 0)
		OPENCV eqn:   K RT [x;y;z;1] = a[px;py; 1]
	
	Let X3 = diag(1;-1;-1) be a matrix that flips y and z (== rotation 180 degrees about x).
	Importantly, det(X3) = 1, so premultiplying the equations by X3 doesn't flip the determinant. Also X3 X3 = I.
	When applied to [px;py;1], it flips py and reverses the direction the camera looks.
	
		X3 K RT = [X3 K X3] X3 RT [x;y;z;1] = X3 a[px;py;1] = a[px;-py;-1]
		
	[X3 K X3] introduces minus signs in K[0,1] and K[0,2].
	X3 RT affects last two rows of R and T.
	
	If we rotate RT 180 degrees about x AND flip py AND negate K[0,1:], then we convert from an opencv camera to ours.
	Finally, the optical centre is a problem. opencv wants us to work in pixels; whereas we want to put the origin in the middle.
	If we are refining a camera, opencv wants the optical centre to be inside the image. So we create an arbitrary 4x4 pixel image
	and add 2 to the 2d coordinates and to the optical centre.
	
	ADDITIONALLY, we offset the 3d coordinates to be closer to the origin, because opencv seems to want it that way. Seriously?
	
	Args:
		x3ds
		x2ds
		Kin Default = None
		distortion Default = None
		solve_distortion Default = False
		solve_principal_point Default = False
		solve_focal_length Default = True
		
	Returns:
		float[3][4]: "P" - solved P matrix
		float[2]: "ks" - k1, k2
		float: "rms" - error from cv2.calibrateCamera
		
	Requires:
		cv2.calibrateCamera
		cv2.solvePnP
		cv2.Rodrigues
	"""
	import cv2
	
	if Kin is None: Kin = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float32)
	else: Kin = Kin.copy()
	distCoeffs = np.zeros(5,dtype=np.float32)
	x2 = np.array([1,-1],dtype=np.float32)
	x3 = np.array([[1],[-1],[-1]],dtype=np.float32)
	Kin[0,1:] = -Kin[0,1:]
	Kin[0,1] = 0 # zero skew!
	fs = 10
	Kin[:2,2] += fs*0.5
	if distortion is not None: # attempt to initialise the distortion...
		f2 = Kin[0,0]**2
		distCoeffs[0] = -distortion[0]*f2
		distCoeffs[1] = -(distortion[1] -3*(distortion[0]**2))*(f2*f2)
	flags = cv2.CALIB_USE_INTRINSIC_GUESS|cv2.CALIB_FIX_ASPECT_RATIO|cv2.CALIB_FIX_K3|cv2.CALIB_ZERO_TANGENT_DIST
	if not solve_principal_point: flags = flags | cv2.CALIB_FIX_PRINCIPAL_POINT
	if not solve_focal_length: flags = flags | cv2.CALIB_FIX_FOCAL_LENGTH
	if not solve_distortion: flags = flags|cv2.CALIB_FIX_K1|cv2.CALIB_FIX_K2
	offset = np.mean(x3ds,axis=0)
	#print 'cf mean/median',offset,np.median(x3ds,axis=0)
	# K R' (x - offset) + K T' = K R' x + K (T' - R' offset); so T = T' - R offset
	# K R' M (x - offset) + K T' = K (R' M) x + K (T' - R' M offset); so R = R' M and T = T' - R offset
	rms,K,dist,rvs,tvs = cv2.calibrateCamera([np.array(x3ds-offset,dtype=np.float32)]*len(x2ds),np.array(x2ds,dtype=np.float32)*x2 + fs*0.5,(fs,fs),Kin,distCoeffs,flags=flags)
	K0 = K.copy()
	for tv in tvs:
		if tv[2] < 0:
			print '(WARNING) inverted camera [shouldn\'t happen], attempting to fix', 'num points %d'%len(x3ds)
	Rs = [cv2.Rodrigues(np.array(rv))[0] for rv in rvs]
	Ts = [np.array(tv,dtype=np.float32) - np.dot(R,offset.reshape(3,1)) for tv,R in zip(tvs,Rs)]
	Ps = np.zeros((len(Ts),3,4),dtype=np.float32)
	Ps[:,:3,:3] = [np.dot(K0,R) for R in Rs]
	K[:2,2] -= fs*0.5
	K[0,1:] = -K[0,1:]
	Ps[:,:3,:3] = [np.dot(K, R * x3) for R in Rs]
	Ps[:,:3,3] = [np.dot(K, T * x3).T for T in Ts]
	# distortion is not the same model as ours. so we solve it again
	print dist
	if distortion is not None or solve_distortion:
		x3s = np.concatenate([np.dot(x3ds,R.T)+T.reshape(-1) for R,T in zip(Rs,Ts)])
		x2s = np.concatenate(x2ds).reshape(-1,2)
		ks = solve_camera_distortion_from_3d(x3s, x2s, np.dot(K,np.eye(3,4,dtype=np.float32)))
		print ks
		f2 = K[0,0]**2
		print 'compare', -ks[0]*f2, -(ks[1] -3*(ks[0]**2))*(f2*f2)
		
		#if not np.allclose(ks,0,0.5,0.5): print 'WARNING bad ks',ks,[k1,k2] #; ks = np.zeros(2,dtype=np.float32)
	else: ks = np.zeros(2,dtype=np.float32)
	return Ps, ks, rms
	
def detect_wand(x2ds_data, x2ds_splits, mats, thresh=20. / 2000., x3d_threshold=1000000.):
	Ps = np.array([m[2] / np.linalg.norm(m[2][0, :3]) for m in mats], dtype=np.float32)
	wand_x3ds = np.array([[160, 0, 0], [0, 0, 0], [-80, 0, 0], [0, 0, -120], [0, 0, -240]], dtype=np.float32)
	x2ds_labels = -np.ones(x2ds_data.shape[0], dtype=np.int32)
	ISCV.label_T_wand(x2ds_data, x2ds_splits, x2ds_labels, 2.0, 0.5, 0.01, 0.07)
	x2ds_labels2 = x2ds_labels.copy()
	count = np.sum(x2ds_labels2 != -1) / 5
	if count < 3: return None, None, None
	x3ds, x3ds_labels, E_x2ds_single, x2ds_single_labels = Recon.solve_x3ds(x2ds_data, x2ds_splits, x2ds_labels2, Ps)
	count = ISCV.project_and_clean(x3ds, Ps, x2ds_data, x2ds_splits, x2ds_labels, x2ds_labels2, thresh ** 2, thresh ** 2, x3d_threshold)
	if count < 3: return None, None, None
	x3ds, x3ds_labels, E_x2ds_single, x2ds_single_labels = Recon.solve_x3ds(x2ds_data, x2ds_splits, x2ds_labels2, Ps)
	assert np.all(x3ds_labels == [0, 1, 2, 3, 4]), 'ERROR: Labels do not match' # skip if somehow not all points seen
	assert np.max(x3ds ** 2) < 1e9, 'ERROR: Values out of bounds' + repr(x3ds)
	mat = rigid_align_points(wand_x3ds, x3ds)
	x3ds = np.dot(wand_x3ds, mat[:3, :3].T) + mat[:, 3]
	return x3ds, x3ds_labels, x2ds_labels2

def boot_cameras_from_wand(wand_frames, cameras_info, lo_focal_threshold=0.5, hi_focal_threshold=4.0, cv_2d_threshold=0.02):
	"""
	Attempt to boot position of cameras from 2d data containing a wand. This is assumed to be 5 marker T wand.
	
	TODO: Generalise size of wand to allow 120mm, 240mm, 780mm etc variations. Also use actual measurements of wand.

	Args:
		wand_frames
		cameras_info
		lo_focal_threshold=0.5
		hi_focal_threshold=4.0
		cv_2d_threshold=0.02
		
	Returns:
		Mat[]: "mats2" - list of GRIP Camera Mats of solved or uninitalised cameras.
		bool[]: "camera_solved" flag to show which cameras have been solved in this process.
		
	Requires:
		ISCV.label_T_wand
		Recon.solve_x3ds
		np.linalg.norm
		
	"""
	
	numCameras = len(cameras_info)
	numFrames = len(wand_frames)
	camera_solved = [False]*numCameras
	# use the wand to boot the first camera
	x2ds_data,x2ds_splits = wand_frames[0]
	x2ds_labels = -np.ones(x2ds_data.shape[0], dtype=np.int32)
	ISCV.label_T_wand(x2ds_data, x2ds_splits, x2ds_labels, 2.0, 0.5, 0.01, 0.07)
	first_x3ds = np.array([[160, 0, 0],[0, 0, 0],[-80, 0, 0],[0, 0, -120],[0, 0, -240]], dtype=np.float32)
	mats2 = [None]*numCameras
	first_good_cameras = [None]*numCameras
	for ci,(c0,c1) in enumerate(zip(x2ds_splits[:-1], x2ds_splits[1:])):
		x2ds = x2ds_data[c0:c1]
		labels = x2ds_labels[c0:c1]
		try:
			order = [list(labels).index(x) for x in range(5)]
		except:
			mats2[ci] = makeUninitialisedMat(ci, cameras_info[ci])
			camera_solved[ci] = False
			continue
		print ('found wand in camera',ci)
		first_good_cameras[ci] = x2ds[order]
		cv2_mat = cv2_solve_camera_from_3d(first_x3ds, x2ds[order])
		rms = cv2_mat[2]
		mats2[ci] = makeMat(cv2_mat[0], cv2_mat[1], cameras_info[ci])
		camera_solved[ci] = True
		if mats2[ci][0][0,0] < lo_focal_threshold or mats2[ci][0][0,0] > hi_focal_threshold or rms > cv_2d_threshold:
			print ('resetting bad camera',ci,'with focal',mats2[ci][0][0,0],'and error',rms)
			mats2[ci] = makeUninitialisedMat(ci,cameras_info[ci])
			camera_solved[ci] = False
	Ps2 = np.array([m[2]/m[0][0,0] for m in mats2],dtype=np.float32)
	x2ds_labels2 = x2ds_labels.copy()
	for ci in xrange(numCameras): # remove unsolved cameras
		if not camera_solved[ci]: x2ds_labels2[x2ds_splits[ci]:x2ds_splits[ci+1]] = -1
	x3ds_ret, x3ds_labels, E_x2ds_single, x2ds_single_labels = Recon.solve_x3ds(x2ds_data, x2ds_splits, x2ds_labels2, Ps2)

	print (x3ds_ret,first_x3ds) # all points should be within 2.5mm of 'true'
	assert(np.allclose(x3ds_ret, first_x3ds, 0.0, 2.5))

	# so, we booted some cameras and they reconstruct the wand in the correct place.
	# unfortunately, there is still an ambiguity: something like the Necker cube (two different ways we can perceive the wand).
	# as soon as the wand moves, we can resolve this
	for mfi in xrange(40,numFrames,20):
		print (mfi)
		x2ds_data,x2ds_splits = wand_frames[mfi]
		x2ds_labels = -np.ones(x2ds_data.shape[0],dtype=np.int32)
		ISCV.label_T_wand(x2ds_data, x2ds_splits, x2ds_labels, 2.0, 0.5, 0.01, 0.07)
		solved_cameras = np.where(camera_solved)[0]
		good_cameras = []
		second_good_cameras = [None]*numCameras
		print (solved_cameras)
		for ci in solved_cameras:
			c0,c1 = x2ds_splits[ci:ci+2]
			x2ds = x2ds_data[c0:c1]
			labels = x2ds_labels[c0:c1]
			try:
				order = [list(labels).index(x) for x in range(5)]
			except:
				continue
			diff = x2ds[order] - first_good_cameras[ci]
			if np.linalg.norm(diff) < 0.02*len(diff): continue # must have moved 'enough'
			good_cameras.append(ci)
			second_good_cameras[ci] = x2ds[order]
		print (good_cameras)
		if len(good_cameras) >= 3: # this is the good frame...
			x2ds_labels2 = x2ds_labels.copy()
			for ci in xrange(numCameras): # remove unsolved cameras
				if not ci in good_cameras: x2ds_labels2[x2ds_splits[ci]:x2ds_splits[ci+1]] = -1
			second_x3ds, second_x3ds_labels, _,_ = Recon.solve_x3ds(x2ds_data, x2ds_splits, x2ds_labels2, Ps2)
			for ci in solved_cameras:
				if ci not in good_cameras:
					print ('resetting bad camera',ci)
					mats2[ci] = makeUninitialisedMat(ci,mats2[ci][5])
					camera_solved[ci] = False
			for ci in good_cameras:
				cv2_mat = cv2_solve_camera_from_3d(np.concatenate((first_x3ds,second_x3ds)), np.concatenate((first_good_cameras[ci],second_good_cameras[ci])))
				rms = cv2_mat[2]
				print (ci,rms)
				mats2[ci] = makeMat(cv2_mat[0],cv2_mat[1],mats2[ci][5])
				camera_solved[ci] = True
			break # finished
	return mats2, camera_solved

def generate_wand_correspondences(wand_frames, mats2, camera_solved, rigid_filter=True, error_thresholds=None, x3d_threshold=1000000.):
	"""
	Args:
		wand_frames
		mats2
		camera_solved
		rigid_filter = True
		error_thresholds = None
		
	Returns:
		x2s_cameras
		x3s_cameras
		frames_cameras
		num_kept_frames
		
	Requires:
		ISCV.undistort_points
		ISCV.label_T_wand
		Recon.solve_x3ds
		ISCV.project_and_clean
		
	"""

	def get_order(labels):
		"""
		Return the x2d index of the five points of the T Wand
		
		Args:
			labels (int[]): 
			
		Returns:
			int[5]: "order" label indexes
			
		"""
		try:
			l = list(labels)
			order = [l.index(x) for x in xrange(5)]
			return order
		except:
			return None
	
	numCameras = len(mats2)
	Ps2 = np.array([m[2]/np.linalg.norm(m[2][0,:3]) for m in mats2],dtype=np.float32)
	x2ds_frames = []
	x2ds_labels_frames = []
	x2ds_splits_frames = []
	x3ds_frames = []
	# TODO wand geo should be passed in? must be compatible with the label_T_wand
	wand_x3ds = np.array([[160,0,0],[0,0,0],[-80,0,0],[0,0,-120],[0,0,-240]],dtype=np.float32)
	thresh = (20./2000.)**2 if error_thresholds is None else error_thresholds**2 # projection must be close to be included for intersection
	num_kept_frames = 0
	for fi,(x2ds_raw_data,x2ds_splits) in enumerate(wand_frames): # intersect over all frames with current solved cameras
		x2ds_data,_ = undistort_dets(x2ds_raw_data, x2ds_splits, mats2)
		x2ds_labels = -np.ones(x2ds_data.shape[0],dtype=np.int32)
		ISCV.label_T_wand(x2ds_data, x2ds_splits, x2ds_labels, 2.0, 0.5, 0.01, 0.07)
		x2ds_labels2 = x2ds_labels.copy()
		for cs,c0,c1 in zip(camera_solved,x2ds_splits[:-1],x2ds_splits[1:]): # remove labels for unsolved cameras
			if not cs: x2ds_labels2[c0:c1] = -1
		count = np.sum(x2ds_labels2 != -1)/5
		if count >= 3: # only use points seen in three solved cameras
			x3ds, x3ds_labels, E_x2ds_single, x2ds_single_labels = Recon.solve_x3ds(x2ds_data, x2ds_splits, x2ds_labels2, Ps2)
			count = ISCV.project_and_clean(x3ds, Ps2, x2ds_data, x2ds_splits, x2ds_labels, x2ds_labels2, thresh, thresh, x3d_threshold)
			if count < 3: continue
			x3ds, x3ds_labels, E_x2ds_single, x2ds_single_labels = Recon.solve_x3ds(x2ds_data, x2ds_splits, x2ds_labels2, Ps2)
			#if not np.all(x3ds_labels == [0,1,2,3,4]): print 'ERROR'; continue # skip if somehow not all points seen
			#if np.max(x3ds**2) > 1e9: print 'ERROR oh oh',x3ds; continue
			if rigid_filter: # enforce x3ds must be a rigid transform of the wand
				mat = rigid_align_points(wand_x3ds, x3ds)
				x3ds = np.dot(wand_x3ds,mat[:3,:3].T) + mat[:,3]
			for cs,c0,c1 in zip(camera_solved,x2ds_splits[:-1],x2ds_splits[1:]): #copy 'cleaned' labels for solved cameras to avoid bad data
				if cs: x2ds_labels[c0:c1] = x2ds_labels2[c0:c1]
			x2ds_frames.append(x2ds_raw_data)
			x2ds_splits_frames.append(x2ds_splits)
			x2ds_labels_frames.append(x2ds_labels) # CBD not x2ds_labels2, otherwise we can't add cameras!
			x3ds_frames.append(x3ds)
			num_kept_frames+=1

	# TODO collapse this into the code above and clean up
	x2s_cameras,x3s_cameras,frames_cameras = [],[],[]
	for ci in xrange(numCameras):
		orders = [get_order(xlf[xsf[ci]:xsf[ci+1]]) for xlf,xsf in zip(x2ds_labels_frames,x2ds_splits_frames)]
		which_frames = np.where([o is not None for o in orders])[0]
		if len(which_frames) == 0:
			x2s,x3s = np.zeros((0,2),dtype=np.float32),np.zeros((0,3),dtype=np.float32)
		else:
			x2s = np.vstack([x2ds_frames[fi][x2ds_splits_frames[fi][ci]:x2ds_splits_frames[fi][ci+1]][orders[fi]] for fi in which_frames])
			x3s = np.vstack([x3ds_frames[fi] for fi in which_frames])
		x2s_cameras.append(x2s)
		x3s_cameras.append(x3s)
		frames_cameras.append(which_frames)

	return x2s_cameras,x3s_cameras,frames_cameras,num_kept_frames



