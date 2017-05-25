#!/usr/bin/env python

import os, sys, re, math
import numpy as np
import IO
from UI import QApp, GLMeshes # TODO move this colin
from GCore import State, Calibrate, Character
from copy import deepcopy

g_transformTypes = {'transform', 'joint', 'locator','ikHandle', 'airField','hikFKJoint'} # derived from transform
g_constraintTypes = {'aimConstraint', 'orientConstraint', 'normalConstraint', 'parentConstraint', 'pointConstraint', 'poleVectorConstraint', 'scaleConstraint', 'tangentConstraint', 'geometryConstraint', 'lookAt'} # derived from constraint
g_transformTypes.update(g_constraintTypes) # constraint derived from transform
g_polyModifierTypes = {'polySmoothFace', 'polyTweak', 'polyTweakUV', 'polyMapCut', 'polyMapDel', 'polyMapSew', 'polyNormal', 'polyMoveUV', 'polyMoveFacetUV', 'polyFlipEdge', 'polySmooth', 'polyReduce', 'polyDelFacet', 'polyDelVertex', 'polyMergeFace', 'polySplit', 'polyAppendVertex', 'polySubdFace', 'polyCollapseF', 'polyCloseBorder', 'polyAppend', 'polyCollapseEdge', 'polyTriangulate', 'polyDelEdge', 'polyMergeEdge', 'polyColorPerVertex', 'polyNormalPerVertex', 'deleteUVSet', 'polySeparate', 'polyTransfer'} # derived from polyModifier
g_polyModifierWorldTypes = {'polySoftEdge', 'polyBevel', 'polyPlanarProj', 'polyProj', 'polyQuad', 'polySubdEdge', 'polyMoveVertex', 'polyMoveEdge', 'polyExtrudeEdge', 'polyMoveFace', 'polyChipOff', 'polyExtrudeFace', 'polySewEdge', 'polySphProj', 'polyCylProj', 'polyMergeVert'}
g_subdModifierTypes = {'subdAddTopology', 'subdAutoProj', 'subdCleanTopology', 'subdLayoutUV', 'subdMapCut', 'subdMapSewMove', 'subdModifier', 'subdModifierUV', 'subdModifierWorld', 'subdPlanarProj', 'subdTweak', 'subdTweakUV' } # derives from subdbase
g_polyModifierTypes.update(g_polyModifierWorldTypes) # polyModifierWorld derives from polyModifier
g_nodeTypes = {}
g_nodeTypes['geometryFilter'] = {\
	'ip':{'type':'compound','tip':'Input','vec':True,'compound':{\
		'ig':{'type':'geometry','tip':'The geometry input'},\
		'gi':{'type':'integer','tip':'The group id input','default':0}}},\
	'og':{'type':'geometry','vec':True,'tip':'Output geometry'},\
	'en':{'type':'float','default':1.0,'tip':'How much of the deformation should be applied.\n0.0 means no deformation at all;\n1.0 means the full deformation.'},\
}
g_nodeTypes['dagNode'] = {\
	'_DAG':{'type':None,'tip':'Is in the DAG'},\
	#'wm':  {'type':'matrix','longname':'worldMatrix','default':np.eye(3,4,dtype=np.float32)},\
	#'m':   {'type':'matrix','longname':'matrix','tip':'Local matrix','default':np.eye(3,4,dtype=np.float32)},\
	#'pm':  {'type':'matrix','longname':'parentMatrix','default':np.eye(3,4,dtype=np.float32)},\
	#'pim':{'longname':'parentInverseMatrix','type':'matrix','default':np.eye(3,4,dtype=np.float32), 'tip':'Inverse of parentMatrix instanced attribute.'},\
	#'wim':{'longname':'worldInverseMatrix','type':'matrix','default':np.eye(3,4,dtype=np.float32), 'tip':'Inverse of worldMatrix instanced attribute.'},\
	'tmp': {'longname':'template','type':'bool', 'default':False,'tip':'Boolean attribute that specifies whether the object is templated.\nTemplated objects are objects in the scene that are not rendered, always drawn in wireframe mode, and can only be selected using "template" selection mode.\nNote: template state is inherited by children of a dagNode.'},\
	'sech': {'longname':'selectionChildHighlighting','type':'bool','default':True, 'tip':'When the global selection preference for "Selection Child Highlight Mode" is set to "Use object highlight setting", this attribute will control the highlighting of children of this node. This control is inherited by this node\'s descendants, so if a root level node has this disabled any selected node in that hierarchy will not highlight its children.'},\
	'v':{'longname':'visibility','default':True,'type':'bool','tip':'Boolean attribute that is set to false for invisible objects in the scene.\nNote: visibility is inherited by children of a dagNode'},\
	'io':{'longname':'intermediateObject','type':'bool','default':False,'tip':'Boolean attribute that specifies whether the dagNode is an intermediate object resulting from a construction history operation.\ndagNodes with this attribute set to true are not visible and/or rendered.\nThis attribute is automatically set so changing this value may cause unpredictable results.\n'},\
	'obcc':{'longname':'objectColorRGB','type':'float3','tip':'The floating point object color.','default':np.zeros(3,dtype=np.float32)},\
	'wfcc':{'longname':'wireColorRGB','type':'float3','tip':'The floating point wire color.','default':np.zeros(3,dtype=np.float32)},\
}
g_nodeTypes['objectSet'] = {\
	'dsm':{'longname':'dagSetMembers','type':None,'tip':'Connections to this attribute specify the DAG nodes or parts (components) of the same that are members of this set.','vec':True},\
	'mwc':{'longname':'memberWireframeColor','type':'short','default':1,'tip':'The index of a user defined color in which the DAG object component members should appear.\nA value of -1 disables use of the color. Values outside the range [-1,7] may give unpredictable results.'},\
	'an':{'longname':'annotation','type':'string'},\
	'ub':{'longname':'usedBy','vec':True,'type':None},\
}
g_nodeTypes['camera'] = {\
	'ncp':{'default':0.1, 'type':'double', 'tip':'Distance from the camera to the near clipping plane'},\
	'fcp':{'default':10000.0, 'type':'double', 'tip':'Distance from the camera to the far clipping plane'},\
	'ow':{'default':10.0, 'type':'double', 'tip':'Orthographic width'},\
	'coi':{'default':5.0, 'type':'double', 'tip':'Distance to the centre of interest'},\
	'fl':{'default':35.0, 'type':'double', 'tip':'The focal length in mm'},\
	'cap':{'default':np.array((3.6,2.4),dtype=np.float32), 'type':'double2', 'tip':'Camera aperture'},\
	'pn':{'default':np.zeros(2,dtype=np.float32), 'type':'double2', 'tip':'2D camera pan'},\
	'sa':{'default':144.0, 'type':'double', 'tip':'The shutter angle for motion blur'},\
	'fs':{'default':5.6, 'type':'double', 'tip':'Camera F/Stop'},\
	'tp':{'default':np.zeros(3,dtype=np.float32), 'type':'double3', 'tip':'World point to tumble about'},\
	'o':{'default':False, 'type':'bool', 'tip':'Activate orthographic projection'},\
	}
g_nodeTypes['transform'] = {\
	'_X':{'type':None,'tip':'Is a transform'},\
	'r':{'default':np.zeros(3,dtype=np.float32),'type':'double3','tip':'Rotation'},\
	's':{'default':np.ones(3,dtype=np.float32),'type':'double3','tip':'Scale'},\
	't':{'default':np.zeros(3,dtype=np.float32),'type':'double3','tip':'Translation'},\
	'rx':{'type':'alias','alias':('r',0)}, 'ry':{'type':'alias','alias':('r',1)}, 'rz':{'type':'alias','alias':('r',2)},\
	'sx':{'type':'alias','alias':('s',0)}, 'sy':{'type':'alias','alias':('s',1)}, 'sz':{'type':'alias','alias':('s',2)},\
	'tx':{'type':'alias','alias':('t',0)}, 'ty':{'type':'alias','alias':('t',1)}, 'tz':{'type':'alias','alias':('t',2)},\
	'ro':{'type':'enum','default':0,'enum':('XYZ', 'YZX', 'ZXY','XZY', 'YXZ', 'ZYX'),'tip':'Rotation order'},\
	'rpt':{'longname':'rotatePivotTranslate','type':'double3','default':np.zeros(3,dtype=np.float32), 'tip':'Rotate pivot correction. Used when moving the rotate pivot point without affecting the overall transformation matrix.'},\
	'rp':{'longname':'rotatePivot','type':'double3','tip':'Point about which to rotate.','default':np.zeros(3,dtype=np.float32)},\
	'sp':{'longname':'scalePivot','type':'double3','tip':'Point about which to scale.','default':np.zeros(3,dtype=np.float32)},\
	'it':{'longname':'inheritsTransform','type':'bool','default':True,'tip':'Attribute that controls whether this transform inherits transformations from its parent transform.\nIf this value is false then the transform will act as though it is in world-space.\nIn other words, transformations to its parent will not affect the world-space position of this transform and the parentMatrix attribute on this transform will return the identity matrix.\nThis is primarily used when grouping objects with their construction history to prevent double transformations on the output object.'},\
}
g_nodeTypes['aimConstraint'] = {\
		'tt':{'type':'double','tip':'Input translate of a target.','longname':'targetTranslate','default':np.zeros(3,dtype=np.float32)},
	}
#g_nodeTypes['parentConstraint'] = {\
#		'cr':
#	}
g_nodeTypes['joint'] = {\
	'jo':{'default':np.zeros(3,dtype=np.float32),'type':'double3','tip':'Joint orient'},\
	'jt':{'default':'xyz','type':'string','tip':'Joint type'},\
	'jot':{'default':'xyz','type':'string','tip':'Joint orient type'},\
	'bps':{'default':np.eye(3,4,dtype=np.float32),'type':'matrix','tip':'joint bind pose for binding skin.\nThis attribute is connected to the dagPose node that stores the bindPose info for the joint.\nThis attribute stores the world matrix of the joint at the time of the bind.\nThe bindPose matrix is undefined if no bindSkin operation has been performed.'},\
	'ds':{'default':0,'type':'enum','enum':['bone','box'],'tip':'This attribute controls how the joint is drawn.\nThe "bone" setting draws the joints as normal bones, as in previous releases.\nThe "Multi-child as Box" draw style indicates that a single box will be drawn to represent the joint and its children,\nin case the joint has multiple children.\nThe box size is chosen as a bounding box of the children joints.\nTypically the "bone" draw style is preferable if you are creating a complex skeletal hierarchy such as human ribs,\nwhile the "Multi-child as Box" draw style is preferable for representing a large bone with multiple children such as the pelvic bone.'},\
	'ssc':{'type':'bool','default':True,'tip':'Indicates whether to compensate for the scale of the parent joint.'},\
	'is':{'default':np.ones(3,dtype=np.float32),'type':'double3','tip':'The scale of the parent joint.'},\
	'isx':{'type':'alias','alias':('is',0)}, 'isy':{'type':'alias','alias':('is',1)}, 'isz':{'type':'alias','alias':('is',2)},\
}
g_nodeTypes['mesh'] = {\
	'vt':{'type':'float3','vec':True,'tip':'This is an internal attribute representing local space vertex position, used for file I/O.\nIt should never be directly manipulated by the user. If it is modified, the results will be unpredictable.'},\
	'ed':{'type':'int3','vec':True,'tip':'Polygon Edges.'},\
	'uv':{'type':'float2','vec':True,'tip':'Polygon uvPoints.'},\
	'fc':{'type':'polyFaces','vec':True,'tip':'Polygon Faces.'},\
	'bw':{'type':'double','default':2.0,'tip':'Controls explicit border width'},\
	'w':{'type':'mesh','tip':'The world space meshes.'},\
	'i':{'type':'mesh','tip':'The input (creation) mesh.'},\
	'o':{'type':'mesh','tip':'The output (local space) mesh.'},\
		
}
g_nodeTypes['subdiv'] = {\
	'vt':{'type':'float3','vec':True,'tip':'Vertices'},\
	'ed':{'type':'int3','vec':True,'tip':'Edges'},\
	'fc':{'type':'polyFaces','vec':True,'tip':'Facets'},\
	'bw':{'type':'double','default':2.0,'tip':'Controls explicit border width'},\
	'cc':{'type':'subd','tip':'cached version of the subd'},\
}
g_nodeTypes['nurbsCurve'] = {\
	'cc':{'type':'nurbsCurve','tip':'Cached curve. Defines geometry of the curve. The properties are defined in this order:\ndegree, number of spans, form (0=open, 1=closed, 2=periodic), rational (yes/no), dimension,\nnumber of knots, list of knot values, number of CVs\nCV positions in x,y,z (and w if rational)'},\
	'degree':{'type':'short','default':0,'tip':'Curve degree'},\
	'spans':{'type':'int','default':0,'tip':'Number of spans'},\
	'form':{'type':'enum','default':0,'enum':['Open','Closed','Periodic'],'tip':'Nurbs curve form'},\
	}
g_nodeTypes['unitConversion'] = {'i':{'type':'Generic'},'o':{'type':'Generic'},'cf':{'type':'float','default':1.0}}
g_nodeTypes['reverse'] = {\
	'i':{'type':'float3','default':np.zeros(3,dtype=np.float32),'tip':'The input value'},\
	'ix':{'tip':'The input value [0]','type':'alias','alias':('i',0)},\
	'iy':{'tip':'The input value [1]','type':'alias','alias':('i',1)},\
	'iz':{'tip':'The input value [2]','type':'alias','alias':('i',2)},\
	'o':{'type':'float3','default':np.zeros(3,dtype=np.float32),'tip':'The output value'},\
	'ox':{'tip':'The output value [0]','type':'alias','alias':('o',0)},\
	'oy':{'tip':'The output value [1]','type':'alias','alias':('o',1)},\
	'oz':{'tip':'The output value [2]','type':'alias','alias':('o',2)},\
						  }
g_nodeTypes['sculpt'] = {'en':{'type':'double'},'dd':{'type':'double'}}
g_nodeTypes['multiplyDivide'] = {\
	'op':{'type':'enum','enum':['No op','Multiply','Divide','Power'],'default':1,'tip':'Controls the operation performed by this node'},\
	'i1':{'tip':'The first input value','type':'float3','default':np.zeros(3,dtype=np.float32)},\
	'i2':{'tip':'The second input value','type':'float3','default':np.zeros(3,dtype=np.float32)},\
	'o':{'tip':'The output value','type':'float3','default':np.zeros(3,dtype=np.float32)},\
	'ox':{'tip':'The output value [0]','type':'alias','alias':('o',0)},\
	'oy':{'tip':'The output value [1]','type':'alias','alias':('o',1)},\
	'oz':{'tip':'The output value [2]','type':'alias','alias':('o',2)},\
}
g_nodeTypes['pairBlend'] = {\
	'c':{'longname':'currentDriver','type':'enum','default':0,'tip':'Index of the current input driver'},\
	'it1':{'longname':'inTranslate1','type':'double3','default':np.zeros(3,dtype=np.float32),'tip':'Input translation 1'},\
	'ir1':{'longname':'inRotate1','type':'double3','default':np.zeros(3,dtype=np.float32),'tip':'Input rotation 1'},\
	'it2':{'longname':'inTranslate2','type':'double3','default':np.zeros(3,dtype=np.float32),'tip':'Input translation 2'},\
	'ir2':{'longname':'inRotate2','type':'double3','default':np.zeros(3,dtype=np.float32),'tip':'Input rotation 2'},\
	'w':{'longname':'weight','type':'double','default':1.0,'tip':'Weight between inputs 1 and 2. Weight 0 = all input 1; weight 1 = all input 2; weight 0.5 = half of each.'},\
	'ro':{'longname':'rotateOrder','type':'enum','enum':('XYZ', 'YZX', 'ZXY','XZY', 'YXZ', 'ZYX'),'default':0,'tip':'This attribute lets us know the order in which rx, ry, and rz are applied from the driven object so that we can output rotation values appropriately.'},\
	'txm':{'longname':'translateXMode','type':'enum','enum':['Blend Inputs','Input 1 Only','Input 2 Only'],'default':'0','tip':'Allows output translate x to be set to only input 1, only input 2, or the blended value'},\
	'tym':{'longname':'translateYMode','type':'enum','enum':['Blend Inputs','Input 1 Only','Input 2 Only'],'default':'0','tip':'Allows output translate y to be set to only input 1, only input 2, or the blended value'},\
	'tzm':{'longname':'translateZMode','type':'enum','enum':['Blend Inputs','Input 1 Only','Input 2 Only'],'default':'0','tip':'Allows output translate z to be set to only input 1, only input 2, or the blended value'},\
	'rm':{'longname':'rotateMode','type':'enum','enum':['Blend Inputs','Input 1 Only','Input 2 Only'],'default':0,'tip':'Allows output rotation to be set to only input 1, only input 2, or the blended value'},\
	'ri':{'longname':'rotInterpolation','type':'enum','enum':['Euler Angle','Quaternion'],'default':0,'tip':'Determines if rotation is calculated by linearly interpolating between Euler rotation values, or spherically interplating between quaternion values.'},\
	'ot':{'longname':'outTranslate','type':'double3','default':np.zeros(3,dtype=np.float32),'tip':'Output translation'},\
	'otx':{'longname':'outTranslate x','type':'alias','alias':('ot',0),'tip':'Output translation [0]'},\
	'oty':{'longname':'outTranslate y','type':'alias','alias':('ot',1),'tip':'Output translation [1]'},\
	'otz':{'longname':'outTranslate z','type':'alias','alias':('ot',2),'tip':'Output translation [2]'},\
	'or':{'longname':'outRotate','type':'double3','default':np.zeros(3,dtype=np.float32),'tip':'Output rotation'},\
	'orx':{'longname':'outRotate x','type':'alias','alias':('or',0),'tip':'Output rotation [0]'},\
	'ory':{'longname':'outRotate y','type':'alias','alias':('or',1),'tip':'Output rotation [1]'},\
	'orz':{'longname':'outRotate z','type':'alias','alias':('or',2),'tip':'Output rotation [2]'},\
}

g_nodeTypes['remapValue'] = {\
	'i':{'longname':'inputValue','type':'float','default':0.0,'tip':'InputValue is the raw input connection to remap.'},\
	'omn':{'type':'float','tip':'This determines the output value which maps to the bottom of the graph or the color black.','default':0},
	'omx':{'type':'float','tip':'This determines the output value at the graph top or white value.','default':1},\
	'imn':{'type':'float','tip':'This determines the input value which maps to the left of the gradients.','default':0},\
	'imx':{'type':'float','tip':'This determines the input value which maps to the right of the gradients.','default':1},\
	'vl':{'type':'compound','vec':True,'tip':'Value defines a range of values used to remap the input value to the outValue.\nThe Input Value parameter, along with the min and max attributes determine where to index into this gradient.','compound':
		{'vlp':{'longname':'value_Position','type':'float','default':0,'tip':'Position of ramp value on normalized 0-1 scale'},\
			'vlfv':{'longname':'value_FloatValue','type':'float','default':0},\
			'vli':{'longname':'value_Interp','type':'enum','enum':('None','Linear','Smooth','Spline'),'tip':'Ramp Interpolation controls the way the intermediate values are calculated. The values are:\nNone: No interpolation is done; the different colors just show up as different bands in the final texture.\nLinear: The values are interpolated linearly in RGB color space.\nSmooth: The values are interpolated along a bell curve, so that each color on the ramp dominates the region around it, then blends quickly to the next color.\nSpline: The values are interpolated with a spline curve, taking neighboring indices into account for greater smoothness.'},\
		}\
	},\
	'ov':{'type':'float','default':0.0,'longname':'outValue','tip':'OutValue is the final remapped value.'},
	'oc':{'type':'float3','default':np.zeros(3,dtype=np.float32),'longname':'outColor','tip':'OutColor is the final remapped color.'},
	'cl':{'type':'compound','vec':True,'longname':'color','compound':{
		'clc':{'longname':'color_Color','type':'float3','default':np.zeros(3,dtype=np.float32),'tip':'Ramp color at the sibling position'},
		'cli':{'longname':'color_Interp','type':'enum','default':0,'enum':('None','Linear','Smooth','Spline'),'tip':'Ramp Interpolation controls the way the intermediate values are calculated. The values are:\nNone: No interpolation is done; the different colors just show up as different bands in the final texture.\nLinear: The values are interpolated linearly in RGB color space.\nSmooth: The values are interpolated along a bell curve, so that each color on the ramp dominates the region around it, then blends quickly to the next color.\nSpline: The values are interpolated with a spline curve, taking neighboring indices into account for greater smoothness.'}
		}\
	},\
}
g_nodeTypes['plusMinusAverage'] = {
	'op':{'longname':'operation','type':'enum','default':1,'enum':('No operation','Sum','Subtract','Average'),'tip':'Operation controls the mathematical operation done by this node. It has four possible values:\nNo operation: The first input is copied to the output. All other inputs are ignored.\nSum: All of the inputs are added together, and the output is set to their sum.\nSubtract: The output is set to the first input, minus all the other inputs.\nAverage: The output is set to the sum of all the inputs, divided by the number of inputs.'},
	'i1':{'longname':'input1D','type':'float','vec':True,'tip':'Input1D is the list of input values. Use this particular list to add, subtract, or average simple numerical values. The results of operations on this list will be placed in Output1D.'},
	'i2':{'longname':'input2D','type':'float2','vec':True,'tip':'Input2D is the list of 2d input values. Use this particular list to add, subtract, or average 2d values, such as UV coordinates. The results of operations on this list will be placed in Output2D.'},
	'i2x':{'longname':'input2Dx','type':'alias','alias':('i2',0),'tip':'The x component of input2d'}, # TODO i1,i2,i3 are arrays...
	'i2y':{'longname':'input2Dy','type':'alias','alias':('i2',1),'tip':'The y component of input2d'},
	'i3':{'longname':'input3D','type':'float3','vec':True,'tip':'Input3D is the list of 3d input values. Use this particular list to add, subtract, or average 3d values, such as XYZ coordinates or colors. The results of operations on this list will be placed in Output3D.'},
	'i3x':{'longname':'input3Dx','type':'alias','alias':('i3',0),'tip':'The x component of input3d'},
	'i3y':{'longname':'input3Dy','type':'alias','alias':('i3',1),'tip':'The y component of input3d'},
	'i3z':{'longname':'input3Dz','type':'alias','alias':('i3',2),'tip':'The z component of input3d'},
	'o1':{'longname':'output1D','type':'float','default':0.0,'tip':'Output1D holds the result of calculations performed on the Input1D list.'},
	'o2':{'longname':'output2D','type':'float2','default':np.zeros(2,dtype=np.float32),'tip':'Output2D holds the result of calculations performed on the Input2D list.'},
	'o2x':{'longname':'output2Dx','type':'alias','alias':('o2',0),'tip':'The X component of the output 2D value'},
	'o2y':{'longname':'output2Dy','type':'alias','alias':('o2',1),'tip':'The Y component of the output 2D value'},
	'o3':{'longname':'output3D','type':'float3','default':np.zeros(3,dtype=np.float32),'tip':'Output3D holds the result of calculations performed on the Input3D list.'},
	'o3x':{'longname':'output3Dx','type':'alias','alias':('o3',0),'tip':'The X component of the output 3D value'},
	'o3y':{'longname':'output3Dy','type':'alias','alias':('o3',1),'tip':'The Y component of the output 3D value'},
	'o3z':{'longname':'output3Dz','type':'alias','alias':('o3',2),'tip':'The Z component of the output 3D value'},
}
g_nodeTypes['animCurve'] = {\
	'ktv':{'type':'float2','vec':True,'tip':'Keyframe time/value pairs'},\
	'i':{'type':'float','tip':'Evaluation input','default':0.0},\
	'o':{'type':'float','tip':'Evaluation output','default':0.0},\
}
g_nodeTypes['blend'] = {\
	'i':{'type':'float','tip':'Inputs that will be blended','default':0.0},\
	'o':{'type':'float','tip':'Blended output','default':0.0},\
	'c':{'type':'int','tip':'Index of the current input driver','default':0},\
}
g_nodeTypes.setdefault('blendWeighted',{}).update(g_nodeTypes['blend'])
g_nodeTypes['groupId'] = {\
	'id':{'type':'int','tip':'System defined id used to uniquely identify an objectGroup in dagObjects','default':0},
}
g_nodeTypes['dagPose'] = {\
	'wm':{'type':'matrix','tip':'This attribute stores the inclusive matrix for the associated member at the time the pose is saved.\nIndices in the worldMatrix multi have a one-to-one correspondence with indices in the members multi.','vec':True},\
	'xm':{'type':'matrix','tip':' Stores the local matrix for the associated member at the time the pose is saved.\nIndices in the xformMatrix multi have a one-to-one correspondence with indices in the members multi.','vec':True},\
	'bp':{'type':'bool','default':False,'tip':'Indicates that the pose node is storing a bindPose.'},\
	'm':{'type':'Message','vec':True,'longname':'members','tip':'Connection to this attribute as a destination signals that the connected item is a member of the pose.'},\
	'p':{'type':'Message','vec':True,'longname':'parents','tip':' Connection to this attribute as a destination signals that the connected item is a parent of the associated member.\nIndices in the parents multi have a one-to-one correspondence with indices in the members multi.'},\
}
g_animCurveTypes = {'animCurveUT', 'animCurveUA', 'animCurveTT', 'animCurveTU', 'animCurveUL', 'animCurveUU', 'resultCurve', 'resultCurveTimeToLinear', 'resultCurveTimeToTime', 'resultCurveTimeToUnitless', 'resultCurveTimeToAngular', 'animCurveTA', 'animCurveTL'}
for nt in g_animCurveTypes: g_nodeTypes.setdefault(nt,{}).update(g_nodeTypes['animCurve']) # nodetypes derived from animCurve
# these types derive ultimately from dagNode (via curveShape->controlPoint->deformableShape->geometryShape->shape)
g_shapeTypes = {'camera','mesh','nurbsCurve','subdiv','nurbsSurface','fosterParent'} # derived from shape
g_nodeTypes.setdefault('transform',{}).update(g_nodeTypes['dagNode']) # transform derives from dagNode
for nt in g_shapeTypes:
	g_nodeTypes.setdefault(nt,{}).update(g_nodeTypes['dagNode']) # shape derives from dagNode
for nt in g_transformTypes:
	if nt != 'transform': g_nodeTypes.setdefault(nt,{}).update(g_nodeTypes['transform']) # derived from transform
#g_nodeTypes['blendShape'] = {'it':{'type':'compound','vec':True,'compound':{ #http://download.autodesk.com/us/maya/2009help/Nodes/blendShape.html
g_nodeTypes['skinCluster'] = {\
	'skm':{'longname':'skinningMethod','type':'enum','enum':('Classical','Dual quaternion','Blended'),'default':0,'tip':'An Enum to set the algorithm method used.\nClassical is the linear vector blended skinning.\nDual quaternion uses a joint space non-linear blend.\nBlended use a weight set to linear interpolate between them.'},\
	'bw':{'longname':'blendWeights','type':'double','default':0.0,'vec':True,'tip':'The blend weights per vertex for blending between dual quaternion and classic skinning.'},\
	'pm':{'longname':'bindPreMatrix','type':'matrix','vec':True,'tip':'The inclusive matrix inverse of the driving transform at the time of bind'},\
	'ma':{'longname':'matrix','type':'matrix','vec':True,'tip':'Driving transforms array'},\
	'gm':{'longname':'geomMatrix','type':'matrix','tip':'The inclusive matrix of the geometry path at the time of the bound.'},\
	'wl':{'longname':'weightList','type':'compound','vec':True,'tip':'Bundle of weights for each CV','compound':{
		'w':{'longname':'weights','type':'double','default':'0.0','vec':'True','tip':'weights for each target'}}},\
}
g_nodeTypes['tweak'] = {\
	'rtw':{'longname':'relativeTweak','type':'bool','default':True,'tip':'If set, the tweaks are relative, otherwise they are constrained to an absolute.'},\
	'pl':{'longname':'plist','type':'compound','vec':True,'tip':'list of 3double points associated with the geometry input of the same index','compound':{'cp':{'longname':'controlPoints','type':'double3','vec':True}}},\
	'vl':{'longname':'vlist','type':'compound','vec':True,'tip':'list of 3float points associated with the geometry input of the same index','compound':{'vt':{'longname':'vertex','type':'float3','vec':True}}},\
}
g_nodeTypes['transformGeometry'] = {\
	'ig':{'type':'geometry','tip':'The input geometry to be transformed'},\
	'txf':{'type':'matrix','tip':'The transform to be applied on geometry.'},\
	'itf':{'type':'bool','tip':'Invert transform before applying on input geometry.'},\
	'fn':{'type':'bool','tip':'Controls if the normals of the geometry should be frozen or not. Applies only to polygonal objects.'},\
	'og':{'type':'geometry','vec':True,'tip':'The ouput transformed geometry'},\
}
g_geometryFilterTypes = {'skinCluster','blendShape','tweak','jiggle','cluster','ffd'} # derived from geometryFilter
for nt in g_geometryFilterTypes:
	g_nodeTypes.setdefault(nt,{}).update(g_nodeTypes['geometryFilter'])

def strip_quotes(s):
	s = s.strip()
	if s == '': return s
	if s[0] == '"' and s[-1] == '"': return s[1:-1]
	return s

def extract_range(an):
	ran = None # not a range
	if '[' in an: # extract the range
		an = an.split('[')
		assert an[1][-1] == ']'
		ran = an[1][:-1].split(':')
		if ran != ['*']: ran = map(int,ran)
		an = an[0]
	return an,ran

def extract_matrix(val):
	if val[0] == '\"xform\"':
		# all the attributes are written inline
		d = {'s':map(float,val[1:4]),'r':map(float,val[4:7]),'ro':int(val[7]),
		't':map(float,val[8:11]),'h':map(float,val[11:14]),'sp':map(float,val[14:17]),
		'st':map(float,val[17:20]),'rp':map(float,val[20:23]),'rpt':map(float,val[23:26]),
		'raq':map(float,val[26:30]),'jpq':map(float,val[30:34]),'is':map(float,val[34:37]),
		'cps':val[37]=='yes'}
		# TODO NOTE raq and jpq are quaternions, which aren't properly handled here
		# NOTE jpq is given as jo in the docs: http://download.autodesk.com/us/maya/2011help/CommandsPython/setAttr.html
		# although jo and jp serve the same role..
		return localMatrix(d),val[38:]
	if map(float,(val[3],val[7],val[11],val[15])) == (0.0,0.0,0.0,1.0):
		print 'ERROR: bad matrix', val
	#assert(val[3] == '0' and val[7] == '0' and val[11] == '0' and val[15] == '1')
	return np.array([[val[0],val[4],val[8],val[12]],[val[1],val[5],val[9],val[13]],[val[2],val[6],val[10],val[14]]],dtype=np.float32),val[16:]

#@profile
def read_MA(filename):
	unexpected = {}
	re_words = re.compile(r'[^"\s]\S*|".+?"')
	re_index = re.compile('\[[0-9]+\]')
	# seen in this file:
	# commands ['addAttr', 'currentUnit', 'setAttr', 'connectAttr', 'createNode', '//Maya', 'dataStructure', 'relationship', 'applyMetadata', 'fileInfo', 'requires', 'select']
	# node types ['unitConversion', 'blendShape', 'tweak', 'polyTweakUV', 'polyMapSewMove', 'shadingEngine', 'locator', 'place2dTexture', 'objectSet', 'blendWeighted', 'displayLayer', 'pointConstraint', 'multiplyDivide', 'nurbsCurve', 'script', 'bump2d', 'mentalrayFramebuffer', 'transform', 'parentConstraint', 'lambert', 'materialInfo', 'camera', 'remapValue', 'plusMinusAverage', 'renderLayerManager', 'hyperGraphInfo', 'animCurveUU', 'hyperView', 'file', 'mentalrayGlobals', 'skinCluster', 'phong', 'hyperLayout', 'joint', 'mesh', 'displayLayerManager', 'groupParts', 'groupId', 'mentalrayOptions', 'dagPose', 'lightLinker', 'partition', 'renderLayer', 'polyNormalPerVertex', 'mentalrayItemsList', 'dx11Shader', 'polySoftEdge']
	# default nodes :time1, :renderPartition, :renderGlobalsList1, :defaultShaderList1, :postProcessList1, :defaultRenderUtilityList1, :defaultRenderingList1, :defaultTextureList1, :initialShadingGroup, :initialParticleSE, :defaultResolution, :defaultLightSet, :defaultObjectSet, :hardwareRenderGlobals, :hardwareRenderingGlobals, :defaultHardwareRenderGlobals

	nodeLists = {'DAG':[],'ALL':{}}

	for k in g_nodeTypes.keys(): nodeLists[k] = []
	root_node = {'name':'__ROOT__','_name':'/root','type':'transform','attrs':{},'cache':{'wm':np.eye(3,4,dtype=np.float32),'wim':np.eye(3,4,dtype=np.float32),'pm':np.eye(3,4,dtype=np.float32),'pim':np.eye(3,4,dtype=np.float32)},'parent':None,'children':{}}
	po = root_node
	pli = []
	for line_number,l in enumerate(open(filename,'r')):
		try:
			l = str.strip(l)
			if l == '' or l[0] == '/': continue
			pli.append(l)
			if not l.endswith(';'): continue
			l = ' '.join(pli)
			pli = []
			l = re_words.findall(l[:-1])
			if l[0] == 'setAttr':
				ai = 1
				vs = None
				if l[ai] == '-k': ai += 2 # (some badly converted files) keyable FIXME
				if l[ai] == '-s': vs = int(l[ai+1]); ai += 2 # it is a multi-attribute array; this is the size (hint) NOTE it may not be contiguous
				if l[ai] == '-l': ai += 2 # lock FIXME
				if l[ai] == '-av': ai += 1 # altered value : don't overwrite with an evaluate else data may be lost FIXME
				if l[ai] == '-k': ai += 2 # keyable FIXME
				if l[ai] == '-ch': ai += 2 # capacity hint: mesh.face - hints the total number of elements in the face edge lists
				if l[ai] == '-cb': ai += 2 # visible in channel box (NB keyable overrides this) FIXME
				# -ca # caching on/off TODO
				# -c # clamp if the value is outside of range TODO
				an = strip_quotes(l[ai]).rpartition('.')
				vi = ai+1
				vt = None # default is a numeric type, but we don't know which (we will guess)
				if vi < len(l) and l[vi] == '-type': vt = strip_quotes(l[vi+1]); vi += 2
				val = l[vi:]
				if len(val) and val[0].startswith('{') and val[-1].endswith('}'):
					val = map(strip_quotes,' '.join(val)[1:-1].split(','))
					g_nodeTypes[node['type']].setdefault(an_hash, {'type':vt}).setdefault('vec', True)
				an0,an = an[0],an[2]
				an,ran = extract_range(an)
				if an0 != '': assert(an0[0] == '.'); an = an0[1:]+'.'+an
				assert not an.startswith('-'),'attr name shouldn\'t start with a -\n'+repr(l)+'\n'+repr(an)
				an_hash = re_index.sub('#',an) # some attributes are structures of attributes; we don't really handle this yet
				if vt is None and g_nodeTypes[node['type']].has_key(an_hash):
					vt = g_nodeTypes[node['type']][an_hash]['type'] # retrieve from the dictionary!
				if ran is None and vs is None and val: # there is a single value
					if vt is None: # try to guess the type
						if val[0] == 'yes' or val[0] == 'no': vt = 'bool'
						elif '.' in val[0]: vt = 'float'
						elif val[0].isdigit() or (val[0].startswith('-') and val[0][1:].isdigit()): vt = 'long'
						else: print 'read_MA #####WARNING couldn\'t guess type of',val
					if vt is not None:
						#print node['name'],node['type'],an_hash,vt,val[:10]
						if vt == 'bool': val = {'no':False,'yes':True,'0':False,'1':True}[val[0]]
						elif vt == 'float' or vt == 'double' or vt == 'doubleAngle' or vt == 'doubleLinear': val = float(val[0])
						elif vt == 'long' or vt == 'enum' or vt == 'int' or vt == 'short':
							if '.' in val[0]:
								print 'read_MA #####WARNING changing type to float', node['name'],node['type'],an_hash,vt,val
								vt = g_nodeTypes[node['type']][an_hash]['type'] = 'float'
								val = float(val[0])
							else:
								val = int(val[0])
						elif vt == 'string': val = strip_quotes(val[0])
						elif vt == 'pointArray': val = np.array(val[1:],dtype=np.float32).reshape(int(val[0]),4)
						elif vt == 'matrix': val,_ = extract_matrix(val)
						elif vt == 'float2' or vt == 'double2': assert(len(val)==2); val = np.array(val,dtype=np.float32)
						elif vt == 'float3' or vt == 'double3': assert(len(val)==3); val = np.array(val,dtype=np.float32)
						elif vt == 'stringArray': assert(len(val) == int(val[0])+1); val = map(strip_quotes,val[1:])
						elif vt == 'Int32Array': val = np.array(val[1:],dtype=np.int32).reshape(int(val[0]))
						elif vt == 'componentList': 
							assert (len(val) == int(val[0])+1)
							val = map(strip_quotes,val[1:])
							if val:
								val = zip(*map(extract_range, val))
								assert (list(val[0]) == [val[0][0]]*len(val[0])), "all components must address the same attribute!"
								val = (val[0][0],[x for y in [range(x[0],x[1]+1) if len(x)>1 else x for x in val[1]] for x in y])
							else: val = (None,[]) # special case, empty list
						elif vt == 'attributeAlias':
							pass # TODO
						elif vt == 'nurbsCurve': # fill in some details for the gui
							node['attrs']['degree'] = int(val[0])
							node['attrs']['spans'] = int(val[1])
							node['attrs']['form'] = int(val[2])
						elif vt == 'nurbsSurface': pass # TODO
						elif vt == 'dataPolyComponent': pass # TODO
						elif vt == 'subd': pass # store as list for now
						else: print 'read_MA WARNING unexpected',vt,l[:10]
				ntt = g_nodeTypes[node['type']].setdefault(an_hash, {'type':vt})
				if vt is not None: ntt['type'] = vt # update the dictionary
				if ran is not None:
					ntt['vec'] = True # TODO eventually, this can become an assert
					#if an_hash == 'vt': print 'read_MA',node['name'],node['type'],an_hash,vt,val[:10]
					if vt is not None and ('float' in vt or 'double' in vt):
						val = [x.replace('1.#INF','inf') for x in val] # apparently this is the new way of representing infinity?
						val = map(float,val) # np.array(val,dtype=np.float32) # TODO np.array causes problems if it's an array where each key is length 1
					elif vt is not None and ('long' in vt or 'int' in vt):
						val = map(int,val)
					node['attrs'].setdefault(an,{})
					if not isinstance(node['attrs'][an],dict):
						print 'read_MA ###ARGH###',node['type'],'.',an,'= [',vt,'] ',node['attrs'][an],'...changing to dict',ran,val[:10]
						node['attrs'][an] = {}
					if vt is None and val and val[0] == 'f':
						vt = ntt['type'] = 'polyFaces' # infer this
					if vt == 'matrix':
						ms = []
						while len(val):
							m,val = extract_matrix(val)
							ms.append(m)
						val = ms
					if vt == 'polyFaces':
						val = (' '+' '.join(val)).split(' f ')[1:]
					if len(ran) == 1:
						node['attrs'][an][ran[0]] = (val if len(val) != 1 else (float(val[0]) if isinstance(val[0],np.float32) else val[0]))
					else:
						lv,lr = len(val),ran[1]+1-ran[0]
						tgt = node['attrs'][an]
						if lv == lr:
							tgt.update(zip(range(ran[0],ran[1]+1),val))
						else:
							ls = lv/lr
							assert(lv == ls*lr)
							if type(val) is np.ndarray:
								tgt.update(zip(range(ran[0],ran[1]+1),val.reshape(lr,ls)))
							else:
								tgt.update(zip(range(ran[0],ran[1]+1),zip(*[iter(val)]*ls)))
							#for r in xrange(lr): tgt[r+ran[0]] = val[r*ls:r*ls+ls]
				elif val != []:
						node['attrs'][an] = val if vs is None else {} #[None]*vs
			elif l[0] == 'addAttr':
				# TODO there are lots of flags here, including -min, -max, -smn, -smx (soft min/max) -ln (longName), -nn (niceName), -is (internalSet)
				# -k (keyable), -en (enumName), -p (parent : yes, attributes can have parent attributes)
				#print 'read_MA','addAttr',l
				v = {'explicit':True}
				ai = l.index('-sn')+1
				an = strip_quotes(l[ai])
				if '-ln' in l:
					li = l.index('-ln')+1
					v['longname'] = strip_quotes(l[li])
				ti = l.index('-at')+1 if '-at' in l else l.index('-dt')+1
				tt = v['type'] = strip_quotes(l[ti])
				#print 'read_MA','addAttr',an
				if '-dv' in l:
					di = l.index('-dv')+1
					val = strip_quotes(l[di])
					if tt == 'double' or tt == 'float': val = float(val)
					elif tt == 'short' or tt == 'int' or tt == 'long' or tt == 'enum': val = int(val)
					elif tt == 'bool': val = {'no':False,'yes':True,'0':False,'1':True}[val]
					else: print 'read_MA ########TYPE',tt,val
				else: # what's the value???
					if tt == 'double' or tt == 'float' or tt == 'doubleLinear' or tt == 'doubleAngle': val = 0.0
					elif tt == 'short' or tt == 'int' or tt == 'long' or tt == 'enum': val = 0
					elif tt == 'bool': val = False
					elif tt == 'string': val = ''
					elif tt == 'message': val = '' # ??
					elif tt == 'float2' or tt == 'double2': val = np.zeros(2,dtype=np.float32)
					elif tt == 'float3' or tt == 'double3': val = np.zeros(3,dtype=np.float32)
					elif tt == 'matrix': val = np.eye(3,4,dtype=np.float32)
					elif tt == 'compound': val = {} # ??
					else: print 'read_MA ########TYPE NO VALUE',tt,node['type'],an,node['name']
				node['attrs'][an] = val
				g_nodeTypes[node['type']].setdefault(an, v)
			elif l[0] == 'connectAttr':
				a1n = strip_quotes(l[1])
				a2n = strip_quotes(l[2])
				assert not a1n[0] == '-'
				assert not a2n[0] == '-'
				nextAvail = False
				lock = 'off'
				ai = 3
				if len(l) > ai and l[ai] == '-l': lock = l[ai+1]; ai += 2
				if len(l) > ai and l[ai] == '-na': nextAvail = True; ai += 1 # attach to the next available index (the target has multiple) FIXME
				if len(l) > ai: print 'read_MA WARNING unexpected flags', l[ai], l
				a1n,a2n = a1n.partition('.'),a2n.partition('.')
				a1on,a2on = a1n[0].split('|'),a2n[0].split('|')
				if a1on[0].startswith(':') or a2on[0].startswith(':'): continue # TODO :miDefaultOptions
				o1,o2 = nodeLists['ALL'][a1on[0]][0]['parent'] if a1on[0] else root_node,nodeLists['ALL'][a2on[0]][0]['parent'] if a2on[0] else root_node
				try:
					for c in a1on: o1 = o1['children'][c] if c else root_node
					for c in a2on: o2 = o2['children'][c] if c else root_node
				except:
					print 'read_MA ??? caused by parent command, probably'
					continue
				o1.setdefault('outs',{}).setdefault(a1n[2],[]).append((o2,a2n[2]))
				o2.setdefault('ins',{})[a2n[2]] = (o1,a1n[2]) # each attribute can only be driven once!
			elif l[0] == 'createNode':
				nt = strip_quotes(l[1])
				ai = 2
				shared = False
				if ai != len(l) and l[ai] == '-s': shared = True; ai += 1 # shared across multiple files: only create if it doesn't already exist
				if ai != len(l) and l[ai] == '-n':
					ni = ai+1
					nn = strip_quotes(l[ni])
					ai += 2
				else:
					import random
					nn = nt+hex(random.randint(0,2**32))
				if ai != len(l) and l[ai] == '-p':
					pn = strip_quotes(l[ai+1])
					if pn[0] == '|': # path to root
						po = root_node
						pn = pn[1:].split('|')
						for p in pn: po = po['children'][p]
					else: # search to the root from the last-referenced parent
						while pn not in po['children']: po = po['parent']
						po = po['children'][pn]
					assert g_nodeTypes[po['type']].has_key('_DAG'), '### should be a transform: '+po['type']
					ai += 2
				else: po = root_node # the root node
				#print 'read_MA  createNode',nn, nt, po['name']
				if shared and po['children'].has_key(nn): node = po['children'][nn]; continue
				if po['children'].has_key(nn): print 'read_MA ERROR: name clash',po['name'],nn,l
				assert(not po['children'].has_key(nn)) # clash of names; TODO should rename the new node here?
				node = po['children'][nn] = {'name':nn,'type':nt,'attrs':{},'parent':po,'children':{},'ins':{'pm':(po,'wm'),'pim':(po,'wim')}}
				# make pm/pim be connections
				po.setdefault('outs',{}).setdefault('wm',[]).append((node,'pm'))
				po['outs'].setdefault('wim',[]).append((node,'pim'))
				g_nodeTypes.setdefault(nt, {}); nodeLists.setdefault(nt, [])
				nodeLists[nt].append(node)
				if g_nodeTypes[nt].has_key('_DAG'): nodeLists['DAG'].append(node)
				nodeLists['ALL'].setdefault(nn,[]).append(node)
			elif l[0] == 'select':
				si = l.index('-ne')+1
				sn = strip_quotes(l[si])
				#print 'read_MA select',sn,l
				if sn.startswith(':') and not root_node['children'].has_key(sn): # defaultNode
					root_node['children'][sn] = {'name':sn,'type':sn,'attrs':{},'parent':root_node,'children':{}}
					g_nodeTypes.setdefault(sn,{})
					nodeLists.setdefault(sn,[]).append(node)
					nodeLists['ALL'].setdefault(sn,[]).append(node)
				node = root_node['children'][sn]
			else: # TODO requires currentUnit fileInfo relationship
				if l[0] not in unexpected:
					unexpected[l[0]] = True
					print 'read_MA ###UNEXPECTED COMMAND', l[0]
		except Exception as E:
			import traceback
			print 'Exception at line',line_number,E.message,traceback.format_exc()
	return root_node,nodeLists

def make_rmat(xyz_degrees, ro=0):
	#[XYZ, YZX, ZXY, XZY, YXZ, ZYX][ro] NB right-to-left matrix order because our matrices are the transpose of maya
	sx,sy,sz = np.sin(np.radians(xyz_degrees))
	cx,cy,cz = np.cos(np.radians(xyz_degrees))
	mx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]],dtype=np.float32)
	my = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]],dtype=np.float32)
	mz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]],dtype=np.float32)
	m1 = [mz,mx,my,my,mz,mx][ro]
	m2 = [my,mz,mx,mz,mx,my][ro]
	m3 = [mx,my,mz,mx,my,mz][ro]
	return np.dot(m1,np.dot(m2,m3))

def decomposeR(R, ro=0):
	'''Decompose a 3x3 rotation matrix into a vector of 3 degrees, taking into account maya rotation order.'''
	if ro == 0 or ro is None:
		cj = math.sqrt(R[0,0]**2 + R[1,0]**2)
		if cj > 1e-30: return np.degrees([math.atan2(R[2,1],R[2,2]),-math.atan2(R[2,0],cj),math.atan2(R[1,0],R[0,0])],dtype=np.float32)
		else:          return np.degrees([-math.atan2(R[1,2],R[1,1]),-math.atan2(R[2,0],cj),0.0],dtype=np.float32)
	elif ro == 1:
		cj = math.sqrt(R[1,1]**2 + R[2,1]**2)
		if cj > 1e-30: return np.degrees([math.atan2(R[2,1],R[1,1]), math.atan2(R[0,2],R[0,0]),-math.atan2(R[0,1],cj)],dtype=np.float32)
		else:          return np.degrees([0.0,-math.atan2(R[2,0],R[2,2]),-math.atan2(R[0,1],cj)],dtype=np.float32)
	elif ro == 2:
		cj = math.sqrt(R[2,2]**2 + R[0,2]**2)
		if cj > 1e-30: return np.degrees([-math.atan2(R[1,2],cj),math.atan2(R[0,2],R[2,2]),math.atan2(R[1,0],R[1,1])],dtype=np.float32)
		else:          return np.degrees([-math.atan2(R[1,2],cj),0.0,-math.atan2(R[0,1],R[0,0])],dtype=np.float32)
	elif ro == 3:
		cj = math.sqrt(R[0,0]**2 + R[2,0]**2)
		if cj > 1e-30: return -np.degrees([math.atan2(R[1,2],R[1,1]),math.atan2(R[2,0],R[0,0]),-math.atan2(R[1,0],cj)],dtype=np.float32)
		else:          return -np.degrees([-math.atan2(R[2,1],R[2,2]),0.0,-math.atan2(R[1,0],cj)],dtype=np.float32)
	elif ro == 4:
		cj = math.sqrt(R[1,1]**2 + R[0,1]**2)
		if cj > 1e-30: return -np.degrees([-math.atan2(R[2,1],cj),math.atan2(R[2,0],R[2,2]),math.atan2(R[0,1],R[1,1])],dtype=np.float32)
		else:          return -np.degrees([-math.atan2(R[2,1],cj),-math.atan2(R[0,2],R[0,0]),0.0],dtype=np.float32)
	elif ro == 5:
		cj = math.sqrt(R[2,2]**2 + R[1,2]**2)
		if cj > 1e-30: return -np.degrees([math.atan2(R[1,2],R[2,2]),-math.atan2(R[0,2],cj),math.atan2(R[0,1],R[0,0])],dtype=np.float32)
		else:          return -np.degrees([0.0,-math.atan2(R[0,2],cj),-math.atan2(R[1,0],R[1,1])],dtype=np.float32)
	else:
		assert False, '###decomposeR with ro='+str(ro)

def matrix_mult(p,c):
	ret = np.dot(p[:3,:3],c[:3,:])
	ret[:,3] += p[:3,3]
	return ret

def recip(v):
	v = np.array(v,dtype=np.float32)
	return v/(v*v+1e-16)

def localMatrix(a):
	# wm = SP^-1 * S * SH * SP * ST * RP^-1 * RA * R * RP * RT * T
	# NOTE, all our matrices are transposed compared to maya; but we operate on the left
	m = np.eye(3,4,dtype=np.float32)
	if a.has_key('sp'):  m[:,3] -= a['sp']
	if a.has_key('s'):   m = (m.T * a['s']).T
	if a.has_key('sh'):  v = a['sh']; m[0,:] += v[0] * m[1,:] + v[1] * m[2,:]; m[1,:] += v[2] * m[2,:]
	if a.has_key('sp'):  m[:,3] += a['sp']
	if a.has_key('spt'): m[:,3] += a['spt']
	if a.has_key('rp'):  m[:,3] -= a['rp']
	if a.has_key('ra'):  m = np.dot(make_rmat(a['ra']), m)
	if a.has_key('r'):   m = np.dot(make_rmat(a['r'], a.get('ro',0)), m)
	if a.has_key('rp'):  m[:,3] += a['rp']
	if a.has_key('rpt'): m[:,3] += a['rpt']
	# TODO raq, jpq, cps from 'matrix'
	if a.has_key('jo'):  m = np.dot(make_rmat(a['jo']), m) # for joints
	if a.has_key('is') and a.get('ssc',True):  m = (m.T * recip(a['is'])).T # parent scale inverse, for joints TODO
	if a.has_key('t') and a['t'] is not '#':   m[:,3] += a['t'] # TODO Azary
	return m


def pickedCB(view,data,clearSelection=True):
	print 'pickedCB',view
	print data
	print clearSelection
	if data is None:
		QApp.app.select(None)
	else:
		primitive_type,pn,pi,distance = data
		if primitive_type is '3d':
			p = view.primitives[pn]
			if isinstance(p,GLMeshes):
				name = p.names[pi]
				print "Picked:", name
				QApp.app.select(name)

def setFrameCB(fi):
	print 'setFrameCB',fi

def matrix_inverse(m):
	ret = np.zeros((3,4),dtype=np.float32)
	try:
		ret[:,:3] = np.linalg.inv(m[:3,:3])
	except:
		print '???exception in matrix_inverse',list(ret.ravel()) # TODO HACK
		ret[:,:3] = np.eye(3) #m[:3,:3].T
	ret[:,3] = -np.dot(ret[:,:3],m[:3,3])
	return ret

def matrix_decompose(m):
	U,S,VT = np.linalg.svd(m[:3,:3]) # U,S,VT
	T = m[:,3]
	R = np.dot(U,VT)
	if np.linalg.det(R) < 0: S[2] *= -1; U[:,2] *= -1; R = np.dot(U,VT)
	return R,S,T

def invalidate_outs(outs_list):
	for node,k in outs_list:
		#print 'invalidating',k,'on node',node['name']
		cache = node.get('cache',{})
		if k in cache:
			cache.pop(k)
			outs = node.get('outs',{})
			if k in outs: invalidate_outs(outs[k])
		#invalidate_node(node) # TODO

def set_node_cache(node, kvs):
	#node.setdefault('cache',{}).update(kvs)
	cache = node.setdefault('cache',{})
	#outs = node.get('outs',{})
	attrs = g_nodeTypes.get(node['type'],{})
	for k,v in kvs:
		alias = attrs.get(k,{}).get('alias',None)
		if alias is not None:
			#if isinstance(attrs[alias[0]],tuple): print 'set_node_cache',k,cache.get(alias[0]),deepcopy(attrs[alias[0]]['default'])
			cache.setdefault(alias[0],deepcopy(attrs[alias[0]]['default']))[alias[1]] = v
		else:
			cache[k] = v
		#if k in outs: invalidate_outs(outs[k])

g_nodeTypes.setdefault('displayLayer',{})['_OUTS'] = {'di'}
def evaluate_displayLayer(node):
	'''outputs di'''
	set_node_cache(node, (('di',True),)) # TODO drawInfo

g_nodeTypes.setdefault('distanceDimShape',{})['_OUTS'] = {'dist'}
def evaluate_distanceDimShape(node):
	#TODO
	set_node_cache(node,(('dist',1.0),)) # put something there

g_nodeTypes.setdefault('distanceBetween',{})['_OUTS'] = {'d'}
def evaluate_distanceBetween(node):
	#TODO
	set_node_cache(node,(('d',1.0),)) # put something there

g_nodeTypes.setdefault('expression',{})['_OUTS'] = {'out'}
def evaluate_expression(node):
	#TODO
	set_node_cache(node,(('out',1.0),)) # put something there

g_nodeTypes.setdefault('blendColors',{})['_OUTS'] = {'op','opr','opg','opb'}
def evaluate_blendColors(node):
	#TODO
	set_node_cache(node,(('op',[1.0,1.0,1.0]),('opr',1.0),('opg',1.0),('opb',1.0))) # put something there

g_nodeTypes['blendWeighted']['_OUTS'] = {'o'}
def evaluate_blendWeighted(node):
	cache = node['cache']
	ins = [x for x in cache.keys() if x.startswith('i')]
	v = 0.0
	v_str = '0'
	for i in ins:
		w_key = 'w'+i[1:]
		w = cache.get(w_key,1.0)
		if w is '#': w = 1.0 # TODO
		v += float(cache[i]) * float(w)
		v_str += '+' + cache.get('#'+i,node['name']+'.'+i) + '*' + cache.get('#'+w_key,node['name']+'.'+w_key)
	set_node_cache(node,(('o',v),('#o',v_str)))

g_nodeTypes['unitConversion']['_OUTS'] = {'o'}
def evaluate_unitConversion(node):
	cache = node['cache']
	v = cache['i']
	if v is None or v is '#':
		print '###WARNING: evaluate_unitConversion, somehow ended up with None', v
		v = 0
	cf = cache.get('cf', 1.0)
	#print v,cf
	set_node_cache(node, (('o',np.array(v,dtype=np.float32)*np.float32(cf)),))

g_nodeTypes['reverse']['_OUTS'] = {'o'}
def evaluate_reverse(node):
	cache = node['cache']
	v = cache.get('i',(0,0,0))
	set_node_cache(node, (('o',np.float32(1.0)-np.array(v,dtype=np.float32),)))

g_nodeTypes['pairBlend']['_OUTS'] = {'ot','or'}
def evaluate_pairBlend(node):
	'''pairBlend reads c,it1,ir1,it2,ir2,w,ro,txm,tym,tzm,rm,ri and writes ot,or'''
	cache = node['cache']
	#pb = g_nodeTypes['pairBlend']
	#for k,v in pb.iteritems(): locals()['_'+k] = cache.get(k,v['default'])
	_w = cache.get('w',1.0)
	z = np.zeros(3,dtype=np.float32)
	_it1 = cache.get('it1',z)
	_it2 = cache.get('it2',z)
	_ir1 = cache.get('ir1',z)
	_ir2 = cache.get('ir2',z)
	_ot = _it1 * (1-_w) + _it2 * _w
	_or = _ir1 * (1-_w) + _ir2 * _w
	set_node_cache(node, (('ot',_ot),('or',_or)))

g_nodeTypes['multiplyDivide']['_OUTS'] = {'o','ox','oy','oz'}
def evaluate_multiplyDivide(node):
	cache = node['cache']
	op_type = cache.get('op',0)
	op = [lambda x,y:x,lambda x,y:x*y,lambda x,y:x/y,lambda x,y:x**y][op_type]
	op_str = ['first','mult','div','pow'][op_type]+'(%s,%s)'%(cache.get('#i1'),cache.get('#i2'))
	i1 = cache.get('i1',np.zeros(3,dtype=np.float32))
	i2 = cache.get('i2',np.ones(3,dtype=np.float32))
	ox = op(cache.get('i1x',i1[0]),cache.get('i2x',i2[0]))
	oy = op(cache.get('i1y',i1[1]),cache.get('i2y',i2[1]))
	oz = op(cache.get('i1z',i1[2]),cache.get('i2z',i2[2]))
	#print ox,oy,oz
	if ox is '#': ox = 0 # HACK
	if oy is '#': oy = 0
	if oz is '#': oz = 0
	o = np.array((ox,oy,oz),dtype=np.float32)
	set_node_cache(node, (('o',o),('#o',op_str)))

g_nodeTypes.setdefault('blendTwoAttr',{})['_OUTS'] = {'o'}
def evaluate_blendTwoAttr(node):
	cache = node['cache']
	#ivals = [int(x[2:-1]) for x in cache if x.startswith('i')]
	i0,i1,ab = float(cache.get('i[0]',0.0)),float(cache.get('i[1]',0.0)),cache.get('ab',0.0)
	ab = float(ab) if ab != '#' else 0.0
	set_node_cache(node, (('o',i0+(i1-i0)*ab),('#o','linterp(%s,%s,%s)'%(cache.get('#i[0]'),cache.get('#i[1]'),cache.get('#ab',ab)))))

g_nodeTypes['transform']['_OUTS'] = {'m','wm','wim'}
def evaluate_transform(node):
	'''transform reads t,s,r,pm etc and writes m,wm,wim.'''
	cache = node['cache']
	#if not cache.has_key('pm'): import pdb; pdb.set_trace()
	if not cache.get('it',True):
		pm = np.eye(3,4,dtype=np.float32)
		set_node_cache(node, (('pm',pm),('pim',pm.copy())))
	pm = cache.get('pm','#') #.get('pm',np.eye(3,4,dtype=np.float32))
	if pm is '#':
		evaluate_transform(node['parent'])
		pm = node['parent']['cache'].get('wm',np.eye(3,4,dtype=np.float32))
		cache['pm'] = pm
		cache['pim'] = matrix_inverse(pm)
		#print 'evaluate_transform',node['name'],'pm is not set: argh!?',pm
	m = localMatrix(cache) if g_nodeTypes[node['type']].has_key('_X') else np.eye(3,4,dtype=np.float32)
	wm = matrix_mult(pm,m)
	#if node['name']=='Root': import pdb; pdb.set_trace()
	set_node_cache(node, (('m',m),('wm',wm),('wim',matrix_inverse(wm))))

g_nodeTypes['transformGeometry']['_OUTS'] = {'og'}
def evaluate_transformGeometry(node):
	print '---evaluate_transformGeometry',node['name']
	cache = node['cache']
	ig = cache.get('ig')
	ig_str = cache.get('#ig',node['ins']['ig'][0]['type'])
	if ig is '#': ig = (np.zeros((0,3),dtype=np.float32),node,None) # ??? HACK TODO
	assert ig != '#',ig_str
	og_str = 'transformGeometry(\'%s\',%s)'%(node['name'],ig_str)
	set_node_cache(node,(('og',ig),('#og',og_str),))

def constraint(cache, keymap, is_orientConstraint = False):
	 # TODO there can be multiple targets with weights; we only support one here
	ld = dict([(k,cache['tg[0].'+v]) for k,v in keymap.iteritems() if 'tg[0].'+v in cache])
	if 'rp' in ld: ld['rpt'] = ld.pop('rp') + ld.get('rpt',0)  # HACK seems to be a bug in maya: rp isn't subtracted!? move it to rpt to emulate the bug
	m = localMatrix(ld)
	if 'pm' in ld: m = matrix_mult(ld['pm'],m)
	if 'ot' in ld: m[:,3] += np.dot(m[:3,:3], ld['ot'])
	if 'or' in ld: m[:3,:3] = np.dot(m[:3,:3], make_rmat(ld['or'],ld.get('ro',0)))

	# apply any offset here, (only for the orientConstraint)
	if is_orientConstraint and 'o' in cache: m[:3,:3] = np.dot(m[:3,:3],make_rmat(cache['o']))
	# TODO this is for the pointConstraint...
	#if 'o' in cache: m[:,3] += np.dot(m[:3,:3],cache['o'] * cache.get('cop',1))

	wm = m.copy()

	# we turn m into a local matrix by premultiplying the constrained parent inverse matrix
	if 'cpim' in cache: m = matrix_mult(cache['cpim'],m)

	# now correct for cjo, crt, crp to get the correct rotation and translation
	if 'cjo' in cache: m = np.dot(make_rmat(cache['cjo'],cache.get('cro',0)).T, m)
	if 'crt' in cache: m[:,3] -= cache['crt']
	#if 'crp' in cache: m[:,3] += np.dot(m[:3,:3], cache['crp']) - cache['crp']
	if 'crp' in cache: m[:,3] -= cache['crp']  # seems to be a bug: rp isn't subtracted!?
	if 'cjo' in cache: m[:,3] = np.dot(make_rmat(cache['cjo'],cache.get('cro',0)), m[:,3])

	# TODO when w = 0, we should use the rest position I think
	#if 'rs' in cache: ... rest scale
	#if 'rst' in cache: ... rest translation
	#if 'rsrr' in cache: ... rest rotation

	return m,wm

g_nodeTypes['parentConstraint']['_OUTS'] = {'cr','ct','crx','cry','crz','ctx','cty','ctz','m','wm','wim'}
def evaluate_parentConstraint(node):
	'''parent constraint reads tg (tt,tr,ts,tro,trp,tpm),tor,tot, cpim,cro,crp,crt and writes cr,ct.'''
	cache = node['cache']
	keymap = {'t':'tt','r':'tr','s':'ts','ro':'tro','rp':'trp','rpt':'trt','jo':'tjo','pm':'tpm','or':'tor','ot':'tot'}
	m,wm = constraint(cache, keymap)
	R,S,T = matrix_decompose(m)
	R = decomposeR(R, cache.get('cro',0))
	set_node_cache(node, (	('cr',np.array(R,dtype=np.float32)),('ct',np.array(T,dtype=np.float32)),\
							('crx',float(R[0])),('cry',float(R[1])),('crz',float(R[2])),\
							('ctx',float(T[0])),('cty',float(T[1])),('ctz',float(T[2])),\
							('m',m),('wm',wm),('wim',matrix_inverse(wm)))) # TODO remove crx cry etc (use aliases)

g_nodeTypes['pointConstraint']['_OUTS'] = {'ct','ctx','cty','ctz','m','wm','wim'}
def evaluate_pointConstraint(node):
	'''point constraint reads tg (tt,trp,trt,tpm),o,cop,crp.crt,cpim and writes ct.'''
	cache = node['cache']
	keymap = {'t':'tt','rp':'trp','rpt':'trt','pm':'tpm'}
	m,wm = constraint(cache, keymap)
	T = m[:,3]
	if 'o' in cache: T += cache['o'] * cache.get('cop',1)
	set_node_cache(node, (	('ct',np.array(T,dtype=np.float32)),\
							('ctx',float(T[0])),('cty',float(T[1])),('ctz',float(T[2])),\
							('m',m),('wm',wm),('wim',matrix_inverse(wm))))

g_nodeTypes['orientConstraint']['_OUTS'] = {'cr','crx','cry','crz','m','wm','wim'}
def evaluate_orientConstraint(node):
	'''orient constraint reads tg (tr,tro,tjo,tpm),o,cro,cjo,cpim and writes cr.'''
	cache = node['cache']
	keymap = {'r':'tr','ro':'tro','jo':'tjo','pm':'tpm'}
	m,wm = constraint(cache, keymap, is_orientConstraint=True)
	R = decomposeR(m[:,:3], cache.get('cro',0))
	# TODO use 'lr' (the previous computed orientation) to figure out the continuity
	set_node_cache(node, (	('cr',np.array(R,dtype=np.float32)),('lr',cache.get('cr',np.array(R,dtype=np.float32))),\
							('crx',float(R[0])),('cry',float(R[1])),('crz',float(R[2])),\
							('m',m),('wm',wm),('wim',matrix_inverse(wm))))

g_nodeTypes['scaleConstraint']['_OUTS'] = {'cs','csx','csy','csz','m','wm','wim'}
def evaluate_scaleConstraint(node):
	'''scale constraint reads tg (ts,tpm),o,rs,cpim and writes cs.'''
	cache = node['cache']
	keymap = {'s':'ts','pm':'tpm'}
	m,wm = constraint(cache, keymap)
	R,S,T = matrix_decompose(m)
	if 'o' in cache: S *= cache['o']
	set_node_cache(node, (	('cs',np.array(S,dtype=np.float32)),\
							('csx',float(S[0])),('csy',float(S[1])),('csz',float(S[2])),\
							('m',m),('wm',wm),('wim',matrix_inverse(wm))))

g_nodeTypes['poleVectorConstraint']['_OUTS'] = {'ct','ctx','cty','ctz','m','wm','wim'}
def evaluate_poleVectorConstraint(node):
	'''poleVector constraint reads tg,ps (tt,trp,trt,tpm),o,cop,crp.crt,cpim and writes ct.'''
	#TODO this should be different from pointConstraint
	cache = node['cache']
	keymap = {'t':'tt','rp':'trp','rpt':'trt','pm':'tpm'}
	m,wm = constraint(cache, keymap)
	T = m[:,3]
	if 'o' in cache: T += cache['o'] * cache.get('cop',1)
	set_node_cache(node, (	('ct',np.array(T,dtype=np.float32)),\
							('ctx',float(T[0])),('cty',float(T[1])),('ctz',float(T[2])),\
							('m',m),('wm',wm),('wim',matrix_inverse(wm))))

g_nodeTypes.setdefault('aimConstraint',{})['_OUTS'] = {'cr','crx','cry','crz','m','wm','wim'}
def evaluate_aimConstraint(node):
	'''aim constraint reads tg (tr,tro,tjo,tpm),o,cro,cjo,cpim and writes cr.'''
	#TODO this should be different from orientConstraint
	cache = node['cache']
	keymap = {'t':'tt','rp':'trp','rpt':'trt','pm':'tpm'}
	m,wm = constraint(cache, keymap)
	# TODO a - aimVector, u - upVector, wu - worldUpVector, wum - worldUpMatrix, wut - worldUpType, ct - constraintTranslate, input, o - offset
	# TODO cv - constraintVector, output!
	R = decomposeR(m[:,:3], cache.get('cro',0))
	set_node_cache(node, (	('cr',np.array(R,dtype=np.float32)),\
							('crx',float(R[0])),('cry',float(R[1])),('crz',float(R[2])),\
							('m',m),('wm',wm),('wim',matrix_inverse(wm))))

g_nodeTypes.setdefault('normalConstraint',{})['_OUTS'] = {'cr','crx','cry','crz','m','wm','wim'}
def evaluate_normalConstraint(node):
	'''normal constraint reads tg (tr,tro,tjo,tpm),o,cro,cjo,cpim and writes cr.'''
	#TODO this should be different from orientConstraint and aimConstraint
	cache = node['cache']
	keymap = {'r':'tr','ro':'tro','jo':'tjo','pm':'tpm'}
	m,wm = constraint(cache, keymap)
	R = decomposeR(m[:,:3], cache.get('cro',0))
	set_node_cache(node, (	('cr',np.array(R,dtype=np.float32)),\
							('crx',float(R[0])),('cry',float(R[1])),('crz',float(R[2])),\
							('m',m),('wm',wm),('wim',matrix_inverse(wm))))

g_nodeTypes['remapValue']['_OUTS'] = {'ov'}
def evaluate_remapValue(node):
	cache = node['cache']
	val = cache.get('i',0)
	imn,omn = cache.get('imn',0),cache.get('omn',0)
	imx,omx = cache.get('imx',1),cache.get('omx',1)
	ov = float(np.clip((val-imn)/float(imx-imn),0,1)*float(omx-omn)+omn)
	set_node_cache(node,(('ov',ov),))

g_nodeTypes.setdefault('addDoubleLinear',{})['_OUTS'] = {'o'}
def evaluate_addDoubleLinear(node):
	cache = node['cache']
	i1 = cache.get('i1',0.0)
	i2 = cache.get('i2',0.0)
	o = float(i1+i2)
	set_node_cache(node,(('o',o),))

g_nodeTypes.setdefault('clamp',{})['_OUTS'] = {'op','opr','opg','opb'}
def evaluate_clamp(node):
	cache = node['cache']
	ipr = cache.get('ipr',0.0)
	ipg = cache.get('ipg',0.0)
	ipb = cache.get('ipb',0.0)
	mnr = cache.get('mnr',0.0)
	mng = cache.get('mng',0.0)
	mnb = cache.get('mnb',0.0)
	mxr = cache.get('mxr',0.0)
	mxg = cache.get('mxg',0.0)
	mxb = cache.get('mxb',0.0)
	opr = np.clip(ipr,mnr,mxr)
	opg = np.clip(ipg,mng,mxg)
	opb = np.clip(ipb,mnb,mxb)
	set_node_cache(node,(('op',np.array((opr,opg,opb),dtype=np.float32)),('opr',float(opr)),('opg',float(opg)),('opb',float(opb))))

g_nodeTypes['plusMinusAverage']['_OUTS'] = {'o1','o2','o3'}
def evaluate_plusMinusAverage(node):
	cache = node['cache']
	#print 'evaluate_plusMinusAverage',cache.keys()
	op = [lambda x:0,lambda x:np.sum(x,axis=0),lambda x:x[0]-np.sum(x[1:],axis=0),lambda x:np.mean][cache.get('op',0)]
	if 'i1' in cache:
		i1 = cache.get('i1')
		o = op(i1)
		set_node_cache(node,(('o1',o),))
	if 'i2' in cache:
		i2 = cache.get('i2')
		o = op(i2)
		set_node_cache(node,(('o2',o),))
	if 'i3' in cache:
		i3 = cache.get('i3')
		if isinstance(i3,dict): print 'evaluate_plusMinusAverage','fixme'; return # HACK
		o = op(i3)
		set_node_cache(node,(('o3',o),))

g_nodeTypes['mesh']['_OUTS'] = {'o','w'}
def evaluate_mesh(node):
	cache = node['cache']
	#print '###---evaluate_mesh',node['name'],sorted(cache.keys())
	vt = cache.get('vt',{})
	pt = cache.get('pt',{})
	if isinstance(vt,dict):
		assert vt.keys() == range(len(vt))
		vt = np.array(vt.values(),dtype=np.float32)
	if pt:
		if max(pt.keys()) > len(vt):
			print 'evaluate_mesh out of range: argh!' # TODO
		else:
			for k,v in pt.iteritems(): vt[k] += v
	ig = cache.get('i',(vt,node,None))
	ig_str = cache.get('#i','meshes[\'%s\']'%node['name'])
	wm = cache.get('wm')
	if ig is '#': ig = (np.zeros((0,3),dtype=np.float32),node,None) # ??? HACK TODO
	if ig[2] is not None: wm = matrix_mult(wm,ig[2])
	wm_str = cache.get('#wm','get_wm(\'%s\')'%node['name'])
	assert ig != '#',ig_str
	#w = np.dot(o,wm[:3,:3].T) + wm[:3,3]  # world geometry is the vertices and a reference to the node for the topology
	set_node_cache(node,(('o',ig),('#o',ig_str),('w',(ig[0],ig[1],wm)),('#w','set_mesh_wm(%s,%s)'%(ig_str,wm_str))))
	fc = cache.get('fc', {}) # faces
	ed = cache.get('ed', {}) # edges
	uv = cache.get('uvst[0].uvsp', None) # vts
	fs,fts = [],[]
	for q in fc.itervalues():
		val = q.split()
		nf = int(val[0])
		face = map(int,val[1:nf+1])
		face = [ed[f][0] if f >= 0 else ed[-1-f][1] for f in face]
		if len(val) > nf+1 and val[nf+1] == 'mu': # a texture face...
			assert(val[nf+2] in '01'),val # TODO texture plane I guess (other values are possible)
			nf_uv = int(val[nf+3])
			face_uv = map(int,val[nf+4:nf+4+nf_uv])
			#face_uv = [ed[f][0] if f >= 0 else ed[-1-f][1] for f in face_uv]
		else: face_uv = [x for x in face]
		if uv is not None and any(f >= len(uv) for f in face_uv): print 'WARNING',face_uv,node['_name'];face_uv=[0]*len(face_uv)
		fs.append(face)
		fts.append(face_uv)
	node['_fs'] = fs
	node['_fts'] = fts
	node['_vts'] = uv

g_nodeTypes['skinCluster']['_OUTS'] = {'og[0]'}
def evaluate_skinCluster(node):
	#print '---evaluate_skinCluster',node['name']
	cache = node['cache']
	ins = node['ins']
	
	ig = cache['ip[0].ig']
	ig_str = cache.get('#ip[0].ig')
	gm = cache.get('gm',np.eye(3,4,dtype=np.float32))
	if ig is '#':
		print 'WARNING: missing geometry in evaluate_skinCluster'
		ig = (np.zeros((0,3),dtype=np.float32),node,None) # ??? HACK TODO
	igm = ig[2] if ig[2] is not None else np.eye(3,4,dtype=np.float32)
	assert not 'ip[1].ig' in cache # what would this mean? why is ip a vector?
	# attrs include: 'dpf', 'gm', 'mi', 'mmi', 'pm', 'ucm', 'wl[%d].w'[0:10448]

	#'skm':{'longname':'skinningMethod','type':'enum','enum':('Classical','Dual quaternion','Blended'),'default':0,'tip':'An Enum to set the algorithm method used.\nClassical is the linear vector blended skinning.\nDual quaternion uses a joint space non-linear blend.\nBlended use a weight set to linear interpolate between them.'},\
	#'bw':{'longname':'blendWeights','type':'double','default':0.0,'vec':True,'tip':'The blend weights per vertex for blending between dual quaternion and classic skinning.'},\
	#'pm':{'longname':'bindPreMatrix','type':'matrix','vec':True,'tip':'The inclusive matrix inverse of the driving transform at the time of bind'},\
	#'ma':{'longname':'matrix','type':'matrix','vec':True,'tip':'Driving transforms array'},\
	#'gm':{'longname':'geomMatrix','type':'matrix','tip':'The inclusive matrix of the geometry path at the time of the bound.'},\
	#'wl':{'longname':'weightList','type':'compound','vec':True,'tip':'Bundle of weights for each CV','compound':{
		#'w':{'longname':'weights','type':'double','default':'0.0','vec':'True','tip':'weights for each target'}}},\
	if not isinstance(cache['pm'],dict):
		cache['pm'] = dict(enumerate(cache['pm']))
	num_nodes = 0 if not cache['pm'] else np.max(cache['pm'].keys())+1
	pm = np.zeros((num_nodes,3,4),dtype=np.float32)
	pm[cache['pm'].keys()] = cache['pm'].values()

	# extract the useful data
	smat = {}
	for attr,skin_weights in cache.iteritems():
		if attr.startswith('wl['):
			vi = int(attr[3:attr.find(']')])
			for ji,jw in skin_weights.iteritems():
				ji,jw = int(ji),float(jw)
				if ji >= len(pm): continue
				tmat = pm[ji]
				tmat = matrix_mult(tmat, gm)
				try:
					smat.setdefault(ji,{})[vi] = np.hstack((jw*(np.dot(tmat[:3,:3], ig[0][vi]) + tmat[:3,3]),jw))
				except Exception as E:
					print 'WARNING evaluate_skinCluster',E,ig_str
	# convert to efficient format
	for ji,v in smat.iteritems(): smat[ji] = np.array(v.keys(),dtype=np.int32), np.array(v.values(),dtype=np.float32)
	node['_smats'] = smat
	
	# evaluate the node
	geo = np.zeros_like(ig[0])
	for ji,(vis,vs) in smat.iteritems():
		mat = cache.get('ma[%d]'%ji,np.eye(3,4,dtype=np.float32))
		geo[vis] += np.dot(vs, mat.T)
	og_str = 'skinCluster(\'%s\',%s)' % (node['name'],ig_str)
	set_node_cache(node,(('og[0]',(geo,ig[1],matrix_inverse(igm))),('#og[0]',og_str),))

g_nodeTypes['blendShape']['_OUTS'] = {'og[0]'}
def evaluate_blendShape(node):
	#print '---evaluate_blendShape',node['name']
	cache = node['cache']
	if 'ip[0].ig' in cache:
		ig = cache.get('ip[0].ig')
		ig_str = cache.get('#ip[0].ig')
		# 'aal', 'ip[0].gi', 'ip[0].ig', 'msg',
		# 'w[%d]'
		# 'it[0].itg[%d].iti[6000].ict'
		# 'it[0].itg[%d].iti[6000].ipt'
		ws = sorted([int(x[2:-1]) for x in cache if x.startswith('w[')])
		wts,cts,pts,wt_names = [],[],[],[]
		rest_vs = ig[0].copy()
		for wi in ws:
			wt = cache.get('w[%d]'%wi)
			ict = cache.get('it[0].itg[%d].iti[6000].ict'%wi,None) # ['vtx', [indices]] could be number other than 6000...w=(n-5000)*0.001 TODO
			if ict is None: continue
			if ict[0] != 'vtx': continue # 'smp' in 'Alexia'
			#assert ict[0] == 'vtx', ict[0] # smp???
			ipt = cache.get('it[0].itg[%d].iti[6000].ipt'%wi,None) # Nx4 vertices
			if ipt is None: continue
			if wt is '#': print 'evaluate_blendShape',node['name'],'blendShape weight',wi,'is missing'; wt = 0.0 # WARNING
			wt = float(wt)
			ict = np.array(ict[1],dtype=np.int32)
			ipt = np.array(ipt,dtype=np.float32)[:,:3]
			wts.append(wt)
			cts.append(ict)
			pts.append(ipt)
			wt_names.append(cache.get('#w[%d]'%wi,'None'))
			try:
				ig[0][ict] += wt * ipt
			except Exception as E:
				print 'WARN evaluate_blendShape',ig_str
		wt_str = '[%s]'%(','.join(wt_names))
		ig_str = 'blendShape(\'%s\',%s,%s)'%(node['name'],ig_str,wt_str)
		set_node_cache(node,(('og[0]',ig),('#og[0]',ig_str),))
		node['_bmat'] = {'vs':rest_vs,'fs':ig[1].get('_fs',None),'fts':ig[1].get('_fts',None),'vts':ig[1].get('_vts',None),'wts':wts,'cts':cts,'pts':pts,'wt_names':wt_names} # for wt,ct,pt in zip(wts,cts,pts): vs[ct] += wt * pt

g_nodeTypes['jiggle']['_OUTS'] = {'og[0]'}
def evaluate_jiggle(node):
	# TODO http://download.autodesk.com/us/maya/2009help/Nodes/jiggle.html
	cache = node['cache']
	if 'ip[0].ig' in cache:
		ig = cache.get('ip[0].ig') # (verts,node,matrix,...)
		ig_str = cache.get('#ip[0].ig')
		set_node_cache(node,(('og[0]',ig),('#og[0]','jiggle(\'%s\',%s)'%(node['name'],ig_str)),))

g_nodeTypes['tweak']['_OUTS'] = {'og[0]'}
def evaluate_tweak(node):
	#print '---evaluate_tweak',node['name']
	cache = node['cache']
	if 'ip[0].ig' in cache:
		ig = cache.get('ip[0].ig') # (verts,node,matrix,...)
		ig_str = cache.get('#ip[0].ig')
		rtw = cache.get('rtw',True) # relative or absolute tweak
		pl = cache.get('pl[0].cp',None)
		vt = cache.get('vl[0].vt',None)
		if vt:
			tweak_ig = ig[0].copy()
			if len(tweak_ig): # TODO getting out of bounds errors here due to missing geometry
				if rtw: tweak_ig[vt.keys()] += vt.values() # TODO should take into account transform here?
				else:   tweak_ig[vt.keys()] = vt.values()
			ig = (tweak_ig,ig[1],ig[2])
			ig_str = 'tweak(\'%s\',%s)'%(node['name'],ig_str)
		if pl:
			#print ig_str
			tweak_ig = ig[0].copy()
			if len(tweak_ig): # TODO getting out of bounds errors here
				if rtw: tweak_ig[pl.keys()] += pl.values()
				else:   tweak_ig[pl.keys()] = pl.values()
			ig = (tweak_ig,ig[1],ig[2])
			ig_str = 'tweak(\'%s\',%s)'%(node['name'],ig_str)
		set_node_cache(node,(('og[0]',ig),('#og[0]',ig_str),))

g_nodeTypes.setdefault('ffd',{})['_OUTS'] = {'og[0]'}
def evaluate_ffd(node):
	#print '---evaluate_ffd',node['name']
	cache = node['cache']
	if 'ip[0].ig' in cache:
		ig = cache.get('ip[0].ig')
		## now actually do the ffd # TODO
		set_node_cache(node,(('og[0]',ig),))

g_nodeTypes.setdefault('wire',{})['_OUTS'] = {'og[0]'}
def evaluate_wire(node):
	#print '---evaluate_wire',node['name']
	cache = node['cache']
	if 'ip[0].ig' in cache:
		ig = cache.get('ip[0].ig')
		## now actually do the wire effect # TODO
		set_node_cache(node,(('og[0]',ig),))

g_nodeTypes.setdefault('curveInfo',{})['_OUTS'] = {'al','cp','xv','yv','zv','wt','kn'}
def evaluate_curveInfo(node):
	cache = node['cache']
	ic = cache.get('ic')
	# TODO measure an arc length
	set_node_cache(node,(('al',0.0),))

g_nodeTypes.setdefault('groupParts',{})['_OUTS'] = {'og'}
def evaluate_groupParts(node):
	#print '---evaluate_groupParts',node['name']
	cache = node['cache']
	ig = cache.get('ig')
	ig_str = cache.get('#ig',node['ins']['ig'][0]['type'])
	if ig is '#': ig = (np.zeros((0,3),dtype=np.float32),node,None) # ??? HACK TODO
	assert ig != '#',ig_str
	ic = cache.get('ic')
	#assert ic[1] == ['*'],'Aha! This groupParts actually does something:'+node['name']+str(ic)
	#ig = (ig[1]['cache'][ic[0]],ig[1],ig[2])
	if ic is not None and ic[1] != ['*']: # this groupParts does something? TODO
		ig_str = 'groupParts(%s,gi=%s)'%(ig_str,cache['gi']) #+ str(cache.get('ic'))#,"'%s'"%ig[1]['name']))
	set_node_cache(node,(('og',ig),('#og',ig_str),))

def animCurve(node, ct):
	cache = node['cache']
	val = cache.get('i',0.0)
	val_str = 'animCurve%s(\'%s\',%s)'%(ct,node['name'],cache.get('#i'))
	ktv = cache.get('ktv',None)
	tan = cache.get('tan',4)
	if tan == 2 and ktv == {0: (0.0, 0.0), 1: (1.0, 1.0), 2: (2.0, 0.0)}: val_str = 'clip01(%s)'%cache.get('#i') # pass through
	elif tan == 2 and ktv == {0: (-2.0, 0.0), 1: (-1.0, 1.0), 2: (0.0, 0.0)}: val_str = 'clip01(minus(%s))'%cache.get('#i') # negate
	#else: print 'evaluate_animCurve'+ct,tan,val,ktv
	if ktv is not None and len(ktv) and len(ktv.values()[0]): val = np.interp(val,*zip(*ktv.values())) # only linear interpolation supported
	set_node_cache(node,(('o',val),('#o',val_str)))

g_nodeTypes['animCurveUU']['_OUTS'] = {'o'}
g_nodeTypes['animCurveUL']['_OUTS'] = {'o'}
g_nodeTypes['animCurveUA']['_OUTS'] = {'o'}
def evaluate_animCurveUU(node): animCurve(node, 'UU')
def evaluate_animCurveUL(node): animCurve(node, 'UL')
def evaluate_animCurveUA(node): animCurve(node, 'UA')

g_nodeTypes.setdefault('condition',{})['_OUTS'] = {'oc','ocr','ocg','ocb'}
def evaluate_condition(node):
	cache = node['cache']
	op = [lambda x,y:x==y,lambda x,y:x!=y,lambda x,y:x>y,lambda x,y:x>=y,lambda x,y:x<y,lambda x,y:x<=y][cache.get('op',0)]
	ft = cache.get('ft',0.0)
	st = cache.get('st',0.0)
	ct = cache.get('ct',(0,0,0))
	cf = cache.get('cf',(0,0,0))
	oc = ct if op else cf
	set_node_cache(node,(('oc',oc),('ocr',oc[0]),('ocg',oc[1]),('ocb',oc[2])))

g_nodeTypes['nurbsCurve']['_OUTS'] = {'o','ws'}
def evaluate_nurbsCurve(node):
	cache = node['cache']
	cc = cache.get('cc', [0,0,0,'no',0,0,0])
	degree,spans,form = map(int,cc[:3]) # degree, spans, form (Open,Closed,Periodic)
	rational = (cc[3] == 'yes')
	dim,nk = map(int,cc[4:6])
	knots = np.array(cc[6:6+nk],dtype=np.float32)
	ncvs = int(cc[6+nk])
	cvs = np.array(cc[7+nk:],dtype=np.float32).reshape(-1,3+rational)
	vs = cvs[:,:3] # we don't support rational curves yet!
	bs = []
	for i in range(cvs.shape[0]-1): bs.append((i,i+1))
	if form == 2: bs.append((cvs.shape[0]-1,0))
	node['_bs'] = bs
	ig_str = 'nurbsCurve[\'%s\']'%(node['name'])
	wm = cache.get('wm')
	wm_str = cache.get('#wm','get_wm(\'%s\')'%node['name'])
	set_node_cache(node,(('o',(vs,node,None)),('#o',ig_str),('ws',(vs,node,wm)),('#w','set_mesh_wm(%s,%s)'%(ig_str,wm_str)),))

g_nodeTypes['nurbsSurface']['_OUTS'] = {'o','ws'}
def evaluate_nurbsSurface(node):
	cache = node['cache']
	cc = cache.get('cc',[0,0,0,0,'no',0,0,0]) 
	degree_u, degree_v, form_u, form_v = map(int, cc[:4]) #degree_u, degree_v, form_u, form_v
	rational = (cc[4] == 'yes') # rational
	nku = int(cc[5]) # knots_u, knots_u*knot_value_float
	knots_u = np.array(cc[6:6+nku],dtype=np.float32)
	nkv = int(cc[6+nku]) # knots_v, knots_v*knot_value_float
	knots_v = np.array(cc[7+nku:7+nku+nkv],dtype=np.float32)
	ncvs = int(cc[7+nku+nkv]) # num_cvs, num_cvs*float
	cvs = np.array(cc[8+nku+nkv:],dtype=np.float32).reshape(-1,3+rational)
	vs = cvs[:,:3] # we don't support rational curves yet!
	fs = []
	stride_u, stride_v = nku-(degree_u-1), nkv-(degree_v-1)
	# form = [0:Open,1:Closed,2:Periodic] TODO think about Open/Closed
	urange = [nku-1, nku-1, nku][form_u]
	vrange = [nkv-1, nkv-1, nkv][form_v]
	for ui in xrange(urange):
		u0,u1 = ((ui)%stride_u)*stride_v,((ui+1)%stride_u)*stride_v
		for vi in xrange(vrange):
			v0,v1 = ((vi)%stride_v),((vi+1)%stride_v)
			fs.append([u0+v0,u1+v0,u1+v1,u0+v1]) # TODO, check the direction of normals
	#print '#####____evaluate_nurbsSurface',ncvs,cvs.shape, nku, nkv, degree_u, degree_v, form_u, form_v, np.max(fs+[0,0,0,0])
	node['_fs'] = fs
	ig_str = 'nurbsSurface[\'%s\']'%(node['name'])
	wm = cache.get('wm')
	wm_str = cache.get('#wm','get_wm(\'%s\')'%node['name'])
	set_node_cache(node,(('o',(vs,node,None)),('#o',ig_str),('ws',(vs,node,wm)),('#w','set_mesh_wm(%s,%s)'%(ig_str,wm_str)),))

g_nodeTypes['subdiv']['_OUTS'] = {'o','ws'}
def evaluate_subdiv(node):
	cache = node['cache']
	cc = cache.get('cc',[0,0]) # num_vertices, num_vertices*(vid,vx,vy,vz), num_faces, num_faces * (face_size, face_size * (vi,)),
	nvs = int(cc[0])
	vs = np.zeros((nvs,3),dtype=np.float32)
	fs = []
	for vi in xrange(nvs):
		vs[int(cc[vi*4+1])] = cc[vi*4+2:vi*4+5]
	nfs = int(cc[nvs*4+1])
	fi = nvs*4+2
	for i in xrange(nfs):
		nf = int(cc[fi])
		fs.append(map(int,cc[fi+1:fi+nf+1]))
		fi += nf+1
	node['_fs'] = fs
	ig_str = 'subdiv[\'%s\']'%(node['name'])
	wm = cache.get('wm')
	wm_str = cache.get('#wm','get_wm(\'%s\')'%node['name'])
	set_node_cache(node,(('o',(vs,node,None)),('#o',ig_str),('ws',(vs,node,wm)),('#w','set_mesh_wm(%s,%s)'%(ig_str,wm_str)),))

g_nodeTypes.setdefault('subdAddTopology',{})['_OUTS'] = {'os'}
def evaluate_subdAddTopology(node):
	#TODO
	cache = node['cache']
	_is = cache.get('is',(None,None,None))
	is_str = cache.get('#is','None')
	set_node_cache(node,(('os',_is),('#os','subdAddTopology(\'%s\',%s)'%(node['name'],is_str))))

g_nodeTypes.setdefault('polySmoothFace',{})['_OUTS'] = {'out'}
def evaluate_polySmoothFace(node):
	cache = node['cache']
	out = cache.get('ip',(None,None,None))
	out_str = cache.get('#ip','None')
	# TODO some smoothing
	#  'mth': 'Type of smoothing algorithm to use 0 - exponential - traditional smoothing 1 - linear - number of faces per edge grows linearly' etc
	set_node_cache(node,(('out',out),('#out','polySmoothFace(\'%s\',%s)'%(node['name'],out_str)),))

g_nodeTypes.setdefault('polySubdFace',{})['_OUTS'] = {'out'}
def evaluate_polySubdFace(node):
	cache = node['cache']
	out = cache.get('ip',(None,None,None))
	out_str = cache.get('#ip','None')
	# TODO something
	set_node_cache(node,(('out',out),('#out','polySubdFace(\'%s\',%s)'%(node['name'],out_str)),))

g_nodeTypes.setdefault('polyTweak',{})['_OUTS'] = {'out'}
def evaluate_polyTweak(node):
	cache = node['cache']
	out = cache.get('ip',(None,None,None))
	out_str = cache.get('#ip','None')
	# TODO 
	set_node_cache(node,(('out',out),('#out','polyTweak(\'%s\',%s)'%(node['name'],out_str)),))

g_nodeTypes.setdefault('polyNormal',{})['_OUTS'] = {'out'}
def evaluate_polyNormal(node):
	cache = node['cache']
	out = cache.get('ip',(None,None,None))
	out_str = cache.get('#ip','None')
	# TODO 
	#  'normalPerVertex (npvx)' Attribute to specify the normal of a vertex, either on the entire vertex on a per face basis. 
	set_node_cache(node,(('out',out),('#out','polyNormalPerVertex(\'%s\',%s)'%(node['name'],out_str)),))

g_nodeTypes.setdefault('polyNormalPerVertex',{})['_OUTS'] = {'out'}
def evaluate_polyNormalPerVertex(node):
	cache = node['cache']
	out = cache.get('ip',(None,None,None))
	out_str = cache.get('#ip','None')
	# TODO 
	#  'normalPerVertex (npvx)' Attribute to specify the normal of a vertex, either on the entire vertex on a per face basis. 
	set_node_cache(node,(('out',out),('#out','polyNormalPerVertex(\'%s\',%s)'%(node['name'],out_str)),))

g_nodeTypes.setdefault('polyTweakUV',{})['_OUTS'] = {'out'}
def evaluate_polyTweakUV(node):
	cache = node['cache']
	out = cache.get('ip',(None,None,None))
	out_str = cache.get('#ip','None')
	# TODO 
	set_node_cache(node,(('out',out),('#out','polyTweakUV(\'%s\',%s)'%(node['name'],out_str)),))

g_nodeTypes.setdefault('polySoftEdge',{})['_OUTS'] = {'out'}
def evaluate_polySoftEdge(node):
	cache = node['cache']
	out = cache.get('ip',(None,None,None))
	out_str = cache.get('#ip','None')
	# TODO 
	set_node_cache(node,(('out',out),('#out','polySoftEdge(\'%s\',%s)'%(node['name'],out_str)),))

g_nodeTypes.setdefault('polyMapSewMove',{})['_OUTS'] = {'out'}
def evaluate_polyMapSewMove(node):
	cache = node['cache']
	out = cache.get('ip',(None,None,None))
	out_str = cache.get('#ip','None')
	# TODO 
	set_node_cache(node,(('out',out),('#out','polyMapSewMove(\'%s\',%s)'%(node['name'],out_str)),))

g_nodeTypes['dagPose']['_OUTS'] = {'wm','xm','p'}
def evaluate_dagPose(node):
	cache = node['cache']
	ins = node['ins']
	m = sorted([(int(x[2:-1]),x) for x in cache if x.startswith('m[')])
	#print '###evaluate_dagPose',node['name'],m
	for mi,mess in m:
		conn = ins[mess][0]
		#print 'evaluate_dagPose',mess,ins[mess][0]['name'],ins[mess][1]
		set_node_cache(node,(('wm[%d]'%mi,conn['cache']['wm']),('xm[%d]'%mi,conn['cache']['m']),))
		set_node_cache(node,(('p[%d]'%mi,conn['parent']),))
	set_node_cache(node,(('#msg','dagPose(\'%s\',\'%s\')'%(node['name'],str([ins[mess][0]['name'] for mi,mess in m]))),))
	#print node['cache'].keys()

g_nodeTypes['cluster']['_OUTS'] = {'og[0]'}
def evaluate_cluster(node): pass # suppress warning
def evaluate_clusterHandle(node): pass # suppress warning
def evaluate_shadingEngine(node): pass # suppress warning
def evaluate_place2dTexture(node): pass # suppress warning
def evaluate_objectSet(node): pass # suppress warning
def evaluate_groupId(node): pass # suppress warning

g_nodeTypes['joint']['_OUTS'] = {'m','wm','wim'}
def evaluate_joint(node): pass # suppress warning
g_nodeTypes['camera']['_OUTS'] = {'m','wm','wim'}
def evaluate_camera(node): pass # suppress warning
g_nodeTypes['locator']['_OUTS'] = {'m','wm','wim'}
def evaluate_locator(node): pass # suppress warning
g_nodeTypes['ikHandle']['_OUTS'] = {'m','wm','wim'}
def evaluate_ikHandle(node): pass # suppress warning
def evaluate_displayLayerManager(node): pass # suppress warning

def evaluate_node_in(node, attr, debug, msg):
	cache = node['cache']
	v = node['ins'][attr]
	in_node, in_attr = v
	if debug: print '->attrIn->',attr,'from',in_node['name'],in_attr
	cv,cvd = None,None
	if not in_node['cache'].has_key(in_attr):
		nt2 = in_node['type']
		if in_attr == 'msg' or in_attr in g_nodeTypes[nt2].get('_OUTS',{}): # if it's an out of the node, power the node
			if debug: print 'case 1'
			evaluate_node(in_node, debug, msg)
		elif in_attr in in_node.get('ins'): # if it's a driven attribute, pull the in
			if debug: print 'case 2'
			evaluate_node_in(in_node, in_attr, debug, msg)
		else:
			if debug: print 'AHEM, not evaluating node',in_node['name'],'[',nt2,'] with attr',in_attr
		if debug: print '  back in',nt,node['name']
	if not in_node['cache'].has_key(in_attr):
		if debug: print 'looking up',in_node['type'],in_attr
		nt_attrs = g_nodeTypes.get(in_node['type'],{})
		nt_attr = nt_attrs.get(in_attr,{})
		alias = nt_attr.get('alias',None)
		if alias is not None:
			if cache.has_key(nt_attr['alias'][0]):
				cv = cache[alias[0]][alias[1]]
				cvd = cache.get('#%s[%d]'%(alias[0],alias[1]),'get_%s(\'%s\')'%(in_attr,in_node['name']))
			else:
				cv = nt_attrs[alias[0]]['default'][alias[1]] # aliased attribute must have a default!?
				cvd = 'get_%s(\'%s\')'%(in_attr,in_node['name'])
		else:
			cv = deepcopy(nt_attr.get('default','#'))
			cvd = 'default'
		if cv is '#':
			assert alias is None
			if in_node != node: # can't satisfy this, skip the warning TODO
				print 'even after evaluate_node, [',node['type'],']',node['name'],'.',attr,'is empty, with input from',in_node['name'],'.',in_attr,'[',in_node['type'],']'
				defaults = g_nodeTypes.get(node['type'],{})
				default_key = defaults.get(attr,{})
				alias = default_key.get('alias',None)
				if alias is not None:
					cv = defaults[alias[0]]['default'][alias[1]] # aliased attribute must have a default!?
					cvd = 'alias[%d]'%(alias[1])
				else:
					cv = deepcopy(default_key.get('default','#')) # HACK try for a default value!?
					cvd = 'missing?'
	else:
		cv = in_node['cache'][in_attr]
		cvd = in_node['cache'].get('#'+in_attr,'get_%s(\'%s\')'%(in_attr,in_node['name']))
	set_node_cache(node, ((attr,cv),('#'+attr,cvd)))

def evaluate_node(node, debug=False, msg=0):
	nt = node['type']
	if debug: print 'in evaluate_node with',nt,node['name']
	#if nt == 'joint':
	#	print 'in evaluate_node with',nt,node['name']
	#	print node.keys()
	cache = node.setdefault('cache',{})
	if cache.get('msg',None) == msg: return # already evaluated
	cache.update((('msg',msg),))
	#parent = node.get('parent',None)
	#if parent is not None: evaluate_node(parent, debug, msg)
	for attr,v in node.get('ins',{}).iteritems():
		if not cache.has_key(attr):
			evaluate_node_in(node, attr, debug, msg)
	if g_nodeTypes[nt].has_key('_DAG') and nt != 'transform':
		if debug: print 'about to eval transform',node['name'],node['type']
		evaluate_transform(node)
	f = 'evaluate_'+nt
	try:
		func = getattr(sys.modules[__name__],f)
	except:
		print '### evaluate_node missing %s' % f
		return
	func(node)

#@profile
def maya_to_state(nodes, use_cache = True):
	'''We want to convert the Maya scenegraph into an equivalent state structure, to use a rig outside of Maya.'''
	# The Maya static info (g_nodeTypes) is converted into a fields structure
	fields = {}
	for k,nv in g_nodeTypes.iteritems():
		tns = sorted(nv.keys()) # TODO sort order
		fields[k] = [(tn, nv[tn]) for tn in tns if tn[0] != '_' and nv[tn]['type']]
	State.setKey('fields',fields)
		
	# the Maya nodes and attributes can be converted into State keys
	display_nodes = []
	def tree_add(display_nodes,po,ps=''):
		adds = [(ps+(' ','+')[v['children']!={}]+v['type']+'_'+v['name'],v) for v in po['children'].itervalues()]
		for k,v in sorted(adds):
			display_nodes.append((k,v))
			tree_add(display_nodes,v,'  '+ps)
	tree_add(display_nodes,nodes)
	#State.setKey('display_nodes',zip(*display_nodes)[0]) # TODO VERY slow, needs fixing

	for o,v in display_nodes:
		a = v['cache' if (use_cache and 'cache' in v) else 'attrs'] # TODO think about this
		# split the attributes into regular and vector-valued
		aa = dict([kv for kv in a.iteritems() if not isinstance(kv[1],dict)])
		va = dict([kv for kv in a.iteritems() if isinstance(kv[1],dict)])
		for k1,v1 in va.iteritems():
			if v1.has_key('name'):
				va[k1] = v1['name']
				if v['type'] != 'dagPose': print 'maya_to_state WARNING attr points to node',v['name'],k1,'[',v['type'],'] =',v1['name']
		for k1,v1 in aa.iteritems():
			if type(v1) is np.float32: aa[k1] = float(v1) #; print 'maya_to_state2 WARNING attr points to float32',v['name'],k1,'[',v['type'],'] =',repr(v1)
			if type(v1) is np.float64: aa[k1] = float(v1) ; print 'maya_to_state2 WARNING attr points to float64',v['name'],k1,'[',v['type'],'] =',repr(v1)
			if type(v1) is tuple and len(v1) == 3 and type(v1[1]) is dict: aa[k1] = (v1[0],v1[1]['name'],v1[2]) #; print v['name'],v['type'],k1
			#elif type(v1) not in [bool,int,float,list,str,np.ndarray]:  print 'typewarn',v['name'],v['type'],k1,type(v1)
		# TODO parent, children, ins, outs, _bs, _vts, _fts, _fs, , _bmat, _smats
		State.addKey(v.get('_name','/root/'+v['name']),{'type':v['type'],'attrs':aa,'vattrs':va})
	State.push('Import MA')
	return fields, display_nodes

def evaluate_scenegraph(nodeLists, msg=0):
	# now can evaluate the scenegraph and populate the cache for each node
	for nt,ntns in nodeLists['ALL'].iteritems(): # visit every node and initialise the cache from attrs
		for node in ntns:
			set_node_cache(node,node['attrs'].items())
	for node in nodeLists['DAG']: # next populate all transform matrices
		cache = node['cache']
		pc = node['parent']['cache'] # parent should be a transform, and have already been evaluated (assume strict ordering)
		node['_name'] = node['parent']['_name']+'/'+node['name']
		set_node_cache(node,(('tmp',cache.get('tmp',pc.get('tmp',False))),('sech',cache.get('sech',pc.get('sech',True))),('v',cache.get('v',pc.get('v',True))))) # 'tmp','sech','v' are special inherited attributes
	for node in nodeLists['DAG']: # now evaluate the rest of the scenegraph
		evaluate_node(node, msg=msg)

def construct_geos(nodeLists, units = 10.0):
	mat_units = np.array([1,1,1,units],dtype=np.float32)
	# evaluate every leaf node (mesh,subdiv,nurbsCurve,joint,transform types) and generate a geometry for each
	# TODO vns=None
	primitives,primitives2D,mats,camera_ids,movies = [],[],[],[],[]
	names,transforms,Vs,VTs,Fs,bones,FTs = [],[],[],[],[],[],[]
	for node in nodeLists['DAG']: # now generate geometries for leaf nodes
		cache = node['cache']
		if not cache.get('v',True): continue # deal with visibility
		if cache.get('io',False): continue # don't generate geo for intermediate objects
		node_name = node['_name']
		nt = node['type']
		wm = cache['wm']
		if nt == 'joint': # draw bones to all the child joints
			child_joints = [(cn,child) for cn,child in node['children'].iteritems() if child['type'] == 'joint']
			vs = [[0,0,0]]
			for cn,child in child_joints: vs.append(child['cache']['m'][:3,3])
			vs = np.array(vs,dtype=np.float32).reshape(-1,3)
			Vs.append(vs*units)
			VTs.append(None)
			bones.append([(0,i) for i in range(1,len(vs))])
			Fs.append([])
			FTs.append(None)
		elif nt in {'subdiv','nurbsSurface','mesh','nurbsCurve'}:
			(vs,node,wm) = cache.get('w' if nt == 'mesh' else 'ws',(None,None,None))
			if isinstance(vs,dict):
				assert (vs.keys() == range(len(vs))), 'ARGH'+repr(vs.keys())+' '+node_name+' '+str(len(vs))
				vs = vs.values()
			elif vs is None: vs = []
			vts = node.get('_vts',None)
			if isinstance(vts,dict):
				assert (vts.keys() == range(len(vts))), 'ARGH'+repr(vts.keys())+' '+node_name+' '+str(len(vts))
				vts = vts.values()
			if wm is None: wm = np.eye(3,4,dtype=np.float32)
			vs = np.array(vs,dtype=np.float32).reshape(-1,3)
			Vs.append(vs*units)
			Fs.append(node.get('_fs',[]))
			FTs.append(node.get('_fts',None))
			VTs.append(vts)
			bones.append(node.get('_bs',None))
		elif nt == 'camera':
			#print cache.keys()
			# TODO much more
			f,ox,oy = cache.get('fl',35.0)/17.5,0,0
			K = np.array([[f,0,ox],[0,f,oy],[0,0,1]],dtype=np.float32)
			RT = matrix_inverse(wm*mat_units)
			P = np.dot(K,RT[:3,:])
			k1,k2 = 0,0
			w,h = 1920,1080
			cid = cache.get('imn')
			mats.append(Calibrate.makeMat(P,(k1,k2),(w,h)))
			movies.append(None)
			camera_ids.append(cid)
			continue # no drawable
		else: # transform, parentConstraint, locator
			if nt not in g_transformTypes and nt not in g_shapeTypes: print 'construct_geos unexpected transform type',nt 
			Vs.append([])
			VTs.append(None)
			Fs.append([])
			FTs.append(None)
			bones.append(None)
		names.append(node_name)
		transforms.append(wm*mat_units)
	primitives.append({'names':names, 'verts':Vs, 'faces':Fs, 'bones':bones, 'transforms':transforms, 'vts':VTs, 'fts':FTs})
	return primitives,primitives2D,mats,camera_ids,movies

def extract_GRIP(nodeLists, units = 10.0):
	mat_units = np.array([1,1,1,units],dtype=np.float32)
	shape_weights = shape_weights_from_skinClusters(nodeLists.get('skinCluster',[]))
	skels = skel_dicts_from_joints(nodeLists.get('joint',[]), mat_units)
	for node in nodeLists['DAG']: # HACK now generate dummy shape_weights for leaf nodes that have joint parents (or grandparents)
		cache = node['cache']
		if not cache.get('v',True): continue # deal with visibility
		if cache.get('io',False): continue # don't generate geo for intermediate objects
		nt = node['type']
		if nt in {'subdiv','nurbsSurface','mesh','nurbsCurve'}:
			node_name = node['_name']
			wm = cache['wm']
			npt,npp = node['parent'],node['parent']['parent']
			if npt['type'] == 'joint':
				shape_weights[node_name] = (None,{npt['_name']:matrix_mult(matrix_inverse(npt['cache']['wm']),wm)*mat_units})
			elif npp['type'] == 'joint':
				shape_weights[node_name] = (None,{npp['_name']:matrix_mult(matrix_inverse(npp['cache']['wm']),wm)*mat_units})
	blendShapes = shapes_from_blendShapes(nodeLists.get('blendShape',[]))
	return {'shape_weights':shape_weights, 'skels':skels, 'blendShapes':blendShapes}

def skel_dicts_from_joints(all_joints, mat_units):
	joint_dict = {}
	for node in all_joints:
		cache = node['cache']
		node_name = node['_name']
		wm = cache['wm']
		ro = ('xyz', 'yzx', 'zxy','xzy', 'yxz', 'zyx')[cache.get('ro',0)]
		dofs = [(ord(roc)-ord('x'),':r'+roc) for roc in ro if cache.get('jt'+roc,True)]
		child_joints = [(cn,child) for cn,child in node['children'].iteritems() if child['type'] == 'joint']
		joint_dict[node_name] = {'name':node_name,
									'G':wm*mat_units,
									'parent':node['parent']['_name'],
									'children':[cn for cn,child in child_joints],
									'dofs':dofs}
	skels = [skel_dict_from_joint_dict(joint_dict)]
	return skels

def find_leaf_shape_name(node):
	shape_name = None
	if node.has_key('outs'):
		out_node = node['outs']['og[0]'][0][0]
		while True:
			out_type = out_node['type']
			#print 'find_leaf_shape_name',out_type
			if out_type == 'groupParts':out_node = out_node['outs']['og'][0][0];continue
			if out_type in g_geometryFilterTypes: out_node = out_node['outs']['og[0]'][0][0];continue
			if out_type in g_polyModifierTypes: out_node = out_node['outs']['out'][0][0];continue
			if out_type in g_subdModifierTypes: out_node = out_node['outs']['os'][0][0];continue
			break
		#print 'find_leaf_shape_name',out_node['type'],out_node['name']
		shape_name = out_node['_name'] # HACK this might not be the shape
	return shape_name

def shapes_from_blendShapes(bs_nodes):
	#return {bs['name']:bs['_bmat'] for bs in bs_nodes}
	ret = {}
	for node in bs_nodes:
		shape_name = find_leaf_shape_name(node)
		if shape_name is not None:
			ret[shape_name] = node.get('_bmat',None)  # TODO fix the scaling for external use
	return ret

def shape_weights_from_skinClusters(sk_nodes):
	shape_weights = {}
	for node in sk_nodes:
		shape_name = find_leaf_shape_name(node)
		if shape_name is not None:
			joint_list = {}
			for attr_name,attr_val in node['ins'].iteritems(): # HACK these might not be the actual controls
				if attr_name.startswith('ma['):
					ind = int(attr_name[3:-1])
					joint_list[attr_val[0]['_name']] = ind
			#print shape_name, joint_list
			shape_weights[shape_name] = node.get('_smats',None), joint_list  # TODO fix the scaling * [10,10,10,1] for externals use
	return shape_weights

def skel_dict_from_joint_dict(joint_dict):
	# TODO: Currently assumes translation only on root
	root_node = None
	for jn, joint in joint_dict.items():
		if joint['parent'] not in joint_dict:
			root_node = jn
			break

	def build_joint_hierarchy(jn, joint_dict):
		if not jn in joint_dict: return []
		out = [joint_dict[jn]]
		for child in joint_dict[jn]['children']:
			out += build_joint_hierarchy(jn+'/'+child, joint_dict)
		return out

	if root_node is not None: joints = build_joint_hierarchy(root_node,joint_dict)
	else: joints = []

	jointNames = [joint['name'] for joint in joints]
	jointIndex = dict(zip(jointNames,range(len(joints))))
	jointParents = [jointIndex[joint['parent']] if (joint['parent'] in jointNames) else -1 for joint in joints]

	jointChans = []
	jointChanSplits = [0]
	dofNames = []
	Gs,Ls,Bs = [],[],[[] for j in joints]
	for ji,joint in enumerate(joints):
		jn = joint['name']
		if ji == 0:
			jointChans += [0,1,2]
			jointChanSplits.append(3)
			dofNames += [jn+':tx',jn+':ty',jn+':tz']
		else:
			jointChanSplits.append(jointChanSplits[-1])
		for di,dn in joint['dofs']: # TODO, check these are in exactly the same order as the ASF export
			jointChans.append(di+3)
			dofNames.append(jn+dn)
		jointChanSplits.append(len(jointChans))
		Gs.append(joint['G'])
		pji = jointParents[ji]
		Ls.append(matrix_mult(matrix_inverse(joints[pji]['G']),joint['G']) if pji != -1 else joint['G'])
		if pji != -1: Bs[pji].append(Ls[-1][:,3])
	numDofs = len(jointChans)
	return { 'name'           : 'test_skel',
			 'numJoints'      : len(jointNames),
			 'jointNames'     : jointNames,  # list of strings
			 'jointIndex'     : jointIndex, # dict of string:int
			 'jointParents'   : np.array(jointParents,dtype=np.int32),
			 'jointChans'     : np.array(jointChans,dtype=np.int32), # 0 to 5 : tx,ty,tz,rx,ry,rz
			 'jointChanSplits': np.array(jointChanSplits,dtype=np.int32),
			 'chanNames'      : dofNames,   # list of strings
			 'chanValues'     : np.zeros(numDofs,dtype=np.float32),
			 'numChans'       : int(numDofs),
			 'Bs'             : Bs,
			 'Ls'             : np.array(Ls, dtype=np.float32),
			 'Gs'             : np.array(Gs, dtype=np.float32)
			}

def dictToMesh(skelDict):
	Vs, Bs, Ts, Names, Faces = [], [], [], [], []
	for ji, joint in enumerate(skelDict['jointNames']):
		bs = [[0,0,0]]
		children_inds = [i for i in xrange(skelDict['numJoints']) if skelDict['jointParents'][i] == ji]
		for j in children_inds:
			bs.append(skelDict['Ls'][j][:,3])
		Vs.append(bs)
		Bs.append([(0,i) for i in range(1,len(bs))])
		Ts.append(skelDict['Gs'][ji])
		Faces.append([])
		Names.append(joint)
	return dict(names=Names,verts=Vs,faces=Faces,bones=Bs,transforms=Ts)

def loadMayaCharacter(filename):
	nodes,nodeLists = read_MA(filename)
	evaluate_scenegraph(nodeLists)
	grip_dict = extract_GRIP(nodeLists)
	primitives,primitives2D,mats,camera_ids,movies = construct_geos(nodeLists)
	maPrimitive = primitives[0]
	skelDict = grip_dict['skels'][0]
	shape_weights = grip_dict['shape_weights']
	print 'loadMayaCharacter',skelDict['numChans']
	mesh_dict = dictToMesh(skelDict)
	skel_mesh = GLMeshes(**mesh_dict)
	print 'loadMayaCharacter',skel_mesh.transforms.shape
	geomInd = [pi for pi,nn in enumerate(maPrimitive['names']) if nn in shape_weights]
	Vs, Bs, Ts, Names, Faces, Vts, Fts = [], [], [], [], [], [], []
	for gi in geomInd:
		Vs.append(np.array(maPrimitive['verts'][gi],dtype=np.float32))
		Bs.append(maPrimitive['bones'][gi])
		Ts.append(maPrimitive['transforms'][gi])
		Names.append(maPrimitive['names'][gi])
		Faces.append(maPrimitive['faces'][gi])
		Vts.append(maPrimitive['vts'][gi])
		Fts.append(maPrimitive['fts'][gi])
	orig_Vs = deepcopy(Vs)
	orig_Bs = deepcopy(Bs)
	orig_Fs = deepcopy(Faces)
	orig_Ts = deepcopy(Ts)
	geom_dict = dict(names=Names,verts=Vs,faces=Faces,bones=Bs,transforms=Ts,vts=Vts,fts=Fts)
	return Character.Character(skeleton={'primitive':skel_mesh,'skel_dict':skelDict},
					 geometry={'geom_dict':geom_dict,'shape_weights':shape_weights}), \
		   {'Vs':orig_Vs, 'Bs':orig_Bs, 'Fs':orig_Fs, 'Ts':orig_Ts}

if __name__ == '__main__':
	filename = sys.argv[1]
	nodes,nodeLists = read_MA(filename)

	#open('nodeTypes','w').write(str(g_nodeTypes))
	print 'MAReader',len(nodes['children'].keys()), 'root objects'

	from PySide import QtGui
	appIn = QtGui.QApplication(sys.argv)
	appIn.setStyle('plastique')
	win = QApp.QApp()
	win.setWindowTitle('Imaginarium Maya File Browser')
	from UI import QGLViewer
	evaluate_scenegraph(nodeLists)
	fields,dnodes = maya_to_state(nodes)
	#IO.save('out.nodes',(nodes,nodeLists)) # TODO TODO this doesn't load because of something that happens in the evaluate_scenegraph (an unsupported back reference, could be a bug)
	grip_data = extract_GRIP(nodeLists)
	#IO.save('out.grip',grip_data)
	primitives,primitives2D,mats,camera_ids,movies = construct_geos(nodeLists)
	QApp.fields = fields
	gl_primitives = []
	for primitive in primitives:
		gl_primitives.append(GLMeshes(**primitive))
	win.qoutliner.set_root('/root')
	QGLViewer.makeViewer(primitives=gl_primitives, primitives2D=primitives2D, timeRange = (1,100), mats=mats, camera_ids=camera_ids, movies=movies, callback=setFrameCB, pickCallback=pickedCB, appIn=appIn, win=win)
