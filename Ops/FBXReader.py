#!/usr/bin/env python

import os.path
from FbxCommon import *
import numpy as np

def matrix_mult(p,c):
    ret = np.dot(p[:3,:3],c[:3,:])
    ret[:,3] += p[:3,3]
    return ret

def matrix_inverse(m):
    ret = np.zeros((3,4),dtype=np.float32)
    try:
        ret[:,:3] = np.linalg.inv(m[:3,:3])
    except:
        print '???exception in matrix_inverse',list(ret.ravel()) # TODO HACK
        ret[:,:3] = np.eye(3) #m[:3,:3].T
    ret[:,3] = -np.dot(ret[:,:3],m[:3,3])
    return ret

def readFile(fbxFilename):
    skeletonDict = None
    animDict = None
    if not os.path.isfile(fbxFilename):
        raise OSError('Specified FBX file (%s) does not exist' % fbxFilename)
    sdkManager, scene = InitializeSdkObjects()
    if LoadScene(sdkManager, scene, fbxFilename):
        skeletonDict = getSkeletonDict(sdkManager, scene)
        # animDict = getAnimationDict()
    else:
        pass

    return (skeletonDict, animDict)

#def fbxVector4_to_

def getSkeletonDict(sdkManager, scene):
    rootNode = scene.GetRootNode()
    #hardcoding pig skeleton
    pigNode = rootNode.GetChild(4).GetChild(2)
    skeleton = processSkeletonHierarchy(pigNode, sdkManager, scene)

    skelDict = {}
    # TODO: do the action below during processSkeletonHierarchy
    skelDict['jointNames'] = [str(skeleton['joints'][i]['name']) for i in range(0,len(skeleton['joints']))]
    skelDict['numJoints'] = len(skelDict['jointNames'])
    skelDict['jointIndex'] = dict(zip(skelDict['jointNames'], range(len(skelDict['jointNames']))))
    skelDict['jointParents'] = np.array([skeleton['joints'][i]['parentIndex'] for i in range(0,len(skeleton['joints']))])
    jointChans = []
    jointChanSplits = [0]
    dofNames = []
    Ls, Bs = [], [[] for j in skelDict['jointNames']]
    dofs = [':rx', ':ry', ':rz']

    Gs = [skeleton['joints'][i]['transform'] for i in range(0,len(skeleton['joints']))]

    for ji, jn in enumerate(skelDict['jointNames']):
        if ji == 0:
            jointChans += [0, 1, 2]
            jointChanSplits.append(3)
            dofNames += [jn + ':tx', jn + ':ty', jn + ':tz']
        else:
            jointChanSplits.append(jointChanSplits[-1])

        jointChans += [3,4,5]
        dofNames += [jn + dn for dn in dofs]
        jointChanSplits.append(len(jointChans))

        pji = skelDict['jointParents'][ji]

        Ls.append(matrix_mult(matrix_inverse(Gs[pji]), Gs[ji]) if pji != -1 else Gs[ji])
        if pji != -1:
            Bs[pji].append(Ls[-1][:, 3])

    skelDict['chanNames'] = dofNames
    skelDict['jointChans'] = np.int32(jointChans)
    skelDict['jointChanSplits'] = np.int32(jointChanSplits)
    skelDict['numChans'] = len(dofNames)
    skelDict['Gs'] = np.array(Gs, dtype=np.float32)
    skelDict['Ls'] = np.array(Ls, dtype=np.float32)
    skelDict['Bs'] = Bs
    # pprint([np.linalg.norm(np.float32(x)) if len(x) else 0.0 for x in Bs ])
    skelDict['chanValues'] = np.zeros(skelDict['numChans'], dtype=np.float32)
    skelDict['rootMat'] = np.eye(3,4)
    return skelDict

def processSkeletonHierarchy(inRootNode, sdkManager, scene):
    skeleton = {}
    skeleton['joints'] = []
    for childIndex in range(0, inRootNode.GetChildCount()):
        currNode = inRootNode.GetChild(childIndex)
        processSkeletonHierarchyRecursively(currNode, 0, -1, skeleton, sdkManager, scene)

    return skeleton

def processSkeletonHierarchyRecursively(inNode, myIndex, inParentIndex, skeleton, sdkManager, scene):
    # TODO: when processing multiple characters in file, use IsSkeletonRoot
    if inNode.GetSkeleton() is not None and inNode.GetSkeleton().IsSkeletonRoot():
        print 'Skeleton root:'
    if inNode.GetNodeAttribute() is not None and inNode.GetNodeAttribute().GetAttributeType() is not None and inNode.GetNodeAttribute().GetAttributeType() == FbxNodeAttribute.eSkeleton:
        curJoint = {}
        curJoint['parentIndex'] = inParentIndex
        curJoint['name'] = inNode.GetName()
        curJoint['translation'] = inNode.EvaluateLocalTranslation()
        curJoint['rotation'] = inNode.EvaluateLocalRotation()
        curJoint['scaling'] = inNode.EvaluateLocalScaling()
        fbxTransform = inNode.EvaluateGlobalTransform()
        transform = np.empty((3, 4), dtype=np.float32)
        for i in range(3):
            for j in range(4):
                transform[i, j] = fbxTransform.Get(j, i)
                if j == 3:
                    transform[i, j] *= (10 / 0.72) # pig is scaled nastily, from cm to mm and 0.72
        curJoint['transform'] = transform

        # TODO: get the scene object to get handle of the animation

        # importer = FbxImporter.Create(sdkManager) # type: FbxImporter
        #
        #
        # framerate = FbxTime.eFrames30
        # animStack = scene.GetSrcObject(FbxCriteria.ObjectType(FbxAnimStack.ClassId), 0)
        # animStackName = animStack.GetName()
        # takeInfo = scene.GetTakeInfo(animStackName) # type: FbxTakeInfo
        # start = takeInfo.mLocalTimeSpan.GetStart() # type: FbxTime
        # end = takeInfo.mLocalTimeSpan.GetStop() # type: FbxTime
        # mAnimationLength = end.GetFrameCount(framerate) - start.GetFrameCount(framerate) + 1

        # frameNumber = 100
        # currTime = FbxTime()
        # currTime.SetFrame(frameNumber, framerate)
        # fTrans = childNode.EvaluateGlobalTransform(currTime)

        # curJoint['animation']

        # print inNode.GetName()
        # print curJoint['translation']
        # print curJoint['rotation']
        # print curJoint['scaling']
        skeleton['joints'].append(curJoint)

    for i in range(0, inNode.GetChildCount()):
        processSkeletonHierarchyRecursively(inNode.GetChild(i), len(skeleton['joints']), myIndex, skeleton, sdkManager, scene)

def getAnimationDict():
    boneNames = None
    asfDofSplits = None
    dofScales = None
    numDofs = None
    pass