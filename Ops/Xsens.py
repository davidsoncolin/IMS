import sys
# if r"E:\dev\GRIP\c\XSens_Receiver\PythonBindings\x64\Debug" not in sys.path:
#     sys.path.append(r"E:\dev\GRIP\c\XSens_Receiver\PythonBindings\x64\Debug")
    
from XSensReceiver import Client
import numpy as np
import quaternion

from pprint import pprint

from PySide import QtCore, QtGui
from UI import QApp, QGLViewer
import Runtime, RenderCallback, Op
from Skeleton import Template
from GCore import Character

Runtime = Runtime.getInstance()

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

def genGs(Rs, Ts):
    Gs = []
    for ji in range(len(Rs)):
        r = quaternion.as_rotation_matrix(Rs[ji])
        t = Ts[ji].reshape(-1, 1)
        G = np.hstack((r, t))
        Gs.append(G)
    return Gs

def makeXSensSkelDict(points):
    # fields: 'name', 'jointChans', 'jointChanSplits', 'chanNames', 'chanValues', 'jointNames', 'jointIndex', 'numJoints',
    # 'jointParents', 'numChans', 'Bs', 'Ls', 'Gs'
    skelDict = {}
    skelDict['jointNames'] = ["Pelvis", "L5", "L3", "T12", "T8", "Neck", "Head", "Right Shoulder", "Right Upper Arm",
                             "Right Forearm", "Right Hand", "Left Shoulder", "Left Upper Arm", "Left Forearm",
                             "Left Hand", "Right Upper Leg", "Right Lower Leg", "Right Foot", "Right Toe",
                             "Left Upper Leg", "Left Lower Leg", "Left Foot", "Left Toe"]
    skelDict['numJoints'] = len(skelDict['jointNames'])
    skelDict['jointIndex'] = dict(zip(skelDict['jointNames'] ,range(len(skelDict['jointNames'] ))))
    skelDict['jointParents'] = [-1, 0, 1, 2, 3, 4, 5, 4, 7, 8, 9, 4, 11, 12, 13, 0, 15, 16, 17, 0, 19, 20, 21] # Eugh
    jointChans = []
    jointChanSplits = [0]
    dofNames = []
    Ls, Bs = [], [[] for j in skelDict['jointNames']]
    dofs = [':rx', ':ry', ':rz']
    T = points[:, :3]
    R = points[:, 3:].astype(np.float)

    R = quaternion.as_quat_array(R)
    Gs = genGs(R, T)
    for ji, jn in enumerate(skelDict['jointNames']):
        if ji == 0:
            jointChans += [0, 1, 2]
            jointChanSplits.append(3)
            dofNames += [jn + ':tx', jn + ':ty', jn + ':tz']
        else:
            jointChanSplits.append(jointChanSplits[-1])

        jointChans += [3, 4, 5]
        dofNames += [jn + dn for dn in dofs]
        jointChanSplits.append(len(jointChans))

        pji = skelDict['jointParents'][ji]

        Ls.append(matrix_mult(matrix_inverse(Gs[pji]), Gs[ji]) if pji != -1 else Gs[ji])
        if pji != -1:
            Bs[pji].append(Ls[-1][:, 3])

    Bs[6].append(np.float32([0,100,0])) # Head
    Bs[10].append(np.float32([0,0,75])) # Right Hand
    Bs[14].append(np.float32([0,0,-75])) # Left Hand
    Bs[18].append(np.float32([75,0,0])) # Right Foot
    Bs[22].append(np.float32([75, 0, 0]))  # Left Foot
    for i in range(len(Bs)):
        if not len(Bs[i]):
            print i
            # Bs[i].append(np.float32([0,100,0]))
    pprint(Bs)
    skelDict['chanNames'] = dofNames
    skelDict['jointChans'] = np.int32(jointChans)
    skelDict['jointChanSplits'] = np.int32(jointChanSplits)
    skelDict['numChans'] = len(dofNames)
    skelDict['Gs'] = np.array(Gs, dtype=np.float32)
    skelDict['Ls'] = np.array(Ls, dtype=np.float32)
    skelDict['Bs'] = Bs
    # pprint([np.linalg.norm(np.float32(x)) if len(x) else 0.0 for x in Bs ])
    skelDict['chanValues'] = np.zeros(skelDict['numChans'], dtype=np.float32)
    return skelDict

class XSensQuat(Op.Op):
    def __init__(self, name='/XSensQuat', locations=''):
        self.fields = [
            ('name', 'name', 'name', 'string', name, {}),
            ('locations', 'locations', 'locations', 'string', locations, {})
        ]

        super(self.__class__, self).__init__(name, self.fields)

        self.client = Client()
        self.client.initialise("127.0.0.1", "9763")
        self.skelDict = None

    def cook(self, location, interface, attrs):
        points = self.client.getQuatData(0)

        if not np.all(np.isnan(points[:23, 0])):

            if self.skelDict is None:
                print "New Skel"
                self.skelDict = makeXSensSkelDict(points[:23, :])
            else:
                T = points[:23, :3]
                R = points[:23, 3:].astype(np.float)
                R = quaternion.as_quat_array(R)
                self.skelDict['Gs'] = np.array(genGs(R, T), dtype=np.float32)



            skelAttrs = {
                'skelDict': self.skelDict,
                'rootMat': np.eye(3,4),
                'originalRootMat': np.eye(3,4),
                'subjectName': self.getName(),
                'boneColour': (0.3, 0.42, 0.66, 1.)
            }
            interface.createChild(interface.name(), 'skeleton', atLocation=interface.parentPath(), attrs=skelAttrs)

import Registry
Registry.registerOp('XSens Skeleton', XSensQuat)