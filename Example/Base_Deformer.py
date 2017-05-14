import numpy as np
import os, sys
from IO import MAReader
import itertools
from UI import GLMeshes, QApp, QGLViewer
from PySide import QtGui, QtCore
from GCore import Character, State

class CubeDeformer:
    type = 'CubeDeformer'

    def __init__(self, name, skel_dict, mappings):
        self.name = name
        self.skel_dict = skel_dict
        self.mappings = mappings

    def findBoneCentre(self, skel_dict, jointIndex):
        children = []
        for i in xrange(skel_dict['numJoints']):
            if skel_dict['jointParents'][i] == jointIndex:
                children.append(i)
        centre = np.zeros(3)
        for child in children:
            centre += skel_dict['Ls'][child][:,3]
        centre /= 2 * len(children)
        return np.dot(skel_dict['Gs'][jointIndex][:,:3], centre.T), np.argmax(np.abs(centre))

    def updateGlobalMapping(self, mapping, init=False):
        if mapping['source_anchor'] <> 'World Origin':
            joint_index = self.skel_dict['jointIndex'][mapping['source_anchor']]
            mapping['global'] = np.vstack((self.skel_dict['Gs'][joint_index],
                                           np.array([0,0,0,1],dtype=np.float32)))
            bone_centre, bone_axis = self.findBoneCentre(self.skel_dict, joint_index)
            mapping['global'][:3,3] += bone_centre
            mapping['inv_global'] = np.linalg.inv(mapping['global'])

    def updateGlobalMappings(self, init=False):
        for mi, mapping in enumerate(self.mappings):
            if mapping['source_anchor'] <> 'World Origin':
                try:
                    joint_index = self.skel_dict['jointIndex'][mapping['source_anchor']]
                    mapping['global'] = np.vstack((self.skel_dict['Gs'][joint_index],
                                                   np.array([0,0,0,1],dtype=np.float32)))
                    bone_centre, bone_axis = self.findBoneCentre(self.skel_dict, joint_index)
                    mapping['global'][:3,3] += bone_centre
                    mapping['inv_global'] = np.linalg.inv(mapping['global'])
                    self.mappings[mi] = mapping
                except KeyError:
                    continue

    def deformPoints(self, points):
        if len(self.mappings) > 0:
            weights = np.zeros(points.shape[0], dtype=np.float32)
            deformation = np.zeros_like(points, dtype=np.float32)
            self.updateGlobalMappings()
            for mapping in self.mappings:
                to_cube_space_mapping = self.combineTransforms(mapping['inv_global'], mapping['inv_local'])
                from_cube_space_mapping = self.combineTransforms(mapping['local'], mapping['global'])
                cube_space_points = self.applyTransform(to_cube_space_mapping, self.pointsToHomogeneous(points))
                deformation_update, update = self.computeDeformation(self.pointsFromHomogeneous(cube_space_points),
                                                                     mapping['source'],mapping['target'],
                                                                     mapping['boundary'])
                global_deformation_update = self.applyTransform(from_cube_space_mapping,
                                                                self.pointsToHomogeneous(deformation_update))
                deformation[update.astype(bool), :] += \
                    self.pointsFromHomogeneous(global_deformation_update[update.astype(bool), :])
                weights[:] += update
            where = np.where(weights > 0)[0]
            deformed_points = points.copy()
            deformed_points[where, :] = deformation[where, :] / weights[where, None]

            return deformed_points
        return points

    @staticmethod
    def pointsToHomogeneous(points):
        out = np.ones((points.shape[0],points.shape[1] + 1), dtype=np.float32)
        out[:,:3] = points
        return out

    @staticmethod
    def pointsFromHomogeneous(points):
        return points[:,:3] / points[:,3].reshape(-1,1)

    @staticmethod
    def computeDeformation(X, source_point, target_point, boundary):
        d_Cube = target_point - source_point
        new_X = np.zeros_like(X, dtype=np.float32)
        step_sizes = np.zeros_like(X, dtype=np.float32)
        for i in range(3):
            step_sizes[:,i] = CubeDeformer.piecewiseMapping(X[:,i],source_point[i],boundary[i])
        where = np.all(step_sizes > 0, 1)
        for i in range(3):
            new_X[where,i] = X[where, i] + np.sign(X[where,i]) * step_sizes[where, i]**2 * d_Cube[i] * \
                CubeDeformer.couplingFactor(X[where, (i+1)%3], source_point[(i+1)%3], boundary[(i+1)%3]) * \
                CubeDeformer.couplingFactor(X[where, (i+2)%3], source_point[(i+2)%3], boundary[(i+2)%3])

        return new_X, where.astype(np.float32)

    @staticmethod
    def couplingFactor(X, abs_x0, abs_b):
        secondRamp = lambda x: np.float32((abs_b - np.abs(x))) / (abs_b - abs_x0)
        return np.piecewise(X, [np.abs(X) < abs_x0], [1,secondRamp])**2

    @staticmethod
    def piecewiseMapping(X, abs_x0, abs_b):
        firstRamp = lambda x: np.float32(np.abs(x)) / abs_x0
        secondRamp = lambda x: np.float32((abs_b - np.abs(x))) / (abs_b - abs_x0)
        return np.piecewise(X, [np.abs(X) < abs_x0], [firstRamp, secondRamp])

    @staticmethod
    def combineTransforms(first_Transform, second_Transform):
        return np.dot(second_Transform, first_Transform)


    @staticmethod
    def applyTransform(transform, points):
        return np.dot(transform,points.T).T
