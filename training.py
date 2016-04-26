# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 16:43:50 2016

@author: owner
"""

import cv2
import numpy as np
from sklearn.preprocessing import normalize
from data_io import saveText3D, readText3D, readText2D, readDepthMap, readDepthAndJoints

def _generateOffsets(n_offsets, dist_max, n_cols):
    
    # generate random offset points
    while True:
        
        # first generates large number of points
        multiplier = 3
        random_offsets = np.random.uniform(-dist_max, dist_max, [n_offsets * multiplier, n_cols])
        
        # distance between offsets and the origin
        offset_norms = np.linalg.norm(random_offsets, axis=1)
        
        # constrain the distance
        accepted_indices = np.asarray(np.where(offset_norms < dist_max)).T
        
        # stop if enough number of offsets are generated
        if n_offsets <= len(accepted_indices):
            
            # randomly choice n_offsets points
            random_offsets = random_offsets[np.random.choice(accepted_indices[:, 0], size=n_offsets, replace=False)]
            
            break
    
    return random_offsets

## Generate training set consists of depth images, offset points, and unit direction vectors toward the true joint position
#  @param data_dir Directory that contains original data file consists of depthmap and true joint positions.
#  @param frame_begin Frame number that reading start with.
#  @param frame_end Frame number that reading stop with.
#  @param frame_step Step size of reading.
#  @param n_offsets Target number of random offset points.
#  @param dist_max Allowable max distance between offset points and true joint positions. (in meters)
#  @param output_dir Output directory. Resultant training set will be contained in here.
def generateTrainingSet(depth_prefix, joints_prefix, frame_begin, frame_end, frame_step, n_offsets, dist_max, sensor, model, output_dir, depth_extension='.png', joints_extension='.dat', visualize=False):
    
    if not (hasattr(sensor, 'project_points3d') and hasattr(sensor, 'create_pointcloud')):
        raise ValueError('sensor must have attributes \'project_points_3d\' and \'create_pointcloud\'')
        
    if not (hasattr(model, 'joints') and hasattr(model, 'adjacents')):
        raise ValueError('model must have attributes \'joints\' and \'adjacents\'')
        
    # 3D data
    n_cols = 3
    
    output_depth_prefix = output_dir + depth_prefix.split('/')[-1]
    output_joints_prefix = output_dir + joints_prefix.split('/')[-1]
    output_offsets_prefix = output_dir + 'offsets_'
    output_offsets_projected_prefix = output_dir + 'offsets_projected_'
    output_directions_prefix = output_dir + 'directions_'

    [depthmaps, true_jointset] = readDepthAndJoints(depth_prefix, joints_prefix, frame_begin, frame_end, frame_step, depth_extension, joints_extension)
    
    frame_number = frame_begin
    for depthmap, true_joints in zip(depthmaps, true_jointset):
        
        offset_points = []
        offset_points_projected = []
        direction_vectors = []
        
        for joint_number, joint in enumerate(true_joints):
            
            # generate random offset points for this joint and its adjacent
            random_offsets = _generateOffsets(n_offsets, dist_max, n_cols)
            random_offsets_adjacent = _generateOffsets(n_offsets, dist_max, n_cols)
                
            # translate to true joint positions
            offsets = random_offsets + joint
            offsets_adjacent = random_offsets_adjacent + true_joints[model.adjacents[joint_number]]
                
            # project to depth plane
            offsets_projected = sensor.project_points3d(offsets)
            offsets_projected_adjacent = sensor.project_points3d(offsets_adjacent)
                
            # unit direction vectors from offset points to true joint position
            directions = normalize(joint - offsets)
            directions_adjacent = normalize(joint - offsets_adjacent)
            
            # put all joints together
            offset_points.append(np.r_[offsets, offsets_adjacent])
            offset_points_projected.append(np.r_[offsets_projected, offsets_projected_adjacent])
            direction_vectors.append(np.r_[directions, directions_adjacent])
        
        # write offsets and direction vectors to file
        saveText3D(output_offsets_prefix + str(frame_number) + joints_extension, np.asarray(offset_points))
        saveText3D(output_offsets_projected_prefix + str(frame_number) + joints_extension, np.asarray(offset_points_projected))
        saveText3D(output_directions_prefix + str(frame_number) + joints_extension, np.asarray(direction_vectors))
        
        # write depth map to file
        cv2.imwrite(output_depth_prefix + str(frame_number) + depth_extension, depthmap)
        
        # write true joint positions to file
        np.savetxt(output_joints_prefix + str(frame_number) + joints_extension, true_joints)
        
        frame_number += 1
        
        if visualize:
            from visualization import visualizeTrainingSet
            visualizeTrainingSet(sensor.create_pointcloud(depthmap), true_joints, offset_points, direction_vectors)
        
## Generate training set consists of depth images, offset points, and unit direction vectors toward the true joint position
#  @param data_dir Directory that contains original data file consists of depthmap and true joint positions.
#  @param frame_begin Frame number that reading start with.
#  @param frame_end Frame number that reading stop with.
#  @param frame_step Step size of reading.
#  @param n_offsets Target number of random offset points.
#  @param dist_max Allowable max distance between offset points and true joint positions. (in meters)
#  @param output_dir Output directory. Resultant training set will be contained in here.
def readTrainingSet(depth_prefix, joints_prefix, offsets_prefix, offsets_projected_prefix, directions_prefix, frame_begin, frame_end, frame_step, depth_extension='.png', joints_extension='.dat', offsets_extension='.dat', offsets_projected_extension='.dat', directions_extension='.dat'):
    
    depth_names = []
    joints_names = []
    offsets_names = []
    offsets_projected_names = []
    directions_names = []
    
    for i in range(frame_begin, frame_end + 1, frame_step):
        depth_names.append(depth_prefix + str(i) + depth_extension)
        joints_names.append(joints_prefix + str(i) + joints_extension)
        offsets_names.append(offsets_prefix + str(i) + offsets_extension)
        offsets_projected_names.append(offsets_projected_prefix + str(i) + offsets_projected_extension)
        directions_names.append(directions_prefix + str(i) + directions_extension)
        
    depthmaps = readDepthMap(depth_names)
    true_joints = readText2D(joints_names)
    offset_points = readText3D(offsets_names)
    offset_points_projected = readText3D(offsets_projected_names)
    direction_vectors = readText3D(directions_names)
    
    depthmaps = np.asarray(depthmaps)
    true_joints = np.asarray(true_joints)
    offset_points = np.asarray(offset_points)
    offset_points_projected = np.asarray(offset_points_projected)
    direction_vectors = np.asarray(direction_vectors)
    
    return depthmaps, true_joints, offset_points, offset_points_projected, direction_vectors
    
def readTestSet(depth_prefix, joints_prefix, frame_begin, frame_end, frame_step, depth_extension='.png', joints_extension='.dat'):
    
    depth_names = []
    joints_names = []
    
    for i in range(frame_begin, frame_end + 1, frame_step):
        depth_names.append(depth_prefix + str(i) + depth_extension)
        joints_names.append(joints_prefix + str(i) + joints_extension)
        
    depthmaps = readDepthMap(depth_names)
    true_joints = readText2D(joints_names)
    
    depthmaps = np.asarray(depthmaps)
    true_joints = np.asarray(true_joints)
    
    return depthmaps, true_joints