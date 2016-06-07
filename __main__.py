# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 00:11:19 2016

@author: owner
"""

from model import SkeletonModel
from decision_tree import RandomTreeWalk
from grid_search import GridSearchCVTree
from training import generateTrainingSet, readTrainingSet
from data_io import readDepthMap
from visualization import visualizeTrainingSet
from pykinect_wrapper import PyKinect

if __name__ == '__main__':
    
    frame_numbers = {'seq_a':(0, 2739), 'seq_b':(2740, 5769), 'seq_c':(5770, 8555)}
    
    frame_begin = frame_numbers['seq_a'][0]
    frame_end = frame_numbers['seq_a'][1]
    frame_step = 1
    n_offsets = 500
    dist_max = 0.5
    source_dir = 'C:/Users/owner/Documents/Python Scripts/RandomTreeWalk/data_set/seq_a/'
    source_depth_prefix = source_dir + 'depth_'
    source_joints_prefix = source_dir + 'joints_'
    train_dir = 'C:/Users/owner/Documents/Python Scripts/RandomTreeWalk/training_set/all/'
    train_depth_prefix = train_dir + 'depth_'
    train_joints_prefix = train_dir + 'joints_'
    train_offsets_prefix = train_dir + 'offsets_'
    train_offsets_projected_prefix = train_dir + 'offsets_projected_'
    train_directions_prefix = train_dir + 'directions_'
    
    n_joints = 12
    valid_depth_range = (100, 4000)
    min_samples_split = 10
    max_leaf_nodes = 32768
    n_leaf_clusters = 5
    n_random_trials = 10
    depth_background_val = 8000
    theta = 100
    tau = 200
    
    n_steps = 64
    step_distance = 0.05
    sensor = PyKinect()
    sensor.set_resolution('320x240')
    
    model = SkeletonModel()    
    
    #generateTrainingSet(source_depth_prefix, source_joints_prefix, frame_begin, frame_end, frame_step, n_offsets, dist_max, sensor, model, train_dir, depth_extension='.png', joints_extension='.dat')

    depthmaps, true_joints, offset_points, offset_points_projected, direction_vectors = readTrainingSet(train_depth_prefix, train_joints_prefix, train_offsets_prefix, train_offsets_projected_prefix, train_directions_prefix, frame_begin, frame_end, frame_step, depth_extension='.png', joints_extension='.dat', offsets_extension='.dat', offsets_projected_extension='.dat', directions_extension='.dat')
    
    tree_walk = RandomTreeWalk(sensor, model, min_samples_leaf=10, n_leaf_clusters=5, theta=100, tau=200, n_random_trials=200, n_steps=128, step_distance=0.05, search='dfs', weights=(1.0, 1.0, 1.5))
    tree_walk.fit(depthmaps, offset_points_projected, direction_vectors)
    tree_walk.save('tree.xml')
    tree_walk.predict(depthmaps[0], visualize=True)