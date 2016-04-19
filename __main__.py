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
    frame_end = 5#frame_numbers['seq_b'][1]
    frame_step = 1
    n_offsets = 500
    dist_max = 0.5
    source_depth_prefix = 'C:/Users/owner/Documents/Python Scripts/RandomTreeWalk/data_set/seq_c/depth_'
    source_joints_prefix = 'C:/Users/owner/Documents/Python Scripts/RandomTreeWalk/data_set/seq_c/joints_'
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
    
    tree_walk = RandomTreeWalk(sensor, model)
    
    """
    param_grid = {'min_samples_split':[100, 200],
                  'max_leaf_nodes':[32768],
                  'n_leaf_clusters':[20, 50],
                  'n_random_trials':[30],
                  'theta':[200, 400],
                  'tau':[20, 50]}
    """             
    param_grid = {'min_samples_split':[100],
                  'max_leaf_nodes':[32768],
                  'n_leaf_clusters':[10],
                  'n_random_trials':[30],
                  'theta':[200],
                  'tau':[50]}
    
    

    depthmaps, true_joints, offset_points, offset_points_projected, direction_vectors = readTrainingSet(train_depth_prefix, train_joints_prefix, train_offsets_prefix, train_offsets_projected_prefix, train_directions_prefix, frame_begin, frame_end, frame_step, depth_extension='.png', joints_extension='.dat', offsets_extension='.dat', offsets_projected_extension='.dat', directions_extension='.dat')
    
    #cloud = sensor.create_pointcloud(depthmaps[0])
    #visualizeTrainingSet(cloud, true_joints[0], offset_points[0], direction_vectors[0], n_offsets_to_render=1000)
    
    """
    gs = GridSearchCVTree(tree_walk, param_grid, n_jobs=3, refit=False,  verbose=2)
    gs.fit(depthmaps, offset_points_projected, direction_vectors, true_joints)
    
    
    with open('best_params_head.dat', 'w') as params_file, open('best_score_head.dat', 'w') as score_file, open('grid_scores.dat', 'w') as grid_file:
        params_file.write(str(gs.best_params_))
        score_file.write(str(gs.best_score_))
        grid_file.write(str(gs.grid_scores_))
    """
    
    #gs.best_estimator_.save('best_estimator_head.xml')
    
    
    #generateTrainingSet(source_depth_prefix, source_joints_prefix, frame_begin, frame_end, frame_step, n_offsets, dist_max, sensor, model, train_dir, depth_extension='.png', joints_extension='.dat', visualize=True)
    
    tree_walk = RandomTreeWalk(sensor, model, theta=300, tau=100, n_random_trials=600)
    #tree_walk.load('tree_seq_a_b.xml')
    tree_walk.fit(depthmaps, offset_points_projected, direction_vectors)
    tree_walk.predict(depthmaps[0], visualize=True)
    
    #tree_walk.load('tree_seq_a_b.xml')
    #tree_walk.predict(depthmaps[0], True)
    
    #tree_walk.score(depthmaps, true_joints)
    
    #test_walk = RandomTreeWalk()
    #test_walk.read('test_tree.xml')
    #test_walk.predict(depthmaps[0], n_steps, step_distance, sensor, model, visualize=True)

    #tree_walk.predict(depthmaps[0], n_steps, step_distance, sensor, model, visualize=True)
    
    