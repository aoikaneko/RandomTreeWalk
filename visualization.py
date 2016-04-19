# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 20:40:44 2016

@author: owner
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualizeTrainingSet(cloud, joints, offset_points, direction_vectors, title=None, n_cloud_to_render=500, n_offsets_to_render=50):
    
    # reduce number of points for rendering
    cloud_reduced = cloud[~np.isnan(cloud).any(axis=1)]
    cloud_reduced = cloud_reduced[np.random.randint(len(cloud_reduced), size=n_cloud_to_render), :]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    if title is not None:
        plt.title(title)

    # cloud
    ax.scatter(cloud_reduced[:, 0], cloud_reduced[:, 1], cloud_reduced[:, 2], s=10, c='r', marker='o', alpha=0.5)

    for i, (joint, offsets, directions) in enumerate(zip(joints, offset_points, direction_vectors)):
        
        # reduce number of offsets for rendering
        remain_indices = np.random.randint(len(offsets), size=n_offsets_to_render)
        offsets_reduced = offsets[remain_indices]
        directions_reduced = directions[remain_indices]
        
        # joint
        ax.scatter(joint[0], joint[1], joint[2], c='b', marker='o')
        ax.text(joint[0], joint[1], joint[2], 'joint' + str(i))
        
        # offsets
        ax.scatter(offsets_reduced[:, 0], offsets_reduced[:, 1], offsets_reduced[:, 2], s=1, c='g', marker='o')
        
        # direcitions
        ax.quiver(joint[0], joint[1], joint[2], directions_reduced[:, 0], directions_reduced[:, 1], directions_reduced[:, 2], color='r', length=0.2, arrow_length_ratio=0.5, alpha=0.3)

    # set bounding box to equalize xyz axis
    limit = cloud_reduced
    max_range = np.array([limit[:, 0].max() - limit[:, 0].min(), limit[:, 1].max() - limit[:, 1].min(), limit[:, 2].max() - limit[:, 2].min()]).max() / 2.0
    mean_x = limit[:, 0].mean()
    mean_y = limit[:, 1].mean()
    mean_z = limit[:, 2].mean()
    ax.set_xlim(mean_x - max_range, mean_x + max_range)
    ax.set_ylim(mean_y - max_range, mean_y + max_range)
    ax.set_zlim(mean_z - max_range, mean_z + max_range)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.ion()
    plt.show()
    
def visualizePredict(cloud, joints, footprints, title=None, n_cloud_to_render=500):
    
    # reduce number of points for rendering
    cloud_reduced = cloud[~np.isnan(cloud).any(axis=1)]
    cloud_reduced = cloud_reduced[np.random.randint(len(cloud_reduced), size=n_cloud_to_render), :]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    if title is not None:
        plt.title(title)

    # cloud
    ax.scatter(cloud_reduced[:, 0], cloud_reduced[:, 1], cloud_reduced[:, 2], s=10, c='r', marker='o', alpha=0.5)

    for joint, footprint in zip(joints, footprints):
        # joint
        ax.scatter(joint[0], joint[1], joint[2], s=120, c='g', marker='o')
        
        # footprint
        ax.plot(footprint[:, 0], footprint[:, 1], footprint[:, 2], c='b', lw=5)
        
    # set bounding box to equalize xyz axis
    limit = cloud_reduced
    max_range = np.array([limit[:, 0].max() - limit[:, 0].min(), limit[:, 1].max() - limit[:, 1].min(), limit[:, 2].max() - limit[:, 2].min()]).max() / 2.0
    mean_x = limit[:, 0].mean()
    mean_y = limit[:, 1].mean()
    mean_z = limit[:, 2].mean()
    ax.set_xlim(mean_x - max_range, mean_x + max_range)
    ax.set_ylim(mean_y - max_range, mean_y + max_range)
    ax.set_zlim(mean_z - max_range, mean_z + max_range)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.ion()
    plt.show()
    
def visualizeResult(cloud, joints, true_joints=None, title=None, n_cloud_to_render=500):
    
    # reduce number of points for rendering
    cloud_reduced = cloud[~np.isnan(cloud).any(axis=1)]
    cloud_reduced = cloud_reduced[np.random.randint(len(cloud_reduced), size=n_cloud_to_render), :]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    if title is not None:
        plt.title(title)

    # cloud
    ax.scatter(cloud_reduced[:, 0], cloud_reduced[:, 1], cloud_reduced[:, 2], s=10, c='r', marker='o', alpha=0.5)

    for i, joint in enumerate(joints):
        # joint
        ax.scatter(joint[0], joint[1], joint[2], c='b', marker='o')
        ax.text(joint[0], joint[1], joint[2], 'estimated' + str(i))
        
    if true_joints is not None:
        for i, true_joint in enumerate(true_joints):
            # true joint
            ax.scatter(true_joint[0], true_joint[1], true_joint[2], c='g', marker='o')
            ax.text(true_joint[0], true_joint[1], true_joint[2], 'truth' + str(i))
        
    # set bounding box to equalize xyz axis
    limit = cloud_reduced
    max_range = np.array([limit[:, 0].max() - limit[:, 0].min(), limit[:, 1].max() - limit[:, 1].min(), limit[:, 2].max() - limit[:, 2].min()]).max() / 2.0
    mean_x = limit[:, 0].mean()
    mean_y = limit[:, 1].mean()
    mean_z = limit[:, 2].mean()
    ax.set_xlim(mean_x - max_range, mean_x + max_range)
    ax.set_ylim(mean_y - max_range, mean_y + max_range)
    ax.set_zlim(mean_z - max_range, mean_z + max_range)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.ion()
    plt.show()