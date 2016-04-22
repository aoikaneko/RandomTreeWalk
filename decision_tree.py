# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 22:35:35 2016

@author: owner
"""

import sys
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from os.path import splitext
from scipy.stats import uniform
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from distutils.util import strtobool
from model import SkeletonModel
from data_io import is_string_like, array2string, string2array1D, string2array2D
from pykinect_wrapper import PyKinect
from pyrdtree_wrapper import _calc_partition_c
from pymyutil_wrapper import convert_16u_to_8u

class TreeNode:
    __slots__ = ['is_leaf', 'level', 'left', 'right', 'samples', 'split_parameter', 'features', 'feature_probabilities']
    
    def __init__(self):
        self.is_leaf = False
        self.level = -1
        self.left = -1
        self.right = -1        
        self.samples = np.array([], np.uint16)
        self.split_parameter = None
        self.features = None
        self.feature_probabilities = None

class DecisionTree:
    
    def __init__(self, min_samples_split=10, max_leaf_nodes=32768, n_leaf_clusters=5, n_random_trials=10, depth_background_val=8.0, theta=100, tau=0.2):
        self._nodes = []
        self._n_leaves = 0
        self._min_samples_split = min_samples_split
        self._max_leaf_nodes = max_leaf_nodes
        self._n_leaf_clusters = n_leaf_clusters
        self._n_random_trials = n_random_trials
        self._depth_background_val = depth_background_val
        self._random_theta = uniform(loc=-theta, scale=theta * 2.0)
        self._random_tau = uniform(loc=-tau, scale=tau * 2.0)
        
    def _set_leaf(self, node, directions):
        
        node.is_leaf = True

        # k-means clustering at leaf node
        leaf_clusters = KMeans(n_clusters=self._n_leaf_clusters, init='random').fit(directions)
        
        # store unit direction vectors at leaf
        node.features = normalize(leaf_clusters.cluster_centers_).astype(np.float32)
        
        # store probabilities of random selection
        labels = leaf_clusters.labels_
        node.feature_probabilities = np.asarray([(labels == label).sum() / len(labels) for label in range(self._n_leaf_clusters)])
        
        self._n_leaves += 1
        
        return self
    
    def _calc_partition_fit(self, sample_frame_ids, sample_offset_ids, depthmaps, offsets, t1, t2, tau):
        
        if len(sample_frame_ids) != len(sample_offset_ids):
            raise ValueError('sample_frame_ids and sample_offset_ids should have same length')
        
        _, rows, cols = depthmaps.shape
        length = len(sample_frame_ids)
        
        partition = _calc_partition_c(length, rows, cols, sample_frame_ids, sample_offset_ids, depthmaps, offsets, t1 ,t2 ,tau, self._depth_background_val)
        
        return partition
        
    def _calc_partition_predict(self, depthmap, x, t1, t2, tau):

        rows, cols = depthmap.shape

        # outside of image gives large constant value
        depth_at_x = self._depth_background_val
        
        # note: x consists of (x_coord_of_depth, y_coord_of_depth)
        # boundary check for initial offset coordinates
        x_valid = (0 <= x[:, 1]) & (x[:, 1] < rows) & (0 <= x[:, 0]) & (x[:, 0] < cols)
        
        # depth values at x
        if x_valid:
            depth_at_x = depthmap[x[:, 1], x[:, 0]]

        t1_div_x = np.floor(x + (t1 / depth_at_x) + 0.5).astype(np.int16)
        t2_div_x = np.floor(x + (t2 / depth_at_x) + 0.5).astype(np.int16)
        
        # outside of image gives large constant value
        depth_at_x_t1 = self._depth_background_val
        depth_at_x_t2 = self._depth_background_val
        
        # boundary check for random offsets
        x_t1_valid = (0 <= t1_div_x[:, 1]) & (t1_div_x[:, 1] < rows) & (0 <= t1_div_x[:, 0]) & (t1_div_x[:, 0] < cols)
        x_t2_valid = (0 <= t2_div_x[:, 1]) & (t2_div_x[:, 1] < rows) & (0 <= t2_div_x[:, 0]) & (t2_div_x[:, 0] < cols)
        
        # depth values at (x + t1/d(x))
        if x_t1_valid:
            depth_at_x_t1 = depthmap[t1_div_x[:, 1], t1_div_x[:, 0]]
        
        # and (x + t2/d(x))
        if x_t2_valid:
            depth_at_x_t2 = depthmap[t2_div_x[:, 1], t2_div_x[:, 0]]
        
        partition = (depth_at_x_t1 - depth_at_x_t2) < tau
        
        return partition
        
    def _calc_ids(self, samples, n_offsets_in_frame):
        return (samples / n_offsets_in_frame).astype(np.uint16), (samples % n_offsets_in_frame).astype(np.uint16)
    
    ## Train dicision tree for random tree walk algorithm. One tree correspond to one joint.
    #  depthmaps is a 3D array with shape [n_frames, rows, cols].
    #  offset_points_projected and direction_vectors are 3D array with shape [n_frames, n_offsets, n_dimensions=3]
    #  @param depthmaps 3D array contains 16-bit, single-channel depth maps
    #  @param offset_points_projected 3D array contains offset points projected on depth map plane.
    #  @param direction_vectors 3D array containes unit direction vectors from offset points to true joint position
    def fit(self, depthmaps, offset_points_projected, direction_vectors):
        
        if len(depthmaps) != len(offset_points_projected):
           raise ValueError('depthmaps and offset_points_projected should have same length.')
        
        if offset_points_projected.shape[:-1] != direction_vectors.shape[:-1]:
            raise ValueError('offset_points_projected and direction_vectors must have same shape except last axis')
            
        n_frames = len(depthmaps)
        n_offsets_in_frame = offset_points_projected.shape[1]
        n_data = n_frames * n_offsets_in_frame

        # root node
        root = TreeNode()
        root.level = 0
        
        # root node containes all training samples
        root.samples = np.append(root.samples, range(n_data))
        
        self._nodes.append(root)
        
        current_node_id = 0
        while current_node_id < len(self._nodes):
            node = self._nodes[current_node_id]
            
            sample_frame_ids, sample_offset_ids = self._calc_ids(node.samples, n_offsets_in_frame)
            
            # check whether current node is leaf
            if self._max_leaf_nodes < self._n_leaves:
                self._set_leaf(node, direction_vectors[sample_frame_ids, sample_offset_ids])
                current_node_id += 1
                continue
            
            best_mse = sys.float_info.max
            best_split = None
            best_left = np.array([], np.uint16)
            best_right = np.array([], np.uint16)
            
            for trial in range(self._n_random_trials):
                t1 = self._random_theta.rvs(size=[2]).astype(np.float32)
                t2 = self._random_theta.rvs(size=[2]).astype(np.float32)
                tau = self._random_tau.rvs(size=[1]).astype(np.float32)

                partition = self._calc_partition_fit(sample_frame_ids, sample_offset_ids, depthmaps, offset_points_projected, t1, t2, tau)
                
                trial_left = node.samples[partition]
                trial_right = node.samples[~partition]
                
                # large deviation; stop this trial
                if len(trial_left) == 0 or len(trial_right) == 0:
                    continue

                left_frame_ids, left_offset_ids = self._calc_ids(trial_left, n_offsets_in_frame)
                right_frame_ids, right_offset_ids = self._calc_ids(trial_right, n_offsets_in_frame)
                
                trial_left_cost = np.linalg.norm(direction_vectors[left_frame_ids, left_offset_ids] - direction_vectors[left_frame_ids, left_offset_ids].mean(axis=0), axis=1).sum()
                trial_right_cost = np.linalg.norm(direction_vectors[right_frame_ids, right_offset_ids] - direction_vectors[right_frame_ids, right_offset_ids].mean(axis=0), axis=1).sum()
                
                mse = trial_left_cost + trial_right_cost
                
                if mse < best_mse:
                    best_mse = mse
                    best_split = np.r_[t1, t2, tau]
                    best_left = trial_left
                    best_right = trial_right
            
            # check both splits have enough number of samples
            if len(best_left) <= self._min_samples_split or len(best_right) <= self._min_samples_split:
                self._set_leaf(node, direction_vectors[sample_frame_ids, sample_offset_ids])
                current_node_id += 1
                continue
            
            # add left and right node
            left = TreeNode()
            right = TreeNode()
            
            left.level = right.level = node.level + 1
            node.split_parameter = best_split
            node.left = len(self._nodes)
            node.right = len(self._nodes) + 1
            
            left.samples = np.append(left.samples, best_left)
            right.samples = np.append(right.samples, best_right)
            
            self._nodes.append(left)
            self._nodes.append(right)
            
            current_node_id += 1
        
        return self
            
    def predict(self, depthmap, startpoint, n_steps, step_distance, sensor):
        
        if (n_steps < 1) or (not isinstance(n_steps, int)):
            raise ValueError('n_steps must be a positive integer')
            
        if step_distance <= 0:
            raise ValueError('step_distance must be a positive number')
            
        if not hasattr(sensor, 'project_points3d'):
            raise ValueError('sensor must have attribute \'project_points_3d\'')
            
        current_position = np.atleast_2d(startpoint.copy())
        current_position_projected = np.floor(sensor.project_points3d(current_position) + 0.5).astype(np.int16)
        position_sum = np.zeros([1, 3], np.float32)
        
        footprints_3d = [current_position.flatten()]
        footprints_2d = [current_position_projected.flatten()]
            
        for step in range(n_steps):
            
            node = self._nodes[0]
            
            while True:
                
                if node.is_leaf:
                    break
                
                # decide whether go to the right or to the left
                t1 = node.split_parameter[0:2]
                t2 = node.split_parameter[2:4]
                tau = node.split_parameter[4:5]
                partition = self._calc_partition_predict(depthmap, current_position_projected, t1, t2, tau)
                
                next_node_id = node.left if partition else node.right
                
                node = self._nodes[next_node_id]
                
            # randomly select k-th cluster
            cluster_id = np.random.choice(self._n_leaf_clusters, p=node.feature_probabilities)
            
            # step into new position
            current_position += node.features[cluster_id] * step_distance
            current_position_projected = np.floor(sensor.project_points3d(current_position) + 0.5).astype(np.int16)
            
            # update joint position sum
            position_sum += current_position
            
            # save footprints
            footprints_3d.append(current_position.flatten())
            footprints_2d.append(current_position_projected.flatten())
        
        average_position = position_sum / n_steps

        return average_position, np.asarray(footprints_3d), np.asarray(footprints_2d)
            
    def save(self, filename, tree_id):
        
        own_outfile = False
        if is_string_like(filename):
            own_outfile = True
            xml_tree = ET.Element('tree')
        elif isinstance(filename, ET.Element):
            xml_tree = ET.SubElement(filename, 'tree')
        else:
            raise ValueError('filename must be a string or ElementTree.Element.')
        
        # set tree attributes
        xml_tree.set('id', str(tree_id))
        xml_tree.set('n_nodes', str(len(self._nodes)))
        
        # TO DO: add n_leaves
        # TO DO: add depth background val
        
        # iterating through nodes
        for node_id, node in enumerate(self._nodes):
            
            # add node as sub-element of tree
            xml_node = ET.SubElement(xml_tree, 'node', {'id':str(node_id), 'is_leaf':str(node.is_leaf), 'left':str(node.left), 'right':str(node.right)})
            
            if node.is_leaf:
                                
                # leaf cluster centers
                xml_features = ET.SubElement(xml_node, 'features', {'rows':str(node.features.shape[0]), 'cols':str(node.features.shape[1])})
                xml_features.text = array2string(node.features)
                
                # cluster probabilities
                xml_feature_probabilities = ET.SubElement(xml_node, 'feature_probabilities', {'length':str(node.feature_probabilities.shape[0])})
                xml_feature_probabilities.text = array2string(node.feature_probabilities)
                
            else:

                # split parameter
                xml_split_parameter = ET.SubElement(xml_node, 'split_parameter', {'length':str(node.split_parameter.shape[0])})
                xml_split_parameter.text = array2string(node.split_parameter)
        
        if own_outfile:
            ET.ElementTree(xml_tree).write(filename)
            
        return self
            
    def load(self, filename):
        
        if is_string_like(filename):
            xml_tree = ET.parse(filename).getroot()
        elif isinstance(filename, ET.Element):
            xml_tree = filename
        else:
            raise ValueError('filename must be a string or ElementTree.Element.')
            
        # clear current nodes for reading
        self._nodes.clear()
            
        # iterating through nodes
        for xml_node in xml_tree.iter('node'):
            
            # create node instance
            node = TreeNode()
            
            # get whether this node is leaf or not
            node.is_leaf = bool(strtobool(xml_node.get('is_leaf')))
            
            # child nodes
            node.left = int(xml_node.get('left'))
            node.right = int(xml_node.get('right'))
            
            if node.is_leaf:
            
                # leaf cluster centers
                xml_features = xml_node.find('features')
                rows = int(xml_features.get('rows'))
                cols = int(xml_features.get('cols'))
                node.features = string2array2D(rows, cols, xml_features.text, np.float32)
                
                # cluster probabilities
                xml_feature_probabilities = xml_node.find('feature_probabilities')
                length = int(xml_feature_probabilities.get('length'))
                node.feature_probabilities = string2array1D(length, xml_feature_probabilities.text)
                
            else:
    
                # split parameter
                xml_split_parameter = xml_node.find('split_parameter')
                length = int(xml_split_parameter.get('length'))
                node.split_parameter = string2array1D(length, xml_split_parameter.text)
                
            # add tree node
            self._nodes.append(node)
            
        return self

class RandomTreeWalk(BaseEstimator):
    
    def __init__(self, sensor=PyKinect(), model=SkeletonModel(), min_samples_split=10, max_leaf_nodes=32768, n_leaf_clusters=5, n_random_trials=10, theta=100, tau=200, n_steps=64, step_distance=0.05, n_joints=12, valid_depth_range=(100, 4000), depth_background_val=8000):
        if not (hasattr(sensor, 'project_points3d') and hasattr(sensor, 'create_pointcloud')):
            raise ValueError('sensor must have attribute \'project_points_3d\' and \'create_pointcloud\'')
        
        if min_samples_split < 0:
            raise ValueError('min_samples_split must be a non-negative number')
            
        if max_leaf_nodes <= 0:
            raise ValueError('max_leaf_nodes must be a positive number')
            
        if n_leaf_clusters <= 0:
            raise ValueError('n_leaf_clusters must be a positive number')
            
        if n_random_trials <= 0:
            raise ValueError('n_random_trials must be a positive number')
        
        if theta <= 0:
            raise ValueError('theta must be a positive number')
            
        if tau <= 0:
            raise ValueError('tau must be a positive number')
        
        if (n_steps < 1) or (not isinstance(n_steps, int)):
            raise ValueError('n_steps must be a positive integer')
            
        if step_distance <= 0:
            raise ValueError('step_distance must be a positive number')
            
        if min_samples_split < n_leaf_clusters:
            print('Warning: n_leaf_clusters must be less than min_samples_split'
                  ' got {0} but set to {1}'.format(n_leaf_clusters, min_samples_split))
            n_leaf_clusters = min_samples_split
        
        self._trees = []
        self.sensor = sensor
        self.model = model
        self.min_samples_split = min_samples_split
        self.max_leaf_nodes = max_leaf_nodes
        self.n_leaf_clusters = n_leaf_clusters
        self.n_random_trials = n_random_trials
        self.theta = theta
        self.tau = tau
        self.n_steps = n_steps
        self.step_distance = step_distance 
        self.n_joints = n_joints
        self.valid_depth_range = valid_depth_range
        self.depth_background_val = depth_background_val
        
    def _visualize_predict(self, depthmap, sensor, joint_positions, footprints_3d, footprints_2d):
        
        # 3d visualization
        from visualization import visualizePredict
        cloud = sensor.create_pointcloud(depthmap)
        visualizePredict(cloud, joint_positions, footprints_3d)
        
        depth_visualize = convert_16u_to_8u(depthmap, False, np.iinfo(np.uint16).max)
        depth_visualize = cv2.cvtColor(depth_visualize, cv2.COLOR_GRAY2BGR)

        for joint_position, footprint in zip(joint_positions, footprints_2d):

            center = np.floor(sensor.project_points3d(np.atleast_2d(joint_position)) + 0.5).astype(np.int16)
            cv2.circle(depth_visualize, (center[0, 0], center[0, 1]), radius=5, color=(0, 255, 0), thickness=cv2.FILLED)
            
            cv2.imshow('Random Tree Walk: press ESC key to exit', depth_visualize)
            cv2.waitKey(0)
            
            for i in range(len(footprint) - 1):
                cv2.line(depth_visualize, (footprint[i, 0], footprint[i, 1]), (footprint[i + 1, 0], footprint[i + 1, 1]), color=(0, 0, 255))
        
            cv2.imshow('Random Tree Walk: press ESC key to exit', depth_visualize)
            cv2.waitKey(0)
        
        while True:
            cv2.imshow('Random Tree Walk: press ESC key to exit', depth_visualize)
            if cv2.waitKey(30) == 27:
                break
        
        cv2.destroyAllWindows()
        
    ## Train dicision tree for random tree walk algorithm.
    #  depthmaps should be 3D array with shape [n_frames, rows, cols].
    #  offset_points and direction_vectors should be 4D array with shape [n_frames, n_joints, n_offsets, n_dimensions=3]
    #  @param depthmaps 3D array that contains 16-bit, single-channel depth maps
    #  @param offset_points 4D array that contains random offset points from specific true joint positions
    #  @param offset_points_projected 4D array that contains offset points projected into image plane
    #  @param direction_vectors 4D array that containes unit direction vectors from offset points to specific true joint positions
    def fit(self, depthmaps, offset_points_projected, direction_vectors):
        
        if len(depthmaps) != len(offset_points_projected):
           raise ValueError('depthmaps and offset_points_projected should have same length.')
        
        if offset_points_projected.shape[1] != self.n_joints or offset_points_projected.shape[3] != 2:
            raise ValueError('offset_points_projected should have shape of [n_frames, n_joints, n_offsets, n_dimensions=2].')
        
        if offset_points_projected.shape[:-1] != direction_vectors.shape[:-1]:
            raise ValueError('offset_points_projected and direction_vectors should have same shape except last axis.')
            
        # set outside of human segmentation to large constant value
        depths_fit = depthmaps.astype(np.float32, copy=False)
        depths_fit[(depths_fit < self.valid_depth_range[0]) | (self.valid_depth_range[1] < depths_fit)] = self.depth_background_val
        
        # convert depth map metric to meters
        depths_fit /= 1000.0
        
        for joint in range(1):#range(self.n_joints):
            # training samples
            offsets_projected_fit = np.floor(offset_points_projected[:, joint, :, :] + 0.5).astype(np.int16)
            directions_fit = direction_vectors[:, joint, :, :]
            
            # train tree for specific joint
            tree = DecisionTree(self.min_samples_split, self.max_leaf_nodes, self.n_leaf_clusters, self.n_random_trials, self.depth_background_val / 1000.0, self.theta, self.tau / 1000.0)
            tree.fit(depths_fit, offsets_projected_fit, directions_fit)
            
            self._trees.append(tree)
            
        return self
        
    def predict(self, depthmap, visualize=False):
        
        # set starting point to a centroid of point cloud
        start_point = np.nanmean(self.sensor.create_pointcloud(depthmap), axis=0)
            
        # set outside of human segmentation to large constant value
        depth_predict = depthmap.astype(np.float32)
        depth_predict[(depth_predict < self.valid_depth_range[0]) | (self.valid_depth_range[1] < depth_predict)] = self.depth_background_val
        
        # convert depth map metric to meters
        depth_predict /= 1000.0
        
        joint_positions = np.full((self.n_joints, 3), start_point)
        footprints_3d = []
        footprints_2d = []
        for tree_id, tree in enumerate(self._trees):
            
            # change start point according to skeletal topology
            start_point = joint_positions[self.model.adjacents[tree_id]]
            
            # predict joint positions
            joint_position, footprint_3d, footprint_2d = tree.predict(depth_predict, start_point, self.n_steps, self.step_distance, self.sensor)
            
            # update joint positions and footprints
            joint_positions[tree_id] = joint_position
            footprints_3d.append(footprint_3d)
            footprints_2d.append(footprint_2d)
            
        if visualize:
            self._visualize_predict(depthmap, self.sensor, joint_positions, footprints_3d, footprints_2d)
        
        return joint_positions
        
    def score(self, depthmaps, true_joints, correct_thresh=0.1):
        
        if depthmaps.shape[0] != true_joints.shape[0] or true_joints.shape[1] != self.n_joints or true_joints.shape[2] != 3:
            raise ValueError('depthmaps must be array-like, shape = (n_frames, rows, cols)\nand true_joints must be array-like, shape(n_frames, n_joints, 3)')
        
        n_total_joints = depthmaps.shape[0] * self.n_joints
        
        n_correct_joints = 0
        
        for depthmap, true_joint in zip(depthmaps, true_joints):
            estimated_joints = self.predict(depthmap)
            estimation_error = np.linalg.norm(estimated_joints - true_joint, axis=1)
            
            # if the estimated joint position is within a 10cm sphere, it is correct
            n_correct_joints += len(estimation_error[estimation_error <= correct_thresh])
            
        return float(n_correct_joints / n_total_joints)
            
    def save(self, filename):
        
        _, ext = splitext(filename)
        
        if ext != '.xml':
            raise ValueError('Currently, only XML format is supported.')
            
        # top element is random tree walk
        xml_random_tree_walk = ET.Element('random_tree_walk', {'n_trees':str(len(self._trees)), 'valid_depth_range_min':str(self.valid_depth_range[0]), 'valid_depth_range_max':str(self.valid_depth_range[1]), 'depth_background_val':str(self.depth_background_val)})
        
        # iterating through trees
        for tree_id, tree in enumerate(self._trees):
            tree.save(xml_random_tree_walk, tree_id)
        
        # write to file
        ET.ElementTree(xml_random_tree_walk).write(filename)
        
        return self
        
    def load(self, filename):
        
        _, ext = splitext(filename)
        
        if ext != '.xml':
            raise ValueError('Currently, only XML format is supported.')
            
        # parse xml from file
        xml_random_tree_walk = ET.parse(filename).getroot()
        
        # set valid depth range
        self.valid_depth_range = (int(xml_random_tree_walk.get('valid_depth_range_min')), int(xml_random_tree_walk.get('valid_depth_range_max')))
        
        # set depth background value
        self.depth_background_val = int(xml_random_tree_walk.get('depth_background_val'))
        
        # clear current trees for reading
        self._trees.clear()
        
        # get tree iterator
        xml_trees = xml_random_tree_walk.iter('tree')
        
        # iterating through trees
        for xml_tree in xml_trees:
            tree = DecisionTree()
            tree.load(xml_tree)
            self._trees.append(tree)
        
        return self