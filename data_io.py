# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 00:02:58 2016

@author: owner
"""

import re
import cv2
import numpy as np
from decimal import Decimal

def is_string_like(obj):
    try:
        obj + ''
    except (TypeError, ValueError):
        return False
    return True
    
def array2string(array):

    if array.ndim != 1 and array.ndim != 2:
        raise ValueError('This function only supports array with ndim=1 or ndim=2')
    
    string = ''
    
    if array.ndim == 1:
        n_elems = array.shape[0]
        
        for n in range(n_elems):
            string += str(Decimal(float(array[n]))) if issubclass(array.dtype.type, np.floating) else str(Decimal(int(array[n])))
            if n != n_elems - 1:
                string += ' '
    else:
        rows = array.shape[0]
        cols = array.shape[1]
        
        for row in range(rows):
            for col in range(cols):
                string += str(Decimal(float(array[row, col]))) if issubclass(array.dtype.type, np.floating) else str(Decimal(int(array([row, col]))))
                if col != (cols - 1):
                    string += ' '
            if row != rows - 1:
                string += '\n'
    
    return string
    
def string2array2D(rows, cols, string, dtype=np.float64):
    
    array = np.zeros([rows, cols], dtype)
    
    string_rows = string.split('\n')
    
    for row, string_row in enumerate(string_rows):
        string_cols = string_row.split(' ')
        for col, string_col in enumerate(string_cols):
            array[row, col] = float(string_col)
    
    return array
    
def string2array1D(length, string, dtype=np.float64):
    
    array = np.zeros([length], dtype)
    
    elems = string.split(' ')
    
    for n, elem in enumerate(elems):
        array[n] = float(elem)
        
    return array

## Save numpy ndarray with ndim==3 to text file
#  @param filename Output text file name
#  @param data Array object to save
def saveText3D(filename, data):
    
    with open(filename, 'wb') as outfile:
        
        # writing a header just for the sake of readability
        outfile.write(bytes('# Array shape: {0}\n'.format(data.shape), 'utf-8'))
        
        # iterating through an array along the last axis
        for data_slice in data:
            
            # writing data into outfile
            np.savetxt(outfile, data_slice)
            
            # writing out a break to indicate different slices
            outfile.write(bytes('# New slice\n', 'utf-8'))
            
## Read random offset points
#  @param filenames File names that contains depth data
#  @param array_shape Array shape of reading data. [n_joints, n_offsets, n_dimensions]
#  @return offset_points list of 3D array that contains random offset points from true joint positions.
def readText3D(filenames):
    
    shape_check = re.compile('\(.*\)')
    
    data = []
    for filename in filenames:
        # check array shape
        with open(filename, 'rb') as infile:
            header = infile.readline().decode('utf-8')
            array_shape_str = shape_check.search(header).group(0)
            array_shape = tuple([int(x) for x in  array_shape_str[1:-1].split(', ')])
        
        data_slice = np.loadtxt(filename)
        data_slice = data_slice.reshape(array_shape)
        data.append(data_slice)
        
    return data

## Read numpy ndarray with ndim==2 from text file
#  @param filenames File names that contains silhouette data
#  @return joints 2d array that contains 3D joint positions
def saveText2D(filename, data):
    
    np.savetxt(filename, data)

## Read numpy ndarray with ndim==2 from text file
#  @param filenames File names that contains silhouette data
#  @return joints 2d array that contains 3D joint positions
def readText2D(filenames):
    
    data = []
    for filename in filenames:
        data_slice = np.loadtxt(filename)
        data.append(data_slice)
        
    return data
    
## Read depth data
#  @param filenames File names that contains depth data
#  @return depthmaps 16-bit, single-channel depth maps
def readDepthMap(filenames):
    
    depthmaps = []
    for filename in filenames:
        img = cv2.imread(filename, cv2.IMREAD_ANYDEPTH)
        depthmaps.append(img)
        
    return depthmaps
    
## Read joint and depth data for pose estimation
#  @param joint_prefix Directory that contains reading file.
#  @param depth_prefix
#  @param frame_begin Frame number that reading start with.
#  @param frame_end Frame number that reading stop with.
#  @param frame_step Step size of reading.
#  @param joint_extension
#  @param depth_extension
#  @return joints 2d array that contains 3D joint positions
#  @return depthmaps List of 16-bit, single-channel depth maps
def readDepthAndJoints(depth_prefix, joints_prefix, frame_begin, frame_end, frame_step, depth_extension, joints_extension):
    
    if(frame_step == 0):
        raise ValueError('frame_step must not be 0')
    
    # file names
    depth_names = []
    joints_names = []    
    
    for i in range(frame_begin, frame_end + 1, frame_step):        
        depth_names.append(depth_prefix + str(i) + depth_extension)
        joints_names.append(joints_prefix + str(i) + joints_extension)
    
    depthmaps = readDepthMap(depth_names)
    joints = readText2D(joints_names)
    
    return depthmaps, joints