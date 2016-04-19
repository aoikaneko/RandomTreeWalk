# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 18:51:01 2016

@author: owner
"""

class SkeletonModel:
    __slots__ = ['joints', 'adjacents']
    
    def __init__(self):
        
        # joints
        j_dict = {'chest':0, 'head':1, 'r-shoulder':2, 'r-elbow':3, 'r-hand':4,
                  'l-shoulder':5, 'l-elbow':6, 'l-hand':7, 'r-knee':8, 'r-foot':9, 'l-knee':10, 'l-foot':11,}
    
        # adjacent joint pairs
        adj_dict = {j_dict['chest']:j_dict['chest'], j_dict['head']:j_dict['chest'],
                    j_dict['r-shoulder']:j_dict['chest'], j_dict['r-elbow']:j_dict['r-shoulder'], j_dict['r-hand']:j_dict['r-elbow'],
                    j_dict['l-shoulder']:j_dict['chest'], j_dict['l-elbow']:j_dict['l-shoulder'], j_dict['l-hand']:j_dict['l-elbow'],
                    j_dict['r-knee']:j_dict['r-knee'], j_dict['r-foot']:j_dict['r-knee'],
                    j_dict['l-knee']:j_dict['l-knee'], j_dict['l-foot']:j_dict['l-knee']}
        
        self.joints = j_dict
        self.adjacents = adj_dict