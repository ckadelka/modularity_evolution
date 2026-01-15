# -*- coding: utf-8 -*-
"""
Created on Sat Jan  3 00:53:29 2026

@author: faith
"""
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
#find it (master_data_file.csv) in the two potential paths
path_in_code_dir = os.path.join(script_dir, "master_data_file.csv")
path_in_parent_dir = os.path.abspath(os.path.join(script_dir, "..", "master_data_file.csv"))

#the directory where the master_data_file.csv is is very important...
GLOBAL_MASTER_DIR = ""

if os.path.exists(path_in_code_dir):
    print("master file is in the code dir")
    GLOBAL_MASTER_DIR = os.path.dirname(path_in_code_dir)
elif os.path.exists(path_in_parent_dir):
    print("master file is in the parent (src) dir")
    GLOBAL_MASTER_DIR = os.path.dirname(path_in_parent_dir)