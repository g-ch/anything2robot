import os
import argparse
import numpy as np
import pickle as pkl
import time
# Add dependencies path
import sys
sys.path.append(os.path.normpath('./auto_design/modules'))

from modules.data_struct import *
from modules.mesh_loader import Custom_Mesh_Loader
from modules.mesh_decomp import Mesh_Decomp
from modules.motor_opt import Motor_Opt, Joint_Connect_Opt, get_bounds
from modules.interference_removal import InterferenceRemoval, RobotOptResult

if __name__=="__main__":
    robot_result = pkl.load(open('./auto_design/results/lynel20240819-162039_robot_result.pkl', 'rb'))
    robot_result.mesh_group.render()