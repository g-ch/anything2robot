from interference_removal import RobotOptResult
import numpy as np
import pickle as pkl
import os
import sys
sys.path.append(os.path.normpath('./auto_design/'))
robot_result = pkl.load(open('/media/clarence/Clarence/anything2robot_data/result/n02085782_28_neutral_res_e300_smoothed_scaled_20241031-030957/result_round7/robot_result.pkl', 'rb'))
robot_result.getMeshSimilarity(stl_dir="/media/clarence/Clarence/anything2robot_data/result/n02085782_28_neutral_res_e300_smoothed_scaled_20241031-030957/result_round7/scaled_model_expected_x_38.974342000000014.stl")
