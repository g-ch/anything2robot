import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import os
import sys

project_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)
sys.path.append(os.path.normpath(os.path.join(project_path, 'auto_design')))
sys.path.append(os.path.normpath(os.path.join(project_path, 'auto_design/modules')))
sys.path.append(os.path.normpath(os.path.join(project_path, 'metamaterial_filling/script')))

def decode_log_pkl(log_pkl_path):
    with open(log_pkl_path, 'rb') as f:
        log_dict = pkl.load(f)
    
    return log_dict


if __name__ == '__main__':
    log_pkl_path = 'result/urdf/n02090622_6293_neutral_res_e300_smoothed_scaled_20241025-205751/result_round1/robot_result.pkl'
    log_dict = decode_log_pkl(log_pkl_path)
    print(log_dict)
