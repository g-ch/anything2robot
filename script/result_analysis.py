import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt


def decode_log_pkl(log_pkl_path):
    with open(log_pkl_path, 'rb') as f:
        log_dict = pkl.load(f)
    
    return log_dict


if __name__ == '__main__':
    log_pkl_path = '/media/clarence/Clarence/anything2robot/result/n02086240_323_neutral_res_e300_smoothed_scaled_20241025-235516/result_round5/round5_variable_exit_code_0.pkl'
    log_dict = decode_log_pkl(log_pkl_path)
    print(log_dict)
