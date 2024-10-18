import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt


def decode_log_pkl(log_pkl_path):
    with open(log_pkl_path, 'rb') as f:
        log_dict = pkl.load(f)
    
    return log_dict


if __name__ == '__main__':
    log_pkl_path = '/media/clarence/Clarence/anything2robot/result/gold_lynel_20241018-170744/result_round3/round3_variable_exit_code_0.pkl'
    log_dict = decode_log_pkl(log_pkl_path)
    print(log_dict)
