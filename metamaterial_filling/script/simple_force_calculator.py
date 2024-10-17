import numpy as np


'''
Calculate the force from the applied torque on the link.
apply_torque: the torque applied on the link. E.g.
array([-0.10141549,  0.19332437, -0.27475526,  1.52129645,  0.        ,
        0.        ]), array([ 0.11239833, -0.21169054,  0.13358129,  2.44728243,  0.        ,
        0.        ])
'''
def calculate_force_from_nodes_and_torques(applied_torque):
    # 