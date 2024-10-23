import sys
import os
sys.path.append(os.path.normpath('./auto_design/'))
import pickle as pkl
import numpy as np
from interference_removal import RobotOptResult, LinkResult, InterferenceRemoval

# auto_design/results/gold_lynel20241010-101912_robot_result.pkl
robot_result = pkl.load(open('./auto_design/results/lynel.pkl', 'rb'))


H = np.array([
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

# Transform the link tree
cur_idx = 0
queue = [robot_result.link_tree]
while queue:
    current_node = queue.pop(0)
    current_link = current_node.val
    for child_node in current_node.children:
        queue.append(child_node)
    if current_link.axis is None or np.linalg.norm(current_link.axis[1]) == 0:
        continue
    
    point = np.append(current_link.axis[0], 1)
    transformed_point = H @ point
    current_link.axis[0] = transformed_point[:3]

    point = np.append(current_link.axis[1], 1)
    transformed_point = H @ point
    current_link.axis[1] = transformed_point[:3]

    if len(current_link.axis) == 3:
        point = np.append(current_link.axis[2], 1)
        transformed_point = H @ point
        current_link.axis[2] = transformed_point[:3]

    for joint_name in current_link.joints.keys():
        point = np.append(current_link.joints[joint_name], 1)
        transformed_point = H @ point
        current_link.joints[joint_name] = transformed_point[:3]



# Change axis of motor_param
def transfor_motor_param(motor_params):
    for motor_param in motor_params:
        point1 = np.append(motor_param[:3], 1)
        point2 = np.append(motor_param[3:6], 1)
        transformed_point1 = H @ point1
        transformed_point2 = H @ point2
        motor_param[:3] = transformed_point1[:3]
        motor_param[3:6] = transformed_point2[:3]

    return motor_params


transformed_motor_param_result = transfor_motor_param(robot_result.motor_results)
interference_removal = InterferenceRemoval(args=robot_result.args, 
                                           mesh_group=robot_result.mesh_group,
                                           motor_param_result=transformed_motor_param_result,
                                           link_tree=robot_result.link_tree,
                                           father_link_dict=robot_result.father_link_dict)

# Change axis of mesh group
interference_removal.mesh_group.voxel_data = np.einsum('ijk->jik', interference_removal.mesh_group.voxel_data)
interference_removal.mesh_group.voxel_data = np.flip(interference_removal.mesh_group.voxel_data, 0)
exchange = interference_removal.mesh_group.x_range
interference_removal.mesh_group.x_range = interference_removal.mesh_group.y_range
interference_removal.mesh_group.y_range = exchange




interference_removal.generate_champ_urdf()
