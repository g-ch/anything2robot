import pickle as pkl
from interference_removal import RobotOptResult, LinkResult


robot_result = pkl.load(open('./results/lynel_robot_result.pkl', 'rb'))
for link_name in robot_result.link_dict:
    print("Link name: ", link_name)
    print("Link Tenon Positions: ", robot_result.link_dict[link_name].tenon_pos)
    print("Link Torques: ", robot_result.link_dict[link_name].applied_torque)