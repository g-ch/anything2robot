import numpy as np


'''
Calculate the force from the applied torque on the link.
@tenon_positions: the position of the tenons on the link from pkl result.
@apply_torque: the torque applied on the link from pkl result. 
@return: the force list on the tenons. Scalar
'''
def calculate_forces_from_nodes_and_torques(tenon_positions, applied_torque):
    # Calculate the center of the tenon positions
    center = np.mean(tenon_positions, axis=0)
    center = center[:3]
    #print("center: ", center)

    forces=[]
    avg_toque = 0
    for i in range(len(applied_torque)):
        avg_toque += np.linalg.norm(applied_torque[i][3:])
    avg_toque /= len(applied_torque)

    for i in range(len(applied_torque)):
        pos = tenon_positions[i][:3]
        distance = np.linalg.norm(pos - center)
        torque = avg_toque #np.linalg.norm(applied_torque[i][3:])
        if distance == 0:
            force = 0
        else:
            force = torque / distance
        forces.append(force)

    return forces


if __name__ == "__main__":
    tenon_positions = [np.array([0.13691549, -0.19332437, 0.27475526, -1., -0., -0.]), np.array([0.07689833, -0.21169054, 0.13358129, 1., 0., 0.])]
    applied_torque = [np.array([-0.10141549, 0.19332437, -0.27475526, 1.52129645, 0., 0.]), np.array([0.11239833, -0.21169054, 0.13358129, 2.44728243, 0., 0.])]
    calculate_forces_from_nodes_and_torques(tenon_positions, applied_torque)