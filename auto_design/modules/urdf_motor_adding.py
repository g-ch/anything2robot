import pickle as pkl
import numpy as np
np.int = int  # Monkey patch for compatibility
np.float = float

import trimesh
import pickle as pkl
from urdfpy import URDF, Link, Joint, Geometry, Visual, Mesh, Collision, Inertial
from scipy.spatial.transform import Rotation as R

import os
import sys
import shutil
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_dir + '/auto_design/modules')
sys.path.append(project_dir + '/auto_design')
sys.path.append(project_dir)

from script.motor_param_lib import MotorParameterLib
from interference_removal import RobotOptResult, LinkResult

motor_param_lib = MotorParameterLib()

def rotation_matrix_from_vectors(vec1, vec2):
    """
    Returns a rotation matrix that aligns vec1 to vec2.
    """
    a = vec1 / np.linalg.norm(vec1)
    b = vec2 / np.linalg.norm(vec2)

    cross = np.cross(a, b)
    dot = np.dot(a, b)

    if np.isclose(dot, 1.0):
        # Vectors are the same
        return np.eye(3)
    elif np.isclose(dot, -1.0):
        # Vectors are opposite — rotate 180° around any perpendicular axis
        perp = np.array([1, 0, 0]) if not np.allclose(a, [1, 0, 0]) else np.array([0, 1, 0])
        axis = np.cross(a, perp)
        axis = axis / np.linalg.norm(axis)
        return R.from_rotvec(np.pi * axis).as_matrix()

    # General case
    skew = np.array([
        [    0, -cross[2],  cross[1]],
        [ cross[2],     0, -cross[0]],
        [-cross[1], cross[0],    0]
    ])

    return np.eye(3) + skew + skew @ skew * ((1 - dot) / (np.linalg.norm(cross) ** 2))


def fix_stl_path_issue(urdf_path, output_path):
    """
    Fix the stl path issue in the urdf file, change absolute path to relative path.
    """
    with open(urdf_path, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if 'clarence' in line or 'package://' in line:
            # Extract the filename from the line
            filename = line.split('"')[1].split('/')[-1]
            print(f"Fixing stl path issue: {filename}")
            # Replace the line with the new path
            lines[i] = f'           <mesh filename="{filename}" />\n'

    # Write the modified lines back to the file
    with open(output_path, 'w') as file:
        file.writelines(lines)

def create_motor_cylinder(tenon_idx):
    """
    Create a cylinder mesh for a motor based on the tenon index and motor library.
    """
    # Choose size based on tenon index
    # Height (cm), Radius (cm), Max Torque (N*M) 
    # self.motor_lib = [        
    #                   #[2.25, 2.15, 0.9], # GIM3505. SITAIWEI
    #                   [3.42, 2.65, 2.5], # MG4005V2. K-Tech
    #                   [3.65, 3.8, 12],   # DM6006. DAMIAO Tech 
    #                   [4.0, 4.8, 20], # DM8006. DAMIAO Tech
    #                   [6.2, 5.56, 120], # DM10010. DAMIAO Tech
    #                  ]  
    sizes = motor_param_lib.get_motor_lib()
    radius = sizes[tenon_idx][1] * 0.01 # cm to m
    height = sizes[tenon_idx][0] * 0.01 # cm to m
    
    # Create cylinder mesh along z-axis
    motor_mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=32)
    return motor_mesh

def get_link_global_transform(robot, link_name):
    """
    Returns the global transformation matrix for a given link by traversing up the kinematic chain.
    """
    # Base case: root link has identity transform
    if link_name == robot.links[0].name:
        return np.eye(4)

    # Find the joint that connects to this link and follow the chain up
    joint_chain = []
    current_link = link_name

    # Print the name of links[0]
    print(f"links[0].name: {robot.links[0].name}")
    
    # Build chain of joints from current link to root
    while current_link != robot.links[0].name:
        found_joint = None
        for joint in robot.joints:
            if joint.child == current_link:
                found_joint = joint
                joint_chain.append(joint)
                current_link = joint.parent
                break
        if not found_joint:
            raise ValueError(f"Link '{current_link}' not connected to tree")

    # Compute transformation by multiplying joint transforms from root to tip
    transform = np.eye(4)
    for joint in reversed(joint_chain):
        transform = transform @ joint.origin

    return transform

def add_motor_to_urdf(urdf_path, pkl_result_path, output_urdf_folder):
    robot = URDF.load(urdf_path)
    robot_result = pkl.load(open(pkl_result_path, 'rb'))

    if not os.path.exists(output_urdf_folder):
        print("Output urdf folder does not exist, shouldn't happen")
        return
    
    extra_links = []
    extra_joints = []
    
    # Iterate through all links in the robot result and add motor to the links with father tenon
    for link_name, link in robot_result.link_dict.items():
        for i, tenon_type in enumerate(link.tenon_type):
            #if 'BODY' in link_name or 'UP' in link_name:
            if True: #NOTE: All the names will be covered but motors maybe duplicated. For visualization only.
                print("Adding motor to link: ", link_name)
                # Extract tenon info
                tenon_root_point = np.array(link.tenon_pos[i][:3])
                tenon_root_dir = np.array(link.tenon_pos[i][3:6])
                tenon_root_dir = tenon_root_dir / np.linalg.norm(tenon_root_dir)
                tenon_idx = link.tenon_idx[i]

                # Create motor mesh if not exists
                mesh_filename = f"motor_{tenon_idx}.stl"
                mesh_path = os.path.join(output_urdf_folder, mesh_filename)

                # Check if the mesh file already exists
                if os.path.exists(mesh_path):
                    # load the mesh directly
                    mesh = trimesh.load(mesh_path)
                else:
                    mesh = create_motor_cylinder(tenon_idx)
                    mesh.export(mesh_path)

                # Compute rotation to align z-axis with tenon_root_dir
                z_axis = np.array([0, 0, 1])
                # print(f"tenon_root_dir: {tenon_root_dir}")
                rot_matrix = rotation_matrix_from_vectors(z_axis, tenon_root_dir)
                print(f"rot_matrix: {rot_matrix}")
                
                # Transform: rotation + translation
                sizes = motor_param_lib.get_motor_lib()
                motor_height = sizes[tenon_idx][0] * 0.01 # cm to m
                transform = np.eye(4)
                transform[:3, :3] = rot_matrix
                transform[:3, 3] = tenon_root_point + motor_height * tenon_root_dir * 0.5

                T_world_motor = transform
                T_world_parent = get_link_global_transform(robot, link_name)

                print(f"T_world_parent: {T_world_parent}")
                print(f"T_world_motor: {T_world_motor}")
                
                T_parent_motor = np.linalg.inv(T_world_parent) @ T_world_motor
                print(f"T_parent_motor: {T_parent_motor}")

                zero_origin = np.eye(4)

                # Create new link
                motor_link_name = f"{link_name}_motor_{i}"
                visual = Visual(
                    geometry=Geometry(mesh=Mesh(filename=mesh_path)),
                    origin=zero_origin
                )
                # Add collision
                collision = Collision(
                    name=f"{motor_link_name}_collision",
                    geometry=Geometry(mesh=Mesh(filename=mesh_path)),
                    origin=zero_origin
                )
                # Add inertial
                inertial = Inertial(
                    mass=0.01,
                    inertia=np.eye(3) * 0.0001,
                    origin=zero_origin
                )

                new_link = Link(
                    name=motor_link_name,
                    visuals=[visual],
                    inertial=inertial,
                    collisions=[collision]
                )

                # Create fixed joint
                joint = Joint(
                    name=f"{motor_link_name}_joint",
                    parent=link_name, #"BODY", #link_name,
                    child=motor_link_name,
                    joint_type='fixed',
                    origin=T_parent_motor
                )

                # Add to robot as extra links and joints
                extra_links.append(new_link)
                extra_joints.append(joint)

    robot_with_motors = URDF(
        name=robot.name,
        links=robot.links + extra_links,
        joints=robot.joints + extra_joints,
        transmissions=robot.transmissions,
        materials=robot.materials
    )

    # List all links
    for link in robot_with_motors.links:
        print(f"Link: {link.name}")

    # List all joints
    for joint in robot_with_motors.joints:
        print(f"Joint: {joint.name} - Parent: {joint.parent} -> Child: {joint.child}")

    # Save the robot
    output_urdf_path = os.path.join(output_urdf_folder, 'robot_with_motors.urdf')
    robot_with_motors.save(output_urdf_path)

    # Copy the stl files to the output folder
    stl_folder = os.path.dirname(urdf_path)
    for mesh_filename in os.listdir(stl_folder):
        if mesh_filename.endswith('.stl'):
            shutil.copy(os.path.join(stl_folder, mesh_filename), os.path.join(output_urdf_folder, mesh_filename))
    
    # Fix the stl path issue
    fix_stl_path_issue(output_urdf_path, output_urdf_path)

    print(f"Done! The urdf with motors is saved to {output_urdf_path}")


if __name__ == "__main__":
    # urdf_path = "/media/clarence/Clarence/anything2robot_data/result/n02086646_422_neutral_res_e300_smoothed_scaled_20241031-014549/result_round1/urdf/n02086646_422_neutral_res_e300_smoothed_scaled20241031-014817.urdf"
    # robot_pkl_path = "/media/clarence/Clarence/anything2robot_data/result/n02086646_422_neutral_res_e300_smoothed_scaled_20241031-014549/result_round1/robot_result.pkl"
    
    robot_pkl_path = "/media/clarence/Clarence/anything2robot_data/gold_lynel_20241201-134522_good/result_round1/robot_result.pkl"
    urdf_path = "/media/clarence/Clarence/anything2robot_data/gold_lynel_20241201-134522_good/result_round1/urdf/gold_lynel20241201-162205.urdf"
    path_fixed_urdf_path = urdf_path.replace('.urdf', '_fixed.urdf')

    # Get the parent parent folder of the urdf path
    output_urdf_folder = os.path.join(os.path.dirname(os.path.dirname(path_fixed_urdf_path)), 'urdf_with_motors')

    # Create the output urdf path if not exists
    if not os.path.exists(output_urdf_folder):
        os.makedirs(output_urdf_folder)

    # Fix the stl path issue
    fix_stl_path_issue(urdf_path, path_fixed_urdf_path)  

    # Add motor to the urdf
    add_motor_to_urdf(path_fixed_urdf_path, robot_pkl_path, output_urdf_folder)

