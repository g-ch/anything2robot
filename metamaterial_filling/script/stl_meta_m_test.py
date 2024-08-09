import os
import argparse
import subprocess
from metamaterial.sixFoldPlatesFillingWithShell import SixFoldPlatesFillingWithShell
import time
import numpy as np
import math
from format_transform.off_to_stl import off_to_stl

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_stl_path', type=str, default='data/FL.stl', help='Input STL file path')
    parser.add_argument('--unit', type=str, default='m', choices=['mm', 'm'], help='Unit of the model. If the unit is in meter, we will scale the model to mm.')
    parser.add_argument('--relative_density', type=float, default=0.1, help='Relative density of the metamaterial given by FEA results')
    parser.add_argument('--shell_thickness', type=float, default=2, help='Thickness of the shell. mm')
    parser.add_argument('--shell_error_bound', type=float, default=0.5, help='Error bound for the shell thickness. mm')
    parser.add_argument('--output_stl_name', type=str, default='FL_final_output_with_shell.stl', help='Output STL file path')
    parser.add_argument('--use_existing_shell', type=bool, default=True, help='Whether to use the existing shell file')

    args = parser.parse_args()

    # Check if the input STL file exists
    if not os.path.exists(args.input_stl_path):
        raise FileNotFoundError(f'Input STL file not found at {args.input_stl_path}')
    
    current_dir = os.path.dirname(os.path.realpath(__file__))
    cmake_build_dir = os.path.join(current_dir, '../build')
    output_folder = os.path.join(current_dir, '../data/output')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    relative_density = args.relative_density
    if relative_density < 0.1:
        relative_density = 0.1
        print('Relative density should be larger than 0.1, set to 0.1')
    elif relative_density >= 1.0:
        raise ValueError('Relative density should be less than 1.0')

    ###### Replace and scale the stl if necessary ######
    replaced_stl_m_name = args.input_stl_path.split('/')[-1].split('.')[0] + '_replaced_m.stl'
    replaced_stl_m_save_path = os.path.join(output_folder, replaced_stl_m_name)
    replaced_stl_m_size_csv = replaced_stl_m_save_path.replace('.stl', '.csv')

    input_stl_path_full_path = os.path.join(current_dir, '../', args.input_stl_path)

    print(f"input_stl_path_full_path: {input_stl_path_full_path}")
    print(f"replaced_stl_m_save_path: {replaced_stl_m_save_path}")

    # Remove the replaced file if it exists
    if os.path.exists(replaced_stl_m_save_path):
        os.remove(replaced_stl_m_save_path)
        os.remove(replaced_stl_m_size_csv)

    if args.unit == 'm':
        # Use gnome-terminal to run the command
        subprocess.run(['gnome-terminal', '--', 'bash', '-c', f'cd {cmake_build_dir}; ./replaceMesh {input_stl_path_full_path} {replaced_stl_m_save_path}'])
    else:
        # Use gnome-terminal to run the command
        subprocess.run(['gnome-terminal', '--', 'bash', '-c', f'cd {cmake_build_dir}; ./replaceMesh {input_stl_path_full_path} {replaced_stl_m_save_path} 0.001'])

    # Wait for the process to finish
    while not os.path.exists(replaced_stl_m_save_path):
        time.sleep(1)
    
    time.sleep(1)

    # Read the size of the replaced STL file from the CSV file. Two lines, the first line is min_x,min_y,min_z, the second line is max_x,max_y,max_z
    with open(replaced_stl_m_size_csv, 'r') as f:
        lines = f.readlines()
        min_x, min_y, min_z = lines[0].split(',')
        max_x, max_y, max_z = lines[1].split(',')
        min_x, min_y, min_z = float(min_x), float(min_y), float(min_z)
        max_x, max_y, max_z = float(max_x), float(max_y), float(max_z)

        print(f"min_x: {min_x} m, min_y: {min_y} m, min_z: {min_z} m")
        print(f"max_x: {max_x} m, max_y: {max_y} m, max_z: {max_z} m")
        # print(f"min_x: {min_x} mm, min_y: {min_y} mm, min_z: {min_z} mm")
        # print(f"max_x: {max_x} mm, max_y: {max_y} mm, max_z: {max_z} mm")
    

    ###### Generate the shell ######
    shell_off_name = args.input_stl_path.split('/')[-1].split('.')[0] + '_shell.off'
    shell_off_save_path = os.path.join(output_folder, shell_off_name)

    # Remove the shell file if it exists and we are not using the existing shell
    do_shell_generation = True
    if os.path.exists(shell_off_save_path):
        if args.use_existing_shell:
            print(f"Using existing shell file at {shell_off_save_path}")
            do_shell_generation = False
        else:
            os.remove(shell_off_save_path)

    if do_shell_generation:
        # Use gnome-terminal to run the command

        shell_thickness = args.shell_thickness / 1000  # Convert to meter so that the shell generation program can be faster
        shell_error_bound = args.shell_error_bound / 1000 # Convert to meter so that the shell generation program can be faster

        subprocess.run(['gnome-terminal', '--', 'bash', '-c', f'cd {cmake_build_dir}; ./shellGeneration {replaced_stl_m_save_path} {shell_off_save_path} {shell_thickness} {shell_error_bound}'])

        # Wait for the process to finish
        while not os.path.exists(shell_off_save_path):
            time.sleep(1)
        
        print(f"Shell OFF file generated at {shell_off_save_path}")
    

    ###### Convert the shell to STL ######
    shell_stl_save_path = shell_off_save_path.replace('.off', '.stl')

    off_to_stl(shell_off_save_path, shell_stl_save_path)

    print(f"Shell STL file generated at {shell_stl_save_path}")

    exit(0)

    ##### TODO: Turn shell and original stl to mm ######

    ########## Generate the final model ##########
    tilt_angle = math.radians(30)
    final_output_stl_path = os.path.join(output_folder, args.output_stl_name)

    # Change min_x etc. to mm
    min_x *= 1000
    max_x *= 1000
    min_y *= 1000
    max_y *= 1000
    min_z *= 1000
    max_z *= 1000

    # Define the dimensions of the board
    safe_scale = 1.5
    height = (max_z - min_z) * safe_scale
    delf_x = max_x - min_x
    delf_y = max_y - min_y
    width = max(delf_x, delf_y) * safe_scale

    interval = 10
    thickness = relative_density / 6.0 * interval

    # Check if thickness is smaller than 0.2
    if thickness < 0.2:
        thickness = 0.2
        print('Thickness should be larger than 0.2, set to 0.2')
        # Correct the interval
        interval = thickness * 6.0 / relative_density
    
    plates_num = int(width / (thickness + interval) / 2)

    print("Generating the final model... Parameters:")
    print(f"height: {height}, width: {width}, thickness: {thickness}, interval: {interval}, plates_num: {plates_num}, tilt_angle: {tilt_angle}")

    sixFoldPlatesFilling = SixFoldPlatesFillingWithShell(height, width, thickness, interval, plates_num, tilt_angle, replaced_stl_m_save_path, shell_stl_save_path, final_output_stl_path)
    sixFoldPlatesFilling.generate_model()

    print(f"Final model generated at {final_output_stl_path}")
