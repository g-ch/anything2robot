from solid import *
from solid.utils import *
import math
import os


class SixFoldPlatesFilling:
    def __init__(self, height, width, thickness, interval, plates_num, tilt_angle, input_stl_path, output_stl_path):
        self.height = height
        self.width = width
        self.thickness = thickness
        self.interval = interval
        self.plates_num = plates_num
        self.tilt_angle = math.radians(tilt_angle)
        self.input_stl_path = input_stl_path
        self.output_stl_path = output_stl_path
    
    # Generate the final model and export to an STL file
    def generate_model(self):
        # Generate the SCAD file
        scad_file = 'final_model.scad'
        scad_render_to_file(self.final_model(), scad_file)
        
        # Use OpenSCAD's command-line interface to convert the SCAD file to an STL file
        os.system(f'openscad -o {self.output_stl_path} {scad_file}')
        

    # Create the board centered at (0,0,0)
    def create_board(self):
        return cube([self.thickness, self.width, self.height], center=True)

    # Tilt the board by 30 degrees along the y-axis and position it
    def tilted_board(self, x_position):
        transform = [
            [math.cos(self.tilt_angle), 0, -math.sin(self.tilt_angle), x_position],
            [0, 1, 0, 0],
            [math.sin(self.tilt_angle), 0, math.cos(self.tilt_angle), 0],
            [0, 0, 0, 1]
        ]
        return multmatrix(transform)(self.create_board())

    # Create multiple boards along +x and -x directions
    def create_all_boards(self):
        return [self.tilted_board(i * (self.thickness + self.interval)) for i in range(-self.plates_num, self.plates_num + 1)]

    # Combine all boards into one mesh iteratively with unions after some boards are generated
    def combined_boards(self):
        combined = union()(*self.create_all_boards())
        for angle in [60, 120, 180, 240, 300]:
            rotated_boards = rotate([0, 0, angle])(self.create_all_boards())
            combined = union()(combined, rotated_boards)
        return combined

    # Cut out the cube space from the combined mesh
    def final_model(self):
        combined = self.combined_boards()
        stl_import = scale([1000, 1000, 1000])(import_stl(self.input_stl_path))
        return intersection()(combined, stl_import)


if __name__ == '__main__':
    # Define the dimensions of the board
    height = 200    # z dimension
    width = 200     # y dimension
    thickness = 0.4 # x dimension
    interval = 15   # Interval between boards
    plates_num = 10
    tilt_angle = 30  # Convert degrees to radians

    # Input STL file path
    input_stl_path = '/home/clarence/ros_ws/metamaterial_ws/src/metamaterial_filling/data/FL_replaced.stl'

    # Output STL file name
    output_stl_path = '/home/clarence/ros_ws/metamaterial_ws/src/metamaterial_filling/data/FL_final_output.stl'

    # Create the SixFoldPlatesFilling object
    sixFoldPlatesFilling = SixFoldPlatesFilling(height, width, thickness, interval, plates_num, tilt_angle, input_stl_path, output_stl_path)
    sixFoldPlatesFilling.generate_model()


# # Define the dimensions of the board
# height = 200    # z dimension
# width = 200     # y dimension
# thickness = 0.4 # x dimension
# interval = 15   # Interval between boards
# plates_num = 10
# tilt_angle = math.radians(30)  # Convert degrees to radians

# # Input STL file path
# input_stl_path = '/home/clarence/ros_ws/metamaterial_ws/src/metamaterial_filling/data/FL_replaced.stl'

# # Output STL file name
# output_stl_path = '/home/clarence/ros_ws/metamaterial_ws/src/metamaterial_filling/data/FL_final_output.stl'


# # Create the board centered at (0,0,0)
# def create_board():
#     return cube([thickness, width, height], center=True)

# # Tilt the board by 30 degrees along the y-axis and position it
# def tilted_board(x_position):
#     transform = [
#         [math.cos(tilt_angle), 0, -math.sin(tilt_angle), x_position],
#         [0, 1, 0, 0],
#         [math.sin(tilt_angle), 0, math.cos(tilt_angle), 0],
#         [0, 0, 0, 1]
#     ]
#     return multmatrix(transform)(create_board())

# # Create multiple boards along +x and -x directions
# def create_all_boards():
#     return [tilted_board(i * (thickness + interval)) for i in range(-plates_num, plates_num + 1)]

# # Combine all boards into one mesh iteratively with unions after some boards are generated
# def combined_boards():
#     combined = union()(*create_all_boards())
#     for angle in [60, 120, 180, 240, 300]:
#         rotated_boards = rotate([0, 0, angle])(create_all_boards())
#         combined = union()(combined, rotated_boards)
#     return combined

# # Cut out the cube space from the combined mesh
# def final_model():
#     combined = combined_boards()
#     stl_import = scale([1000, 1000, 1000])(import_stl(input_stl_path))
#     return intersection()(combined, stl_import)


# Generate the final model and export to an STL file
# if __name__ == '__main__':
#     # Generate the SCAD file
#     scad_file = 'final_model.scad'
#     scad_render_to_file(final_model(), scad_file)
    
#     # Use OpenSCAD's command-line interface to convert the SCAD file to an STL file
#     os.system(f'openscad -o {output_stl_path} {scad_file}')