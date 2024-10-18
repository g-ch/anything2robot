from solid import *
from solid.utils import *
import math
import os


class SixFoldPlatesFillingWithShell:
    def __init__(self, height, width, thickness, interval, plates_num, tilt_angle, input_stl_path, input_smaller_model_stl_path, output_stl_path):
        self.height = height
        self.width = width
        self.thickness = thickness
        self.interval = interval
        self.plates_num = plates_num
        self.tilt_angle = math.radians(tilt_angle)
        self.input_stl_path = input_stl_path
        self.input_smaller_model_stl_path = input_smaller_model_stl_path
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

        stl_import = import_stl(self.input_stl_path)
        stl_smaller_model_import = import_stl(self.input_smaller_model_stl_path)

        shell_model = difference()(stl_import, stl_smaller_model_import)
        intersection_model = intersection()(combined, stl_import)

        return union()(shell_model, intersection_model)


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
    output_stl_path = '/home/clarence/ros_ws/metamaterial_ws/src/metamaterial_filling/data/FL_final_output_with_shell.stl'

    # Input smaller model STL file path to generate the shell
    input_smaller_model_stl_path = '/home/clarence/ros_ws/metamaterial_ws/src/metamaterial_filling/data/FL_shell.stl'

    # Create the SixFoldPlatesFilling object
    sixFoldPlatesFilling = SixFoldPlatesFillingWithShell(height, width, thickness, interval, plates_num, tilt_angle, input_stl_path, input_smaller_model_stl_path, output_stl_path)
    sixFoldPlatesFilling.generate_model()
