from solid import *
from solid.utils import *
import math
import os


def mesh_difference(target_mesh_path, tool_mesh_path, result_save_path):
    # Generate the SCAD file
    scad_file = 'difference_model.scad'
    scad_render_to_file(difference()(import_stl(target_mesh_path), import_stl(tool_mesh_path)), scad_file)

    try:
        # Use OpenSCAD's command-line interface to convert the SCAD file to an STL file
        os.system(f'openscad -o {result_save_path} {scad_file}')
    except Exception as e:
        print(f"OpenSCAD difference operation Error: {e}")
        return False


# Test the function
if __name__ == '__main__':
    target_mesh_path = 'target.stl'
    tool_mesh_path = 'tool.stl'
    result_save_path = 'result.stl'
    mesh_difference(target_mesh_path, tool_mesh_path, result_save_path)
    print(f"Mesh difference saved to {result_save_path}")
    