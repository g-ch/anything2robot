'''
This script is used to convert the off file to stl file.
'''

import meshio
import argparse

def off_to_stl(off_file, stl_file):
    # Read the off file
    mesh = meshio.read(off_file)

    # Write the stl file
    meshio.write(stl_file, mesh)

if __name__ == "__main__":
    # Get the path of the off file and the out stl file path with Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("off_file", help="The path to the off file")
    parser.add_argument("stl_file", help="The path to the stl file")

    args = parser.parse_args()

    off_to_stl(args.off_file, args.stl_file)

    # # Read the off file
    # mesh = meshio.read(args.off_file)

    # # Write the stl file
    # meshio.write(args.stl_file, mesh)
    