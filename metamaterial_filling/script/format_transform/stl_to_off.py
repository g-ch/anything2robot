'''
This script is used to convert the stl file to off file.
'''

import meshio
import argparse


def stlToOff(stl_file: str, off_file: str):
    '''
    This function converts the stl file to off file.

    Parameters:
    stl_file (str): The path to the stl file.
    off_file (str): The path to the off file.
    '''
    # Read the stl file
    mesh = meshio.read(stl_file)

    # Write the off file
    meshio.write(off_file, mesh)


if __name__ == "__main__":
    # Get the path of the stl file and the out off file path with Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("stl_file", help="The path to the stl file")
    parser.add_argument("off_file", help="The path to the off file")

    args = parser.parse_args()

    stl_file = args.stl_file
    off_file = args.off_file

    stlToOff(stl_file, off_file)
    
