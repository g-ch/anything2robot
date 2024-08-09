#!/usr/lib/freecad/bin/freecadcmd-python3

'''
@Author: Gang Chen
@Date: 2024-06-11
@Description: This script is used to convert a stl file to a iges file using FreeCAD
@Prerequisite: FreeCAD. Install FreeCAD using the following command:
    sudo apt-get install freecad
@Usage: python3 stl_to_iges.py input_file output_file tolerance 
    input_file: The path to the input stl file
    output_file: The path to the output iges file
    tolerance: The tolerance value
@Note: Tested on FreeCAD 0.18 in Ubuntu 20.04
'''


FREECADPATH = '/usr/lib/freecad-python3/lib/' # path to your FreeCAD.so or FreeCAD.pyd file,
import sys
sys.path.append(FREECADPATH)

import FreeCAD as App
import argparse
import os


def main(input_file, output_file, tolerance):
    App.newDocument("Unnamed")
    App.setActiveDocument("Unnamed")
    App.ActiveDocument=App.getDocument("Unnamed")

    import Mesh
    Mesh.insert(input_file,"Unnamed")

    import Part
    FreeCAD.getDocument("Unnamed").addObject("Part::Feature","FL001")
    __shape__=Part.Shape()
    __shape__.makeShapeFromMesh(FreeCAD.getDocument("Unnamed").getObject("FL").Mesh.Topology,tolerance)
    FreeCAD.getDocument("Unnamed").getObject("FL001").Shape=__shape__
    FreeCAD.getDocument("Unnamed").getObject("FL001").purgeTouched()
    del __shape__
    App.ActiveDocument.addObject('Part::Feature','FL001').Shape=App.ActiveDocument.FL001.Shape.removeSplitter()
    App.ActiveDocument.ActiveObject.Label=App.ActiveDocument.FL001.Label


    App.ActiveDocument.recompute()
    import Part
    __s__=App.ActiveDocument.FL001001.Shape
    __s__=Part.Solid(__s__)
    __o__=App.ActiveDocument.addObject("Part::Feature","FL001001_solid")
    __o__.Label="FL002 (Solid)"
    __o__.Shape=__s__
    del __s__, __o__
    __objs__=[]
    __objs__.append(FreeCAD.getDocument("Unnamed").getObject("FL001001_solid"))
    import Part
    Part.export(__objs__,output_file)

    del __objs__



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="The path to the input stl file")
    parser.add_argument("output_file", help="The path to the output iges file")
    parser.add_argument("tolerance", help="The tolerance value")

    args = parser.parse_args()

    # Check if the input file exists and is a stl file
    if not os.path.exists(args.input_file):
        raise Exception("The input file does not exist")

    if not args.input_file.endswith(".stl"):
        raise Exception("The input file is not a stl file")
    
    # Check if the output file is a iges file
    if not args.output_file.endswith(".iges"):
        raise Exception("The output file is not named as an .iges file")
    

    # Create a temp folder to store the intermediate files if not exists
    temp_folder = "temp"
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    # Cp the input file to the temp folder and rename it to FL.stl
    input = os.path.join(temp_folder, "FL.stl")
    os.system("cp {} {}".format(args.input_file, input))

    input = u"{}".format(input)
    output = u"{}".format(temp_folder + "/FL.igs")
    tolerance = float(args.tolerance)

    main(input, output, tolerance)

    # Cp the output file to the output folder
    os.system("cp {} {}".format(output, args.output_file))

    # Remove the temp folder
    os.system("rm -rf {}".format(temp_folder))




