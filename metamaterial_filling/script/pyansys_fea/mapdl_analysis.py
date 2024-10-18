from matplotlib import pyplot as plt
import numpy as np
from ansys.mapdl.core import launch_mapdl
from ansys.mapdl import core as pymapdl
import os

exec_loc = '/root/ansys_inc/v232/ansys/bin/ansys232'
mapdl = launch_mapdl(exec_loc)
print(mapdl)

mapdl.clear()

###############################################################################
# Load a igs file
# cad_file = '/home/clarence/ros_ws/metamaterial_ws/src/metamaterial_filling/data/FL_replaced_scaled.iges'
cad_file = '/home/clarence/ros_ws/metamaterial_ws/src/robot_design/pyansys/model1.igs'

# Check if the corresponding msh file exists
if cad_file.endswith(".igs"):
    msh_file = cad_file.replace(".igs", ".msh")
else:
    msh_file = cad_file.replace(".iges", ".msh")

msh_exists = os.path.exists(msh_file)

if msh_exists:
    print("Mesh file found:", msh_file)
else:
    print("Mesh file not found:", msh_file)

# Remove .msh in the file name. Otherwise, the file name will be saved as .msh.msh
msh_file = msh_file.replace(".msh", "")


''' Use the following method to replace the characters in Window otherwise there might be problem in Windows '''
# cmd = '''
# *dim,iges_file,string,248
# *set,iges_file(1),'QQQ'
# '''.replace("QQQ",cad_file)
# mapdl.input_strings(cmd)
 
if not msh_exists:
    mapdl.aux15()
    # mapdl.igesin('iges_file(1)')
    mapdl.igesin(cad_file)

    #######

    mapdl.lplot(
        show_line_numbering=False,
        background="k",
        line_width=3,
        color="w",
        show_axes=False,
        show_bounds=True,
        title="",
        cpos="xz",
    )

    print(mapdl.geometry)

    mapdl.prep7() # Enters the model creation preprocessor
    if mapdl.geometry.vnum.size == 0:
        print("No volume found in the imported model! Exiting...")
        mapdl.exit()
        exit()
        # ''' Make a volume with the areas. Doesn't work. If the volume can be generated, it is generated . '''
        # mapdl.asel("S", "AREA", "", 1, mapdl.geometry.n_area)
        # mapdl.va("ALL")


    ###############################################################################
    # Generate a mesh

    print("Generating a mesh ...")

    mapdl.et(1, "SOLID285")
    '''
    SOLID285	3-D 4-Node Tetrahedral Structural Solid with Nodal Pressures
    check: https://www.mm.bme.hu/~gyebro/files/ans_help_v182/ans_elem/elem_matsupp.html
    '''

    # mapdl.esize(0.1)  # Mesh size
    mapdl.smrtsize(1) # From 1 to 10: Fine to Coarse Mesh Size
    mapdl.vmesh("all")

    mapdl.eplot(vtk=True, show_edges=True, show_axes=False, line_width=2, background="w")

    if mapdl.mesh.nodes.size == 0:
        print("Meshing failed!")
        mapdl.exit()
        exit()

    print("Meshing successful!")

    # Save mesh to a msh file
    mapdl.cdwrite("db", msh_file, "msh")


else:
    mapdl.cdread("db", msh_file, "msh")

    mapdl.eplot(vtk=True, show_edges=True, show_axes=False, line_width=2, background="w")

    mapdl.prep7() #Enters the model creation preprocessor

    print("Mesh loaded from file:", msh_file)


###############################################################################
# Material properties

# Define custom units using the UNITS command
mapdl.units('USER', 'millimeter', 'gram', 'second', 'newton', 'degree', 'ampere')
# mapdl.units("SI")  # SI - International system (m, kg, s, K).

# Define material properties
mapdl.mp('EX', 1, 210e3)  # Elastic modulus (also EY, EZ)
mapdl.mp('PRXY', 1, 0.3)  # Major Poisson’s ratios
mapdl.mp('DENS', 1, 0.0078) # Mass density.

###############################################################################
# Add load and constraints

# Get the maximum X coordinate value in the model
print(mapdl.mesh.nodes[:, 0])
max_x = np.max(mapdl.mesh.nodes[:, 0])

print("MAX_X")
print(max_x)

mapdl.nsel("S","LOC", "X", max_x*0.95, max_x)  # Select node in a range of x
mapdl.nplot(1) # Plot the selected nodes

mapdl.d("ALL", "UX")  # Add DOF constraints to selected nodes
mapdl.d("ALL", "UY")  # Add DOF constraints to selected nodes
mapdl.d("ALL", "UZ")  # Add DOF constraints to selected nodes
# mapdl.d("ALL", "ROTX")  # Add DOF constraints to selected nodes
# mapdl.d("ALL", "ROTY")  # Add DOF constraints to selected nodes
# mapdl.d("ALL", "ROTZ")  # Add DOF constraints to selected nodes

mapdl.allsel(mute=True)  # Select all nodes to find the min_x in the next step

min_x = np.min(mapdl.mesh.nodes[:, 0])
print("MIN_X")
print(min_x)

mapdl.nsel("S","LOC", "X", min_x)  # Select node
mapdl.nplot(1) # Plot the selected nodes


mapdl.cp(2, "UZ", "ALL")  # Next, couple the DOF for these nodes.

mapdl.f("ALL", "FZ", 1000) # Add Force
mapdl.allsel(mute=True)


###############################################################################
# Solve the Static Problem
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Solve the static analysis
print("Solving the static problem ...")

mapdl.run("/SOLU")
mapdl.antype("STATIC")
mapdl.solve()
mapdl.finish(mute=True)

print("Static problem solved!")

# grab the result from the ``mapdl`` instance
result = mapdl.result
result.plot_principal_nodal_stress(
    0,
    "SEQV",
    lighting=False,
    background="w",
    show_edges=True,
    text_color="k",
    add_text=False,
)

nnum, stress = result.principal_nodal_stress(0)
von_mises = stress[:, -1]  # von-Mises stress is the right most column

# Must use nanmax as stress is not computed at mid-side nodes
max_stress = np.nanmax(von_mises)

mapdl.post_processing.plot_nodal_displacement("Z")


# stop mapdl
mapdl.exit()