from matplotlib import pyplot as plt
import numpy as np
from ansys.mapdl.core import launch_mapdl
from ansys.mapdl import core as pymapdl
import os
import argparse


def static_fea_analysis(msh_file, elastic=210e3, poisson_ratio=0.3, display=1):

    exec_loc = '/root/ansys_inc/v232/ansys/bin/ansys232'
    mapdl = launch_mapdl(exec_loc)
    print(mapdl)

    mapdl.clear()

    mapdl.cdread("db", msh_file, "msh")

    if display:
        mapdl.eplot(vtk=True, show_edges=True, show_axes=False, line_width=2, background="w")

    mapdl.prep7() #Enters the model creation preprocessor

    print("Mesh loaded from file:", msh_file)


    ###############################################################################
    # Material properties

    # Define custom units using the UNITS command
    mapdl.units('USER', 'millimeter', 'gram', 'second', 'newton', 'degree', 'ampere')
    # mapdl.units("SI")  # SI - International system (m, kg, s, K).

    # Define material properties
    mapdl.mp('EX', 1, elastic)  # Elastic modulus (also EY, EZ)
    mapdl.mp('PRXY', 1, poisson_ratio)  # Major Poisson’s ratios
    # mapdl.mp('DENS', 1, 0.0078) # Mass density.

    ###############################################################################
    # Add load and constraints
    nodes = mapdl.mesh.nodes

    print(mapdl.mesh.nodes[:, 0])
    max_x = np.max(mapdl.mesh.nodes[:, 0]) # Get the maximum X coordinate value in the model

    print("MAX_X")
    print(max_x)

    mapdl.nsel("S","LOC", "X", max_x*0.95, max_x)  # Select node in a range of x

    if display:
        mapdl.nplot(1) # Plot the selected nodes

    mapdl.d("ALL", "UX")  # Add DOF constraints to selected nodes
    mapdl.d("ALL", "UY")  # Add DOF constraints to selected nodes
    mapdl.d("ALL", "UZ")  # Add DOF constraints to selected nodes
    mapdl.allsel(mute=True)  # Select all nodes to find the min_x in the next step

    min_x = np.min(mapdl.mesh.nodes[:, 0])
    print("MIN_X")
    print(min_x)

    mapdl.nsel("S","LOC", "X", min_x)  # Select node
    
    if display:
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
    if display:
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
    nnum, displacements = result.nodal_displacement(0)

    # Get max stress
    von_mises = stress[:, -1]  # von-Mises stress is the right most column
    max_stress = np.nanmax(von_mises)

    # Get max displacement
    print(displacements)
    displacement_magnitude = np.linalg.norm(displacements[:, :3], axis=1)
    max_displacement = np.max(displacement_magnitude)

    if display:
        mapdl.post_processing.plot_nodal_displacement("Z")

    # stop mapdl
    mapdl.exit()

    return max_stress, max_displacement


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Static FEA Analysis")
    parser.add_argument("--msh_file", type=str, help="Path to the mesh file")
    parser.add_argument("--display", type=int, default=1, help="Display the models")
    parser.add_argument("--elastic", type=float, default=210e3, help="Elastic modulus")
    parser.add_argument("--poisson_ratio", type=float, default=0.3, help="Poisson ratio")

    args = parser.parse_args()
    msh_file = args.msh_file

    if not os.path.exists(msh_file):
        raise FileNotFoundError(f"File not found: {msh_file}")
    
    if not msh_file.endswith(".msh"):
        raise ValueError("File must be a .msh file")
    
    # Remove the file extension
    msh_file = msh_file.replace(".msh", "")

    max_stress, max_displacement = static_fea_analysis(msh_file, args.elastic, args.poisson_ratio, args.display)

    print("Done!")
    print(f"Max von Mises stress: {max_stress}")
    print(f"Max displacement: {max_displacement}")
