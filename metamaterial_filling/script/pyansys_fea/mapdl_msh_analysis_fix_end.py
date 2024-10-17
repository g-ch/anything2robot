from matplotlib import pyplot as plt
import numpy as np
from ansys.mapdl.core import launch_mapdl
from ansys.mapdl import core as pymapdl
import os
import argparse

inserted_indices_set = set()

'''
@breif: Find the nearest n nodes to a given point in 3D space
@param: nodes: numpy array: 3D coordinates of the nodes
@param: point3d: list: 3D coordinates of the point
@param: n: int: Number of nearest nodes to find. Default: 1
@return: list: List of indices of the nearest n nodes
'''
def find_nearest_n_nodes(nodes, point3d, n=1):
    """
    Find the nearest n nodes to a given point in 3D space.
    """
    # Calculate the distance between the point and all nodes
    distances = np.linalg.norm(nodes - point3d, axis=1)

    # Get the indices of the n smallest distances
    # nearest_n_indices = np.argsort(distances)[:n]
    nearest_indices = np.argsort(distances)

    # Get the n nearest nodes. If the node is already inserted, skip it
    nearest_n_indices = []
    for i in range(len(nearest_indices)):
        if nearest_indices[i] in inserted_indices_set:
            continue
        nearest_n_indices.append(nearest_indices[i])
        inserted_indices_set.add(nearest_indices[i])

        if len(nearest_n_indices) == n:
            break
    
    nearest_n_indices = np.array(nearest_n_indices)
    
    return nearest_n_indices


'''
@breif: Do Fea using Ansys. Make sure to have Ansys installed in the system and license available. This function fixex one end and applies specific orces on specific nodes.
@param: msh_file: str: Path to the input stl file
@param: elastic: float: Elastic modulus. MPa
@param: poisson_ratio: float: Poisson ratio
@param: fixed_end: int: define the fixed end. 0:-x most, 1:+x most, 2:-y most, 3:+y most, 4:-z most, 5:+z most. Default: 4
@param: forces_nodes: list: List of nodes where the forces are applied. The format is [ [node1_x, node1_y, node1_z], [node2_x, node2_y, node2_z], ...]
@param: forces: list: List of forces applied at the nodes. The format is [ [F1_x, F1_y, F1_z], [F2_x, F2_y, F2_z], ...]
@param: closest_node_num_per_force: int: Number of closest nodes to the forces_nodes to apply the forces. These nodes will form a component, where the force will be applied. Default: 1
@param: display: bool: Display the FEA results. Default: True
@return: float: Max von Mises stress and max displacement
'''
def static_fea_analysis(msh_file, elastic=210e3, poisson_ratio=0.3, fixed_end=4, forces_nodes=None, forces=None, closest_node_num_per_force=1, display=True):
    # Check if the msh file exists
    if not os.path.exists(msh_file + ".msh"):
        raise FileNotFoundError(f"File not found: {msh_file + '.msh'}")
    
    # Check if forces_nodes has the same length as forces.
    if forces_nodes is not None and forces is not None:
        if len(forces_nodes) != len(forces):
            raise ValueError("forces_nodes and forces must have the same length")
    else:
        raise ValueError("forces_nodes and forces must not be None")
    
    ###############################################################################
    # Load the mesh file
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

    # Define the fixed end based on the user input
    if fixed_end == 0:
        # Fixed end at the -x most. Use ratio 0.05 to select the nodes at the -x most
        min_x = np.min(mapdl.mesh.nodes[:, 0])
        max_x = np.max(mapdl.mesh.nodes[:, 0])
        mapdl.nsel("S","LOC", "X", min_x, min_x + 0.05*(max_x - min_x))  # Select node in a range of x
    elif fixed_end == 1:
        # Fixed end at the +x most. Use ratio 0.05 to select the nodes at the +x most
        min_x = np.min(mapdl.mesh.nodes[:, 0])
        max_x = np.max(mapdl.mesh.nodes[:, 0])
        mapdl.nsel("S","LOC", "X", max_x - 0.05*(max_x - min_x), max_x)
    elif fixed_end == 2:
        # Fixed end at the -y most. Use ratio 0.05 to select the nodes at the -y most
        min_y = np.min(mapdl.mesh.nodes[:, 1])
        max_y = np.max(mapdl.mesh.nodes[:, 1])
        mapdl.nsel("S","LOC", "Y", min_y, min_y + 0.05*(max_y - min_y))
    elif fixed_end == 3:
        # Fixed end at the +y most. Use ratio 0.05 to select the nodes at the +y most
        min_y = np.min(mapdl.mesh.nodes[:, 1])
        max_y = np.max(mapdl.mesh.nodes[:, 1])
        mapdl.nsel("S","LOC", "Y", max_y - 0.05*(max_y - min_y), max_y)
    elif fixed_end == 4:
        # Fixed end at the -z most. Use ratio 0.05 to select the nodes at the -z most
        min_z = np.min(mapdl.mesh.nodes[:, 2])
        max_z = np.max(mapdl.mesh.nodes[:, 2])
        mapdl.nsel("S","LOC", "Z", min_z, min_z + 0.05*(max_z - min_z))
    elif fixed_end == 5:
        # Fixed end at the +z most. Use ratio 0.05 to select the nodes at the +z most
        min_z = np.min(mapdl.mesh.nodes[:, 2])
        max_z = np.max(mapdl.mesh.nodes[:, 2])
        mapdl.nsel("S","LOC", "Z", max_z - 0.05*(max_z - min_z), max_z)
    else:
        raise ValueError("Invalid fixed_end value. Must be in the range [0, 5]")    

    if display:
        mapdl.nplot(1) # Plot the selected nodes

    mapdl.d("ALL", "UX")  # Add DOF constraints to selected nodes
    mapdl.d("ALL", "UY")  # Add DOF constraints to selected nodes
    mapdl.d("ALL", "UZ")  # Add DOF constraints to selected nodes
    mapdl.allsel(mute=True)  # Select all nodes to find the min_x in the next step


    # Add forces based on the user input
    nodes = mapdl.mesh.nodes
    # print(nodes)
    for i in range(len(forces_nodes)):
        nearest_n_indices = find_nearest_n_nodes(nodes, forces_nodes[i], closest_node_num_per_force)

        # print(f"Nearest node to the point {forces_nodes[i]} is {nearest_n_indices}")
        
        # Convert to one-based indexing for MAPDL
        nearest_n_indices_from_1 = nearest_n_indices + 1
        
        # Select nodes
        mapdl.nsel('S', 'NODE', vmin=nearest_n_indices_from_1[0], vmax=nearest_n_indices_from_1[0])
        for node_seq in nearest_n_indices_from_1[1:]:
            mapdl.nsel('A', 'NODE', vmin=node_seq, vmax=node_seq)

         # Check if the component was created correctly
        num_nodes = mapdl.get(entity='NODE', item1='COUNT')
        # print(f"Number of nodes selected for component {component_name}: {num_nodes}")
        if num_nodes == 0:
            raise ValueError(f"No nodes were selected for component {component_name}.")

        # Create the component if there are more than one nodes to be selected for a force
        if closest_node_num_per_force > 1:
            # Create the component
            component_name = f"COMP_{i+1}"
            mapdl.cm(component_name, 'NODE')

            # Couple the nodes in this component using explicit set numbers
            mapdl.cp(f"NEXT", "UX", component_name)
            mapdl.cp(f"NEXT", "UY", component_name)
            mapdl.cp(f"NEXT", "UZ", component_name)

        # Apply force to one of the nodes in the component
        force_value = forces[i]
        mapdl.f(nearest_n_indices_from_1[0], 'FX', force_value[0])
        mapdl.f(nearest_n_indices_from_1[0], 'FY', force_value[1])
        mapdl.f(nearest_n_indices_from_1[0], 'FZ', force_value[2])


        if display:
            mapdl.nplot(1) # Plot the selected nodes
        
        # Clear the selection
        mapdl.allsel()


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
    # print(displacements)
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
    parser.add_argument("--display", type=bool, default=False, help="Display the FEA results")
    parser.add_argument("--elastic", type=float, default=210e3, help="Elastic modulus")
    parser.add_argument("--poisson_ratio", type=float, default=0.3, help="Poisson ratio")

    parser.add_argument("--fixed_end", type=int, default=2, help="define the fixed end. 0:-x most, 1:+x most, 2:-y most, 3:+y most, 4:-z most, 5:+z most")
    parser.add_argument("--forces_nodes", type=float, nargs='+', default=[[0, 0, 180], [0, 0, 130]], help="List of nodes where the forces are applied. The format is [node1_x, node1_y, node1_z, node2_x, node2_y, node2_z, ...]")
    parser.add_argument("--forces", type=float, nargs='+', default=[[0, 1000, 0], [10000, 0, 0]], help="List of forces applied at the nodes. The format is [F1_x, F1_y, F1_z, F2_x, F2_y, F2_z, ...]")
    parser.add_argument("--closest_node_num_per_force", type=int, default=10, help="Number of closest nodes to the forces_nodes to apply the forces")

    args = parser.parse_args()
    msh_file = args.msh_file

    if not os.path.exists(msh_file):
        raise FileNotFoundError(f"File not found: {msh_file}")
    
    if not msh_file.endswith(".msh"):
        raise ValueError("File must be a .msh file")
    
    # Remove the file extension! Note: This is necessary for the MAPDL script
    msh_file = msh_file.replace(".msh", "")

    max_stress, max_displacement = static_fea_analysis(msh_file=msh_file, elastic=args.elastic, poisson_ratio=args.poisson_ratio, fixed_end=args.fixed_end, forces_nodes=args.forces_nodes, forces=args.forces, closest_node_num_per_force=args.closest_node_num_per_force, display=args.display)

    print("Done!")
    print(f"Max von Mises stress: {max_stress}")
    print(f"Max displacement: {max_displacement}")
