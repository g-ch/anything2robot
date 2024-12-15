import trimesh
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_mesh_with_arrows_and_colorbar(mesh, arrow_starts, arrow_ends, costs, colormap_name='viridis'):
    """
    Plot a trimesh mesh with arrows and a color bar representing the costs.
    
    Parameters:
        mesh (trimesh.Trimesh): The mesh to visualize.
        arrow_starts (ndarray): Array of shape (N, 3) specifying the start points of arrows.
        arrow_ends (ndarray): Array of shape (N, 3) specifying the end points of arrows.
        costs (ndarray): A 1D array of costs to be mapped to colors.
        colormap_name (str): Name of the matplotlib colormap to use.
    """
    # Normalize costs to the range [0, 1]
    normalized_costs = (costs - np.min(costs)) / (np.max(costs) - np.min(costs) + 1e-8)
    
    # Get the colormap
    colormap = colormaps[colormap_name]
    
    # Map normalized costs to colors (RGBA; use RGB only)
    colors = colormap(normalized_costs)[:, :3]
    
    # Calculate arrow directions
    arrow_directions = arrow_ends - arrow_starts
    
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    length = np.linalg.norm(arrow_directions[0])
    
    # Plot the mesh
    ax.plot_trisurf(
        mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
        triangles=mesh.faces, color='gray', alpha=0.5
    )
    
    # Plot the arrows with mapped colors
    for start, direction, color in zip(arrow_starts, arrow_directions, colors):
        ax.quiver(
            start[0], start[1], start[2],
            direction[0], direction[1], direction[2],
            color=color, length=length, normalize=True
        )
    
    # Add a color bar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=np.min(costs), vmax=np.max(costs)))
    sm.set_array([])  # Dummy data for the colorbar
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, aspect=15)
    cbar.set_label('Cost Value', rotation=270, labelpad=15)
    
    # Set plot limits for better visualization
    mesh_bounds = mesh.bounds
    ax.set_xlim(mesh_bounds[0][0], mesh_bounds[1][0])
    ax.set_ylim(mesh_bounds[0][1], mesh_bounds[1][1])
    ax.set_zlim(mesh_bounds[0][2], mesh_bounds[1][2])
    
    # Show the plot
    plt.show()




def visualize_mesh_voxels_vectors(mesh, voxels, voxel_size, min_bound, start_point, direction_vectors, results):
    '''
    @Description: Visualize the mesh, occupied voxels, and direction vectors
    @Input:
        mesh: The trimesh mesh to visualize
        voxels: The 3D array of voxels
        voxel_size: The size of the voxels
        min_bound: The minimum bounds of the voxel grid
        start_point: The starting point of the direction vectors
        direction_vectors: The list of direction vectors
        results: The list of whether the rays hit an occupied voxel
    '''
    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the mesh
    ax.add_collection3d(Poly3DCollection(mesh.triangles, alpha=0.2, color='gray'))
    
    # Plot the occupied voxels
    occupied_voxel_coords = np.array(np.nonzero(voxels)).T * voxel_size + min_bound
    for voxel in occupied_voxel_coords:
        # Plot the voxel as a small cube
        ax.bar3d(voxel[0], voxel[1], voxel[2], voxel_size, voxel_size, voxel_size, color='blue', alpha=0.5)

    # Plot the direction vectors
    for vec, hit in zip(direction_vectors, results):
        end_point = start_point + vec  # Calculate the end point of the vector
        
        # Choose color based on whether the direction is occupied
        color = 'red' if hit >= 1 else 'green'
        
        # Plot the vector
        ax.quiver(
            start_point[0], start_point[1], start_point[2], 
            vec[0], vec[1], vec[2], 
            color=color, length=1.0, normalize=True
        )
    
    # Set the limits for better visualization
    ax.set_xlim(min_bound[0], min_bound[0] + voxel_size * voxels.shape[0])
    ax.set_ylim(min_bound[1], min_bound[1] + voxel_size * voxels.shape[1])
    ax.set_zlim(min_bound[2], min_bound[2] + voxel_size * voxels.shape[2])
    
    plt.show()


if __name__ == "__main__":
    import trimesh

    # Load a mesh
    mesh = trimesh.load('/home/clarence/git/anything2robot/anything2robot/result/gold_lynel_20241124-161201_good/result_round1/urdf/BODY.stl')  # Replace with your mesh file

    # Define arrow parameters
    arrow_starts = np.array([
        [0, 0, 0],
        [1, 1, 1],
        [0.5, 0.5, 0.5]
    ])
    arrow_ends = np.array([
        [1, 0, 0],
        [1, 2, 1],
        [0.5, 0.5, 1]
    ])
    costs = np.array([0.1, 0.5, 0.9])  # Costs associated with the arrows
    
    # Call the function
    plot_mesh_with_arrows_and_colorbar(mesh, arrow_starts, arrow_ends, costs, colormap_name='plasma')

