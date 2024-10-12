import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure
import open3d as o3d

def voxelize_stl(stl_file, resolution):
    mesh = pv.read(stl_file)
    
    # Get the bounding box of the mesh
    bounds = mesh.bounds
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    
    # Calculate the size of the mesh in each dimension
    x_size = xmax - xmin
    y_size = ymax - ymin
    z_size = zmax - zmin
    
    # Determine the density based on the mesh size and the desired resolution
    max_size = max(x_size, y_size, z_size)
    density = max_size / resolution
    
    # Voxelize the mesh with the calculated density
    voxels = pv.voxelize(mesh, density=density)

    voxels.plot(show_edges=True)

    # Shift points to ensure all are positive (temporary shift)
    shift_vector = np.array([-xmin, -ymin, -zmin])
    grid = voxels.cell_centers().points + shift_vector

    # Initialize an empty voxel grid
    voxel_grid = np.zeros((resolution, resolution, resolution), dtype=int)
    
    # Fill the voxel grid based on the shifted grid points
    for point in grid:
        x, y, z = (point / density).astype(int)
        if 0 <= x < resolution and 0 <= y < resolution and 0 <= z < resolution:
            voxel_grid[x, y, z] = 1
            
    return voxel_grid, shift_vector, density


def shift_voxel_back(voxel_grid, shift_vector, density, resolution):
    # Get voxel coordinates
    filled = np.argwhere(voxel_grid == 1)
    
    # Convert voxel indices back to original coordinates using the inverse shift
    original_points = (filled / density) - shift_vector
    
    return original_points

def display_voxel_grid(points, title="Voxel Grid"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    
    # Plot the filled voxels
    if points.size > 0:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='blue', alpha=0.3, label="Voxelized Mesh")
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

'''
@brief Iterate through the voxel grid to find a candidate cube that is filled
@param voxel_grid: The voxel grid
@param n: The voxel size (number) of the minimal candidate cube
@return: The coordinates of the candidate cube
'''
def find_candidate_cube(voxel_grid, n):
    shape = voxel_grid.shape
    for x in range(shape[0] - n):
        for y in range(shape[1] - n):
            for z in range(shape[2] - n):
                if np.all(voxel_grid[x:x+n, y:y+n, z:z+n] == 1):
                    return x, y, z
    return None

'''
@brief Find the max dimensions of the cube and remove the voxels
@param voxel_grid: The voxel grid
@param x0: The x coordinate of the cube. Starting point.
@param y0: The y coordinate of the cube. Starting point.
@param z0: The z coordinate of the cube. Starting point.
@param n: The size of the cube
@param s: The size of the cross-section area, which is the number of voxels to remove to have a tube. We also use s as the shell thickness.
@return: The voxel grid after removing the voxels
'''
def find_max_dimensions_and_remove_voxels(voxel_grid, x0, y0, z0, n, s):
    # Initialize the dimensions a, b, c to n
    x = x0 + n//2
    y = y0 + n//2
    z = z0 + n//2
    a, b, c = n // 2, n // 2, n // 2
    
    while True:
        expanded = False
        # Try expanding in the x direction
        if x - a - 1 >= 0 and x + a + 1 < voxel_grid.shape[0]:
            if np.all(voxel_grid[x - a - 1:x + a + 2, y - b:y + b + 1, z - c:z + c + 1] == 1):
                a += 1
                expanded = True
        if not expanded:
            break
    
    while True:
        expanded = False
        # Try expanding in the y direction
        if y - b - 1 >= 0 and y + b + 1 < voxel_grid.shape[1]:
            if np.all(voxel_grid[x - a:x + a + 1, y - b - 1:y + b + 2, z - c:z + c + 1] == 1):
                b += 1
                expanded = True
        if not expanded:
            break
        
    while True:
        expanded = False
        # Try expanding in the z direction
        if z - c - 1 >= 0 and z + c + 1 < voxel_grid.shape[2]:
            if np.all(voxel_grid[x - a:x + a + 1, y - b:y + b + 1, z - c - 1:z + c + 2] == 1):
                c += 1
                expanded = True
        if not expanded:
            break

    print(f"Max dimensions: {a*2}, {b*2}, {c*2}")

    # Shrink the cube by 5 voxels and remove the voxels
    voxel_grid[x - a + s:x + a - s, y - b + s:y + b - s, z - c + s:z + c - s] = 0

    return voxel_grid

'''
@brief Count the number of voxels to remove in a direction
@param voxel_grid: The voxel grid
@param center: The center of the cube
@param direction: The direction to count the voxels
@param s: The size of the cross-section area, which is the number of voxels to remove to have a tube. We also use s as the shell thickness.
@return: The number of voxels to remove
'''
def count_voxels_to_remove(voxel_grid, center, direction, s):
    cx, cy, cz = center
    dx, dy, dz = direction
    count = 0
    
    while True:
        cx += dx
        cy += dy
        cz += dz
        
        if not in_bounds(voxel_grid, cx, cy, cz):
            break
        
        # Define the cross-section area
        cross_section = voxel_grid[cx-s//2:cx+s//2+1, cy-s//2:cy+s//2+1, cz-s//2:cz+s//2+1]
        
        # Count the filled voxels in the cross-section
        count += np.sum(cross_section)
        
        # Stop if the cross-section becomes empty
        if count == 0:
            break
    
    return count


'''
@brief Remove voxels in the direction that has the fewest voxels
@param voxel_grid: The voxel grid
@param center: The center of the cube
@param s: The size of the cross-section area, which is the number of voxels to remove to have a tube. We also use s as the shell thickness.
@return: The voxel grid after removing the voxels
'''
def remove_voxels_in_direction(voxel_grid, center, s):
    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    best_direction = None
    fewest_voxels = float('inf')
    
    for direction in directions:
        count = count_voxels_to_remove(voxel_grid, center, direction, s)
        if count < fewest_voxels:
            fewest_voxels = count
            best_direction = direction
            
    if best_direction:
        voxel_grid = remove_voxels(voxel_grid, center, best_direction, s)
    
    return voxel_grid

def remove_voxels(voxel_grid, center, direction, s):
    cx, cy, cz = center
    dx, dy, dz = direction
    
    while True:
        cx += dx
        cy += dy
        cz += dz
        
        if not in_bounds(voxel_grid, cx, cy, cz):
            break
        
        # Set the cross-section area to 0 (remove voxels)
        voxel_grid[cx-s//2:cx+s//2+1, cy-s//2:cy+s//2+1, cz-s//2:cz+s//2+1] = 0
        
    return voxel_grid

def in_bounds(voxel_grid, x, y, z):
    return 0 <= x < voxel_grid.shape[0] and 0 <= y < voxel_grid.shape[1] and 0 <= z < voxel_grid.shape[2]



def voxel_grid_to_mesh_marching_cubes(voxel_grid, shift_vector, density):
    # Perform marching cubes to get vertices and faces
    verts, faces, _, _ = measure.marching_cubes(voxel_grid)
    
    # Adjust vertices to match the original coordinate system
    verts = (verts * density) - shift_vector
    
    # Convert vertices and faces to a pyvista mesh
    faces = np.hstack([[3] + list(face) for face in faces])
    mesh = pv.PolyData(verts, faces)
    
    return mesh    
    


'''
@brief Add inner cubes to the mesh
@param stl_file: The path to the STL file
@param resolution: The resolution of the voxel grid. Int. E.g. 100
@param n: The voxel size (number) of the minimal candidate cube
@param s: The size of the cross-section area, which is the number of voxels to remove to have a tube. We also use s as the shell thickness.
@param new_mesh_save_path: The path to save the new mesh
'''
def add_inner_cubes(stl_file, resolution, n, s, new_mesh_save_path):
    # For demonstration, using a synthetic cube grid
    voxel_grid, shift_vector, density = voxelize_stl(stl_file, resolution)
    
    # Display the initial voxel grid
    # original_points = shift_voxel_back(voxel_grid, shift_vector, density, resolution)
    # display_voxel_grid(original_points, title="Initial Voxel Grid")
    
    # count = 0
    # max_candidate_size = n * 2
    # min_candidate_size = n

    # for candidate_size in range(max_candidate_size, min_candidate_size, -4):
    #     print(f"Candidate size: {candidate_size}")
    #     while True:
    #         count += 1
    #         print(f"Iteration {count}")
    #         candidate_cube = find_candidate_cube(voxel_grid, candidate_size)
    #         if not candidate_cube:
    #             break
    #         print(f"Candidate cube: {candidate_cube}")
    #         x, y, z = candidate_cube
    #         voxel_grid = find_max_dimensions_and_remove_voxels(voxel_grid, x, y, z, candidate_size, s)

            # Find the center of the cube after hollowing and make a corridor from cube to the surface
            # cx, cy, cz = x + candidate_size//2, y + candidate_size//2, z + candidate_size//2
            # voxel_grid = remove_voxels_in_direction(voxel_grid, (cx, cy, cz), s)
    
    # Display the voxel grid after processing
    processed_points = shift_voxel_back(voxel_grid, shift_vector, density, resolution)
    display_voxel_grid(processed_points, title="Voxel Grid After Processing")

    # Generate mesh from voxel grid using marching cubes and save as STL
    mesh = voxel_grid_to_mesh_marching_cubes(voxel_grid, shift_vector, density)

    # Do laplacian smoothing
    mesh.smooth(n_iter=3)

    mesh.save(new_mesh_save_path)


if __name__ == "__main__":
    # Running the visualization with resolution, n, and s.
    new_mesh_save_path = "/home/clarence/git/anything2robot/anything2robot/urdf/processed_mesh_marching_cubes.stl"
    add_inner_cubes(stl_file="/home/clarence/git/anything2robot/anything2robot/urdf/gold_lynel20241010-134328_good/BODY_UP.stl", resolution=100, n=20, s=4, new_mesh_save_path=new_mesh_save_path)
