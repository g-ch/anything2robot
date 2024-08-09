'''
@Author: Clarence
@Date: 2023/12/09
'''

import pyvista as pv
import numpy as np

# Path to your STL file and NumPy file
model_name = 'jkhk'
model_name = 'n02108915_895'
stl_file_path = './model/sample_models/' + model_name + '_res_e300_smoothed.stl'
numpy_file_path = './model/sample_models/' + model_name + '_joints.npy'

# Load the STL file
mesh = pv.read(stl_file_path)

# Load the NumPy file containing 3D points
points = np.load(numpy_file_path)
print(points)

# Convert the NumPy array of points into a PyVista PolyData
point_cloud = pv.PolyData(points)

# Create a plotter
plotter = pv.Plotter()

# Add the mesh to the plotter
plotter.add_mesh(mesh, color='lightblue', opacity=0.5)

# Add the point cloud to the plotter
# You can change the color and point size as needed
plotter.add_points(point_cloud, color='red', point_size=5)
plotter.add_axes()
# Show the plotter
plotter.show()
