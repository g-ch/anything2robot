import open3d as o3d

# Load Mesh from stl file
mesh = o3d.io.read_triangle_mesh("./urdf/lynel/tmp/BODY_ideal.stl")
mesh.scale(100, center=(0, 0, 0))

cur_scene = o3d.t.geometry.RaycastingScene()
_ = cur_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

# Visualize the mesh, add grid and axis
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])

o3d.visualization.draw_geometries([mesh, axis])
