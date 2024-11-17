import trimesh
import pymeshfix


def repair_mesh(mesh, output_path):
    # Load the STL file into a Trimesh object
    mesh = trimesh.load_mesh(mesh)

    # Analyze the mesh for errors
    print("Is the mesh watertight?", mesh.is_watertight)

    # Convert the Trimesh object to vertex and face arrays
    vertices = mesh.vertices
    faces = mesh.faces

    # Create a MeshFix object from the vertex and face arrays
    meshfix = pymeshfix.MeshFix(vertices, faces)

    # Plot the input mesh (optional, requires vtkInterface)
    # meshfix.plot()

    # Repair the mesh
    meshfix.repair()

    # Access the repaired mesh as a Trimesh object
    repaired_vertices = meshfix.v
    repaired_faces = meshfix.f

    # Convert the repaired vertices and faces back into a Trimesh object
    repaired_mesh = trimesh.Trimesh(vertices=repaired_vertices, faces=repaired_faces)

    # Recheck the repaired mesh for errors
    print("Is the repaired mesh watertight?", repaired_mesh.is_watertight)

    # Save the repaired mesh to a new STL file
    repaired_mesh.export(output_path)

    print(f"Repaired mesh saved as {output_path}")


if __name__ == '__main__':
    # Example usage
    input_path = '/media/clarence/Clarence/lynel/Gold_Lynel_0819082531.stl'
    output_path = '/media/clarence/Clarence/lynel/Gold_Lynel_0819082531_fixed.stl'

    repair_mesh(input_path, output_path)


