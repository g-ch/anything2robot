    
import trimesh
import os

if __name__ == '__main__':
    stl_path = "/home/cc/git/anything2robot/metamaterial_filling/data/output/BODY_replaced_smaller.stl"
    smaller_stl_save_path = "/home/cc/git/anything2robot/urdf/gold_lynel20241010-134328_good/BODY_UP_smaller.stl"
    
    mesh_to_downsample = trimesh.load(stl_path)

    #### FOR OLD VERSION OF TRIMESH
    # down_sample_ratio = 10
    # ori_face_num = len(mesh_to_downsample.faces)
    # print(f"Original face number: {ori_face_num}")
    # down_sampled_face_num = int(ori_face_num / down_sample_ratio)
    # down_sampled_face_num = max(down_sampled_face_num, 1000) # At least 1000 points
    # print(f"Expected Downsampled point number: {down_sampled_face_num}")
    # down_sampled_mesh = mesh_to_downsample.simplify_quadratic_decimation(down_sampled_face_num)
    # down_sampled_mesh.export(smaller_stl_save_path)
    

    ### FOR NEW VERSION OF TRIMESH
    down_sample_ratio = 0.1
    mesh_to_downsample = mesh_to_downsample.simplify_quadric_decimation(down_sample_ratio)
    # Smooth the mesh
    mesh_to_downsample = trimesh.smoothing.filter_mut_dif_laplacian(mesh_to_downsample, iterations=3)
    # Downsample the mesh again
    mesh_to_downsample = mesh_to_downsample.simplify_quadric_decimation(down_sample_ratio)
    # Smooth the mesh
    mesh_to_downsample = trimesh.smoothing.filter_mut_dif_laplacian(mesh_to_downsample, iterations=3)
    # Downsample the mesh again
    mesh_to_downsample = mesh_to_downsample.simplify_quadric_decimation(down_sample_ratio)
    # Smooth the mesh
    mesh_to_downsample = trimesh.smoothing.filter_mut_dif_laplacian(mesh_to_downsample, iterations=3)
    # Downsample the mesh again
    mesh_to_downsample = mesh_to_downsample.simplify_quadric_decimation(down_sample_ratio)


    mesh_to_downsample.export(smaller_stl_save_path)