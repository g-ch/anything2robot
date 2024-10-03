import json
import os
import numpy as np
import pyvista as pv
import argparse

def quit_press_callback():
    global quit_requested
    quit_requested = True

def save_press_callback():
    global save_requested
    save_requested = True

'''
Visualize the animal3d mesh and keypoint_3d
'''
def visualize_animal3d(mesh_path, keypoint_3d):
    global quit_requested
    global save_requested
    quit_requested = False
    save_requested = False

    mesh = pv.read(mesh_path)
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color=[0.5, 0.5, 0.5], opacity=0.5)
    keypoint_3d = np.array(keypoint_3d)
    plotter.add_points(keypoint_3d, color='red', point_size=10)
    
    # Add a callback function to handle key presses
    plotter.add_key_event('q', quit_press_callback)
    plotter.add_key_event('s', save_press_callback)

    plotter.add_axes()
    plotter.add_text("Press 'q' to quit. Press 's' to save. Close the window to go to the next image.", position='upper_left', font_size=15, color='red')
    
    plotter.show(auto_close=False)
    
    # Replace the problematic line with:
    return quit_requested, save_requested


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='/home/clarence/git/anything2robot/Animal3d')
    parser.add_argument('--json_file_name', type=str, default='test.json')
    parser.add_argument('--save_to_new_json', type=bool, default=True)
    parser.add_argument('--new_json_path', type=str, default='neutral_pose_images.json')
    args = parser.parse_args()

    dataset_path = args.dataset_path
    json_path = os.path.join(dataset_path, args.json_file_name)
    new_json_path = os.path.join(dataset_path, args.new_json_path)
    save_to_new_json = args.save_to_new_json

    # Check if the new json file exists. If so, add "_1" to the end of the new json file name
    if os.path.exists(new_json_path) and save_to_new_json:
        new_json_path = new_json_path.replace('.json', '_1.json')
        print(f"New JSON file saved at {new_json_path}")

    # Read the old JSON file containing all the data
    with open(json_path, 'r') as file:
        data = json.load(file)

    # Extract image path and keypoint_3d for each item in the data
    extracted_data = []
    for item in data['data']:
        extracted_item = {
            'img_path': item['img_path'],
            'keypoint_3d': item['keypoint_3d']
        }
        extracted_data.append(extracted_item)


    if save_to_new_json:
        new_data = []

    for item in extracted_data:
        print(f"Image path: {item['img_path']}")

        obj_path = os.path.join(dataset_path, item['img_path'].replace('images', 'obj_files').replace('.JPEG', '.obj'))
        quit_requested, save_requested = visualize_animal3d(obj_path, item['keypoint_3d'])
        
        if quit_requested:
            print("Visualization stopped by user.")
            break
        
        if save_requested:
            new_data.append(item)
            print(f"Saved {item['img_path']} to new JSON file.")

    if save_to_new_json and new_data:
        with open(new_json_path, 'w') as file:
            json.dump(new_data, file)
        print(f"New JSON file saved at {new_json_path}")

