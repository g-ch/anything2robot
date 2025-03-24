import numpy as np
from stl import mesh
import argparse
import os

def translate_stl(input_file, output_file, translation_vector):
    """
    Translate (move) an STL file by the given translation vector.
    
    Parameters:
    input_file (str): Path to the input STL file
    output_file (str): Path to save the translated STL file
    translation_vector (tuple/list): (x, y, z) translation vector
    """
    try:
        # Load the STL file
        print(f"Loading mesh from {input_file}...")
        model = mesh.Mesh.from_file(input_file)
        
        # Get current mesh properties
        old_min = model.vectors.min(axis=(0, 1))
        old_max = model.vectors.max(axis=(0, 1))
        old_center = (old_min + old_max) / 2
        
        print(f"Original mesh bounds: min={old_min}, max={old_max}, center={old_center}")
        
        # Convert translation vector to numpy array
        translation = np.array(translation_vector, dtype=float)
        
        # Apply translation to all vertices
        model.vectors += translation
        
        # Get new mesh properties
        new_min = model.vectors.min(axis=(0, 1))
        new_max = model.vectors.max(axis=(0, 1))
        new_center = (new_min + new_max) / 2
        
        print(f"Applied translation: {translation}")
        print(f"New mesh bounds: min={new_min}, max={new_max}, center={new_center}")
        
        # Save the translated model
        model.save(output_file)
        
        print(f"Successfully translated {input_file} and saved to {output_file}")
        return True
        
    except Exception as e:
        print(f"Error translating STL file: {e}")
        return False

def move_to_origin(input_file, output_file, center_at_origin=True):
    """
    Move an STL file so that its minimum corner or center is at the origin.
    
    Parameters:
    input_file (str): Path to the input STL file
    output_file (str): Path to save the translated STL file
    center_at_origin (bool): If True, center the model at origin.
                            If False, move the minimum corner to origin.
    """
    try:
        # Load the STL file
        print(f"Loading mesh from {input_file}...")
        model = mesh.Mesh.from_file(input_file)
        
        # Get current mesh bounds
        min_corner = model.vectors.min(axis=(0, 1))
        max_corner = model.vectors.max(axis=(0, 1))
        center = (min_corner + max_corner) / 2
        
        print(f"Original mesh bounds: min={min_corner}, max={max_corner}, center={center}")
        
        # Calculate translation vector based on mode
        if center_at_origin:
            # Move center to origin
            translation = -center
            print(f"Moving center to origin...")
        else:
            # Move minimum corner to origin
            translation = -min_corner
            print(f"Moving minimum corner to origin...")
        
        # Apply translation
        model.vectors += translation
        
        # Get new mesh bounds
        new_min = model.vectors.min(axis=(0, 1))
        new_max = model.vectors.max(axis=(0, 1))
        new_center = (new_min + new_max) / 2
        
        print(f"Applied translation: {translation}")
        print(f"New mesh bounds: min={new_min}, max={new_max}, center={new_center}")
        
        # Save the translated model
        model.save(output_file)
        
        print(f"Successfully translated {input_file} and saved to {output_file}")
        return True
        
    except Exception as e:
        print(f"Error translating STL file: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Translate (move) an STL file")
    parser.add_argument("input_file", help="Input STL file path")
    parser.add_argument("output_file", nargs="?", help="Output STL file path (default: input_translated.stl)")
    
    # Create a mutually exclusive group for translation options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--translate", "-t", nargs=3, type=float, metavar=("X", "Y", "Z"),
                       help="Translation vector (x, y, z)")
    group.add_argument("--center", "-c", action="store_true",
                       help="Center the model at the origin (0,0,0)")
    group.add_argument("--origin", "-o", action="store_true",
                       help="Move the minimum corner to the origin (0,0,0)")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.isfile(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        return
    
    # Set default output filename if not provided
    if args.output_file is None:
        base, ext = os.path.splitext(args.input_file)
        args.output_file = f"{base}_translated{ext}"
    
    # Perform the translation based on the chosen option
    if args.translate:
        translate_stl(args.input_file, args.output_file, args.translate)
    elif args.center:
        move_to_origin(args.input_file, args.output_file, center_at_origin=True)
    elif args.origin:
        move_to_origin(args.input_file, args.output_file, center_at_origin=False)

if __name__ == "__main__":
    main()