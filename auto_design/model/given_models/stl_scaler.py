import numpy as np
from stl import mesh
import sys
import os

def scale_stl(input_file, output_file, scale_factor):
    """
    Scale an STL file by the given scale factor.
    
    Parameters:
    input_file (str): Path to the input STL file
    output_file (str): Path to save the scaled STL file
    scale_factor (float or tuple): Scale factor to apply. Can be a single number for uniform scaling,
                                  or a tuple of (x_scale, y_scale, z_scale) for non-uniform scaling
    """
    try:
        # Load the STL file
        model = mesh.Mesh.from_file(input_file)
        
        # Convert scale_factor to a tuple if it's a single number
        if isinstance(scale_factor, (int, float)):
            scale_factor = (scale_factor, scale_factor, scale_factor)
        elif len(scale_factor) != 3:
            raise ValueError("Scale factor must be either a single number or a tuple of 3 values (x, y, z)")
        
        # Scale the model
        model.vectors = model.vectors * np.array(scale_factor)
        
        # Save the scaled model
        model.save(output_file)
        
        print(f"Successfully scaled {input_file} by {scale_factor} and saved to {output_file}")
        return True
    except Exception as e:
        print(f"Error scaling STL file: {e}")
        return False

def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  For uniform scaling: python scale_stl.py input.stl scale_factor [output.stl]")
        print("  For non-uniform scaling: python scale_stl.py input.stl x_scale y_scale z_scale [output.stl]")
        print("\nExamples:")
        print("  python scale_stl.py model.stl 2.0")
        print("  python scale_stl.py model.stl 2.0 scaled_model.stl")
        print("  python scale_stl.py model.stl 2.0 1.5 1.0 scaled_model.stl")
        return
    
    input_file = sys.argv[1]
    
    # Check if the input file exists
    if not os.path.isfile(input_file):
        print(f"Error: Input file '{input_file}' not found")
        return
    
    # Determine if it's uniform or non-uniform scaling
    if len(sys.argv) >= 5 and all(is_number(sys.argv[i]) for i in range(2, 5)):
        # Non-uniform scaling
        x_scale = float(sys.argv[2])
        y_scale = float(sys.argv[3])
        z_scale = float(sys.argv[4])
        scale_factor = (x_scale, y_scale, z_scale)
        output_idx = 5
    else:
        # Uniform scaling
        scale_factor = float(sys.argv[2])
        output_idx = 3
    
    # Determine output filename
    if len(sys.argv) > output_idx:
        output_file = sys.argv[output_idx]
    else:
        # Create default output filename
        base, ext = os.path.splitext(input_file)
        if isinstance(scale_factor, tuple):
            suffix = f"_scaled_{scale_factor[0]}_{scale_factor[1]}_{scale_factor[2]}"
        else:
            suffix = f"_scaled_{scale_factor}"
        output_file = f"{base}{suffix}{ext}"
    
    # Scale the STL file
    scale_stl(input_file, output_file, scale_factor)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

if __name__ == "__main__":
    main()