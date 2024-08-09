import vtk
import os
import argparse

def read_vtu(file_path):
    # Read the vtu file
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(file_path)
    reader.Update()

    # Get the output of the reader
    unstructured_grid = reader.GetOutput()
    
    # Get points from the unstructured grid
    points = unstructured_grid.GetPoints()
    number_of_points = unstructured_grid.GetNumberOfPoints()
    
    # Extract points coordinates
    points_coordinates = []
    for i in range(number_of_points):
        points_coordinates.append(points.GetPoint(i))
    
    # Get cells (elements) from the unstructured grid
    # cells = unstructured_grid.GetCells()
    # cell_types = unstructured_grid.GetCellTypesArray()
    number_of_cells = unstructured_grid.GetNumberOfCells()
    
    elements = []
    for i in range(number_of_cells):
        cell = unstructured_grid.GetCell(i)
        cell_point_ids = cell.GetPointIds()
        element = [cell_point_ids.GetId(j) for j in range(cell_point_ids.GetNumberOfIds())]
        elements.append({
            # 'cell_type': cell_types.GetValue(i),
            'points': element
        })
    
    return points_coordinates, elements


def fix_length_non_zero_start_format_number(value, total_length):
    # Convert the number to a string with initial precision of 10 decimal places
    formatted_value = f"{value:.10f}"  # Start with 10 decimal places
    # Reduce decimal places until the formatted value fits into the specified total length
    if len(formatted_value) > total_length:
        for precision in range(9, -1, -1):
            formatted_value = f"{value:.{precision}f}"
            if len(formatted_value) <= total_length:
                break
    # Strip leading spaces
    formatted_value = formatted_value.lstrip()
    return formatted_value


def write_msh_file(in_file_path, out_file_path):    
    file_path = in_file_path
    points, elements = read_vtu(file_path)

    # Write a .msh file
    out_file = out_file_path
    file = open(out_file, "w")

    # Write the header
    file.write(f"/COM,VTU TO MESH COPY RIGHT GANG CHEN CLARENCE      UP20230616       14:23:48\n")
    file.write("/PREP7\n")
    file.write("/NOPR\n")
    file.write("/TITLE,\n")
    file.write("*IF,_CDRDOFF,EQ,1,THEN     !if solid model was read in\n")
    file.write("_CDRDOFF=             !reset flag, numoffs already performed\n")
    file.write("*ELSE              !offset database for the following FE model\n")
    file.write(f"NUMOFF,NODE,{len(points):9d}\n")
    file.write(f"NUMOFF,ELEM,{len(elements):9d}\n")
    file.write(f"NUMOFF,TYPE,{1:9d}\n")
    file.write("*ENDIF\n")
    file.write("*SET,_RETURN ,  0.000000000000    \n")
    file.write("*SET,_STATUS ,  0.000000000000    \n")
    file.write("*SET,_UIQR   ,  65.00000000000    \n")
    file.write(f"*SET,__FLOATPARAMETER__,  {fix_length_non_zero_start_format_number(len(elements), 14)}    \n")
    file.write("*SET,__PYMAPDL_SESSION_ID__,'a3dc293362534a46b696cb38fc6     '\n")
    file.write("*SET,__TMPVAR__,  1.000000000000    \n")
    file.write(" DOF,DELETE\n")
    file.write("ETBLOCK,        1,        1\n")
    file.write("(2i9,19a9)\n")
    file.write("        1      285        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0\n")
    file.write("       -1\n")
    file.write(f"NBLOCK,6,SOLID,{len(points):9d},{len(points):9d}\n")
    file.write("(3i9,6e21.13e3)\n")

    # Write points/nodes
    for i, point in enumerate(points):
        file.write(f"{i+1:9d}{1:9d}{1:9d}{point[0]:21.13E}{point[1]:21.13E}{point[2]:21.13E}\n")

    file.write("N,UNBL,LOC,       -1,\n")
    file.write(f"EBLOCK,19,SOLID,{len(elements):9d},{len(elements):9d}\n")
    file.write("(19i10)\n")

    # Write elements
    for i, element in enumerate(elements):
        file.write("         1         1         1         1         0         0         0         0         4         0")
        file.write(f"{i+1:10d}")
        for point in element['points']:
            file.write(f"{point+1:10d}")
        file.write("\n")
    
    file.write("        -1\n")


    # Write the footer. Settings for the analysis. Use the default settings for now.
    file.write("CMBLOCK,__NODE__,NODE,       2  ! users node component definition\n")
    file.write("(8i10)\n")
    file.write(f"{1:10d}{-len(points):10d}\n")
    file.write("EXTOPT,ATTR,      0,      0,      0\n")
    file.write("EXTOPT,ESIZE,  0,  0.0000    \n")
    file.write("EXTOPT,ACLEAR,      0\n")
    file.write("TREF,  0.00000000\n")
    file.write("IRLF,  0\n")
    file.write("BFUNIF,TEMP,_TINY\n")
    file.write("ACEL,  0.00000000    ,  0.00000000    ,  0.00000000\n")
    file.write("OMEGA,  0.00000000    ,  0.00000000    ,  0.00000000\n")
    file.write("DOMEGA,  0.00000000    ,  0.00000000    ,  0.00000000\n")
    file.write("CGLOC,  0.00000000    ,  0.00000000    ,  0.00000000\n")
    file.write("CGOMEGA,  0.00000000    ,  0.00000000    ,  0.00000000\n")
    file.write("DCGOMG,  0.00000000    ,  0.00000000    ,  0.00000000\n")

    file.write("KUSE,     0\n")
    file.write("TIME,  0.00000000\n")
    file.write("ALPHAD,  0.00000000\n")
    file.write("BETAD,  0.00000000\n")
    file.write("DMPRAT,  0.00000000\n")
    file.write("DMPSTR,  0.00000000\n")

    file.write("CRPLIM, 0.100000000    ,   0\n")
    file.write("CRPLIM,  0.00000000    ,   1\n")
    file.write("NCNV,     1,  0.00000000    ,     0,  0.00000000    ,  0.00000000\n")
    file.write("NEQIT,     0\n")

    file.write("ERESX,DEFA\n")
    file.write("/GO\n")
    file.write("FINISH\n")


    file.close()
    print(f"File saved to {out_file}")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert VTU to ANSYS MSH")
    parser.add_argument("vtu_file", type=str, help="Path to the VTU file")
    parser.add_argument("msh_file", type=str, help="Path to save the MSH file")
    
    args = parser.parse_args()
    vtu_file = args.vtu_file
    msh_file = args.msh_file
    
    write_msh_file(vtu_file, msh_file)

    print("Done!")
