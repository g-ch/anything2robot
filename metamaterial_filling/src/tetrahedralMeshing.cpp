/***
 * This file is an example of how to use the CGAL library to generate a tetrahedral mesh from a given polyhedron.
 * ****/

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>

#include <CGAL/Mesh_triangulation_3.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Mesh_criteria_3.h>

#include <CGAL/Polyhedral_mesh_domain_with_features_3.h>
#include <CGAL/make_mesh_3.h>
#include <CGAL/IO/output_to_vtu.h>
#include <CGAL/Polyhedron_3.h>

#include <fstream>
#include <iostream>
#include <math.h>

// Domain 
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Mesh_polyhedron_3<K>::type Polyhedron;
typedef CGAL::Polyhedral_mesh_domain_with_features_3<K> Mesh_domain;

typedef K::Point_3 Point;


#ifdef CGAL_CONCURRENT_MESH_3
typedef CGAL::Parallel_tag Concurrency_tag;
#else
typedef CGAL::Sequential_tag Concurrency_tag;
#endif

// Triangulation
typedef CGAL::Mesh_triangulation_3<Mesh_domain,CGAL::Default,Concurrency_tag>::type Tr;

typedef CGAL::Mesh_complex_3_in_triangulation_3<
  Tr,Mesh_domain::Corner_index,Mesh_domain::Curve_index> C3t3;

// Criteria
typedef CGAL::Mesh_criteria_3<Tr> Mesh_criteria;


// To avoid verbose function and named parameters call
using namespace CGAL::parameters;


// Function to compute the range of nodes
void compute_ranges(const Polyhedron& polyhedron, 
                    double& min_x, double& max_x, 
                    double& min_y, double& max_y, 
                    double& min_z, double& max_z) {
  if (polyhedron.size_of_vertices() == 0) {
    std::cerr << "The polyhedron is empty." << std::endl;
    return;
  }

  // Initialize min and max values with extreme values
  min_x = min_y = min_z = std::numeric_limits<double>::infinity();
  max_x = max_y = max_z = -std::numeric_limits<double>::infinity();

  for (auto it = polyhedron.vertices_begin(); it != polyhedron.vertices_end(); ++it) {
    const Point& p = it->point();
    if (p.x() < min_x) min_x = p.x();
    if (p.x() > max_x) max_x = p.x();
    if (p.y() < min_y) min_y = p.y();
    if (p.y() > max_y) max_y = p.y();
    if (p.z() < min_z) min_z = p.z();
    if (p.z() > max_z) max_z = p.z();
  }
}



int main(int argc, char*argv[])
{
  //  const char* fname = (argc>1)?argv[1]:"data/fandisk.off";

  // Check if there are at least 5 arguments. If not, print the usage and return. Note surface_accuracy will affect the number of elements.
  if (argc < 5) {
    std::cerr << "Usage: " << argv[0] << " <input_file (.off)>  <output_file (.vtu)>  <desired_number_of_elements>  <surface_accuracy (0-1.0) 0.5 suggested>" << std::endl;
    return EXIT_FAILURE;
  }

  const char* fname = argv[1];
  const char* output_file = argv[2];
  int desired_number_of_elements = std::stoi(argv[3]); // Desired number of elements in the mesh. Note that this number is not guaranteed to be achieved.
  double surface_accuracy = std::stod(argv[4]); // Surface accuracy (0-1). 0.5 is suggested.

  std::ifstream input(fname);
  Polyhedron polyhedron;
  input >> polyhedron;
  if(input.fail()){
    std::cerr << "Error: Cannot read file " <<  fname << std::endl;
    return EXIT_FAILURE;
  }

  if (!CGAL::is_triangle_mesh(polyhedron)){
    std::cerr << "Input geometry is not triangulated." << std::endl;
    return EXIT_FAILURE;
  }

  // Get the xyz range of the nodes
  double min_x, max_x, min_y, max_y, min_z, max_z;
  compute_ranges(polyhedron, min_x, max_x, min_y, max_y, min_z, max_z);

  double x_range = max_x - min_x;
  double y_range = max_y - min_y;
  double z_range = max_z - min_z;

  std::cout << "X range: " << x_range << std::endl;
  std::cout << "Y range: " << y_range << std::endl;
  std::cout << "Z range: " << z_range << std::endl;

  // Calculate the parameters for meshing. The autofilling parameters can be improved. 
  double max_edge_size = pow((x_range * y_range * z_range) / desired_number_of_elements, 1.0/3.0) * 3; // Suppose that the domain is a cube then use three to correct roughly.
  double max_facet_size = max_edge_size;

  double divisor = 1000.0;

  // Map surface_accuracy (0-1) to divisor 10 to 1000 using a linear function
  if (surface_accuracy < 0.1) {
    divisor = 10.0;
  } else if (surface_accuracy > 0.9) {
    divisor = 1000.0;
  } else {
    divisor = 10.0 + (surface_accuracy - 0.1) * 990.0 / 0.8;
  }

  double max_facet_distance = std::min({x_range, y_range, z_range}) / divisor; // 0.1% of the smallest range
  double max_cell_size = max_edge_size;

  std::cout << "Max edge size: " << max_edge_size << std::endl;

  // Create domain
  Mesh_domain domain(polyhedron);

  // Get sharp features
  domain.detect_features();



  // Mesh criteria
  /******************
   * edge_size: This parameter sets the maximum allowed length of the edges in the generated mesh. In your case, edge_size = 0.025 means that no edge in the mesh will be longer than 0.025 units.
   * facet_angle: This parameter sets the minimum allowed angle (in degrees) of the facets in the mesh. facet_angle = 25 ensures that the angles of the facets are at least 25 degrees. This helps in avoiding very sharp triangles which can be undesirable for numerical stability and mesh quality.
   * facet_size: This parameter sets the maximum allowed size of the facets. facet_size = 0.01 means that the longest edge of any facet (triangle) will not be longer than 0.01 units.
   * facet_distance: This parameter sets the maximum allowed distance from the center of a facet to the surface approximation. facet_distance = 0.001 ensures that the facets conform closely to the surface of the domain, with a maximum deviation of 0.001 units.
   * cell_radius_edge_ratio: This parameter sets the maximum allowed ratio of the circumscribed radius of the cells (tetrahedra) to the length of their shortest edge. cell_radius_edge_ratio = 3 ensures that the cells are not too elongated. A lower ratio results in better-shaped (more isotropic) tetrahedra.
   * cell_size: This parameter sets the maximum allowed size of the cells. cell_size = 0.01 ensures that the circumradius of any cell will not exceed 0.01 units. This helps in controlling the overall size of the tetrahedra in the mesh.
  ***************/
  Mesh_criteria criteria(edge_size = max_edge_size,
                          facet_angle = 30, facet_size = max_facet_size, facet_distance = max_facet_distance,
                          cell_radius_edge_ratio = 2, cell_size = max_cell_size);


  // Mesh generation
  std::cout << "Generating mesh..." << std::endl;
  
  C3t3 c3t3 = CGAL::make_mesh_3<C3t3>(domain, criteria);
  
  std::cout << "Meshing done." << std::endl;

  // Output
  std::ofstream file(output_file);
  CGAL::output_to_vtu(file, c3t3, CGAL::IO::ASCII); // USE ASCII. Otherwise, the file is hard to be converted to .msh format

  std::cout << "Mesh saved to: " << output_file << std::endl;

  return EXIT_SUCCESS;
}