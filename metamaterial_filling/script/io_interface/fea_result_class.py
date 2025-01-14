'''
This file contains the class definition for the FEA_Opt_Result class.
self.success_flag: True if the optimization was successful, False otherwise
self.best_relative_density: The best relative density found by the optimization. Float
self.young_modulus: The young's modulus of the optimized structure. Float
self.von_mises: The von mises stress of the optimized structure. NumPy array. Shape: (num_nodes,)
self.displacement_magnitude: The displacement magnitude of the optimized structure. NumPy array. Shape: (num_nodes,)
self.nodes: The nodal coordinates of the optimized structure. NumPy array. Shape: (num_nodes, 3)
self.max_allowed_stress_material: The maximum allowed stress in the structure. Float
self.max_allowed_displacement: The maximum allowed displacement in the structure. Float
self.nodes_seq_exceeding_stress: Sequence of nodes that exceed the maximum allowed stress. List of integers.
'''

class FEA_Opt_Result:
    def __init__(self, filename):
        self.filename = filename
        
        # success_flag, best_relative_density, young_modulus, von_mises, displacement_magnitude, nodes
        self.success_flag = None
        self.best_relative_density = None
        self.young_modulus = None
        self.von_mises = None
        self.displacement_magnitude = None
        self.nodes = None

        self.max_allowed_stress_material = None
        self.max_allowed_displacement = None

        self.nodes_seq_exceeding_stress = None

    def set_result(self, success_flag, best_relative_density, young_modulus, von_mises, displacement_magnitude, nodes, max_allowed_stress_material, max_allowed_displacement, nodes_seq_exceeding_stress):
        self.success_flag = success_flag
        self.best_relative_density = best_relative_density
        self.young_modulus = young_modulus
        self.von_mises = von_mises
        self.displacement_magnitude = displacement_magnitude
        self.nodes = nodes

        self.max_allowed_stress_material = max_allowed_stress_material
        self.max_allowed_displacement = max_allowed_displacement

        self.nodes_seq_exceeding_stress = nodes_seq_exceeding_stress
    
    def get_result(self):
        return self.success_flag, self.best_relative_density, self.young_modulus, self.von_mises, self.displacement_magnitude, self.nodes, self.max_allowed_stress_material, self.max_allowed_displacement, self.nodes_seq_exceeding_stress

    def show_result(self):
        print('success_flag:', self.success_flag)
        print('best_relative_density:', self.best_relative_density)
        print('young_modulus:', self.young_modulus)
        print('max_allowed_stress_material:', self.max_allowed_stress_material)
        print('max_allowed_displacement:', self.max_allowed_displacement)
        print('nodes_seq_exceeding_stress:', self.nodes_seq_exceeding_stress)

        print('von_mises:', self.von_mises)
        print('displacement_magnitude:', self.displacement_magnitude)
        print('nodes:', self.nodes)

        

   