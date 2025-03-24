


'''
Motor parameter library. This is used to store the motor parameters that is used in the optimization process.
'''
class MotorParameterLib:
    def __init__(self):
        self.tenon_height = 1.5 # cm. Tenon is the connection part between the motor and the link.

        # Height (cm), Radius (cm), Max Torque (N*M) 
        self.motor_lib = [        
                          #[2.25, 2.15, 0.9], # GIM3505. SITAIWEI
                          [3.42, 2.65, 2.5], # MG4005V2. K-Tech
                          [3.65, 3.8, 12],   # DM6006. DAMIAO Tech 
                          [4.0, 4.8, 20], # DM8006. DAMIAO Tech
                          [6.2, 5.56, 120], # DM10010. DAMIAO Tech
                         ]  
        
        # TODO: CONSIDER MOTOR WEIGHT INDIVIDUALLY

        # This is the connector length between two motors in a 2 DOF joint. L shape. Unit: cm
        self.connector_lib = []
        for i in range(len(self.motor_lib)):
            # connector_length is radius + height/2 + 0.5. 0.5 is a margin.
            connector_length = self.motor_lib[i][1] + self.motor_lib[i][0]/2 + 0.5
            self.connector_lib.append([connector_length, connector_length])

        # Add self.tenon_height to the motor height to consider the tenon height
        for i in range(len(self.motor_lib)):
            self.motor_lib[i][0] += self.tenon_height
    
    def get_motor_lib(self):
        return self.motor_lib
    
    def get_connector_lib(self):
        return self.connector_lib
