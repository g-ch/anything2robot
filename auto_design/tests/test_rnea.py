import pinocchio as pin
import numpy as np
import sys
from pinocchio.visualize import MeshcatVisualizer


urdf_dir = "C:\\Users\\MECHREV\\OneDrive\\git\\auto_design\\urdf\\lynel\\tmp\\lynel_ideal.urdf"
# urdf_dir = "C:\\Users\\MECHREV\\OneDrive\\git\\auto_design\\test.urdf"
mesh_dir = "C:\\Users\\MECHREV\\OneDrive\\git\\"

model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_dir, mesh_dir)
viz = MeshcatVisualizer(model, collision_model, visual_model)
viz.initViewer()
viz.loadViewerModel()


# Build a data frame associated with the model
data = model.createData()

# Sample a random joint configuration, joint velocities and accelerations
# q = 1.57 * np.ones((model.nq, 1))
# q = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0, 1.57, 0.0, -1.57, 0.0]).reshape(-1, 1)
q = np.array([0.0, 0.0, 0.0, -1.57, 1.57, 1.57, -1.57, 1.57, -1.57, -1.57, 1.57]).reshape(-1, 1)
v = np.zeros((model.nv, 1))  # in rad/s 
a = np.zeros((model.nv, 1))  # in rad/s² 

viz.display(q)

# Computes the inverse dynamics (RNEA) for all the joints of the robot
tau = pin.rnea(model, data, q, v, a)

# Print out to the vector of joint torques (in N.m)
print("Sampled Configuration: ", q, v.flatten(), a.flatten())
print("Joint torques: " + str(tau))