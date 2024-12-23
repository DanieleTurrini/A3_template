from orc.A3_template.BwRS import is_in_BwRS
import numpy as np
from example_robot_data.robots_loader import load

qMin = np.array([-2.0*np.pi,-2.0*np.pi])#POS_BOUNDS_SCALING_FACTOR * robot.model.lowerPositionLimit
qMax = -qMin#POS_BOUNDS_SCALING_FACTOR * robot.model.upperPositionLimit
vMax = np.array([10.0,10.0])#VEL_BOUNDS_SCALING_FACTOR * robot.model.velocityLimit

q_bound = np.array([qMin[0],qMin[1],-vMax[0],-vMax[1],qMax[0],qMax[1],vMax[0],vMax[1]])

TORQUE_BOUNDS_SCALING_FACTOR = 0.5
tau_min = (np.array([-10.0, -10.0])*TORQUE_BOUNDS_SCALING_FACTOR).tolist()
tau_max = (np.array([10.0, 10.0])*TORQUE_BOUNDS_SCALING_FACTOR).tolist()
tau_bound = np.array([tau_min[0],tau_min[1],tau_max[0],tau_max[1]])

N_sim = 200
dt = 0.010 # time step MPC
N = int(N_sim/8)  # time horizon MPC

robot = load("double_pendulum")




# Define bounds and step size
pos_bounds = [(-2.0 * np.pi, 2.0 * np.pi), (-2.0 * np.pi, 2.0 * np.pi)]  # (min, max) for each pos dimension
v_bounds = [(-3.0, 3.0), (-3.0, 3.0)]  # (min, max) for each vel dimension
step = 0.5

# Create grids for positions and velocities
pos_grid = np.array(np.meshgrid(*pos_bounds)).T.reshape(-1, len(pos_bounds))  # 2D positions
vel_grid = np.array(np.meshgrid(*v_bounds)).T.reshape(-1, len(v_bounds))  # 2D velocities

# Combine position and velocity grids to form the dataset
dataset = np.array([np.hstack((p, v)) for p in pos_grid for v in vel_grid])

print(len(dataset))


'''# Generate a Dataset
num_samples = 500
pos = np.random.uniform(-2.0*np.pi,2.0*np.pi, size=(num_samples, 2))
vel = np.random.uniform(-3*vMax,3*vMax, size=(num_samples, 2))
# Initialize states as a 2D array
states = np.zeros((num_samples, 4))

# Fill in the states array
for i in range(num_samples):
    states[i] = [pos[i, 0], pos[i, 1], vel[i, 0], vel[i, 1]]

labels = np.array([is_in_BwRS(robot,state,N,dt, q_bound, tau_bound) for state in states])
np.savez("training_data.npz", states=states, labels=labels)'''

