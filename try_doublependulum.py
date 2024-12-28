import os
import numpy as np
import matplotlib.pyplot as plt
from adam.casadi.computations import KinDynComputations
import casadi as cs
from time import time as clock
from time import sleep
from termcolor import colored
from A3_template.train import create_casadi_function

import orc.utils.plot_utils as plut
from example_robot_data.robots_loader import load
import orc.A3_template.conf_doublep as conf_doublep
from orc.utils.robot_simulator import RobotSimulator
from orc.utils.robot_loaders import loadPendulum
from orc.utils.robot_wrapper import RobotWrapper

print("Load robot model")
robot = load("double_pendulum")

print("Create KinDynComputations object")
joints_name_list = [s for s in robot.model.names[1:]] # skip the first name because it is "universe"
nq = len(joints_name_list)  # number of joints
nx = 2*nq # size of the state variable
kinDyn = KinDynComputations(robot.urdf, joints_name_list)
ADD_SPHERE = 0
SPHERE_POS = np.array([0.2, -0.10, 0.5])
SPHERE_SIZE = np.ones(3)*0.1
SPHERE_RGBA = np.array([1, 0, 0, 1.])

# WITH THIS CONFIGURATION THE SOLVER ENDS UP VIOLATING THE JOINT LIMITS
# ADDING THE TERMINAL CONSTRAINT FIXES EVERYTHING!
# BUT SO DOES:
# - DECREASING THE POSITION WEIGHT IN THE COST
# - INCREASING THE ACCELERATION WEIGHT IN THE COST
# - INCREASING THE MAX NUMBER OF ITERATIONS OF THE SOLVER
DO_WARM_START = True
SOLVER_TOLERANCE = 1e-4
SOLVER_MAX_ITER = 40

DO_PLOTS = True
SIMULATOR = "pinocchio" #"mujoco" or "pinocchio" or "ideal"
POS_BOUNDS_SCALING_FACTOR = 0.2
VEL_BOUNDS_SCALING_FACTOR = 3.0
TORQUE_BOUNDS_SCALING_FACTOR = 0.5
qMin = np.array([-2.0*np.pi,-2.0*np.pi])#POS_BOUNDS_SCALING_FACTOR * robot.model.lowerPositionLimit
qMax = -qMin#POS_BOUNDS_SCALING_FACTOR * robot.model.upperPositionLimit
vMax = np.array([10.0,10.0])#VEL_BOUNDS_SCALING_FACTOR * robot.model.velocityLimit
dt_sim = 0.002
N_sim = 200
q0 = np.array([0, 0])  # initial joint configuration
dq0= np.zeros(nq)  # initial joint velocities

dt = 0.010 # time step MPC
N = 50 #int(N_sim/10)  # time horizon MPC
q_des = np.array([-np.pi, 0])
w_p = 1e2   # position weight
w_v = 1e-8  # velocity weight
w_a = 1e-8  # acceleration weight
w_final_v = 0e0 # final velocity cost weight
w_t = 1e2
USE_TERMINAL_CONSTRAINT = 1

if(SIMULATOR=="mujoco"):
    from orc.utils.mujoco_simulator import MujocoSimulator
    print("Creating simulator...")
    simu = MujocoSimulator("ur5", dt_sim)
    simu.set_state(q0, dq0)
else:
    r = RobotWrapper(robot.model, robot.collision_model, robot.visual_model)
    simu = RobotSimulator(conf_doublep, r)
    simu.init(q0, dq0)
    simu.display(q0)
    
print("Create optimization parameters")
''' The parameters P contain:
    - the initial state (first 12 values)
    - the target configuration (last 6 values)
'''

opti = cs.Opti()
param_x_init = opti.parameter(nx)
param_q_des = opti.parameter(nq)
cost = 0

# create the dynamics function
q   = cs.SX.sym('q', nq)
dq  = cs.SX.sym('dq', nq)
ddq = cs.SX.sym('ddq', nq)
state = cs.vertcat(q, dq)
rhs    = cs.vertcat(dq, ddq)
f = cs.Function('f', [state, ddq], [rhs])

# create a Casadi inverse dynamics function
H_b = cs.SX.eye(4)     # base configuration
v_b = cs.SX.zeros(6)   # base velocity
bias_forces = kinDyn.bias_force_fun()
mass_matrix = kinDyn.mass_matrix_fun()
# discard the first 6 elements because they are associated to the robot base
h = bias_forces(H_b, q, v_b, dq)[6:]
M = mass_matrix(H_b, q)[6:,6:]
tau = M @ ddq + h
inv_dyn = cs.Function('inv_dyn', [state, ddq], [tau])

# pre-compute state and torque bounds
lbx = qMin.tolist() + (-vMax).tolist()
ubx = qMax.tolist() + vMax.tolist()
tau_min = (np.array([-10.0, -10.0])*TORQUE_BOUNDS_SCALING_FACTOR).tolist() #(-robot.model.effortLimit).tolist()
tau_max = (np.array([10.0, 10.0])*TORQUE_BOUNDS_SCALING_FACTOR).tolist() #robot.model.effortLimit.tolist()
print('lbx',lbx)
print('ubx',ubx)
print('tau_min',tau_min)
print('tau_max',tau_max)

# LOAD THE NN AND TRANSFORM IT IN A CASADI FUNCTION
A3_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(A3_dir, "nn_models")
input_size = 4  # Adjust this to match your neural network's input size

# Load the CasADi function
back_reach_set_fun = create_casadi_function(robot_name="double_pendulum", NN_DIR=model_dir, input_size=input_size)

# create all the decision variables
X, U = [], []
X += [opti.variable(nx)] # do not apply pos/vel bounds on initial state
for k in range(1, N+1): 
    X += [opti.variable(nx)]
    opti.subject_to(opti.bounded(lbx, X[-1], ubx))
for k in range(N): 
    U += [opti.variable(nq)]
    opti.subject_to(opti.bounded(tau_min, U[-1], tau_max))

print("Add initial conditions")
opti.subject_to(X[0] == param_x_init)
for k in range(N):   

    # print("Compute cost function")
    cost += w_p * (X[k][:nq] - param_q_des).T @ (X[k][:nq] - param_q_des)
    cost += w_v * X[k][nq:].T @ X[k][nq:]
    cost += w_a * U[k].T @ U[k]

    # print("Add dynamics constraints")
    opti.subject_to(X[k+1] == X[k] + dt * f(X[k], U[k]))

    # print("Add physical constraints")
    opti.subject_to(X[k][0] >= -3.4)

    # print("Add torque constraints")
    opti.subject_to( opti.bounded(tau_min, inv_dyn(X[k], U[k]), tau_max))

# add the final cost
cost += w_final_v * X[-1][nq:].T @ X[-1][nq:]

if(USE_TERMINAL_CONSTRAINT):
    #opti.subject_to(X[-1][nq:] == 0.0)
    opti.subject_to(back_reach_set_fun(X[-1]) >= 0.5)

# print("Constraints added to the optimization:")
# for constraint in cs.vertsplit(opti.g, 1):  # List all constraints
#     print(constraint)

opti.minimize(cost)

print("Create the optimization problem")
opts = {
    "error_on_fail": False,
    "ipopt.print_level": 0,
    "ipopt.tol": SOLVER_TOLERANCE,
    "ipopt.constr_viol_tol": SOLVER_TOLERANCE,
    "ipopt.compl_inf_tol": SOLVER_TOLERANCE,
    "print_time": 0,                # print information about execution time
    "detect_simple_bounds": True,
    "ipopt.nlp_scaling_method": "gradient-based",
    "ipopt.max_iter": 1000,
    "ipopt.hessian_approximation": "limited-memory" 
}
opti.solver("ipopt", opts)

# set up simulation environment
if(SIMULATOR=="mujoco" and ADD_SPHERE):
    simu.add_sphere(pos=SPHERE_POS, size=SPHERE_SIZE, rgba=SPHERE_RGBA)

# Solve the problem to convergence the first time
x = np.concatenate([q0, dq0])
opti.set_value(param_q_des, q_des)
opti.set_value(param_x_init, x)

### List all constraints -> this part used only for debugging
# print("Constraints added to the optimization:")
# for i, constraint in enumerate(cs.vertsplit(opti.g, 1)):
#    print(f"Constraint {i}: {constraint}")

sol = opti.solve()

print("Last state:", opti.debug.value(X[-1]))
print("Last control:", opti.debug.value(U[-1]))

opts["ipopt.max_iter"] = SOLVER_MAX_ITER
opti.solver("ipopt", opts)

data = np.zeros((N_sim, 8))

print("Start the MPC loop")
for i in range(N_sim):
    start_time = clock()

    if(DO_WARM_START):
        # use current solution as initial guess for next problem
        for t in range(N):
            opti.set_initial(X[t], sol.value(X[t+1]))
        for t in range(N-1):
            opti.set_initial(U[t], sol.value(U[t+1]))
        opti.set_initial(X[N], sol.value(X[N]))
        opti.set_initial(U[N-1], sol.value(U[N-1]))
        # initialize dual variables
        lam_g0 = sol.value(opti.lam_g)
        opti.set_initial(opti.lam_g, lam_g0)

    print("Time step", i, "State", x)
    opti.set_value(param_x_init, x)
    try:
        sol = opti.solve()
    except:
        sol = opti.debug
        # print("Convergence failed!")
    end_time = clock()

    print("Comput. time: %.3f s"%(end_time-start_time), 
          "Iters: %3d"%sol.stats()['iter_count'], 
          "Tracking err: %.3f"%np.linalg.norm(q_des - x[:nq]))

    if USE_TERMINAL_CONSTRAINT:
        final_state = sol.value(X[-1])  # Extract the numerical value of X[-1] from the solution
        terminal_value = back_reach_set_fun(final_state)  # Evaluate the neural network output
        print("TERMINAL CONSTRAINT VALUE: ", terminal_value)

    tau = inv_dyn(sol.value(X[0]), sol.value(U[0])).toarray().squeeze()

    # Store all the data
    data[i][0] = x[0] # joint 1 coord
    data[i][1] = x[1] # joint 2 coord
    data[i][2] = x[2] # joint 1 vel
    data[i][3] = x[3] # joint 2 vel
    data[i][4] = tau[0] #sol.value(U[1][0]) # joint 1 torque
    data[i][5] = tau[1] #sol.value(U[1][1]) # joint 2 torque
    data[i][6] = sol.value(U[0][0])
    data[i][7] = sol.value(U[0][1])

    if(SIMULATOR=="mujoco"):
        # do a proper simulation with Mujoco
        simu.step(tau, dt)
        x = np.concatenate([simu.data.qpos, simu.data.qvel])
    elif(SIMULATOR=="pinocchio"):
        # do a proper simulation with Pinocchio
        simu.simulate(tau, dt, int(dt/dt_sim))
        x = np.concatenate([simu.q, simu.v])
    elif(SIMULATOR=="ideal"):
        # use state predicted by the MPC as next state
        x = sol.value(X[1])
        simu.display(x[:nq])

    if( np.any(x[:nq] > qMax)):
        print(colored("\nUPPER POSITION LIMIT VIOLATED ON JOINTS", "red"), np.where(x[:nq]>qMax)[0])
    if( np.any(x[:nq] < qMin)):
        print(colored("\nLOWER POSITION LIMIT VIOLATED ON JOINTS", "red"), np.where(x[:nq]<qMin)[0])

    if( np.any(x[0] <= -3.4)):
        print(colored("\nIMPACT DETECTED", "red"))

    if( np.any(x[nq:] > vMax)):
        print(colored("\nUPPER VELOCITY LIMIT VIOLATED ON JOINTS", "red"), np.where(x[:nq]>qMax)[0])
    if( np.any(x[nq:] < -vMax)):
        print(colored("\nLOWER VELOCITY LIMIT VIOLATED ON JOINTS", "red"), np.where(x[:nq]<qMin)[0])

    if( np.any(tau > tau_max)):
        print(colored("\nUPPER TORQUE LIMIT VIOLATED ON JOINTS", "red"), np.where(x[:nq]>qMax)[0])
    if( np.any(tau < tau_min)):
        print(colored("\nLOWER TORQUE LIMIT VIOLATED ON JOINTS", "red"), np.where(x[:nq]<qMin)[0])

# Plots 
# plot joint trajectories
if(DO_PLOTS):
    time = np.arange(0, (N_sim)*dt_sim, dt_sim)
    # Plot of the joint coordinates
    plt.figure(figsize=(10, 6))
    for i in range(nq):
        plt.plot(time, data[:,i], label=f'q {i}', alpha=0.7)
        plt.plot(time, np.full_like(time, q_des[i]), linestyle='dotted',label=f'q {i} desired' )
    plt.xlabel('Time [s]')
    plt.ylabel('Joint coordinates [m]')
    plt.title('Joint coordinates')
    plt.legend()
    plt.grid(True)

    # Plot of joints velocities
    plt.figure(figsize=(10, 6))
    for i in range(nq):
        plt.plot(time, data[:,i+nq], label=f'vel_q {i}', alpha=0.7)
        plt.plot(time, np.full_like(time, vMax[i]), linestyle='dotted',label=f'Maximum vel joint {i}' )
        plt.plot(time, np.full_like(time, -vMax[i]), linestyle='dotted',label=f'Minimum vel joint {i}' )
    plt.xlabel('Time [s]')
    plt.ylabel('Joint velocities [m/s]')
    plt.title('Joint velocities')
    plt.legend()
    plt.grid(True)

    # Plot of joints torques
    plt.figure(figsize=(10, 6))
    for i in range(nq):
        plt.plot(time, data[:,i+4], label=f'torques {i}', alpha=0.7)
        plt.plot(time, np.full_like(time, tau_max[i]), linestyle='dotted',label=f'Maximum torque joint {i}' )
        plt.plot(time, np.full_like(time, tau_min[i]), linestyle='dotted',label=f'Minimum torque joint {i}' )
    plt.xlabel('Time [s]')
    plt.ylabel('Joint torques [Nm]')
    plt.title('Joint torques')
    plt.legend()
    plt.grid(True)

    # Plot of joints torques
    plt.figure(figsize=(10, 6))
    for i in range(nq):
        plt.plot(time, data[:,i+6], label=f'Control U {i}', alpha=0.7)
        plt.plot(time, np.full_like(time, tau_max[i]), linestyle='dotted',label=f'Maximum torque joint {i}' )
        plt.plot(time, np.full_like(time, tau_min[i]), linestyle='dotted',label=f'Minimum torque joint {i}' )
    plt.xlabel('Time [s]')
    plt.ylabel('Joint Control [Nm]')
    plt.title('Joint Control')
    plt.legend()
    plt.grid(True)
   
    plt.show()