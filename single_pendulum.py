import numpy as np
import matplotlib.pyplot as plt
from adam.casadi.computations import KinDynComputations
import casadi as cs
from time import time as clock
from termcolor import colored

from utils import plot_utils as plut
from example_robot_data.robots_loader import load, load_full
from utils.robot_simulator import RobotSimulator
from utils.robot_wrapper import RobotWrapper 

SOLVER_TOLERANCE = 1e-4
DO_PLOTS = True
DO_WARM_START = True
SOLVER_MAX_ITER = 3

m = 1.0       # Mass of the pendulum (kg)
L = 1.0       # Length of the pendulum (m)
g = 9.81      # Gravitational acceleration (m/s^2)

# Define position, velocity, and torque bounds
qMin = np.array([-np.pi])  # pendulum can rotate fully
qMax = np.array([np.pi])
vMax = np.array([100.0])    # max angular velocity
tauMax = np.array([100.0])   # max torque

lbx = np.concatenate([qMin, -vMax])
ubx = np.concatenate([qMax, vMax])

nq = 1             # number of joints of a single pendolum
nx = 2*nq          # size of state variables

dt_sim = 0.002
N_sim = 300
dt = 0.010          # time step MPC
N = int(N_sim/10)           # horizon length MPC ### TODO: ###

q0 = np.array([3.14])#np.zeros(nq)   # initial joint configuration
dq0= np.array([0]) #np.zeros(nq)   # initial joint velocities

p_ee_des = np.array([0.0, 0.0, 1.0]) # desired end-effector position -> vertical configuration

data = np.zeros((N_sim, 3))

# Cost weight
w_p = 1e2           # position weight
w_v = 1e-1          # velocity weight
w_a = 1e-5          # acceleration weight
w_final_v = 0e0     # final velocity cost weight
USE_TERMINAL_CONSTRAINT = 0 ### TODO: 1 ###
USE_FINAL_COST = 0

# Create optimization parameters
opti = cs.Opti()
param_x_init = opti.parameter(nx)
param_p_ee_des = opti.parameter(3)
cost = 0

# Set initial condition
# opti.set_value(param_x_init, [2,0])

# create the dynamics function
q   = cs.SX.sym('q', nq)
dq  = cs.SX.sym('dq', nq)
ddq = cs.SX.sym('ddq', nq)
state = cs.vertcat(q, dq)
rhs    = cs.vertcat(dq, ddq)
f = cs.Function('f', [state, ddq], [rhs])

# Define the mass matrix and bias forces
M = cs.SX(m * L**2)                  # Scalar inertia for 1 DoF
h = cs.SX(m * g * L * cs.sin(q))     # Gravity term

# Define torque (inverse dynamics)
tau = M * ddq + h

# Create the CasADi function
inv_dyn = cs.Function('inv_dyn', [state, ddq], [tau])

# Forward kinematics function
# the angle q = 0 correspond to the vertical position
def forward_kinematics(q):
    x = L * cs.sin(q)
    y = 0
    z = L * cs.cos(q)      
    return cs.vertcat(x,y,z)

# Create all decision variables
X, U = [], []
X += [opti.variable(nx)]  # Initial state (no bound on initial state)
for k in range(1, N+1):
    X += [opti.variable(nx)]
    opti.subject_to(opti.bounded(lbx, X[-1], ubx))
for k in range(N):
    U += [opti.variable(nq)]

# Add initial condition
opti.subject_to(X[0] == param_x_init)

for k in range(N):
    # Compute end-effector position
    p_ee_actual = forward_kinematics(X[k][:nq])

    # Position cost
    cost += w_p * (p_ee_actual - param_p_ee_des).T @ (p_ee_actual - param_p_ee_des)
    # Velocity cost
    cost += w_v * (X[k][nq:]).T @ (X[k][nq:])
    # Control effort cost
    cost += w_a * (U[k]).T @ (U[k])

    # Add dynamics constraints
    opti.subject_to(X[k+1] == X[k] + dt * f(X[k], U[k]))

    # Add torque constraints
    opti.subject_to( opti.bounded(-tauMax, inv_dyn(X[k], U[k]), tauMax))

if (USE_FINAL_COST):
    cost += w_final_v * X[-1][nq:].T @ X[-1][nq:] # terminal cost in velocity

# Final velocity cost (optional)
if (USE_TERMINAL_CONSTRAINT):
    opti.subject_to(X[-1][nq:] == 0.0)  # Terminal velocity = 0
    
opti.minimize(cost)

# Set solver options
opts = {
    "error_on_fail": False,
    "ipopt.print_level": 0,
    "ipopt.tol": SOLVER_TOLERANCE,
    "ipopt.constr_viol_tol": SOLVER_TOLERANCE,
    "ipopt.compl_inf_tol": SOLVER_TOLERANCE,
    "print_time": 0,             
    "detect_simple_bounds": True,
    "ipopt.max_iter": 1000
}
opti.solver("ipopt", opts)

# Set parameters and solve in an MPC loop
x = np.concatenate([q0, dq0])
opti.set_value(param_p_ee_des, p_ee_des)
opti.set_value(param_x_init, x)
sol = opti.solve()

opts["ipopt.max_iter"] = SOLVER_MAX_ITER
opti.solver("ipopt", opts)
mean_comp_time = 0

# Start the MPC loop
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

    try:
        sol = opti.solve()
    except:
        print("Convergence failed!")
        sol = opti.debug.value

    end_time = clock()
    mean_comp_time += (end_time-start_time)/N_sim

    # Extract the optimal control input
    u_opt = sol.value(U[0])

    # Simulate the pendulum with the optimal control input
    x = x + dt_sim * f(x, u_opt).full().flatten()

    # Update initial state for the next iteration
    opti.set_value(param_x_init, x)

    data[i,0] = x[0]
    data[i,1] = x[1]
    data[i,2] = u_opt

    # Log results (e.g., position, velocity, control input)
    print("Comput. time: %.3f s"%(end_time-start_time), 
          "Iters: %3d"%sol.stats()['iter_count'], 
          "Tracking err: %.3f"%np.linalg.norm(p_ee_des-forward_kinematics(x[:nq])),
          "Return status", sol.stats()["return_status"])
    #print(f"Step {i}, Position: {x[0]:.3f}, Velocity: {x[1]:.3f}, Torque: {u_opt:.3f}")

# Plots
if DO_PLOTS:
    time = np.arange(0, N_sim * dt_sim, dt_sim)

    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Position vs. Time
    plt.subplot(3, 1, 1)  # 3 rows, 1 column, first subplot
    plt.plot(time, data[:, 0], label="Position (rad)", color='b')
    plt.xlabel("Time (s)")
    plt.ylabel("Position (rad)")
    plt.title("Pendulum Position Over Time")
    plt.grid()
    plt.legend()
    
    # Subplot 2: Velocity vs. Time
    plt.subplot(3, 1, 2)  # 3 rows, 1 column, second subplot
    plt.plot(time, data[:, 1], label="Velocity (rad/sec)", color='g')
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/sec)")
    plt.title("Pendulum Velocity Over Time")
    plt.grid()
    plt.legend()
    
    # Subplot 3: Torque vs. Time
    plt.subplot(3, 1, 3)  # 3 rows, 1 column, third subplot
    plt.plot(time, data[:, 2], label="Torque (Nm)", color='r')
    plt.xlabel("Time (s)")
    plt.ylabel("Torque (Nm)")
    plt.title("Control Torque Over Time")
    plt.grid()
    plt.legend()
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
