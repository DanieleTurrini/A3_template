import numpy as np
import matplotlib.pyplot as plt
from adam.casadi.computations import KinDynComputations
import casadi as cs
from time import time as clock
from termcolor import colored
import conf_doublep as config_doublep
import display

from orc.utils import plot_utils as plut
from example_robot_data.robots_loader import load, load_full
from orc.utils.robot_simulator import RobotSimulator
from orc.utils.robot_wrapper import RobotWrapper     

print("Load robot model")
robot, _, urdf, _ = load_full("double_pendulum")

print("Create KinDynComputations object")
joints_name_list = [s for s in robot.model.names[1:]]   # skip the first name because it is "universe"
#end_effector_frame_name = "link2"
nq = len(joints_name_list)                              # number of joints
nx = 2*nq                                               # size of the state variable
kinDyn = KinDynComputations(urdf, joints_name_list)
#forward_kinematics_ee = kinDyn.forward_kinematics_fun(end_effector_frame_name)

DO_WARM_START = True
SOLVER_TOLERANCE = 1e-4
SOLVER_MAX_ITER = 3
DO_PLOTS = True
SAVE_DATA = False

SIMULATOR = "pinocchio"
VEL_BOUNDS_SCALING_FACTOR = 1.0
CONTROL_BOUNDS_SCALING_FACTOR = 0.5 ### TODO: 0.45 ###

qMin = np.array([-2*np.pi, -2*np.pi])
qMax = np.array([2*np.pi, 2*np.pi])
vMax = VEL_BOUNDS_SCALING_FACTOR * np.array([10.0, 10.0])
tauMin = -CONTROL_BOUNDS_SCALING_FACTOR * np.array([10.0, 0])
tauMax = CONTROL_BOUNDS_SCALING_FACTOR * np.array([10.0, 0])

dt_sim = 0.002
N_sim = 200

dt = 0.010          # time step MPC
N = int(N_sim/2)           # horizon length MPC ### TODO: ###

q0 = np.array([-np.pi,0]) # initial joint configuration
dq0= np.zeros(nq)   # initial joint velocities

q_des = np.array([0.0,0.0]) # desired joint position

# COST FUNCTION WEIGTHS
w_p = 1e2           # position weight
w_v = 0 # 1e-6          # velocity weight
w_a = 1e-8          # acceleration weight
w_final_v = 0e0     # final velocity cost weight
USE_TERMINAL_CONSTRAINT = 0 ### TODO: 1 ###

# Initialize the Simulation
r = RobotWrapper(robot.model, robot.collision_model, robot.visual_model)
simu = RobotSimulator(config_doublep, r)
simu.init(q0, dq0)
simu.display(q0)

print("Create optimization parameters")
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
tau_min = (tauMin).tolist()
tau_max = (tauMax).tolist()
print('lbx',lbx)
print('ubx',ubx)
print('tau_min',tau_min)
print('tau_max',tau_max)

# create all the decision variables
X, U = [], []
X += [opti.variable(nx)] # do not apply pos/vel bounds on initial state
for k in range(1, N+1): 
    X += [opti.variable(nx)]
    opti.subject_to( opti.bounded(lbx, X[-1], ubx) )
for k in range(N): 
    U += [opti.variable(nq)]

print("Add initial conditions")
opti.subject_to(X[0] == param_x_init)

for k in range(N):  
    # print("Compute cost function")
    cost += w_p * (X[k][:nq] - param_q_des).T @ (X[k][:nq] - param_q_des)
    cost += w_v * X[k][nq:].T @ X[k][nq:]
    cost += w_a * U[k].T @ U[k]

    # print("Add dynamics constraints")
    opti.subject_to(X[k+1] == X[k] + dt * f(X[k], U[k]))

    # print("Add torque constraints")
    opti.subject_to( opti.bounded(tau_min, inv_dyn(X[k], U[k]), tau_max))

# add the final cost
cost += w_final_v * X[-1][nq:].T @ X[-1][nq:]

if(USE_TERMINAL_CONSTRAINT):
    opti.subject_to(X[-1][nq:] == 0.0)

opti.minimize(cost)

print("Create the optimization problem")
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

# Solve the problem to convergence the first time
x = np.concatenate([q0, dq0])
opti.set_value(param_q_des,q_des)
opti.set_value(param_x_init, x)
sol = opti.solve()
opts["ipopt.max_iter"] = SOLVER_MAX_ITER
opti.solver("ipopt", opts)

# initialization of data matrix
data = np.zeros((N_sim, 6))

mean_comp_time = 0
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
        # print("Convergence failed!")
        sol = opti.debug
    end_time = clock()
    mean_comp_time += (end_time-start_time)/N_sim

    print("Comput. time: %.3f s"%(end_time-start_time), 
          "Iters: %3d"%sol.stats()['iter_count'], 
          "Norm of dq: %.3f"%np.linalg.norm(x[nq:]),"Return status", sol.stats()["return_status"])
    # "Tracking err: %.3f"%np.linalg.norm(p_ee_des-forward_kinematics_ee(cs.DM.eye(4), x[:nq])[:3,3].toarray().squeeze()),
    
    tau = inv_dyn(sol.value(X[0]), sol.value(U[0])).toarray().squeeze()

    data[i][0] = x[0] # joint 1 coord
    data[i][1] = x[1] # joint 2 coord
    data[i][2] = x[2] # joint 1 vel
    data[i][3] = x[3] # joint 2 vel
    data[i][4] = sol.value(U[1][0]) # joint 1 torque
    data[i][5] = sol.value(U[1][1]) # joint 1 torque

    if(SIMULATOR=="pinocchio"):
        # do a proper simulation with Pinocchio
        simu.simulate(tau, dt, int(dt/dt_sim))
        x = np.concatenate([simu.q, simu.v])
    elif(SIMULATOR=="ideal"):
        # use state predicted by the MPC as next state
        x = sol.value(X[1])
        simu.display(x[:nq])

print("Mean computation time: %.3f"%mean_comp_time)

if( np.any(x[:nq] > qMax)):
        print(colored("\nUPPER POSITION LIMIT VIOLATED ON JOINTS", "red"), np.where(x[:nq]>qMax)[0])
if( np.any(x[:nq] < qMin)):
        print(colored("\nLOWER POSITION LIMIT VIOLATED ON JOINTS", "red"), np.where(x[:nq]<qMin)[0])

if(SAVE_DATA):  
    np.savetxt('data.txt', (data))

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
    plt.xlabel('Time [s]')
    plt.ylabel('Joint velocities [m/s]')
    plt.title('Joint velocities')
    plt.legend()
    plt.grid(True)

    # Plot of joints torques
    plt.figure(figsize=(10, 6))
    for i in range(nq):
        plt.plot(time, data[:,i+4], label=f'torques {i}', alpha=0.7)
        plt.plot(time, np.full_like(time, tauMax[i]), linestyle='dotted',label=f'Maximum {i} torque' )
        plt.plot(time, np.full_like(time, tauMin[i]), linestyle='dotted',label=f'Minimum {i} torque' )
    plt.xlabel('Time [s]')
    plt.ylabel('Joint torques [Nm]')
    plt.title('Joint torques')
    plt.legend()
    plt.grid(True)
   
    plt.show()