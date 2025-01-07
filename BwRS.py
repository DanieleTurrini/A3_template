import numpy as np
import matplotlib.pyplot as plt
from adam.casadi.computations import KinDynComputations
from example_robot_data.robots_loader import load
import casadi as cs



def is_in_BwRS (x_init, N):
    '''
    robot: robot model
    x_init: state to be checked (if belongs to the N-step backward reachable set)
    N: number of steps
    '''
    SOLVER_TOLERANCE = 1e-4
    dt = 0.010 # time step MPC
    VEL_BOUNDS_SCALING_FACTOR = 1.0
    TORQUE_BOUNDS_SCALING_FACTOR = 0.6
    qMin = np.array([-2.0*np.pi,-2.0*np.pi])
    qMax = -qMin
    vMax = np.array([10.0,10.0])*VEL_BOUNDS_SCALING_FACTOR
    vMin = -vMax
    tauMin = np.array([-1.0, -1.0])*TORQUE_BOUNDS_SCALING_FACTOR
    tauMax = -tauMin
    # Definition of a physical constraint that  doesn't allow the double pendulum to go beyond 
    # a certain configuration (in terms of joint angles)
    DELTA = 0.05
    q_lim = np.array([-(np.pi+DELTA),-(0.0+DELTA)])

    robot = load("double_pendulum")
    joints_name_list = [s for s in robot.model.names[1:]] # skip the first name because it is "universe"
    nq = len(joints_name_list)  # number of joints
    nx = 2*nq # size of the state variable
    kinDyn = KinDynComputations(robot.urdf, joints_name_list) 
    # BUILDING THE OPTIMIZATION PROBLEM
    opti = cs.Opti()
    param_x_init = opti.parameter(nx)
    param_q_des = opti.parameter(nq)
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
    lbx = qMin.tolist() + vMin.tolist()
    ubx = qMax.tolist() + vMax.tolist()
    tau_min = tauMin.tolist()
    tau_max = tauMax.tolist()
    # create all the decision variables
    X, U = [], []
    X += [opti.variable(nx)] # do not apply pos/vel bounds on initial state
    for k in range(1, N+1):
        X += [opti.variable(nx)]
        opti.subject_to( opti.bounded(lbx, X[-1], ubx) )
    for k in range(N): 
        U += [opti.variable(nq)]
        # Add dynamics constraints
        opti.subject_to(X[k+1] == X[k] + dt * f(X[k], U[k]))
        # Add torque constraints
        opti.subject_to( opti.bounded(tau_min, inv_dyn(X[k], U[k]), tau_max))
       # Add physical constraints
        opti.subject_to(X[k][:nq] >= q_lim)
    # Add initial condition of the state
    opti.subject_to(X[0] == param_x_init)
    # Constrain the final set to be a stationary point with zero velocity:
    opti.subject_to(X[-1][nq:] == 0.0)
    
    cost = 1  # No optimization, just checking feasibility
    opti.minimize(cost)

    # Create the optimization problem
    opts = {
        "error_on_fail": False,
        "ipopt.print_level": 0,
        "ipopt.tol": SOLVER_TOLERANCE,
        "ipopt.constr_viol_tol": SOLVER_TOLERANCE,
        "ipopt.compl_inf_tol": SOLVER_TOLERANCE,
        "print_time": 0,                # print information about execution time
        "detect_simple_bounds": True,
        "ipopt.max_iter": 1000
    }
    opti.solver("ipopt", opts)
    try:
        opti.set_value(param_x_init, x_init)
        sol = opti.solve()
        #per fare una prova plotto direttamente l'ultimo stato che il solver ha calcolato:
        #print(sol.value(X[-1]))
        #print("1")
        #return True  # Feasible solution
        return 1
    except:
        sol = opti.debug
        #per fare una prova plotto direttamente l'ultimo stato che il solver ha calcolato:
        #print(sol.value(X[-1]))
        #print("0")
        """X_values = np.array([sol.value(X[i]) for i in range(len(X))])
        U_values = np.array([sol.value(U[i]) for i in range(len(U))])

        print(X_values)
        print(U_values)
        for i in range(len(U)):
            print(inv_dyn(sol.value(X[i]),sol.value(U[i])))"""
        
        #return False  # Unfeasible solution
        return 0
