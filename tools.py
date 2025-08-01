# tools.py
# This module contains utility functions and classes for solving DC Optimal Power Flow (DCOPF) problems
# using various optimization techniques including interior point methods, neural networks, and projection methods.

# ── standard library imports ─────────────────────────────────────────
import os
import sys
import math
import random
import pickle
from copy import deepcopy   # used for the best-checkpoint snapshot

# ── third-party packages ─────────────────────────────────────
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import cvxpy as cp
from scipy.io import loadmat

# Set random seeds for reproducibility
torch.manual_seed(0)          # reproducibility ─ remove if you want stochastic runs
np.random.seed(0)


def get_para_input():
    """
    Load DCOPF problem data from MATLAB file and organize into a structured dictionary.
    
    This function reads the DC Optimal Power Flow data from a MATLAB file and extracts
    all the necessary matrices and vectors for the optimization problem formulation.
    
    File Requirements:
    - File path: 'raw_para/dcopf_data.mat'
    - Contains matrices: Aineq, Bineq, Aeq, Beq, Q
    - Contains vectors: bbineq, pd, c, pg_ref
    - Contains scalar: fval
    
    Returns:
    --------
    para_input : dict
        Dictionary containing all problem parameters:
        - Aineq: inequality constraint matrix (m × ng)
        - Bineq: inequality constraint matrix for state variables
        - Aeq: equality constraint matrix
        - Beq: equality constraint matrix for state variables
        - Q: quadratic cost matrix
        - bbineq: inequality constraint right-hand side vector
        - pd: power demand vector
        - c: linear cost vector
        - pg_ref: reference generator power (optimal solution from MATLAB)
        - fval: optimal objective value from MATLAB solver
    """
    # Load data from MATLAB file
    data = loadmat('raw_para/dcopf_data.mat')
    
    # Extract constraint matrices
    Aineq  = data['Aineq']           # Inequality constraint matrix (m × ng)
    Bineq  = data['Bineq']           # Inequality constraint matrix for input variables (m x nbus)
    Aeq    = data['Aeq']             # Equality constraint matrix
    Beq    = data['Beq']             # Equality constraint matrix for input variables
    Q      = data['Q']               # Quadratic cost matrix
    
    # Extract vectors (ensure they are 1-dimensional)
    bbineq = data['bbineq'].ravel()  # Inequality constraint RHS, make 1-D
    pd     = data['pd'].ravel()      # Power demand vector
    c      = data['c'].ravel()       # Linear cost vector
    pg_ref = data['pg_opt'].ravel()  # Reference generator power (MATLAB solution)
    
    # Extract scalar value
    fval   = data['fval'].item()     # Optimal objective value from MATLAB
    
    # Organize all parameters into a structured dictionary
    para_input = {
        'Aineq': Aineq,
        'Bineq': Bineq,
        'Aeq': Aeq,
        'Beq': Beq,
        'Q': Q,
        'bbineq': bbineq,
        'pd': pd,
        'c': c,
        'pg_ref': pg_ref,
        'fval': fval
    }
    return para_input

def get_para_ind(para_input):
    """
    Transform the original problem parameters into independent variable formulation.
    
    This function performs variable elimination to reduce the problem dimensionality
    by expressing dependent variables in terms of independent variables. The process:
    1. Separates variables into dependent (u_dep) and independent (u_ind) sets
    2. Uses equality constraints to express u_dep = R_u * u_ind + R_x * x + R_c
    3. Substitutes this relationship into inequality constraints
    4. Creates a new problem formulation with only independent variables
    
    Parameters:
    -----------
    para_input : dict
        Original problem parameters containing Aineq, Bineq, bbineq, Aeq, Beq
        
    Returns:
    --------
    para_ind : dict
        Transformed problem parameters for independent variable formulation:
        - A_all: inequality constraint matrix for independent variables
        - B_all: inequality constraint matrix for state variables
        - bb_all: inequality constraint right-hand side
        - F_u: mapping matrix from independent to full variables
        - F_x: mapping matrix from state to full variables  
        - F_c: constant offset in variable mapping
    """
    # Extract original constraint matrices and vectors
    Aineq = para_input['Aineq'] 
    Bineq = para_input['Bineq'] 
    bbineq = para_input['bbineq'].ravel() 
    Aeq = para_input['Aeq']
    Beq = para_input['Beq']
    bbeq = np.zeros((Aeq.shape[0], 1)).ravel()  # Zero RHS for equality constraints

    # Print dimensions for debugging
    print(f"Aineq.shape: {Aineq.shape}")
    print(f"Bineq.shape: {Bineq.shape}")
    print(f"bbineq.shape: {bbineq.shape}")
    print(f"Aeq.shape: {Aeq.shape}")
    print(f"Beq.shape: {Beq.shape}")
    print(f"bbeq.shape: {bbeq.shape}")

    # Define dependent variable indices (currently only index 0)
    index_c = [0]  # dependent variables (can be any index among 0~ng-1)

    # Separate constraint matrices for dependent and independent variables
    A_eq_M1 = []   # Equality constraints for dependent variables
    A_eq_M2 = []   # Equality constraints for independent variables
    A_ineq_M1 = [] # Inequality constraints for dependent variables
    A_ineq_M2 = [] # Inequality constraints for independent variables
    
    for iloop in range(Aeq.shape[1]):
        if iloop in index_c:
            # Collect columns for dependent variables
            A_eq_M1.append(Aeq[:, iloop])
            A_ineq_M1.append(Aineq[:, iloop])
        else:
            # Collect columns for independent variables
            A_eq_M2.append(Aeq[:, iloop])
            A_ineq_M2.append(Aineq[:, iloop])
    
    # Stack columns into matrices
    A_eq_AD = np.column_stack(A_eq_M1)    # Equality constraints for dependent vars
    A_eq_AI = np.column_stack(A_eq_M2)    # Equality constraints for independent vars
    A_ineq_AD = np.column_stack(A_ineq_M1) # Inequality constraints for dependent vars
    A_ineq_AI = np.column_stack(A_ineq_M2) # Inequality constraints for independent vars
    
    # Calculate transformation matrices using equality constraints
    # u_dep = R_u * u_ind + R_x * x + R_c
    R_u = -np.linalg.inv(A_eq_AD) @ A_eq_AI  # Mapping from independent to dependent
    R_x = -np.linalg.inv(A_eq_AD) @ Beq      # Mapping from input to dependent
    R_c = -np.linalg.inv(A_eq_AD) @ bbeq     # Constant offset

    # Transform inequality constraints by substituting the dependent variable relationship
    A_all = A_ineq_AI - A_ineq_AD @ np.linalg.inv(A_eq_AD) @ A_eq_AI
    B_all = Bineq - A_ineq_AD @ np.linalg.inv(A_eq_AD) @ Beq
    bb_all = (bbineq - A_ineq_AD @ np.linalg.inv(A_eq_AD) @ bbeq).ravel()

    # Print transformed constraint dimensions
    print(f"A_all.shape: {A_all.shape}")
    print(f"B_all.shape: {B_all.shape}")
    print(f"bb_all.shape: {bb_all.shape}")

    # Get dimensions
    nI, nD = A_ineq_AI.shape[1], A_ineq_AD.shape[1]  # Independent and dependent variable counts

    # Create mapping matrices to reconstruct full variable vector
    # F_u: maps independent variables to full variables
    # F_x: maps input variables to full variables  
    # F_c: constant offset for full variables
    F_u = np.zeros(((nI + nD), nI))
    F_x = np.zeros(((nI + nD), R_x.shape[1]))
    F_c = np.zeros(((nI + nD), 1))

    # Fill mapping matrices based on variable indices
    for i in range(nI + nD):
        if i in index_c:
            # For dependent variables, use the transformation relationship
            row_idx = index_c.index(i)
            F_u[i] = R_u[row_idx]
            F_x[i] = R_x[row_idx]
            F_c[i] = R_c[row_idx]
        else:
            # For independent variables, use identity mapping
            F_u[i, i - sum(elem < i for elem in index_c)] = 1

    # Return transformed problem parameters
    para_ind = {
        'A_all': A_all,
        'B_all': B_all,
        'bb_all': bb_all,
        'F_u': F_u,
        'F_x': F_x,
        'F_c': F_c.ravel()
    }
    return para_ind
import numpy as np
import cvxpy as cp


def _as_1d_numpy(vec, name):
    """
    Convert *anything* that looks like a vector into a 1-D NumPy array.

    This utility function ensures consistent vector handling throughout the codebase
    by converting various input formats (lists, arrays, tensors) into flattened
    NumPy arrays. This prevents shape-related errors in matrix operations.

    Parameters:
    -----------
    vec  : array-like
        Input vector that can be in various formats (list, numpy array, tensor, etc.)
    name : str
        Variable name used only for informative error messages.

    Returns:
    --------
    ndarray
        Flattened (n,) array.

    Raises:
    -------
    ValueError
        If input is a scalar (0-dimensional array).
    """
    arr = np.asarray(vec)
    if arr.ndim == 0:
        raise ValueError(f"'{name}' must be a vector, not a scalar.")
    return arr.ravel()          # collapses (n,), (1,n) or (n,1) ➜ (n,)


def find_optimal_solution(x, para_input):
    """
    Solve the original DCOPF optimization problem using CVXPY.
    
    This function solves the standard DC Optimal Power Flow problem:
        minimize   ½ uᵀ Q u + cᵀ u
        subject to A_eq u + B_eq x  = 0
                  A_in u + B_in x + b_in ≤ 0
    
    The problem is solved using the GUROBI solver through CVXPY interface.
    
    Parameters:
    -----------
    x          : array-like (n_x,)
        State variables (e.g., power demands, network parameters)
    para_input : dict
        Problem parameters containing:
        - Aineq: inequality constraint matrix
        - Bineq: inequality constraint matrix for state variables  
        - bbineq: inequality constraint right-hand side
        - Aeq: equality constraint matrix
        - Beq: equality constraint matrix for state variables
        - Q: quadratic cost matrix
        - c: linear cost vector

    Returns:
    --------
    u_opt  : ndarray
        Optimal decision variables (generator powers)
    f_opt  : float
        Optimal objective value

    Raises:
    -------
    ValueError
        If there are dimension mismatches in the constraint matrices.
    RuntimeError
        If the solver fails to find an optimal solution.
    """
    # Ensure x and bbineq are 1-D — all shapes welcome upstream
    x       = _as_1d_numpy(x,       "x")
    bb_in   = _as_1d_numpy(para_input["bbineq"], "bbineq")

    # Pull remaining data straight out of the dict
    A_in  = para_input["Aineq"]     # Inequality constraint matrix
    B_in  = para_input["Bineq"]     # Inequality constraint matrix for state vars
    A_eq  = para_input["Aeq"]       # Equality constraint matrix
    B_eq  = para_input["Beq"]       # Equality constraint matrix for state vars
    Q     = para_input["Q"]         # Quadratic cost matrix
    c     = para_input["c"]         # Linear cost vector

    # --- dimensions check (helps catch silent broadcasting bugs) ----------
    if B_in.shape[0] != bb_in.size:
        raise ValueError(
            f"Row dimension mismatch: Bineq has {B_in.shape[0]} rows "
            f"but bbineq has length {bb_in.size}"
        )
    if B_in.shape[1] != x.size:
        raise ValueError(
            f"Column mismatch: Bineq has {B_in.shape[1]} columns "
            f"but x has length {x.size}"
        )
    if A_eq.shape[1] != A_in.shape[1]:
        raise ValueError(
            "Number of decision variables implied by Aeq and Aineq differ."
        )

    # ----------------------------------------------------------------------
    m = A_in.shape[1]               # number of decision variables
    u = cp.Variable(m)

    # Define objective function: quadratic + linear terms
    objective   = 0.5 * cp.quad_form(u, Q) + c @ u
    
    # Define constraints: equality and inequality
    constraints = [
        A_eq @ u + B_eq @ x == 0,           # Equality constraints
        A_in @ u + B_in @ x + bb_in <= 0    # Inequality constraints
    ]

    # Create and solve the optimization problem
    prob = cp.Problem(cp.Minimize(objective), constraints)

    # Use GUROBI solver with high precision settings
    try:
        prob.solve(solver=cp.GUROBI, eps_abs=1e-7, eps_rel=1e-7)
    except (cp.SolverError, AttributeError):
        print(f"Status: {prob.status}")
        exit()

    return u.value, prob.value

    
def find_optimal_solution_ind(x, para_ind, para_input):
    """
    Solve the DCOPF problem using the independent variable formulation.
    
    This function solves the transformed problem where dependent variables
    have been eliminated using equality constraints:
    
        minimize   (1/2) uᵀ Q u + cᵀ u
        subject to u           = F_u u_ind + F_x x + F_c     (affine mapping)
                   A_all u_ind + B_all x + bb_all ≤ 0        (inequalities)
    
    The advantage of this formulation is reduced problem dimensionality
    and potentially better numerical stability.
    
    Parameters:
    -----------
    x         : array-like, shape (n_x,) or (n_x,1) or (1,n_x)
        State variables
    para_ind  : dict
        Independent variable formulation parameters:
        - A_all: inequality constraint matrix for independent variables
        - B_all: inequality constraint matrix for state variables
        - bb_all: inequality constraint right-hand side
        - F_u: mapping matrix from independent to full variables
        - F_x: mapping matrix from state to full variables
        - F_c: constant offset in variable mapping
    para_input: dict
        Original problem parameters containing Q and c

    Returns:
    --------
    u_opt  : ndarray (m,)
        Optimal decision vector u (full dimension)
    f_opt  : float
        Optimal objective value

    Raises:
    -------
    ValueError
        If there are dimension mismatches in the constraint matrices.
    RuntimeError
        If the solver fails to find an optimal solution.
    """
    # ── constants ──────────────────────────────────────────────────────────
    x       = _as_1d_numpy(x,                "x")
    bb_all  = _as_1d_numpy(para_ind["bb_all"], "bb_all")
    F_c     = _as_1d_numpy(para_ind["F_c"],    "F_c")

    # Extract constraint matrices and mapping matrices
    A_all   = para_ind["A_all"]     # Inequality constraints for independent vars
    B_all   = para_ind["B_all"]     # Inequality constraints for state vars
    F_u     = para_ind["F_u"]       # Mapping from independent to full vars
    F_x     = para_ind["F_x"]       # Mapping from state to full vars

    # Extract cost function parameters
    Q       = para_input["Q"]       # Quadratic cost matrix
    c       = para_input["c"]       # Linear cost vector

    # ── dimension guards (fail fast, no silent broadcasting) ──────────────
    m, k = F_u.shape                    # m = dim(u), k = dim(u_ind)
    if F_x.shape[0] != m or F_c.size != m:
        raise ValueError("F_x and F_c must have the same number of rows as F_u.")

    s = A_all.shape[0]                  # number of inequality rows
    if A_all.shape[1] != k:
        raise ValueError("A_all must have as many columns as F_u has columns (u_ind size).")
    if B_all.shape != (s, x.size):
        raise ValueError(
            f"B_all shape should be ({s}, {x.size}) to match A_all rows and x length."
        )
    if bb_all.size != s:
        raise ValueError(f"bb_all length {bb_all.size} must equal the row count of A_all {A_all.shape[0]}.")

    # ── cvxpy variables & constants ───────────────────────────────────────
    u_ind = cp.Variable(k)              # Independent variables
    u     = cp.Variable(m)              # Full variables

    # Convert numpy arrays to CVXPY constants for efficiency
    x_c      = cp.Constant(x)
    bb_all_c = cp.Constant(bb_all)
    F_c_c    = cp.Constant(F_c)

    # ── objective & constraints ───────────────────────────────────────────
    objective = 0.5 * cp.quad_form(u, Q) + c @ u
    constraints = [
        A_all @ u_ind + B_all @ x_c + bb_all_c <= 0,  # Inequality constraints
        u == F_u @ u_ind + F_x @ x_c + F_c_c          # Variable mapping
    ]

    prob = cp.Problem(cp.Minimize(objective), constraints)

    # ── solve, with robust fall-back ──────────────────────────────────────
    try:
        prob.solve(solver=cp.GUROBI, eps_abs=1e-7, eps_rel=1e-7)
    except (cp.SolverError, AttributeError):
        prob.solve(solver=cp.ECOS, abstol=1e-7, reltol=1e-7)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Solver failed with status '{prob.status}'.")

    return u.value.ravel(), prob.value


def generate_u0(x, para_ind, para_input):
    """
    Generate an interior point solution for the DCOPF problem.
    
    This function solves an auxiliary optimization problem to find a strictly
    feasible point (interior point) that satisfies all inequality constraints
    with strict inequality. The interior point is used as a starting point
    for iterative optimization algorithms.
    
    The auxiliary problem is:
        minimize M * ua
        subject to A_all * u + B_all * x + bb_all - ua <= 0
    
    where M is a large number and ua is a slack variable. The solution u
    is an interior point if ua < 0.
    
    Parameters:
    -----------
    x : array-like
        State variables
    para_ind : dict
        Independent variable formulation parameters
    para_input : dict
        Original problem parameters (not used in this function)
        
    Returns:
    --------
    u0 : ndarray
        Interior point solution (strictly feasible)
    ua : float
        Slack variable value (should be negative for interior point)
        
    Raises:
    -------
    ValueError
        If no interior point can be found (ua >= 0).
    """
    # Ensure inputs are 1-dimensional
    x = _as_1d_numpy(x, "x")
    bb_all = _as_1d_numpy(para_ind["bb_all"], "bb_all")
    
    # Extract constraint matrices
    A_all = para_ind["A_all"]     # Inequality constraint matrix
    B_all = para_ind["B_all"]     # Inequality constraint matrix for state vars

    # Large penalty parameter for the auxiliary problem
    M = 1e6
    
    # ── cvxpy variables & constants ───────────────────────────────────────
    ua = cp.Variable(1)           # Slack variable
    u = cp.Variable(A_all.shape[1])  # Decision variables

    # Define auxiliary optimization problem
    objective = cp.Minimize(M * ua)
    constraints = [
        A_all @ u + B_all @ x + bb_all - ua <= 0
    ]

    prob = cp.Problem(objective, constraints)

    # ── solve, with robust fall-back based on the value of ua ────────────
    prob.solve(solver=cp.GUROBI, eps_abs=1e-7, eps_rel=1e-7)
    
    # Check if interior point was found
    if ua.value >= 0:
        raise ValueError("No interior point can be found.")
    else:
        return u.value.ravel(), ua.value





def get_optimality_gap(u_pre, u_star):       
    """
    Calculate the optimality gap between predicted and optimal solutions.
    
    The optimality gap measures how close the predicted solution is to the
    true optimal solution in terms of Euclidean distance. This is a fundamental
    metric for evaluating the quality of optimization predictions.
    
    Parameters:
    -----------
    u_pre : array-like, shape (u_dim,)
        Predicted decision variables (generator powers)
    u_star : array-like, shape (u_dim,)
        Optimal decision variables (ground truth)
        
    Returns:
    --------
    optimality_gap : float
        Euclidean norm of the difference between predicted and optimal solutions
        ||u_pre - u_star||₂
    """
    optimality_gap = np.linalg.norm(u_pre - u_star)
    return optimality_gap

def get_optimality_deviation_rate(u_pre, u_star):
    """
    Calculate the relative optimality deviation rate.
    
    This metric normalizes the optimality gap by the norm of the optimal solution,
    providing a relative measure of prediction accuracy. This is useful when
    comparing performance across different problem instances with varying scales.
    
    Parameters:
    -----------
    u_pre : array-like, shape (u_dim,)
        Predicted decision variables (generator powers)
    u_star : array-like, shape (u_dim,)
        Optimal decision variables (ground truth)
        
    Returns:
    --------
    optimality_deviation_rate : float
        Relative optimality gap: ||u_pre - u_star||₂ / ||u_star||₂
    """
    optimality_deviation_rate = np.linalg.norm(u_pre - u_star) / np.linalg.norm(u_star)
    return optimality_deviation_rate

def get_feasibility_gap(x, u_pre, Aineq, Bineq, bbineq, Aeq, Beq, bbeq):
    """
    Calculate the feasibility gap of a predicted solution.
    
    The feasibility gap measures how much the predicted solution violates
    the problem constraints. It consists of two components:
    1. Inequality violation: sum of positive constraint violations
    2. Equality violation: sum of absolute constraint violations
    
    A solution is feasible if and only if the feasibility gap is zero.
    
    Parameters:
    -----------
    x : array-like, shape (x_dim,)
        State variables (power demands, network parameters)
    u_pre : array-like, shape (u_dim,)
        Predicted decision variables (generator powers)
    Aineq : ndarray, shape (m_ineq, u_dim)
        Inequality constraint matrix
    Bineq : ndarray, shape (m_ineq, x_dim)
        Inequality constraint matrix for state variables
    bbineq : array-like, shape (m_ineq,)
        Inequality constraint right-hand side
    Aeq : ndarray, shape (m_eq, u_dim)
        Equality constraint matrix
    Beq : ndarray, shape (m_eq, x_dim)
        Equality constraint matrix for state variables
    bbeq : array-like, shape (m_eq,)
        Equality constraint right-hand side
        
    Returns:
    --------
    feasibility_gap : float
        Total constraint violation: ineq_gap + eq_gap
        where:
        - ineq_gap = sum(max(0, Aineq @ u_pre + Bineq @ x + bbineq))
        - eq_gap = sum(|Aeq @ u_pre + Beq @ x + bbeq|)
    """
    # Calculate constraint residuals
    rhs_ineq = Aineq @ u_pre + Bineq @ x + bbineq  # Inequality constraint residuals
    rhs_eq = Aeq @ u_pre + Beq @ x + bbeq          # Equality constraint residuals
    
    # Calculate inequality violation (only positive violations count)
    ineq_gap = np.sum(np.maximum(0, rhs_ineq))
    
    # Calculate equality violation (absolute values of all violations)
    eq_gap = np.sum(np.abs(rhs_eq))
    
    # Total feasibility gap
    feasibility_gap = ineq_gap + eq_gap
    return feasibility_gap


class CustomNN(nn.Module):
    """
    Custom Neural Network for DCOPF optimization with multiple feasibility strategies.
    
    This neural network implements different approaches to ensure constraint feasibility:
    
    1. **loop2**: Uses generalized gauge-map projection inside forward pass
    2. **loop**: Uses traditional gauge-map projection inside forward pass  
    3. **penalty**: Plain feed-forward network with constraint violation penalties
    4. **projection**: Feed-forward network with optional post-processing projection
    
    The network architecture consists of:
    - Multiple hidden layers with ReLU activation
    - Output layer with method-specific activation functions
    - Xavier weight initialization for better training stability
    
    For gauge-map methods (loop2, loop), the network learns to predict feasible
    solutions directly by incorporating constraint satisfaction into the forward pass.
    For penalty methods, constraint violations are penalized in the loss function.
    
    Attributes:
    -----------
    method : str
        Feasibility strategy ('loop2', 'loop', 'penalty', 'projection')
    A_all : torch.Tensor
        Inequality constraint matrix for gauge-map methods
    B_all : torch.Tensor
        Inequality constraint matrix for state variables
    bb_all : torch.Tensor
        Inequality constraint right-hand side
    F_u : torch.Tensor
        Mapping matrix from independent to full variables
    F_x : torch.Tensor
        Mapping matrix from state to full variables
    F_c : torch.Tensor
        Constant offset in variable mapping
    net : nn.Sequential
        The neural network architecture
    """

    def __init__(self, para_ind_tensor, para_input_tensor, sita):
        """
        Initialize the CustomNN with problem parameters and configuration.
        
        Parameters:
        -----------
        para_ind_tensor : dict
            Problem parameters converted to PyTorch tensors for gauge-map methods
        para_input_tensor : dict
            Original problem parameters converted to PyTorch tensors
        sita : dict
            Configuration dictionary containing:
            - method: feasibility strategy ('loop2', 'loop', 'penalty', 'projection')
            - number_of_hidden_layers: number of hidden layers
            - number_of_hidden_nodes: number of neurons per hidden layer (int or list)
        """
        super().__init__()
        self.method = sita['method']

        # Set up problem-specific constants for gauge-map methods
        if self.method == 'loop2' or self.method == 'loop':
            # Extract constraint matrices and mapping matrices for gauge-map projection
            self.A_all = para_ind_tensor['A_all']           # (m, u0_dim) - inequality constraints
            self.B_all = para_ind_tensor['B_all']           # (m, x_dim) - state variable constraints
            self.bb_all = para_ind_tensor['bb_all'].reshape(-1, 1)  # (m, 1) - constraint RHS
            self.F_u  = para_ind_tensor['F_u']              # Mapping from independent to full vars
            self.F_x  = para_ind_tensor['F_x']              # Mapping from state to full vars
            self.F_c  = para_ind_tensor['F_c'].reshape(-1, 1)  # Constant offset

            # Set input/output dimensions based on problem structure
            input_dim  = self.B_all.shape[1]        # x-dimension (state variables)
            output_dim = self.A_all.shape[1]        # u0-dimension (independent variables)

        else:  # penalty | projection methods
            # For penalty/projection methods, use original problem dimensions
            input_dim  = para_input_tensor['Beq'].shape[1]  # x-dimension
            output_dim = para_input_tensor['Aeq'].shape[1]  # u-dimension (full variables)

        # ===== build the MLP common to all three methods =========
        hidden_cfg = sita['number_of_hidden_nodes']
        if isinstance(hidden_cfg, int):
            # If single integer provided, use same number for all layers
            hidden_cfg = [hidden_cfg] * sita['number_of_hidden_layers']

        # Construct hidden layers with ReLU activation
        layers, prev = [], input_dim
        for l in range(sita['number_of_hidden_layers']):
            # Get number of units for current layer (use last value if list too short)
            units = hidden_cfg[l] if l < len(hidden_cfg) else hidden_cfg[-1]
            layers += [nn.Linear(prev, units), nn.ReLU()]
            prev = units
        
        # Add output layer with method-specific activation functions
        if self.method == 'penalty' or self.method == 'projection':
            # For penalty/projection: linear output + ReLU (ensure non-negative outputs)
            layers.append(nn.Linear(prev, output_dim))
            layers.append(nn.ReLU())
        elif self.method == 'loop2':
            # For loop2: linear output only (no activation, allows negative values)
            layers.append(nn.Linear(prev, output_dim))
        elif self.method == 'loop':
            # For loop: linear output + tanh (bounded outputs in [-1, 1])
            layers.append(nn.Linear(prev, output_dim))
            layers.append(nn.Tanh())
        
        # Create sequential network
        self.net = nn.Sequential(*layers)

        # Xavier weight initialization for better training convergence
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x, u0=None):
        """
        Forward pass through the neural network.
        
        For penalty/projection methods, this is a standard feed-forward pass.
        For gauge-map methods (loop2, loop), this includes constraint projection
        to ensure feasibility of the output.
        
        Parameters:
        -----------
        x : torch.Tensor, shape (batch, x_dim)
            Input state variables (power demands, network parameters)
        u0 : torch.Tensor, shape (batch, u0_dim), optional
            Interior point solution (only needed for gauge-map methods)
            
        Returns:
        --------
        torch.Tensor, shape (batch, u_dim)
            Predicted decision variables (generator powers)
            
        Raises:
        -------
        AssertionError
            If gauge-map projection fails to produce feasible solution
        """
        # Standard feed-forward pass for penalty/projection methods
        if self.method == 'penalty' or self.method == 'projection':
            return self.net(x)

        # ---------- gauge map methods (loop2, loop) ----------
        # Get raw network output (independent variables)
        v   = self.net(x)                              # (B, u0_dim)
        
        # Calculate constraint residuals for gauge-map projection
        vA  = v @ self.A_all.T                         # (B, m) - constraint violations
        base = -(u0 @ self.A_all.T + x @ self.B_all.T + self.bb_all.T)  # (B, m) - constraint bounds

        # Calculate scaling ratios for projection
        ratio = vA / (base + 1e-9)                     # avoid division by zero
        
        if self.method == 'loop2':
            # Generalized gauge-map: use maximum of ratios and 1
            phy_sv, _ = torch.max(
                torch.cat([ratio, torch.ones_like(ratio[:, :1])], dim=1),
                dim=1, keepdim=True
            )                                              # (B, 1)
            scaled_v = v / phy_sv                          # (B, u0_dim)
            
        elif self.method == 'loop':
            # Traditional gauge-map: scale by maximum absolute value
            phy_v = torch.max(torch.abs(v), dim=1, keepdim=True)[0]  # (B, 1) - max abs value
            phy_sv = torch.max(ratio, dim=1, keepdim=True)[0]        # (B, 1) - max ratio
            scaled_v = v * (phy_v / phy_sv)                          # (B, u0_dim)
        
        # Add interior point to get feasible independent variables
        u_ind = scaled_v + u0

        # Optional safety check to verify feasibility (can comment out for speed)
        with torch.no_grad():
            viol = (self.A_all @ u_ind.T + self.B_all @ x.T + self.bb_all).T
            assert (viol <= 1e-4).all(), "Gauge-map feasibility violated"

        # Lift back to full variable space using affine mapping
        u_full = u_ind @ self.F_u.T + x @ self.F_x.T + self.F_c.T
        return u_full

def recover(u_pre, x_np, para_ind, para_input):
    """
    Project a predicted solution back to the feasible set using optimization.
    
    This function solves a projection problem to find the closest feasible point
    to the predicted solution. The projection problem is:
    
        minimize   0.5 * ||u - u_pre||²
        subject to Aineq * u + Bineq * x + bbineq <= 0
                  Aeq * u + Beq * x + bbeq = 0
    
    This is useful for the 'projection' method where the neural network output
    may not be feasible and needs to be projected onto the constraint set.
    
    Parameters:
    -----------
    u_pre : array-like, shape (u_dim,)
        Predicted decision variables (may be infeasible)
    x_np : array-like, shape (x_dim,)
        State variables (power demands, network parameters)
    para_ind : dict
        Independent variable formulation parameters (not used in this function)
    para_input : dict
        Original problem parameters containing constraint matrices
        
    Returns:
    --------
    u_proj : ndarray, shape (u_dim,)
        Projected feasible solution (closest feasible point to u_pre)
        
    Raises:
    -------
    ValueError
        If input arrays are not 1-dimensional
    """
    # Validate input dimensions
    if u_pre.ndim != 1 or x_np.ndim != 1:
        raise ValueError("u_pre and x_np must be 1d arrays")
    
    # Ensure inputs are 1-dimensional
    x       = _as_1d_numpy(x_np,       "x")
    bb_in   = _as_1d_numpy(para_input["bbineq"], "bbineq")

    # Extract constraint matrices
    A_in  = para_input["Aineq"]     # Inequality constraint matrix
    B_in  = para_input["Bineq"]     # Inequality constraint matrix for state vars
    A_eq  = para_input["Aeq"]       # Equality constraint matrix
    B_eq  = para_input["Beq"]       # Equality constraint matrix for state vars
    
    # Set up optimization problem
    m = A_in.shape[1]               # number of decision variables
    u = cp.Variable(m)

    # Define constraints
    constraints = [
        A_eq @ u + B_eq @ x == 0,           # Equality constraints
        A_in @ u + B_in @ x + bb_in <= 0    # Inequality constraints
    ]

    # Define objective: minimize distance to predicted solution
    prob = cp.Problem(
        cp.Minimize(0.5 * cp.sum_squares(u - u_pre)),
        constraints
    )

    # Solve the projection problem
    prob.solve(solver=cp.GUROBI, eps_abs=1e-7, eps_rel=1e-7)
    
    u_proj = u.value
    return u_proj


def test(model, test_data, para_ind, para_input, sita):
    """
    Evaluate the trained model on test data and compute performance metrics.
    
    This function computes comprehensive evaluation metrics including:
    - Loss: Mean squared error between predicted and true solutions
    - Optimality gap: Euclidean distance between predicted and optimal solutions
    - Optimality deviation rate: Relative optimality gap
    - Feasibility gap: Constraint violation measure
    
    For the 'projection' method, predicted solutions are projected to the feasible
    set before computing metrics.
    
    Parameters:
    -----------
    model : CustomNN
        The trained neural network model
    test_data : dict
        Test dataset containing:
        - 'x': state variables, shape (n_samples, x_dim)
        - 'u': optimal solutions, shape (n_samples, u_dim)  
        - 'u0': interior points, shape (n_samples, u0_dim)
    para_ind : dict
        Independent variable formulation parameters
    para_input : dict
        Original problem parameters
    sita : dict
        Model configuration including method type
        
    Returns:
    --------
    dict
        Dictionary containing test metrics:
        - val_loss: float, average MSE loss
        - val_opt_gap: float, average optimality gap
        - val_opt_dev: float, average optimality deviation rate
        - val_feas_gap: float, average feasibility gap
    """
    model.eval()  # Set model to evaluation mode
    criterion = nn.MSELoss(reduction='mean')
    
    # Convert test data to PyTorch tensors
    x_tensor = torch.tensor(test_data['x'], dtype=torch.float64)
    u_tensor = torch.tensor(test_data['u'], dtype=torch.float64)
    u0_tensor = torch.tensor(test_data['u0'], dtype=torch.float64)
    
    # Process all samples at once
    with torch.no_grad():
        # Get predictions for all samples
        if (sita['method'] == 'loop2' or sita['method'] == 'loop'):
            # Gauge-map methods need interior points
            preds = model(x_tensor, u0_tensor)
        else:
            # Penalty/projection methods only need state variables
            preds = model(x_tensor)
        
        # Calculate MSE loss
        loss = criterion(preds, u_tensor)
        
        # Calculate detailed metrics for each sample
        opt_gaps = []
        opt_devs = []
        feas_gaps = []
        
        for i in range(len(preds)):
            u_pre = preds[i].cpu().numpy()
            u_gt = u_tensor[i].cpu().numpy()
            x_np = x_tensor[i].cpu().numpy()

            # For projection method, project to feasible set before evaluation
            if sita['method'] == 'projection':
                u_pre = recover(u_pre, x_np, para_ind, para_input)
            
            # Compute evaluation metrics
            opt_gaps.append(get_optimality_gap(u_pre, u_gt))
            opt_devs.append(get_optimality_deviation_rate(u_pre, u_gt))
            feas_gaps.append(get_feasibility_gap(
                x_np, u_pre,
                para_input['Aineq'], para_input['Bineq'], para_input['bbineq'],
                para_input['Aeq'], para_input['Beq'], para_input['bbeq']
            ))
    
    # Return average metrics
    return {
        'val_loss': loss.item(),
        'val_opt_gap': np.mean(opt_gaps),
        'val_opt_dev': np.mean(opt_devs),
        'val_feas_gap': np.mean(feas_gaps)
    }

def train_and_test(training_data, validation_data, para_ind, para_input, sita, print_loss=False):
    """
    Train and evaluate a neural network model for DCOPF optimization.
    
    This function implements a complete training pipeline including:
    - Data preparation and tensor conversion
    - Model initialization with Xavier weight initialization
    - Training loop with gradient descent optimization
    - Validation and early stopping based on optimality deviation rate
    - Comprehensive metric tracking and best model checkpointing
    
    The training supports different feasibility strategies:
    - **loop2/loop**: Gauge-map methods with constraint projection in forward pass
    - **penalty**: Constraint violations penalized in loss function
    - **projection**: Standard training with optional post-processing projection
    
    Parameters:
    -----------
    training_data : dict
        Training dataset containing:
        - 'x': state variables, shape (n_train, x_dim)
        - 'u': optimal solutions, shape (n_train, u_dim)
        - 'u0': interior points, shape (n_train, u0_dim)
    validation_data : dict
        Validation dataset with same structure as training_data
    para_ind : dict
        Independent variable formulation parameters
    para_input : dict
        Original problem parameters
    sita : dict
        Training configuration containing:
        - method: feasibility strategy ('loop2', 'loop', 'penalty', 'projection')
        - batch_size: training batch size
        - learning_rate: Adam optimizer learning rate
        - number_of_epochs: total training epochs
        - number_of_hidden_layers: network architecture
        - number_of_hidden_nodes: neurons per layer
    print_loss : bool, optional
        Whether to print training progress (default: False)
        
    Returns:
    --------
    dict
        Training results containing:
        - model_para: dict, best model checkpoint including:
            - model_state: model state dictionary
            - optimizer_state: optimizer state dictionary
            - epoch: best epoch number
            - val_loss: validation loss at best epoch
            - val_opt_gap: validation optimality gap
            - val_opt_dev: validation optimality deviation rate
            - val_feas_gap: validation feasibility gap
        - model_metrics: dict, training history including:
            - train_loss: list of training losses
            - val_loss: list of validation losses
            - val_opt_gap: list of validation optimality gaps
            - val_opt_dev: list of validation optimality deviation rates
            - val_feas_gap: list of validation feasibility gaps
    """
    # Convert training data to PyTorch tensors
    x_train = torch.tensor(training_data['x'], dtype=torch.float64)
    u_train = torch.tensor(training_data['u'], dtype=torch.float64)
    u0_train = torch.tensor(training_data['u0'], dtype=torch.float64)
    
    # Convert problem parameters to PyTorch tensors for efficient computation
    para_ind_tensor = {k: torch.tensor(v, dtype=torch.float64) for k, v in para_ind.items()}
    para_input_tensor = {k: torch.tensor(v, dtype=torch.float64) for k, v in para_input.items()}
    
    # Create data loader for batch training
    train_dataset = TensorDataset(x_train, u_train, u0_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=sita['batch_size'],
        shuffle=False,  # Keep order for reproducibility
        drop_last=False
    )
    
    # Flag to control when validation starts (last 20 epochs)
    test_flag = False
    
    # Initialize model, loss function, and optimizer
    model = CustomNN(para_ind_tensor, para_input_tensor, sita).double()
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=sita['learning_rate'])
    
    # Initialize training history tracking
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_opt_gap": [],
        "val_opt_dev": [],
        "val_feas_gap": []
    }
    
    # Track best model based on validation optimality deviation rate
    best_dev = float('inf')
    best_snap = None
    
    # Training loop
    for epoch in range(sita['number_of_epochs']):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for xb, ub, u0b in train_loader:
            optimizer.zero_grad()
            
            # Forward pass (method-specific)
            if (sita['method'] == 'loop2' or sita['method'] == 'loop'):
                # Gauge-map methods need interior points
                preds = model(xb, u0b)
            else:
                # Penalty/projection methods only need state variables
                preds = model(xb)
            
            # Calculate base MSE loss
            loss = criterion(preds, ub)
            
            # Add constraint violation penalties for penalty method
            if sita['method'] == 'penalty':
                lambda_ineq = 1  # Penalty weight for inequality violations
                lambda_eq = 1    # Penalty weight for equality violations
                
                # Calculate inequality constraint violations
                bb = para_input_tensor['bbineq'].unsqueeze(1)  # Reshape for broadcasting
                rhs_ineq = (
                    para_input_tensor['Aineq'] @ preds.T
                    + para_input_tensor['Bineq'] @ xb.T
                    + bb  # Broadcast to (588, batch_size)
                )
                
                # Penalty for positive inequality violations
                vio_ineq = torch.sum(torch.clamp_min(rhs_ineq, 0)) / preds.shape[0]
                
                # Calculate equality constraint violations
                rhs_eq = (
                    para_input_tensor['Aeq'] @ preds.T 
                    + para_input_tensor['Beq'] @ xb.T 
                    + para_input_tensor['bbeq'].view(-1, 1)
                )
                
                # Penalty for absolute equality violations
                vio_eq = torch.sum(torch.abs(rhs_eq)) / preds.shape[0]
                
                # Add constraint penalties to loss
                loss = loss + lambda_ineq * vio_ineq + lambda_eq * vio_eq
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        
        # Calculate average training loss for this epoch
        train_loss = running_loss / len(train_dataset)
        history["train_loss"].append(train_loss)
        
        # Start validation in last 20 epochs to save computation
        if epoch >= sita['number_of_epochs'] - 20:
            test_flag = True

        # Validation phase
        if test_flag:
            test_metrics = test(model, validation_data, para_ind, para_input, sita)
            
            # Store validation metrics
            for metric, value in test_metrics.items():
                history[metric].append(value)
            
            # Update best model checkpoint based on optimality deviation rate
            if test_metrics['val_opt_dev'] < best_dev:
                best_dev = test_metrics['val_opt_dev']
                best_snap = {
                    "epoch": epoch,
                    "model_state": deepcopy(model.state_dict()),
                    "optimizer_state": deepcopy(optimizer.state_dict()),
                    "val_loss": test_metrics['val_loss'],
                    "val_opt_gap": test_metrics['val_opt_gap'],
                    "val_opt_dev": test_metrics['val_opt_dev'],
                    "val_feas_gap": test_metrics['val_feas_gap']
                }
            
            # Print training progress if requested
            if print_loss:
                print(f"Epoch {epoch+1:4d}/{sita['number_of_epochs']:4d}  "
                    f"train {train_loss:.4e} | val {test_metrics['val_loss']:.4e} | "
                    f"opt-dev {test_metrics['val_opt_dev']:.3e}")
        
    # Return best model checkpoint and training history
    return {
        "model_para": best_snap,
        "model_metrics": history
    }
