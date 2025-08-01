"""
Power System Dataset Generation for 200-Bus System
=================================================

This Python script creates a comprehensive dataset for a 200-bus power system optimization problem.
The script generates various data files that are essential for training and validating machine learning
models for power system optimization.

OUTPUT FILES:
1. para_input.pkl - Parameters of the original optimization problem:
   min ½ uᵀ Q u + c u
   s.t. Aeq@u + Beq@x == 0; Aineq@u + Bineq@x + bbineq <= 0

2. para_ind.pkl - Parameters of the reformulated problem constraints:
   s.t. Aall@uind + Ball@x + bb_all <= 0, u = F_u@uind + F_x@x + F_c

3. x_data.csv - Input data (power demand) with added noise for robustness

4. u_data.csv - Optimal solutions of the original problem (generation strategy)

5. u0_data.csv - Interior points from the interior point problem:
   min M * ua
   s.t. A_all * u + B_all * x + bb_all - 1 * ua <= 0

6. ua_data.csv - Slack variables of the interior point problem

7. fval_data.csv - Objective values of the original problem

FINAL USEFUL OUTPUTS:
1. train_data.pkl - Training dataset (x_train, u_train, fval_train, u0_train, ua_train)
2. val_data.pkl - Validation dataset (x_val, u_val, fval_val, u0_val, ua_val)
3. para_input.pkl, para_ind.pkl - Problem parameters
"""

# Standard library imports
import pickle
import os
import copy
import sys
import time

# Scientific computing imports
import numpy as np
import pandas as pd

# Optimization and machine learning imports
import cvxpy as cp
import torch
import torch.nn as nn
import torch.optim as optim

# Visualization imports
import matplotlib.pyplot as plt

# Custom utility functions
from tools import get_para_input, get_para_ind, find_optimal_solution, find_optimal_solution_ind, generate_u0

# =============================================================================
# CONFIGURATION FLAGS AND PARAMETERS
# =============================================================================

# Control flags for different stages of data generation
flag_generate_para_input = True    # Generate original problem parameters
flag_generate_para_ind = True      # Generate reformulated problem parameters
flag_generate_x_data = True        # Generate power demand data and solutions
flag_split_data = True             # Split data into training and validation sets

# Dataset size configuration
num_x = 200                        # Total number of data points to generate (can be 500 or 1000)
num_x_limit = 100                  # Maximum number of data points to solve in one run (for memory management)

# =============================================================================
# SECTION 1: GENERATE ORIGINAL PROBLEM PARAMETERS (para_input)
# =============================================================================

if flag_generate_para_input:
    print("Generating original problem parameters...")
    # Generate the parameters for the original optimization problem
    # This includes matrices Q, c, Aeq, Beq, Aineq, Bineq, bbineq
    para_input = get_para_input()
    
    # Save parameters to pickle file for future use
    with open('para_input.pkl', 'wb') as f:
        pickle.dump(para_input, f)
    print("Original problem parameters saved to para_input.pkl")

# Load parameters from pickle file (regardless of whether we just generated them)
print("Loading original problem parameters...")
with open('para_input.pkl', 'rb') as f:
    para_input = pickle.load(f)

# =============================================================================
# SECTION 2: GENERATE REFORMULATED PROBLEM PARAMETERS (para_ind)
# =============================================================================

if flag_generate_para_ind:
    print("Generating reformulated problem parameters...")
    # Generate parameters for the reformulated problem
    # This reformulation helps with dimension reduction
    para_ind = get_para_ind(para_input)
    
    # Validate the reformulation by comparing solutions
    print("Validating reformulation accuracy...")
    x_single = para_input['pd'].reshape(1, -1)  # Use original power demand as test case
    
    # Solve using reformulated problem
    u_single, fval = find_optimal_solution_ind(x_single, para_ind, para_input)
    
    # Solve using original problem for comparison
    u_single_ref, fval_ref = find_optimal_solution(x_single, para_input)
    
    # Calculate differences to verify accuracy
    diff = np.linalg.norm(u_single - u_single_ref)
    fval_diff = np.abs(fval - fval_ref)
    
    print(f"Solution difference between reformulated and original: {diff}")
    print(f"Objective value difference: {fval_diff}")
    
    # Save reformulated parameters
    with open('para_ind.pkl', 'wb') as f:
        pickle.dump(para_ind, f)
    print("Reformulated problem parameters saved to para_ind.pkl")

# Load reformulated parameters
print("Loading reformulated problem parameters...")
with open('para_ind.pkl', 'rb') as f:
    para_ind = pickle.load(f)

# =============================================================================
# SECTION 3: GENERATE POWER DEMAND DATA AND OPTIMAL SOLUTIONS
# =============================================================================

if flag_generate_x_data:
    print("Generating power demand data and optimal solutions...")
    
    # Check if data generation was previously started
    # This allows for resuming interrupted data generation
    if os.path.exists('x_data.csv'):
        print("Found existing data files. Checking progress...")
        x_data_old = np.loadtxt('x_data.csv', delimiter=',')
        num_x_remain = num_x - x_data_old.shape[0]
        print(f"Already generated {x_data_old.shape[0]} data points. {num_x_remain} remaining.")
    else:
        print("No existing data found. Starting fresh generation...")
        num_x_remain = num_x

    # Continue data generation if there are remaining data points
    if num_x_remain > 0:
        # Determine how many data points to solve in this run
        num_x_solve = min(num_x_limit, num_x_remain)
        
        # Calculate the range of indices for this batch
        index_start = num_x - num_x_remain
        index_end = index_start + num_x_solve
        ix_solve = list(range(index_start, index_end))
        
        print(f"Solving {num_x_solve} data points (indices {index_start} to {index_end-1})...")
        
        # Initialize lists to store new data
        x_data_new = []      # Power demand data
        u_data_new = []      # Optimal generation strategies
        fval_data_new = []   # Objective function values
        u0_data_new = []     # Interior points
        ua_data_new = []     # Slack variables
        
        count = 0
        for ix in ix_solve:
            print(f"Solving {ix}th data point, {num_x_solve - count} remaining in this run...")
            
            # Generate power demand with random variation (±10%)
            # This creates realistic scenarios with demand uncertainty
            x_single = para_input['pd'] * np.random.uniform(0.9, 1.1, para_input['pd'].shape)
            x_data_new.append(x_single)
            
            # Find optimal solution for this power demand
            u_single, fval = find_optimal_solution(x_single, para_input)
            u_data_new.append(u_single)
            fval_data_new.append(fval)
            
            # Generate interior point and slack variables
            # These are useful for interior point methods and barrier functions
            u0_single, ua = generate_u0(x_single, para_ind, para_input)
            u0_data_new.append(u0_single)
            ua_data_new.append(ua)
            
            print(f"Completed {ix}th data point, {num_x_solve - count} remaining in this run")
            count += 1

        # Convert lists to numpy arrays for efficient storage
        x_data_new = np.array(x_data_new)
        u_data_new = np.array(u_data_new)
        fval_data_new = np.array(fval_data_new)
        u0_data_new = np.array(u0_data_new)
        ua_data_new = np.array(ua_data_new)
        
        # Save new data points to CSV files
        # Use append mode if files exist, write mode if they don't
        print("Saving new data points to CSV files...")
        
        with open('x_data.csv', 'a' if os.path.exists('x_data.csv') else 'w') as f:
            np.savetxt(f, x_data_new, delimiter=',')
        with open('u_data.csv', 'a' if os.path.exists('u_data.csv') else 'w') as f:
            np.savetxt(f, u_data_new, delimiter=',')
        with open('fval_data.csv', 'a' if os.path.exists('fval_data.csv') else 'w') as f:
            np.savetxt(f, fval_data_new, delimiter=',')
        with open('u0_data.csv', 'a' if os.path.exists('u0_data.csv') else 'w') as f:
            np.savetxt(f, u0_data_new, delimiter=',')
        with open('ua_data.csv', 'a' if os.path.exists('ua_data.csv') else 'w') as f:
            np.savetxt(f, ua_data_new, delimiter=',')
            
        print(f"Successfully saved {num_x_solve} new data points")
        
    else:
        # All data points have been generated
        print("All data points have been generated. Performing final validation...")
        
        # Verify that all CSV files have the correct number of rows
        x_data = np.loadtxt('x_data.csv', delimiter=',')
        u_data = np.loadtxt('u_data.csv', delimiter=',')
        fval_data = np.loadtxt('fval_data.csv', delimiter=',')
        u0_data = np.loadtxt('u0_data.csv', delimiter=',')
        ua_data = np.loadtxt('ua_data.csv', delimiter=',')
        
        # Check consistency across all files
        if (x_data.shape[0] == num_x and u_data.shape[0] == num_x and 
            fval_data.shape[0] == num_x and u0_data.shape[0] == num_x and 
            ua_data.shape[0] == num_x):
            print("✓ All data files are complete and consistent")
        else:
            print("✗ Inconsistent data files detected. Some data points may be missing.")
            print(f"Expected {num_x} rows, found: x_data={x_data.shape[0]}, u_data={u_data.shape[0]}, "
                  f"fval_data={fval_data.shape[0]}, u0_data={u0_data.shape[0]}, ua_data={ua_data.shape[0]}")
            exit()

# =============================================================================
# SECTION 4: SPLIT DATA INTO TRAINING AND VALIDATION SETS
# =============================================================================

if flag_split_data:
    print("Splitting data into training and validation sets...")
    
    # Load all generated data
    x_data = np.loadtxt('x_data.csv', delimiter=',')
    u_data = np.loadtxt('u_data.csv', delimiter=',')
    fval_data = np.loadtxt('fval_data.csv', delimiter=',')
    u0_data = np.loadtxt('u0_data.csv', delimiter=',')
    ua_data = np.loadtxt('ua_data.csv', delimiter=',')
    
    # Calculate split sizes (80% training, 20% validation)
    n_total = x_data.shape[0]
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    
    print(f"Total data points: {n_total}")
    print(f"Training set size: {n_train} ({n_train/n_total*100:.1f}%)")
    print(f"Validation set size: {n_val} ({n_val/n_total*100:.1f}%)")
    
    # Split training data (first 80%)
    x_train = x_data[:n_train]
    u_train = u_data[:n_train]
    fval_train = fval_data[:n_train]
    u0_train = u0_data[:n_train]
    ua_train = ua_data[:n_train]

    # Split validation data (last 20%)
    x_val = x_data[n_train:]
    u_val = u_data[n_train:]
    fval_val = fval_data[n_train:]
    u0_val = u0_data[n_train:]
    ua_val = ua_data[n_train:]
    
    # Save training data to pickle file
    print("Saving training data...")
    with open('train_data.pkl', 'wb') as f:
        pickle.dump({
            'x': x_train,      # Power demand for training
            'u': u_train,      # Optimal generation strategies for training
            'fval': fval_train, # Objective values for training
            'u0': u0_train,    # Interior points for training
            'ua': ua_train     # Slack variables for training
        }, f)
    
    # Save validation data to pickle file
    print("Saving validation data...")
    with open('val_data.pkl', 'wb') as f:
        pickle.dump({
            'x': x_val,        # Power demand for validation
            'u': u_val,        # Optimal generation strategies for validation
            'fval': fval_val,   # Objective values for validation
            'u0': u0_val,      # Interior points for validation
            'ua': ua_val       # Slack variables for validation
        }, f)
    
    print("✓ Data generation and splitting completed successfully!")
    print("Final outputs:")
    print("  - train_data.pkl: Training dataset")
    print("  - val_data.pkl: Validation dataset")
    print("  - para_input.pkl: Original problem parameters")
    print("  - para_ind.pkl: Reformulated problem parameters")

