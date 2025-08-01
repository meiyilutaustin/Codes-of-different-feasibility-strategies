Power System Optimization Dataset Generation and Neural Network Training
=====================================================================

please cite "Li, Meiyi, and Javad Mohammadi. "Toward rapid, optimal, and feasible power dispatch through generalized neural mapping." 2024 IEEE Power & Energy Society General Meeting (PESGM). IEEE, 2024." for LOOP2;
please cite "Li, Meiyi, Soheil Kolouri, and Javad Mohammadi. "Learning to solve optimization problems with hard linear constraints." IEEE Access 11 (2023): 59995-60004."for Loop.

This project implements a comprehensive framework for generating training datasets for power system optimization problems and training neural networks to solve them efficiently. The code is organized into two main sections that work together to create a machine learning solution for DC Optimal Power Flow (DCOPF) problems.

PROJECT OVERVIEW
================
The project addresses the challenge of solving power system optimization problems in real-time using neural networks. Traditional optimization solvers are computationally expensive for real-time applications, so this project generates training data from solved optimization problems and trains neural networks to predict optimal solutions quickly.

RAW DATA SOURCE
===============
The raw_para folder contains:
- dcopf_data.mat: MATLAB data file containing a 200-bus power system case (case_ACTIVSg200)
- test.m: MATLAB script that generates the optimization problem coefficients from MATPOWER case files

The raw data includes:
- Power system topology (buses, generators, branches)
- Generator cost coefficients (quadratic and linear terms)
- Power demand data
- Constraint matrices for equality and inequality constraints

SECTION 0: DATASET GENERATION (section0.py)
===========================================
This script creates a comprehensive training dataset for machine learning models by solving multiple instances of the power system optimization problem.

Key Functions:
1. Problem Parameter Generation:
   - get_para_input(): Loads and organizes raw data into optimization problem format
   - get_para_ind(): Reformulates the problem using variable elimination for dimensionality reduction

2. Data Generation Process:
   - Generates 200 power demand scenarios with ±10% random variation
   - Solves the optimization problem for each scenario using CVXPY
   - Computes optimal generation strategies, objective values, and interior points
   - Creates slack variables for interior point methods

3. Output Files Generated:
   - para_input.pkl: Original problem parameters (Q, c, Aeq, Beq, Aineq, Bineq, bbineq)
   - para_ind.pkl: Reformulated problem parameters with reduced dimensionality
   - x_data.csv: Power demand scenarios (input data)
   - u_data.csv: Optimal generation strategies (output data)
   - fval_data.csv: Objective function values
   - u0_data.csv: Interior points for barrier functions
   - ua_data.csv: Slack variables
   - train_data.pkl: Training dataset (80% of data)
   - val_data.pkl: Validation dataset (20% of data)

4. Problem Formulation:
   Original Problem:
     min ½ uᵀ Q u + cᵀ u
     s.t. Aeq @ u + Beq @ x = 0
          Aineq @ u + Bineq @ x + bbineq ≤ 0

   Reformulated Problem:
     min ½ u_indᵀ Q_ind u_ind + c_indᵀ u_ind
     s.t. A_all @ u_ind + B_all @ x + bb_all ≤ 0

SECTION 1: NEURAL NETWORK TRAINING (section1.py)
================================================
This script trains neural networks to learn the mapping from power demand to optimal generation strategies using different constraint handling methods.

Key Features:
1. Multiple Feasibility Strategies:
   - loop2: Generalized gauge approach for constraint handling
   - penalty: Penalty method for constraint satisfaction
   - projection: Projection-based constraint handling using solvers
   - loop: Gauge-based constraint handling

2. Neural Network Architecture:
   - Custom neural network class (CustomNN) defined in tools.py
   - Configurable number of hidden layers and nodes
   - Supports different activation functions and training parameters

3. Training Process:
   - Grid search over hyperparameters (layers, nodes, learning rate, epochs, batch size)
   - Uses PyTorch for neural network implementation
   - Implements custom loss functions that incorporate constraint satisfaction
   - Saves best models based on validation performance

4. Model Organization:
   Models are saved in a hierarchical folder structure:
   - Main folder: method name (loop2, penalty, projection, loop)
   - Subfolder: model configuration (layers_nodes_lr_epochs_batch)
   - Files: model_para.pkl (model parameters), model_metrics.pkl (training metrics)

5. Performance Metrics:
   - Optimality gap: Measures how close the predicted solution is to the true optimum
   - Feasibility gap: Measures constraint violation
   - Optimality deviation rate: Percentage deviation from optimal solution

TOOLS MODULE (tools.py)
=======================
Contains utility functions and classes:
- Problem setup and data loading functions
- Optimization solvers using CVXPY
- Neural network architecture (CustomNN class)
- Training and testing functions
- Performance evaluation metrics
- Constraint handling methods

WORKFLOW
========
1. Raw data preparation (MATLAB/MATPOWER) → dcopf_data.mat
2. Dataset generation (section0.py) → Training/validation datasets
3. Neural network training (section1.py) → Trained models
4. Model evaluation and deployment

APPLICATIONS
============
This framework is designed for:
- Real-time power system optimization
- Fast solution prediction for varying demand scenarios
- Integration with power system control systems
- Research in machine learning for power systems

TECHNICAL REQUIREMENTS
======================
- Python 3.x with NumPy, Pandas, PyTorch, CVXPY(gurobi)
- MATLAB/MATPOWER for raw data generation
- Sufficient computational resources for dataset generation (can be memory-intensive)

The project demonstrates how machine learning can be applied to power system optimization, providing a balance between computational speed and solution quality for real-time applications.
