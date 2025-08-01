# =============================================================================
# Neural Network Training Script for Optimization Problems
# =============================================================================
# 
# This file is used to train neural network models for solving optimization problems
# with different feasibility strategies.
# 
# FEASIBILITY STRATEGIES:
# - loop2: generalized gauge approach to handle constraints
# - penalty: Penalty method for constraint satisfaction
# - projection: Projection-based constraint handling using solvers
# - loop: gauge-based constraint handling
# 
# NEURAL NETWORK PARAMETERS:
# - number_of_hidden_layers: Number of hidden layers in the network
# - number_of_hidden_nodes: Number of neurons in each hidden layer
# - learning_rate: Learning rate for the optimizer
# - number_of_epochs: Number of training epochs
# - batch_size: Batch size for training
# 
# FILE ORGANIZATION:
# Models are saved in folders organized by feasibility strategy:
# - Main folder: method name (loop2, penalty, projection, loop)
# - Subfolder: model configuration (layers_nodes_lr_epochs_batch)
# - Files in each subfolder:
#   * model_para.pkl: Model parameters and state
#   * model_metrics.pkl: Training results (loss, accuracy, etc.)
# =============================================================================

# ── standard library ─────────────────────────────────────────
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
import time
from tools import CustomNN, train_and_test, test, get_optimality_gap, get_optimality_deviation_rate, get_feasibility_gap

# Set random seeds for reproducibility
torch.manual_seed(0)          # reproducibility ─ remove if you want stochastic runs
np.random.seed(0)


# =============================================================================
# SECTION 0: HYPERPARAMETER CONFIGURATION
# =============================================================================
# Define the ranges of hyperparameters to test during model training
# These lists control the grid search over different model configurations

number_of_hidden_nodes_list = [20] 
number_of_hidden_layers_list = [1]
learning_rate_list = [1e-4]
number_of_epochs_list = [500]
batch_size_list = [500]
method_list = ['loop']  # Feasibility strategy to use

# =============================================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
# =============================================================================

# Load training and validation datasets
# These contain the input-output pairs for training the neural network
with open('train_data.pkl', 'rb') as f:
    training_data = pickle.load(f)
with open('val_data.pkl', 'rb') as f:
    validation_data = pickle.load(f)

# Load problem parameters that define the optimization problem structure
# para_input: Contains matrices and vectors defining the optimization problem
with open('para_input.pkl', 'rb') as f:
    para_input = pickle.load(f)

# Ensure bbeq parameter exists - if not, initialize it as zero vector
# bbeq represents the right-hand side of equality constraints
if 'bbeq' not in para_input:
    para_input['bbeq'] = np.zeros((para_input['Aeq'].shape[0],1)).ravel()

# para_ind: Contains indices and other problem-specific parameters
with open('para_ind.pkl', 'rb') as f:
    para_ind = pickle.load(f)

# =============================================================================
# SECTION 2: MODEL TRAINING LOOP
# =============================================================================
# Grid search over all hyperparameter combinations
# For each combination, train a model and save the results

for method in method_list:
    for number_of_hidden_nodes in number_of_hidden_nodes_list:
        for number_of_hidden_layers in number_of_hidden_layers_list:
            for learning_rate in learning_rate_list:
                for number_of_epochs in number_of_epochs_list:
                    for batch_size in batch_size_list:
                        
                        # Create parameter dictionary for current configuration
                        # This will be passed to the training function
                        sita = {
                            'method': method,
                            'number_of_hidden_nodes': number_of_hidden_nodes,
                            'number_of_hidden_layers': number_of_hidden_layers,
                            'learning_rate': learning_rate,
                            'number_of_epochs': number_of_epochs,
                            'batch_size': batch_size
                        }
                        
                        # Record training start time for performance monitoring
                        start_time = time.time()
                        
                        # Train and test the model with current configuration
                        # This function handles the entire training process
                        results = train_and_test(
                            training_data,
                            validation_data,
                            para_ind,
                            para_input,
                            sita,
                            print_loss=False
                        )
                        
                        # Calculate and display training time
                        end_time = time.time()
                        print(f"Time taken: {end_time - start_time} seconds")
                        
                        # =============================================================================
                        # SECTION 3: MODEL SAVING
                        # =============================================================================
                        # Create directory structure for saving the trained model
                        
                        # Main folder named after the feasibility method
                        folder_name = method
                        
                        # Subfolder named with model configuration parameters
                        # Format: layers_nodes_lr_epochs_batch
                        subfolder_name = f"{number_of_hidden_layers}_{number_of_hidden_nodes}_{learning_rate}_{number_of_epochs}_{batch_size}"
                        
                        # Create directories if they don't exist
                        os.makedirs(folder_name, exist_ok=True)
                        os.makedirs(os.path.join(folder_name, subfolder_name), exist_ok=True)
                        
                        # Save model parameters (weights, biases, optimizer state)
                        with open(os.path.join(folder_name, subfolder_name, 'model_para.pkl'), 'wb') as f:
                            pickle.dump(results['model_para'], f)
                            
                        # Save training metrics (loss, accuracy, etc.)
                        with open(os.path.join(folder_name, subfolder_name, 'model_metrics.pkl'), 'wb') as f:
                            pickle.dump(results['model_metrics'], f)
                        
                        # Print summary of saved model configuration
                        print(f"method: {method} | number_of_hidden_layers: {number_of_hidden_layers} | "
                              f"number_of_hidden_nodes: {number_of_hidden_nodes} | learning_rate: {learning_rate} | "
                              f"number_of_epochs: {number_of_epochs} | batch_size: {batch_size} | model has been saved")


                 
