"""File containing all utility functions for NetPlot"""
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import seaborn as sns
import matplotlib.pyplot as plt
import math
import sys

import numpy as np

import resnet
import other_resnet

cuda_ = "cuda:0"
device = torch.device(cuda_ if torch.cuda.is_available() else "cpu")

def loadNetwork(path, arch):
    """ Loads the network from the specified path """
    if arch in resnet.__dict__:
        model = resnet.__dict__[arch]()
        model.load_state_dict(torch.load(path, map_location=device))
        return model
    elif arch in other_resnet.__dict__:
        model = other_resnet.__dict__[arch]()
        model.load_state_dict(torch.load(path, map_location=device))
        return model

def processNetwork(model):
    """ Creates and returns a dictionary containing {module name : weights} pairs 
    
    This function takes a model as an argument and creates a dictionary corresponding to that model.
    The dictionary keys are the names of each module in the model and the value for each key is a 
    numpy array made up of the weights for that module.

    Args:
        model: Neural network that has been loaded using loadNetwork()
    """
    module_dict = {}

    # Loop through modules in network and add {name:array} pairs to dictionary
    print("Creating dictionary...")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            module_dict[name] = [module.weight.detach().numpy(), module.weight.detach().numpy().shape]
        elif isinstance(module, torch.nn.Linear):
            module_dict[name] = [module.weight.detach().numpy(), module.weight.detach().numpy().shape]

    # Delete all layers that are not 4 dimensional
    to_delete = []
    for name, module in module_dict.items():
        if (len(module[1]) < 4) or ((module[1][-1] != 1) and (module[1][-1] != 3)):
            to_delete.append(name)

    for name in to_delete:
        del module_dict[name]

    # Now we loop through and transform each module into an array that can be plotted more easily
    print("Cleaning up...")
    for name, module in module_dict.items():
        x_dim = module[1][0]
        y_dim = module[1][1]
        conv_dim = module[1][-1]
        final_x_dim = x_dim*conv_dim
        final_y_dim = y_dim*conv_dim

        if conv_dim == 3:
            weights = np.array(module[0]).reshape(x_dim*y_dim, conv_dim, conv_dim)

            top_weights = weights[:, 0].flatten()
            middle_weights = weights[:, 1].flatten()
            bottom_weights = weights[:, 2].flatten()
                
            weights = np.append(top_weights, np.append(middle_weights, bottom_weights)).reshape(conv_dim, top_weights.shape[0])
            final_weights = np.zeros((final_y_dim, final_x_dim))
            # Currently only works when conv_dim = 3 because for loop is hardcoded
            # Could probably also be vectorized/sped up
            for i in range(y_dim):
                final_weights[i*3] = weights[0][i*final_x_dim:(i+1)*final_x_dim]
                final_weights[i*3+1] = weights[1][i*final_x_dim:(i+1)*final_x_dim]
                final_weights[i*3+2] = weights[2][i*final_x_dim:(i+1)*final_x_dim]
        else:
            final_weights = module[0].reshape(x_dim, y_dim)
        
        module_dict[name] = final_weights
    
    return module_dict

#TODO: def plotLayer(layer_name):
    # Plots a single layer

def plotNetwork(module_dict, arch, max_dim):
    """Creates a set of heatmaps corresponding to the weights in each layer of the original network

    Args:
        module_dict: Dictionary returned by the processNetwork() function
        arch (str): The network architecture being used
        max_dim (int): Will be automated soon so don't worry about this
    """
    # Not a great way of doing it but it'll do for now
    min_val = 0
    max_val = 0
    for name, module in module_dict.items():
        if np.amin(module) < min_val:
            min_val = np.amin(module)
        if np.amax(module) > max_val:
            max_val = np.amax(module)

    list_keys = list(module_dict)
    num_layers = len(module_dict)
    num_cols = 8
    num_rows = math.ceil(num_layers/8)
    fig, axes = plt.subplots(num_cols, num_rows, figsize=(num_cols*10, num_rows*10))

    for i, ax in zip(range(num_layers), axes.flat):
        sns.heatmap(module_dict[list_keys[i]], xticklabels=False, yticklabels=False, center=0.00, cmap="coolwarm", square=True, cbar=False, ax=ax)
        #axes[i].set_title(list_keys[i])
        ax.set(ylim=(0, max_dim*3))
        ax.set(xlim=(0, max_dim*3))
        ax.set_title(list_keys[i])
    
    if not os.path.exists('plots'):
        os.makedirs('plots')

    fig.savefig('plots/{architecture}full_network.png'.format(architecture=arch), transparent=True)

def plotDifference(path1, path2, architecture, max_dim):
    """Plots the change in weights between two neural networks

    Plots the difference in weights between the two networks passed as arguments. 
    Note that we consider the order of the networks to be chronological, meaning we 
    subtract the weights of the first network from the weights of the second network.

    Args:
        path1 (str): Location of the .th or .pth file for the first network
        path2 (str): Location of the .th or .pth file for the second network
        architecture (str): The network architecture being used (architectures must match)
        max_dim (int): Will be automated soon so don't worry about this
    """
    network1 = loadNetwork(path1, architecture)
    network2 = loadNetwork(path2, architecture)
    network1_dict = processNetwork(network1)
    network2_dict = processNetwork(network2)
    difference_dict = {}

    for name, module in network1_dict.items():
        if network1_dict[name].shape == network2_dict[name].shape:
            difference_dict[name] = np.subtract(network2_dict[name], network1_dict[name])
        else:
            print("Input networks must be of the same architecture")
            break
    
    plotNetwork(difference_dict, architecture, max_dim)
