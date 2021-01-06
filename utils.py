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

import numpy as np

import resnet
import other_resnet

# Device set to either GPU or CPU
model_names = sorted(name for name in resnet.__dict__
	if name.islower() and not name.startswith("__")
					 and name.startswith("resnet")
					 and callable(resnet.__dict__[name]))

cuda_ = "cuda:0"
device = torch.device(cuda_ if torch.cuda.is_available() else "cpu")	

parser = argparse.ArgumentParser(description='Heatmap plotting for Neural Networks')
parser.add_argument('--path', default='', type=str, metavar='PATH',
					help='Path to .th file (default: none)')
parser.add_argument('--save-dir', dest='save_dir',
					help='The directory used to save the plot(s)',
					default='save_temp', type=str)
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet56', type=str, 
                    help='Currently only supports ResNet models. (default: resnet56)')

def main():
    global args
    args = parser.parse_args()

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

def loadNetwork(path, arch):
    if arch in resnet.__dict__:
        model = resnet.__dict__[arch]()
        model.load_state_dict(torch.load(path, map_location=device))
        return model
    elif arch in other_resnet.__dict__:
        model = other_resnet.__dict__[arch]()
        model.load_state_dict(torch.load(path, map_location=device))
        return model

def processNetwork(model):
    module_dict = {}

    # Loop through modules in network and add {name:array} pairs to dictionary
    print("Creating dictionary...")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            module_dict[name] = [module.weight.detach().numpy(), module.weight.detach().numpy().shape]
        elif isinstance(module, torch.nn.Linear):
            module_dict[name] = [module.weight.detach().numpy(), module.weight.detach().numpy().shape]

    # Delete all layers that are not 4D or don't have dimensions (a, a, b, b)
    to_delete = []
    for name, module in module_dict.items():
        if (len(module[1]) < 4) or ((module[1][-1] != 1) and (module[1][-1] != 3)):
            to_delete.append(name)
            print(module[1][-1])
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
            print(final_weights.shape)
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

# def plotLayer(layer_name):
    # Plots a single layer

def plotNetwork(module_dict, arch, max_dim):
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
    
    if not os.path.exists('plots'):
        os.makedirs('plots')
    fig.savefig('plots/{architecture}full_network.png'.format(architecture=arch), transparent=True)
