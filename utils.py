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

import numpy as np

import resnet

# Device set to either GPU or CPU
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

def loadNetwork(path):
    model = resnet.__dict__[args.arch]()
    model.load_state_dict(torch.load(path, map_location=device))

def processNetwork():
    module_dict = {}

    # Loop through modules in network and add {name:array} pairs to dictionary
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            module_dict[name] = [module.weight.detach().numpy(), module.weight.detach().numpy().shape]
        elif isinstance(module, torch.nn.Linear):
            module_dict[name] = [module.weight.detach().numpy(), module.weight.detach().numpy().shape]

    # Delete all layers that are not 4D or don't have dimensions (a, a, b, b)
    # In most cases this is just the first Conv2d layer and linear layer
    to_delete = []
    for name, module in module_dict.items():
        if (len(module[1]) < 4) or (module[1][0] != module[1][1]):
            to_delete.append(name)
            
    for name in to_delete:
        del module_dict[name]

    # Now we loop through and transform each module into an array that can be plotted more easily
    for name, module in module_dict.items():
        layer_dim = module[1][0]
        conv_dim = module[1][-1]
        final_dim = layer_dim*conv_dim
        weights = np.array(module[0]).reshape(layer_dim**2, conv_dim, conv_dim)
        
        top_weights = []
        middle_weights = []
        bottom_weights = []
            
        for i in range(len(weights)):
            top_weights = np.append(top_weights, weights[i][0])
            middle_weights = np.append(middle_weights, weights[i][1])
            bottom_weights = np.append(bottom_weights, weights[i][2])
            
        weights = np.append(top_weights, np.append(middle_weights, bottom_weights)).reshape(conv_dim, top_weights.shape[0])
        final_weights = np.zeros((final_dim, final_dim))
        # Currently only works when conv_dim = 3 because for loop is hardcoded
        for i in range(layer_dim):
            final_weights[i*3] = weights[0][i*final_dim:(i+1)*final_dim]
            final_weights[i*3+1] = weights[1][i*final_dim:(i+1)*final_dim]
            final_weights[i*3+2] = weights[2][i*final_dim:(i+1)*final_dim]
        
        module_dict[name] = final_weights

def plot_layer(layer_name):
    # Plots a single layer

def plot_network():
    # Plots entire network
