import argparse
import os
import shutil
import time

import torch
import torch.utils.data

import utils

cuda_ = "cuda:0"
device = torch.device(cuda_ if torch.cuda.is_available() else "cpu")	

parser = argparse.ArgumentParser(description='NetPlot')
parser.add_argument('--model_path', dest='PATH', type=str, metavar='PATH', required=True,
					help='Path to state dict file from current directory')
parser.add_argument('--arch', '-a', metavar='ARCH', required=True, type=str, 
                    help='Currently only supports ResNet models')

def main():
	global args
	args = parser.parse_args()
	MODEL_ARCH = (args.arch).lower()
	MODEL_PATH = args.path

	if not os.path.exists(MODEL_PATH):
		print("No such path exists")
	else:
		model = utils.loadNetwork(MODEL_PATH, MODEL_ARCH)
		model_dict = utils.processNetwork(model)
		utils.plotNetwork(model_dict, MODEL_ARCH, 512)

if __name__ == '__main__':
	main()