import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score

import os
import json
import argparse

import resnet

parser = argparse.ArgumentParser(description='ResNet56 pruning experiment testing properties')
parser.add_argument('--model_dir', type=str,  default='trained_models', help='directory of trained models. Should all be of same model type')
parser.add_argument('--arch', type=str, default="resnet56", help="model type to load pretrained weights into")
parser.add_argument('--log_dir', type=str, default="accuracy_logs.json", help='directory to save accuracy logs from pretrained models')
args = parser.parse_args()

def main():
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

	test_loader = torch.utils.data.DataLoader(
		datasets.CIFAR10(root='./data', train=False, transform=transform_test, download=True),
		batch_size=128, shuffle=False,
		num_workers=0)

	#save files need the format <pruning style><compression rate in 3 numbers>.pth for example SNIP010.pth for SNIP style pruning to 10% weight retention
	model_names = []
	with os.scandir(args.model_dir) as folder:
		for file in folder:
			model_names.append(file.name[:-3])

	#collect accuracies
	accuracies = []
	for prune_style in model_names:
		model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
		model.to(device)

		#some nonsense needed for weight loading since I forgot to remove the weight masks before saving the torch models
		if prune_style[0:4] == "SNIP": 
			for layer in model.modules():
				if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
					layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight)) 
	
		filename = args.model_dir + os.sep + prune_style + ".th"
		print("Testing Model: {} from {}".format(prune_style, filename))
		pretrained = torch.load(filename, map_location=torch.device(device))
		model.load_state_dict(pretrained['state_dict'])
		model.eval()
		model.to(device)
		test_loss = 0
		correct = 0
		total = 0
		with torch.no_grad():
			for batch_idx, (inputs, targets) in enumerate(test_loader):
				inputs, targets = inputs.to(device), targets.to(device)
				outputs = model(inputs)
				_, predicted = outputs.max(1)
				total += targets.size(0) 
				correct += predicted.eq(targets).sum().item()
		
		acc = 100. * correct / total
		accuracies.append(acc)
		print("    Accuracy: {}%\n".format(acc))

	#write to json file.
	output_json = {}
	for i,model_name in enumerate(model_names):
		prune_style = model_name[:-3]
		if not (prune_style in output_json):
			output_json[prune_style] = {}
		output_json[prune_style]['compression' + model_name[-3:]] = {'accuracy': accuracies[i]}

	with open(args.log_dir, 'w') as output:
		json.dump(output_json, output, indent=1)
	print("Testing complete, results saved to {}".format(args.log_dir))


if __name__ == '__main__':
	main()