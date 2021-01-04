import json
import matplotlib.pyplot as plt
import numpy as np

def create_individual_plots(filepath, compression_list):

	accuracies, keys = _get_accuracies_and_keys(filepath, compression_list)
	int_compressions = list(map(_to_int, compression_list))
	x_ticks = list(map(_add_percent, int_compressions))
	x = np.arange(len(x_ticks))
	for k in keys:
		change = _get_acc_change(accuracies[k])

		plt.figure(figsize=(16,4))
		plt.suptitle(k + " Compression")
		plt.subplot(121)
		plt.ylim(85,100)
		plt.plot(x, accuracies[k])
		plt.xticks(x, x_ticks)
		plt.xlabel("Percent of Weights Retained")
		plt.ylabel("Accuracy on Test Set")
		plt.subplot(122)
		plt.grid(True)
		plt.plot(x, change, 'o--')
		plt.xticks(x, x_ticks)
		plt.ylim(-5,3)
		plt.yticks(np.arange(-5,4), list(map(_add_percent, np.arange(-5,4))))
		plt.xlabel("Percent of Weights Retained")
		plt.ylabel("Change in Accuracy")

def create_comparison_plots(filepath, compression_list):

		accuracies, keys = _get_accuracies_and_keys(filepath, compression_list)
		int_compressions = list(map(_to_int, compression_list))
		x_ticks = list(map(_add_percent, int_compressions))
		x = np.arange(len(x_ticks))

		plt.figure(figsize=(16,4))
		plt.suptitle("Comparison of Pruning Techniques")	
		plt.subplot(121)
		plt.ylim(85,100)
		for k in keys:
			plt.plot(x, accuracies[k], label=k)
		plt.xticks(x, x_ticks)
		plt.xlabel("Percent of Weights Retained")
		plt.ylabel("Accuracy on Test Set")
		plt.legend()
		plt.subplot(122)
		plt.grid(True)
		for k in keys:
			change = _get_acc_change(accuracies[k])
			plt.plot(x, change, 'o--', label=k)
		plt.xticks(x, x_ticks)
		plt.ylim(-5,3)
		plt.yticks(np.arange(-5,4), list(map(_add_percent, np.arange(-5,4))))
		plt.xlabel("Percent of Weights Retained")
		plt.ylabel("Change in Accuracy")	
		plt.legend()		

def _to_int(n):
    return int(n)

def _add_percent(n):
	return str(n) + "%"

def _get_acc_change(accuracies):
	first_ind = accuracies.index(next(filter(lambda x: x!=0, accuracies)))
	unpruned_acc = accuracies[first_ind]
	acc_change = [x - unpruned_acc for x in accuracies]
	return acc_change

def _get_accuracies_and_keys(filepath, compression_list):
	accuracies = {}
	key_list = []
	with open(filepath, 'r') as logs:
		data = json.load(logs)
		key_list = list(data.keys())
		for k in key_list:
			accuracies[k] = []
			for compression in compression_list:
				compress_key = 'compression' + compression
				if compress_key in data[k]:
					accuracies[k].append(data[k][compress_key]['accuracy'])
				else:		
					if accuracies[k] == []:
						accuracies[k].append(92.33)
					else:
						print("{}{} missing. Check that all models were put through test.py," \
						"a default value of previous accuracy will be used".format(k, compression))
						accuracies[k].append(accuracies[k][-1])
	return accuracies, key_list