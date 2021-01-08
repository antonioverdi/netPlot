<p align="center">
    <br>
    <img src="https://github.com/antonioverdi/Text-Generation-GUI/blob/master/docs/imgs/happy-robot.png" width="200"/>
    <br>
<p>
<h1 align="center">
<p> Graphically map out neural network structures </p>
</h1>

## Table of Contents

- [Quick Start](#quick-start)
- [Copyright and License](#copyright-and-license)

## Quick Start
### Option 1: Start with path to model
The first option is to start with some model state dict saved as either a `.th` or `.pth` file. 
```Python
model = loadNetwork(file_path, architecture)
```
*Note:* A list of all supported network architectures can be printed out using the `supportedNetworks()` function found in `utils`.

Once we have loaded the model as shown above, it can then simply be processed and plotted using a call to the `processNetwork()` and `plotNetwork()` in `utils` as follows:
```Python
import utils

# Creating a dictionary of {name:weights} pairs
model_dict = utils.processNetwork(model)

# Plotting all trainable layers within the network using the dictionary we created in the previous line
utils.plotNetwork(model_dict)
```

### Option 2: Start with preloaded model
If a model has already been created and either trained or loaded with a state dict, the model can then simply be processed and plotted using a call to the `processNetwork()` and `plotNetwork()` in `utils`. For example, let us say we have already loaded and trained some model named 'test_model". We would then process and plot the network with:
```Python
import utils

# Creating a dictionary of {name:weights} pairs
model_dict = utils.processNetwork(model)

# Plotting all trainable layers within the network using the dictionary we created in the previous line
utils.plotNetwork(model_dict)
```

## Copyright and License
