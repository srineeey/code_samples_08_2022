
"""
function for calculating neuron activation layer sizes.
needed for torch_net_class
needs to be updated each time a new type of layer is being introduced
net_struct = list of dictionaries with layer attributes and parameters
"""

def calc_layer_sizes(input_shape, net_struct, log_file=None):
    #layer_sizes will have shape
    #[[channels,length,width],...,dense_neurons,...]
    layer_sizes = [input_shape]
    #print(f"input_shape: {input_shape}")

    #go through each layer
    for i in range(len(net_struct)):
        new_layer_size = []
        #print(i)
        #print(net_struct[i]["type"])

        if net_struct[i]["type"] == nn.Linear:
            new_layer_size = net_struct[i]["layer_pars"]["out_features"]

        elif net_struct[i]["type"] == nn.Flatten:
            new_layer_size = int(np.prod(layer_sizes[-1]))

        elif net_struct[i]["type"] == misc_modules.Reshape:
            new_layer_size = net_struct[i]["layer_pars"]["new_shape"]

        elif net_struct[i]["type"] == misc_modules.NpSplitReImToChannel:
            channel = net_struct[i]["layer_pars"]["channel_axis"]
            prev_layer_size = layer_sizes[-1].copy()
            new_layer_size = prev_layer_size
            new_layer_size[channel] *= 2

        elif net_struct[i]["type"] == misc_modules.PermuteAxes:
            perm = net_struct[i]["layer_pars"]["perm"]
            prev_layer_size = layer_sizes[-1].copy()
            new_layer_size = prev_layer_size[perm]

        elif net_struct[i]["type"] == circ_padding.CircularPadding:
            perm = net_struct[i]["layer_pars"]["padding"]
            prev_layer_size = layer_sizes[-1].copy()
            new_layer_size = prev_layer_size

            for layer_d in range(len(prev_layer_size)):
                new_layer_size[layer_d] += 2*net_struct[i]["layer_pars"]["padding"][layer_d]

        elif net_struct[i]["type"] == circ_padding.AsymmetricCircularPadding:
            perm = net_struct[i]["layer_pars"]["padding"]
            prev_layer_size = layer_sizes[-1].copy()
            new_layer_size = prev_layer_size

            for layer_d in range(len(prev_layer_size)):
                new_layer_size[layer_d] += net_struct[i]["layer_pars"]["padding"][layer_d]


        #elif (net_struct[i]["type"] == nn.Conv2d) or (net_struct[i]["type"] == nn.MaxPool2d):
        #elif net_struct[i]["type"] in [nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d]:
        elif net_struct[i]["type"] in [nn.Conv1d, nn.Conv2d, nn.Conv3d, conv4d.Conv4d, nn.MaxPool2d, nn.AvgPool2d]:
        #elif net_struct[i]["type"] in [nn.Conv1d, nn.Conv2d, nn.Conv3d, Conv4d, nn.MaxPool2d, nn.AvgPool2d]:

            kernel_shape = net_struct[i]["layer_pars"]["kernel_size"]

            #if net_struct[i]["type"] == nn.Conv2d:
            if net_struct[i]["type"] in [nn.Conv1d, nn.Conv2d, nn.Conv3d, conv4d.Conv4d]:
                stride = [1 for i in range(len(kernel_shape))]
            else:
                stride = kernel_shape

            padding_mode = "zeros"
            padding = [0 for i in range(len(kernel_shape))]
            dilation = [1 for i in range(len(kernel_shape))]

            if "stride" in net_struct[i]["layer_pars"]:
                stride = net_struct[i]["layer_pars"]["stride"]

            if "padding" in net_struct[i]["layer_pars"]:
                padding = net_struct[i]["layer_pars"]["padding"]

            if "padding_mode" in net_struct[i]["layer_pars"]:
                padding_mode = net_struct[i]["layer_pars"]["padding_mode"]

                #circular padding required specific values
                #bug of pytorch
                #works only for odd kernel sizes!!!
                if padding_mode == "circular":
                    padding = [int((kernel_l-1))//2 for kernel_l in kernel_shape]

                    if not(np.all(padding == net_struct[i]["layer_pars"]["padding"])):
                           print("padding size possibly incorrect!")

            if "dilation" in net_struct[i]["layer_pars"]:
                dilation = net_struct[i]["layer_pars"]["dilation"]

            #print(kernel_shape)
            #new_layer_size = []

            for d in range(len(kernel_shape)):
                #get length of the previous layer in dimension d
                #layer_sizes[n][0] = number of channels!
                #print(f"last layer {layer_sizes[-1]}")
                prev_layer_l = int(layer_sizes[-1][d+1])
                kernel_l = int(kernel_shape[d])

                #padding_l = int(padding[d])

                if type(stride) == int:
                    stride_l = stride
                elif type(stride) == list:
                    stride_l = int(stride[d])

                if type(padding) == int:
                    padding_l = padding
                elif type(padding) == list:
                    padding_l = int(padding[d])

                if type(dilation) == int:
                    dilation_l = dilation
                elif type(dilation) == list:
                    dilation_l = int(dilation[d])

                if (prev_layer_l + 2*padding_l - dilation_l*(kernel_l - 1) - 1) % stride_l != 0:
                    print(f"Input {layer_sizes[-1]} maybe not compatible with:")
                    print(net_struct[i]['layer_pars'])
                    #raise ValueError(f'Input {layer_sizes[-1]}, kernel {kernel_shape}, stride {stride} and padding {padding} in layer {i} not compatible!')

                #actual computation

                new_layer_size_l = int(np.floor( (prev_layer_l + 2*padding_l - dilation_l*(kernel_l - 1) - 1)/(stride_l) + 1.))
                new_layer_size.append( new_layer_size_l )

                #print(f"new layer {new_layer_size}")

            #if net_struct[i]["type"] == nn.Conv2d:
            if net_struct[i]["type"] in [nn.Conv1d, nn.Conv2d, nn.Conv3d, conv4d.Conv4d]:
                new_layer_size = [net_struct[i]["layer_pars"]["out_channels"]] + new_layer_size

            elif net_struct[i]["type"] in [nn.MaxPool2d, nn.AvgPool2d]:
                prev_channels = layer_sizes[-1][0]
                new_layer_size = [prev_channels] + new_layer_size

        elif net_struct[i]["type"] == nn.ConvTranspose2d:
            kernel_shape = net_struct[i]["layer_pars"]["kernel_size"]
            stride = net_struct[i]["layer_pars"]["stride"]
            padding = net_struct[i]["layer_pars"]["padding"]

            #new_layer_size = []
            for d in range(len(kernel_shape)):
                #get length of the previous layer in dimension d
                #layer_sizes[n][0] = number of channels!
                prev_layer_l = int(layer_sizes[-1][d+1])
                kernel_l = int(kernel_shape[d])
                if type(stride) == int:
                    stride_l = stride
                elif type(stride) == list:
                    stride_l = int(stride[d])

                #actual conputation
                #for dilation = 1
                new_layer_size.append( (prev_layer_l - 1)*stride_l + kernel_l - 2*padding)

                #new_layer_size.append( (prev_layer_l - 1)*stride_l - 2*padding + 1)

            new_layer_size = [net_struct[i]["layer_pars"]["out_channels"]] + new_layer_size

        #elif (net_struct[i]["type"] == nn.BatchNorm1d) or (net_struct[i]["type"] == nn.Dropout) or (net_struct[i]["type"] == nn.Softmax):

        elif net_struct[i]["type"] in [nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d]:
            """Append channel sizes from last layer"""
            new_layer_size = layer_sizes[-1].copy()
            d = len(net_struct[i]["layer_pars"]["output_size"])
            new_layer_size[-d:] = net_struct[i]["layer_pars"]["output_size"][:]

        elif net_struct[i]["type"] in [nn.BatchNorm1d, nn.Dropout, nn.Softmax]:
            new_layer_size = layer_sizes[-1]

        else:
            print("custom layer operation not defined, assuming previous layer_size for")
            print(net_struct[i]["type"])
            new_layer_size = layer_sizes[-1]

        #append newly calculated neuron activation shape to layer_sizes
        if np.any(np.array(new_layer_size) <= 0):
            #raise ValueError(f'Negative layer size found in {new_layer_size}!')
            print(f'Negative layer size found in {new_layer_size}!')

        layer_sizes.append(new_layer_size)

    return layer_sizes






"""Neural Network class"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch import cat
from torch import stack

import pytorch_lightning as pl

from utils import *
#import matplotlib.pyplot as plt

"""
the neural network class inherits from nn.Module
the default functions
init (constructor) and forward
have to be defined manually
"""

#class CustomNet(nn.Module):
class CustomNet(pl.LightningModule):
    """
    class constructor
    typically the neural networks variables and structure are initialized here
    super().__init__() is necessary to call the mother class constructor
    """

    #def __init__( self, hparams, dataset, cnn_struct, classifier1_struct, classifier2_struct, input_size, fc_input_size, output_size, device):
    def __init__(self, hparams):

        super(CustomNet, self).__init__()

        """ADD HYPER PARS"""

        self.hparams = hparams

        #self.name = "CustomNet"
        self.name = self.hparams["name"]

        self.loss_type = self.hparams["loss"]
        self.optimizer_type = self.hparams["optimizer"]

        self.optimizer_kwargs = self.hparams["optimizer_kwargs"]
        self.loss_kwargs = self.hparams["loss_kwargs"]

        self.train_sampler = []
        self.val_sampler = []
        self.test_sampler = []

        self.optimizer = None
        self.loss = None

        self.dataset = None
        #self.dataset = dataset
        #self.dataset = hparams["dataset"]


        self.net_struct = self.hparams["net_struct"]

        self.input_size = self.hparams["input_size"]
        self.output_size = self.hparams["output_size"]

        #self.device = None
        #self.device = device
        #self.device = hparams["device"]

        self.tb_logger = None

        """
        Construct neural network layers from subclasses
        """
        self.net = Net("CNN", self.net_struct, self.input_size, self.output_size)


    """
    the forward function is called each the the network is propagating forward
    takes in input data and spits out the predicted output
    """

    def forward(self, input_data):

        output = self.cnn(input_data)

        return output

    """
    initialize weights
    Using initialization routine from the torch.nn.init package
    """

    def init_weights(self, init_routine):
        #print(f"Initializing weights of {self.name} with method {init_routine}\n")
        for i, layer in enumerate(self.layers):
            if type(layer) == nn.Linear:
                #torch.nn.init.xavier_normal_(layer.weight)
                init_routine(layer.weight)
                #if self.net_struct[i]["bias"] == True:
                    #layer.bias.data.fill_(0.01)


    def set_layers(self, net_struct):
        self.net_struct = net_struct
        self.layer_sizes = calc_layer_sizes(self.input_size, self.net_struct) #inherit from utils
        self.layers = nn.ModuleList()

        #print(f"Initializing {self.name}:\n")
        for layer in self.net_struct:
            #print(f"Adding {layer}\n")
            self.layers.append(layer["type"](**layer["layer_pars"]))


    def prepare_dataset_splits(self, dataset, split_indices):

        self.dataset = dataset

        self.train_sampler = SubsetRandomSampler(split_indices["train"])
        self.val_sampler = SubsetRandomSampler(split_indices["val"])
        self.test_sampler = SubsetRandomSampler(split_indices["test"])


class Net(CustomNet):

    def __init__(self, name, net_struct, input_size, output_size):
        super(CustomNet, self).__init__()
        self.name = name
        #self.net_struct = net_struct
        self.input_size = input_size
        self.output_size = output_size

        self.set_layers(net_struct)

        """
        self.layer_sizes = calc_layer_sizes(self.input_size, self.net_struct) #inherit from utils
        self.layers = nn.ModuleList()

        print(f"Initializing {self.name}:\n")
        for layer in self.net_struct:
            print(f"Adding {layer}\n")
            self.layers.append(layer["type"](**layer["layer_pars"]))
        """
        self.init_weights(torch.nn.init.xavier_normal_)


    def forward(self, input_data):

        x = input_data
        print(x.shape)

        """
        iterate through all layers and perform calculation
        """

        for layer_i in range(len(self.layers)):
            #print(layer_i)
            print(x.shape)
            z = self.layers[layer_i](x)
            if "act_func" in self.net_struct[layer_i]:
                x = self.net_struct[layer_i]["act_func"](z)
            else:
                x = z

            #print(x)

        return x





"""Constructing NN and training procedure"""

"""Manually set network structure"""
"""
    This list can be loaded into the constructor of the Net neural network class, to automatically generate the network structure
    type = pointer to the layer function'
    layer_pars = parameters which must be given to the layer function in order to initialize it
    act_func = activation function to be applied directly after feeding to the corresponding layer
    dropout = certain neurons cna be dropped out if specified
"""

cnn_struct = []
input_size = dataset.output_size
#print(f"input size {input_size}")
#target_size = len(target_attributes)
#print(f"target size {target_size}")
#output_size = target_size
output_size = len(target_attributes)

i0 = input_size[0]
#[ [[in_channels, out_channels], [kernel_size], [stride], padding], ... ]
kernel_pars = [
    [[i0,4],[3,9],[1,1],0],
    [[4,4],[1,4],[1,4],0],
    [[4,8],[3,13],[1,1],0],
    [[8,8],[1,4],[1,4],0],
    [[8,16],[3,4],[1,1],0],
    [[16,16],[2,2],[2,2],0],
    [[16,16],[2,3],[1,1],0]
]



#fixed_net_struct.append( {"type": layer_type, "layer_pars": {"in_channels": kernel_par[0][0], "out_channels": kernel_par[0][1], "kernel_size": kernel_par[1], "stride": kernel_par[2], "padding": kernel_par[3], "bias": True}} )
for i, kernel_par in enumerate(kernel_pars):
    if i%2 == 0:
        layer_type = nn.Conv2d
        cnn_struct.append( {"type": layer_type, "layer_pars": {"in_channels": kernel_par[0][0], "out_channels": kernel_par[0][1], "kernel_size": kernel_par[1], "stride": kernel_par[2], "padding": kernel_par[3], "bias": True}} )
        #fixed_net_struct.append( {"type": nn.PReLU, "layer_pars": {}} )
        cnn_struct.append( {"type": nn.ReLU, "layer_pars": {}} )
    else:
        #layer_type = nn.MaxPool2d
        layer_type = nn.AvgPool2d
        cnn_struct.append( {"type": layer_type, "layer_pars": {"kernel_size": kernel_par[1], "stride": kernel_par[2], "padding": kernel_par[3]}} )


conv_sizes = utils.calc_layer_sizes(input_size, cnn_struct)
print(conv_sizes)

# set NN structure and hyperparameters
hyper_parameters = {}
hyper_parameters["name"] = "CustomNet"
hyper_parameters["cnn_struct"] = cnn_struct
# ...etc...


# create model and train!
model = custom_torch_net_class_lightning.CustomNet(hyper_parameters)
hparams = model.hparams

model.prepare_dataset_splits(dataset, split_indices)

#training procedure
trainer.fit(model)

NN END
