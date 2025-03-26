'''Generates a customized MLP model for hyperparameter search.

Description
===========

Customized MLP model for hyperparameter search. This file is part of the paper:

    *Scoop: An Optimization Algorithm for Profiling Attacks against Higher-Order Masking.*
'''



import torch.nn as nn

class MLPModel(nn.Module):
    '''.. class:: MLPModel(n_classes, signal_length, n_linear=2, linear_size=1000, activation='ReLU', input_bn=False, dense_bn=False)

   A Multi-Layer Perceptron (MLP) model for classification tasks.

   :param n_classes: Number of output classes (must be > 0).
   :param signal_length: Length of the input signal (must be > 0).
   :param n_linear: Number of linear layers (must be > 0; default: 2).
   :param linear_size: Size of each linear layer (must be > 0; default: 1000).
   :param activation: Activation function to use; one of 'ReLU', 'SeLU', 'ELU', or 'Tanh' (default: 'ReLU').
   :param input_bn: If True, applies batch normalization on the input (default: False).
   :param dense_bn: If True, applies batch normalization after each linear layer (default: False).

   :raises ValueError: If any of the parameter constraints are violated.

   .. method:: forward(x)

      Perform the forward pass of the MLP model.

      :param x: Input tensor.
      :returns: Log-probabilities for each class.
'''
    def __init__(self, n_classes, signal_length, n_linear=2, linear_size=1000, activation='ReLU', input_bn=False, dense_bn=False):
        super(MLPModel, self).__init__()
        self.linear_blocks = nn.ModuleList()
        if n_classes < 0:
            raise ValueError('Number of classes must be greater than 0')
        if signal_length < 0:
            raise ValueError('Signal length must be greater than 0')
        if n_linear < 0:
            raise ValueError('Number of linear layers must be greater than 0')
        if linear_size < 1:
            raise ValueError('Linear size must be greater than 0')
        if activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'SeLU':
            self.activation = nn.SELU()
        elif activation == 'ELU':
            self.activation = nn.ELU()
        elif activation == 'Tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError('Invalid activation function, must be one of ReLU, SeLU, ELU, Tanh')
        if input_bn not in [True, False]:
            raise ValueError('Input batch norm must be a boolean')
        if dense_bn not in [True, False]:
            raise ValueError('Dense batch norm must be a boolean')
        
        self.signal_length = signal_length
        if input_bn:
            self.linear_blocks.append(nn.BatchNorm1d(1))
        self.linear_blocks.append(nn.Flatten())
        self.linear_blocks.append(nn.Linear(signal_length, linear_size))
        if dense_bn:
            self.linear_blocks.append(nn.BatchNorm1d(linear_size))
        self.linear_blocks.append(self.activation)
        for _ in range(n_linear-1):
            self.linear_blocks.append(nn.Linear(linear_size, linear_size))
            if dense_bn:
                self.linear_blocks.append(nn.BatchNorm1d(linear_size))
            self.linear_blocks.append(self.activation)

        self.classifier = nn.Sequential(
            nn.Linear(linear_size, n_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        for layer in self.linear_blocks:
            x = layer(x)
        x = self.classifier(x)
        return x