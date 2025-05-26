import numpy as np
#those impors are currently unnecessary
'''from core.Convolutional import Convolutional
from core.Flatten import Flatten
from core.Softmax import Softmax
from core.ReLU import ActivationLayer
from core.MaxPool import MaxPool
from core.FullyConnected import FullyConnected'''

class Model:
    def __init__(self, layers):
        self.layers = layers
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    def backward(self, labels):
        for layer in reversed(self.layers):
            #here we insert the label to the softmax + Cross entropy loss head. In the next steps variable label will hold actually grad_data
            labels = layer.backward(labels)
    def update(self, lr):
        for layer in self.layers:
            layer.update_parameters(lr)