import numpy as np
from utils.activation_functions import relu

class ActivationLayer:
    def __init__(self, activation_function = relu):
        self.avf = activation_function

    def forward(self,x):
        self.grad_data = self.avf.backward(x)
        return self.avf.forward(x)

    def backward(self, grad_output):
        return self.grad_data * grad_output

    def update_parameters(self, lr):
        pass





