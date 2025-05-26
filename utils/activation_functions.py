import numpy as np

class ActivationFunction:
    def __init__(self , name, forward, backward):
        self.name = name
        self.forward = forward
        self.backward = backward
    
    def forward(self , x):
        return self.forward(x)
    def backward(self, x):
        return self.backward(x)

# Define the relu instance of the class

def relu_forward(x):
    return np.maximum(0,x)
def relu_backward(x):
    return (x>0).astype(float)

relu = ActivationFunction("relu",relu_forward,relu_backward)
