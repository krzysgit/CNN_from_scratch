import numpy as np
from training.train import to_one_hot

def softmax(x, axis = -1):
    x_max = np.max(x , axis=axis, keepdims=True)
    e_x = np.exp(x-x_max)
    sum_e_x = np.sum(e_x, axis=axis , keepdims=True)
    return e_x / sum_e_x

class Softmax:
    def __init__(self, temperature=1):
        self.temperature = temperature
    def forward(self, x):
        self.probabilities = softmax(x/self.temperature)
        return self.probabilities
    def backward(self, y_true):
            batch_size = y_true.size
            return (self.probabilities-to_one_hot(y_true, self.probabilities.shape[1]))/(batch_size*self.temperature)
    def update_parameters(self, lr):
         pass

