import numpy as np
from utils.pooling_helpers import forward,backward

class MaxPool:
    def __init__(self, in_channels, kernel_size, stride):
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
    def forward(self, x):
        self.input_image_size = x.shape[3]
        output, self.mask =forward(x, self.kernel_size, self.stride)
        return output
    def backward(self, grad_data):
        return backward(grad_data, self.mask, self.stride, self.input_image_size)
    def update_parameters(self, lr):
        pass