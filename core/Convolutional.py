import numpy as np
from utils.convolutional_helpers import convolution, loss_wrt_kernels, loss_wrt_data

class Convolutional:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        #initialize kernels and biases using He
        self.kernels = np.random.randn(self.out_channels,self.in_channels, kernel_size, kernel_size)*np.sqrt(2/(self.in_channels*kernel_size*kernel_size))
        self.biases = np.zeros((self.out_channels,1,1))
    def forward(self, x):
        if x.ndim == 3:
            x= x[:,None,:,:]
        self.forward_input = x
        self.input_padded = np.pad(x, ((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)))
        self.input_shape = self.input_padded.shape
        x_convoluted = convolution(self.input_padded, self.kernels, self.stride)
        x_biased = x_convoluted + self.biases
        return x_biased
    def backward(self, grad_data):
        #obecnie poniższa zmienna jest 1D array, trzeba ją jakoś przeformatować
        self.loss_wrt_biases = np.sum(grad_data, axis=(0,2,3)).reshape((self.out_channels,1,1))
        self.loss_wrt_kernels = loss_wrt_kernels(grad_data, self.input_padded, self.stride, self.kernel_size)
        loss_wrt_padded_input = loss_wrt_data(grad_data, self.kernels, self.stride, self.input_shape)
        p = self.padding
        self.loss_wrt_input = loss_wrt_padded_input[:,:,p:-p,p:-p]
        return self.loss_wrt_input
    def update_parameters(self, lr):
        self.biases -= lr * self.loss_wrt_biases
        self.kernels -= lr * self.loss_wrt_kernels