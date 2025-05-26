import numpy as np

class FullyConnected:
    def __init__(self, in_features, out_features):
        #The weights are initialized using He initialization
        self.W = np.random.randn(in_features, out_features)*np.sqrt(2/in_features)
        self.b = np.zeros((1,out_features))
        
    def forward(self, x):
        self.input = x
        self.logits = x @ self.W + self.b
        return self.logits

    def backward(self, grad_data):
        self.loss_wrt_logits = grad_data @ self.W.T
        self.loss_wrt_weights = self.input.T @ grad_data
        self.loss_wrt_bias = grad_data.sum(axis=0)
        return self.loss_wrt_logits
    def update_parameters(self, lr):
        self.W -= lr * self.loss_wrt_weights
        self.b -= lr * self.loss_wrt_bias