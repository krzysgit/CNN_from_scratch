class Flatten:
    def forward(self,x):
        self.original_shape = x.shape
        return x.reshape(self.original_shape[0],-1)
    #backward inputs a matrix and outputs a tensor of dimensions num_batch*num_channels*height*width
    def backward(self, grad_data):
        return grad_data.reshape(self.original_shape)        
    def update_parameters(self, lr):
        pass