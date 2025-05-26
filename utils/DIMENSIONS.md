# The dimensions of paticular inputs and outputs

## ReLU

### Forward

num_features -> num features

### Backward

num_features -> num features

## Fully connected layer

### Forward

W: (in_features, out_features)
b: (1, out_features)
(batch_size, in_features) -> (batch_size, out_features)

### Backward

grad_data: (batch_size, out_features)
loss_wrt_logits: (batch_size, in_features)

## Flatten

### Forward

(batch_size, num_channels, height, width) -> (batch_size, num_channels\*height\*width)

### Backward

(batch_size, num_channels\*height\*width) -> (batch_size, num_channels, height, width)

## Convolutional

bias: (out_features,1,1)

### Padding

(batch_size, num_channels, height, width) -> (batch_size, num_channels, height + 2\*padding, width + 2\*padding)

### convolution

kernels: (out_channels, in_channels, kernel_size, kernel_size)

(batch_size, in_channels, height, width) -> (batch_size, out_channels, (height - kernel_size)/stride + 1, (width - kernel_size)/stride +1)

### Backpropagation

grad_data: (batch_size, out_channels, (height - kernel_size)/stride + 1, (width - kernel_size)/stride +1)
Loss_wrt_kernels: (out_channels, in_channels, kernel_size, kernel_size)
Loss_wrt_input: (batch_size, in_channels, height, width)

## Max pooling

### forward

Input: (batch_size, num_channels, height, width)

## Softmax

(batch_size, num_classes) -> (batch_size, num_classes)
