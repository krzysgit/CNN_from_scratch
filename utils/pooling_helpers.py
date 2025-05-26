import numpy as np
from numpy.lib.stride_tricks import as_strided

class MyCustomError(Exception):
    pass

def forward(x, kernel_size, stride):
    batch_size, num_channels, image_size, _ = x.shape
    #compute the size of the strided image
    strided_size = (image_size-kernel_size)/stride +1
    if strided_size%1 == 0:
        strided_size = int(strided_size)
    else:
        raise MyCustomError("incorrect layer parameters")

    window_shape = (batch_size, num_channels, strided_size, strided_size, kernel_size, kernel_size)
    batch_stride, channel_stride, change_row_stride, change_column_stride = x.strides
    window_strides = (batch_stride, channel_stride,
                      stride*change_row_stride,
                      stride*change_column_stride,
                      change_row_stride,
                      change_column_stride
                      )
    window_view = as_strided(x, shape= window_shape, strides= window_strides)
    output = np.max(window_view, axis = (-1,-2))
    #we create a mask pointing to the maximal elements in each window
    mask = (window_view == output[...,None,None])
    return output, mask

def backward(grad_data, mask, stride, image_size):
    grad_data_expanded = grad_data[...,None, None]
    grad_windows = np.where(mask, grad_data_expanded, 0.0)
    # can be alternatively writen as: grad_windows = grad_data_expanded*mask
    strided_size = grad_data.shape[3]
    kernel_size = mask.shape[5]
    batch_size, num_channels = grad_data.shape[:2]
    output = np.zeros((batch_size, num_channels,image_size,image_size))

    window_shape = (batch_size, num_channels, strided_size, strided_size, kernel_size, kernel_size)
    batch_stride, channel_stride, change_row_stride, change_column_stride = output.strides
    window_strides = (batch_stride, channel_stride,
                      stride*change_row_stride,
                      stride*change_column_stride,
                      change_row_stride,
                      change_column_stride
                      )
    window_view = as_strided(output, shape= window_shape, strides= window_strides)
    np.add.at(window_view, tuple(), grad_windows)
    return output
