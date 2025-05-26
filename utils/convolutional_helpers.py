import numpy as np
from numpy.lib.stride_tricks import as_strided

def signle_position_conv(window, kernels):
    return np.einsum('bihw,oihw->bo',window, kernels)

def convolution(x, kernels, stride):
    original_shape = x.shape
    out_channels = kernels.shape[0]
    batch_size = original_shape[0]
    kernel_size = kernels.shape[2]
    #following height is assumed with padding alredy done. additionally we assume height = width
    height = original_shape[2]
    #compute the height/width of the output matrix
    output_size = (height - kernel_size)/stride +1
    if output_size %1 == 0:
        output_size = int(output_size)
    #initialize output matrix with zeros
    output = np.zeros((batch_size, out_channels, output_size, output_size))
    #loop across all possible positions of the kernels updating output
    for i,j in np.ndindex(output_size,output_size):
        #below we track the left upper cornen of the kernel and extract slices based on its position
        current_window = x[:,:,
                           i*stride:i*stride+kernel_size,
                           j*stride:j*stride+kernel_size,
                           ]
        #the widow has dimensions (batch_size, num_channels, kernel_size, kernel_size)
        #we pass the current window to the single postition conv and update the output
        output[:,:,i,j] = signle_position_conv(current_window, kernels)
    return output

def loss_wrt_kernels(grad_data, forward_input, stride, kernel_size):
    in_channels = forward_input.shape[1]
    out_channels = grad_data.shape[1]
    output_picture_size = grad_data.shape[2]
    output = np.zeros((out_channels,in_channels,kernel_size,kernel_size))
    # we iterate over the locations of the kernel. for example if it is 5x5 we iterate over 5x5 array
    for i,j in np.ndindex(kernel_size,kernel_size):
        stride_window = forward_input[:,:,
                                      i:i+stride*output_picture_size:stride,
                                      j:j+stride*output_picture_size:stride]
        output[:,:,i,j] = np.einsum('biwh,bowh->oi', stride_window, grad_data)
    return output
def loss_wrt_data(grad_data, kernels, stride, input_shape):
    #I learned to use the window sliding package, so it is applied in this function instead of a for loop
    batches, C_out, size_out, _ = grad_data.shape
    C_out_k, C_in, kernel_size, _ = kernels.shape
    size_input = input_shape[2]
    #sanity check
    assert C_out == C_out_k
    # we apparently have to flip the kernels to vectorise the operation, will see why next
    flipped_kernels = kernels[:,:,::-1,::-1]
    # we now have to calculate and initialize the sparse matrix with grad_data inside. next it will be convoluted with grad data to get what is desired
    grad_up_size = 2*kernel_size + (size_out-1)*stride-1
    grad_up = np.zeros((batches, C_out,grad_up_size,grad_up_size),dtype=grad_data.dtype)
    grad_up[:,:,
            kernel_size-1:kernel_size+stride*(size_out-1):stride,
            kernel_size-1:kernel_size+stride*(size_out-1):stride,
            ] = grad_data
    #create the view of each kernel*kernel window
    window_viev_shape = (batches, C_out, size_input, size_input, kernel_size, kernel_size)
    batch_stride, channel_stride, change_row_stride, change_column_stride= grad_up.strides
    window_viev_strides = (batch_stride, channel_stride,
                           change_row_stride,change_column_stride,change_row_stride,change_column_stride
                           )
    grad_patches = as_strided(grad_up,shape=window_viev_shape,strides=window_viev_strides)
    
    loss_wrt_inputs = np.einsum('bohwxy,oixy->bihw',grad_patches,flipped_kernels)
    #finally
    return loss_wrt_inputs

        
        