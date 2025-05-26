A lightweight, educational implementation of a Convolutional Neural Network (CNN) built entirely from scratch using NumPy. No deep learning frameworks used in the model architecture. Includes support for training, evaluating, and loading pretrained models.

## Features

-   Forward and backward pass implemented manually
-   Convolution, ReLU, MaxPooling, and Fully Connected layers
-   Softmax + Cross-Entropy loss
-   SGD optimizer
-   Training loop with mini-batching
-   Support for saving and loading models
-   Dataloader and defaut dataset implemented from torch and torchvision.

## Installation

```bash
git clone https://github.com/krzysgit/CNN_from_scratch.git
cd numpy-cnn
pip install -r requirements.txt
```

## Pretrained Model

Download from: `saved_models/`

Trained on MNIST for 2 epochs, ~94.3% accuracy.
