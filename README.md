# neural-network
A lightweight neural network library written in ANSI-C supporting prediction and backpropagation for Convolutional- (CNN) and Fully Connected  (FCN) neural networks.

# Features

## Different layer types:
- `Input layer`: 1-dimensional layer used e.g. as input for a FCN.
- `Dense layer`: 1-dimensional fully connected hidden layer of a FCN, having weights and an activation function.
- `Dropout layer`: 1-dimensional hidden layer of a FCN to reduce overfitting by randomly deactivating some nodes.
- `Output layer`: 1-dimensional layer which represents the outputs of a FCN, having weights and an activation function.
- `Input2D layer`: 2-dimensional layer used e.g. as input for a CCN.
- `Conv2D layer`: 2-dimensional layer of a CNN, having for each produced output channel (feature map) one 3D-kernel (3D-filter).
  For now only filter rows, filter columns, filter numbers and filter stride is supported. The activation function type is limitted to only ReLU and Leaky ReLU.
- `Max Pooling2D layer`: 2-dimensional layer of a CNN, downsampling the input by taking the maximum value over an input window.
  For now only the most common window size of 2x2 with a stride of 2 is supported.
- `Avr Pooling2D layer`: 2-dimensional layer of a CNN, downsampling the input by taking the average value over an input window.
  For now only the most common window size of 2x2 with a stride of 2 is supported.
- `Flatten2D layer`: 2-dimensional layer of a CNN. This layer is used to make the transition from a CNN (2-dimensional) to a FCN (1-dimensional).

## Different activation functions:
- `Sigmoid`: s-shaped curve, only for Dense- and Output layers.
- `ReLU`: Rectified Linear Unit curve, for Dense-, Output- and Conv2d layers.
- `Leaky ReLU`: Leaky Rectified Linear Unit curve, for Dense-, Output- and Conv2d layers.
- `Tanh`: hyperbolic tangent curve, only for Dense- and Output layers.
- `Softsign`: softsign curve, only for Dense- and Output layers.
- `Softmax`: softmax function, only for the Output layer.
