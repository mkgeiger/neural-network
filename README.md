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

## Different loss functions:
- `MSE`: mean squared error
- `Categorical cross entropy`: softmax loss (used always in combination with the Softmax activation in Output layers)

## Different random function types (used to initialize weights, biases and filter values when training begins):
- `Uniform` (uniform distributed): general purpose, lightweight models
- `Normal` (gaussian distributed): general purpose
- `Glorot uniform`: general purpose, sigmoid, tanh, ReLU activations, classification
- `Glorot normal`: sensitive training, sigmoid, tanh or softmax activations, regression
- `HE uniform`: wide Layers, ReLU and variants (Leaky ReLU), classification
- `HE normal`: very deep networks, ReLU and variants (Leaky ReLU), regression

## Different optimizer function types (used to adjust weights, biases and filter values during the training):
- `SGD`: stochastic gradient descent
- `SGD with decay`: stochastic gradient descent with decay
- `Adapt`: stochastic gradient descent with adapt
- `Momentum`: stochastic gradient descent with momentum
- `RMSProp`: root mean square propagation
- `Adagrad`: adaptive gradient algorithm
- `Adam`: adaptive moment estimation
  
