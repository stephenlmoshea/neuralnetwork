# Neural network
Artificial neural network for PHP. Features backpropagtion learning using gradient descent, momentum and the sigmoid activation function.

# About
The library allows you to build and multi-layer neural networks. You first define the structure for the network. The network is then built. Interconnection strengths are represented using an adjacency matrix and initialised to small random values.  Traning data is then presented to the network incrementally. The neural network uses an online backpropagation training algorithm that uses gradient descent to descend the error curve to adjust interconnection strengths.  

You can provide learning rate to affect the speed at which the neural network converges to an optimal solution. You can provide a momentum parameter to avoid the error curve from coverging to a non optimal solution called local minima.  The correct size for the momentum parameter will help to find the global minima but to large a value will prevent the neural network from ever converging to a solution.
