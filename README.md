# Neural network
Artificial neural network for PHP. Features backpropagtion learning using gradient descent, momentum and the sigmoid activation function.

# About
The library allows you to build and train multi-layer neural networks. You first define the structure for the network. The number of input, output, layers and hidden nodes. The network is then constructed. Interconnection strengths are represented using an adjacency matrix and initialised to small random values.  Traning data is then presented to the network incrementally. The neural network uses an online backpropagation training algorithm that uses gradient descent to descend the error curve to adjust interconnection strengths. The aim of the training algorithm is to adjust the interconnection strengths in order to reduce the global error. The global error being the difference between the target output and actual output. 

You can provide a learning rate and momentum parameter.  The learning rate will affect the speed at which the neural network converges to an optimal solution. The momentum parameter will help gradient descent to avoid converging to a non optimal solution on the error curve called local minima.  The correct size for the momentum parameter will help to find the global minima but too large a value will prevent the neural network from ever converging to a solution.

## Installation
```bash
$ git clone https://github.com/stephenlmoshea/neuralnetwork.git
$ cd neuralnetwork
$ composer install
```
## Example

```php
//Create network with 2 input nodes, 2 hidden nodes, and 1 output node
//and set activation function
$network = new FeedForward([2, 2, 1], new Sigmoid());

//Define learning rate and momentum parameters for backpropagation algorithm
$ann = new Backpropagation($network, 0.7, 0.3);


//Provide XOR training data
$trainingSet = [
                    [0,0,0],
                    [0,1,1],
                    [1,0,1],
                    [1,1,0]
                ];

//Keep training the neural network until it converges
do {
    $ann->initialise();
    $result = $ann->train($trainingSet);
} while (!$result);

//Present [0,0] as network inputs and get the network output
$network->activate([0, 0]);
$outputs = $network->getOutputs();

//Present [0,1] as network inputs and get the network output
$network->activate([0, 1]);
$outputs = $network->getOutputs();
$this->assertTrue($outputs[0] > 0.9);

//Present [1,0] as network inputs and get the network output 
$network->activate([1, 0]);
$outputs = $network->getOutputs();
$this->assertTrue($outputs[0] > 0.9);

//Present [1,1] as network inputs and get the network output
$network->activate([1, 1]);
$outputs = $network->getOutputs();
$this->assertTrue($outputs[0] < 0.1);
```
