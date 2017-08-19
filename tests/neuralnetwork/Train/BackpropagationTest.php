<?php
namespace neuralnetwork\Test;

use neuralnetwork\Network\FeedForward;
use neuralnetwork\Train\Backpropagation;
use neuralnetwork\Activation\Sigmoid;

class BackpropagationTest extends \PHPUnit_Framework_TestCase
{
    /**
     * Test it learns the OR function
     */
    public function testItLearnsORFunction()
    {
        $network = new FeedForward([2, 2, 1], new Sigmoid());
        $ann = new Backpropagation($network, 0.7, 0.3, 0.0001);
        $trainingSet = [
                            [0,0,0],
                            [0,1,1],
                            [1,0,1],
                            [1,1,1]
                        ];
        do {
            $ann->initialise();
            $result = $ann->train($trainingSet);
        } while (!$result);
        
        $network->activate([0, 0]);
        $outputs = $network->getOutputs();
        $this->assertTrue($outputs[0] < 0.1);
        
        $network->activate([0, 1]);
        $outputs = $network->getOutputs();
        $this->assertTrue($outputs[0] > 0.9);
         
        $network->activate([1, 0]);
        $outputs = $network->getOutputs();
        $this->assertTrue($outputs[0] > 0.9);
        
        $network->activate([1, 1]);
        $outputs = $network->getOutputs();
        $this->assertTrue($outputs[0] > 0.9);
    }
    
    /**
     * Test it learns the AND function
     */
    public function testItLearnsANDFunction()
    {
        $network = new FeedForward([2, 2, 1], new Sigmoid());
        $ann = new Backpropagation($network, 0.9, 0.3);
        $trainingSet = [
                            [0,0,1],
                            [0,1,0],
                            [1,0,0],
                            [1,1,1]
                        ];
        do {
            $ann->initialise();
            $result = $ann->train($trainingSet);
        } while (!$result);
        
        $network->activate([0, 0]);
        $outputs = $network->getOutputs();
        $this->assertTrue($outputs[0] > 0.9);
        
        $network->activate([0, 1]);
        $outputs = $network->getOutputs();
        $this->assertTrue($outputs[0] < 0.1);
         
        $network->activate([1, 0]);
        $outputs = $network->getOutputs();
        $this->assertTrue($outputs[0] < 0.1);
        
        $network->activate([1, 1]);
        $outputs = $network->getOutputs();
        $this->assertTrue($outputs[0] > 0.9);
    }
    
    /**
     * Test it learns the XOR function with two hidden neurons
     */
    public function testItLearnsXORFunctionWithTwoHiddenUnits()
    {
        $network = new FeedForward([2, 2, 1], new Sigmoid());
        $ann = new Backpropagation($network, 0.7, 0.3);
        $trainingSet = [
                            [0,0,0],
                            [0,1,1],
                            [1,0,1],
                            [1,1,0]
                        ];

        do {
            $ann->initialise();
            $result = $ann->train($trainingSet);
        } while (!$result);
        
        $network->activate([0, 0]);
        $outputs = $network->getOutputs();
        $this->assertTrue($outputs[0] < 0.1);
        
        $network->activate([0, 1]);
        $outputs = $network->getOutputs();
        $this->assertTrue($outputs[0] > 0.9);
         
        $network->activate([1, 0]);
        $outputs = $network->getOutputs();
        $this->assertTrue($outputs[0] > 0.9);
        
        $network->activate([1, 1]);
        $outputs = $network->getOutputs();
        $this->assertTrue($outputs[0] < 0.1);
    }
    
    /**
     * Test it learns the XOR function with three hidden neurons
     */
    public function testItLearnsXORFunctionWithThreeHiddenUnits()
    {
        $network = new FeedForward([2, 3, 1], new Sigmoid());
        $ann = new Backpropagation($network, 0.7, 0.3);
        $trainingSet = [
                            [0,0,0],
                            [0,1,1],
                            [1,0,1],
                            [1,1,0]
                        ];

        do {
            $ann->initialise();
            $result = $ann->train($trainingSet);
        } while (!$result);
        
        $network->activate([0, 0]);
        $output = $network->getOutputs();
        $this->assertTrue($output[0] < 0.1);
        
        $network->activate([0, 1]);
        $output = $network->getOutputs();
        $this->assertTrue($output[0] > 0.9);
         
        $network->activate([1, 0]);
        $output = $network->getOutputs();
        $this->assertTrue($output[0] > 0.9);
        
        $network->activate([1, 1]);
        $output = $network->getOutputs();
        $this->assertTrue($output[0] < 0.1);
    }
    
    /**
     * Test it learns the XOR function with four hidden neurons
     */
    public function testItLearnsXORFunctionWithFourHiddenUnits()
    {
        $network = new FeedForward([2, 4, 1], new Sigmoid());
        $ann = new Backpropagation($network, 0.7, 0.3);
        $trainingSet = [
                            [0,0,0],
                            [0,1,1],
                            [1,0,1],
                            [1,1,0]
                        ];

        do {
            $ann->initialise();
            $result = $ann->train($trainingSet);
        } while (!$result);
        
        $network->activate([0, 0]);
        $output = $network->getOutputs();
        $this->assertTrue($output[0] < 0.1);
        
        $network->activate([0, 1]);
        $output = $network->getOutputs();
        $this->assertTrue($output[0] > 0.9);
         
        $network->activate([1, 0]);
        $output = $network->getOutputs();
        $this->assertTrue($output[0] > 0.9);
        
        $network->activate([1, 1]);
        $output = $network->getOutputs();
        $this->assertTrue($output[0] < 0.1);
    }
    
    /**
     * Test it learns the XOR function with three input nodes
     */
    public function testItLearnsXORFunctionWithThreeInputNodes()
    {
        $network = new FeedForward([3, 4, 1], new Sigmoid());
        $ann = new Backpropagation($network, 0.7, 0.3, 0.005, 5000);
        $trainingSet = [
                            [0,0,0,0],
                            [0,0,1,1],
                            [0,1,0,1],
                            [0,1,1,0],
                            [1,0,0,1],
                            [1,0,1,0],
                            [1,1,0,0],
                            [1,1,1,1]
                        ];

        do {
            $ann->initialise();
            $result = $ann->train($trainingSet);
        } while (!$result);
        
        $network->activate([0, 0, 0]);
        $output = $network->getOutputs();
        $this->assertTrue($output[0] < 0.1);
        
        $network->activate([0, 1, 0]);
        $output = $network->getOutputs();
        $this->assertTrue($output[0] > 0.9);
         
        $network->activate([1, 1, 1]);
        $output = $network->getOutputs();
        $this->assertTrue($output[0] > 0.9);
        
        $network->activate([1, 1, 0]);
        $output = $network->getOutputs();
        $this->assertTrue($output[0] < 0.1);
    }
}