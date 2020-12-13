<?php
namespace neuralnetwork\Test;

use neuralnetwork\Network\FeedForward;
use neuralnetwork\Train\Backpropagation;
use neuralnetwork\Activation\Sigmoid;
use neuralnetwork\Activation\HyperbolicTangent;

class BackpropagationTest extends \PHPUnit_Framework_TestCase
{

    public function testCalculateNodeDeltasWithTwoOutputs()
    {
        $nodePerLayer = [2, 2, 2];
        $activation = new Sigmoid();
        $network = new FeedForward($nodePerLayer, $activation);
        $ann = new Backpropagation($network, 0.7, 0.3,0.005,1);
        $ann->initialise();
        $this->initialiseNetworkWithTwoOutputs($network);

        $trainingSet = [0,0,0,0];

        $network->activate($trainingSet);
        $ann->calculateNodeDeltas($trainingSet);

        $expectedNodeDeltas = [0,0,0.00062326980735828, -0.00030381413438377, -0.12624651653667, -0.12468201071659];

        $this->assertEquals($ann->getNodeDeltas(), $expectedNodeDeltas);
    }

    public function testCalculateGradientsWithTwoOutputs()
    {
        $nodePerLayer = [2, 2, 2];
        $activation = new Sigmoid();
        $network = new FeedForward($nodePerLayer, $activation);
        $ann = new Backpropagation($network, 0.7, 0.3,0.005,1);
        $ann->initialise();
        $this->initialiseNetworkWithTwoOutputs($network);

        $trainingSet = [0,0,0,0];

        $network->activate($trainingSet);
        $ann->calculateNodeDeltas($trainingSet);
        $ann->calculateGradients();

        $expectedGradients = [];
        $expectedGradients[0][2] = 0;
        $expectedGradients[0][3] = -0;
        $expectedGradients[1][2] = 0;
        $expectedGradients[1][3] = -0;
        $expectedGradients[2][4] = -0.062176480401586;
        $expectedGradients[2][5] = -0.061405960405238;
        $expectedGradients[3][4] = -0.062176480401586;
        $expectedGradients[3][5] = -0.061405960405238;

        $this->assertEquals($ann->getGradients(), $expectedGradients);

        $expectedBiasGradients = [];
        $expectedBiasGradients[0][2] = 0.00062326980735828;
        $expectedBiasGradients[0][3] = -0.00030381413438377;
        $expectedBiasGradients[1][4] = -0.12624651653667;
        $expectedBiasGradients[1][5] = -0.12468201071659;

        $this->assertEquals($ann->getBiasGradients(), $expectedBiasGradients);
    }

    public function testCalculateWeightUpdatesWithTwoOutputs()
    {
        $nodePerLayer = [2, 2, 2];
        $activation = new Sigmoid();
        $network = new FeedForward($nodePerLayer, $activation);
        $ann = new Backpropagation($network, 0.7, 0.3,0.005,1);
        $ann->initialise();
        $this->initialiseNetworkWithTwoOutputs($network);

        $trainingSet = [0,0,0,0];

        $network->activate($trainingSet);
        $ann->calculateNodeDeltas($trainingSet);
        $ann->calculateGradients();
        $ann->calculateWeightUpdates();

        $expectedWeightUpdates = [];
        $expectedWeightUpdates[0][2] = 0;
        $expectedWeightUpdates[0][3] = 0;
        $expectedWeightUpdates[1][2] = 0;
        $expectedWeightUpdates[1][3] = 0;
        $expectedWeightUpdates[2][4] = -0.04352353628111;
        $expectedWeightUpdates[2][5] = -0.042984172283667;
        $expectedWeightUpdates[3][4] = -0.04352353628111;
        $expectedWeightUpdates[3][5] = -0.042984172283667;

        $this->assertEquals($ann->getWeightUpdates(), $expectedWeightUpdates);

        $expectedBiasWeightUpdates = [];
        $expectedBiasWeightUpdates[0][2] = 0.00043628886515079;
        $expectedBiasWeightUpdates[0][3] = -0.00021266989406864;
        $expectedBiasWeightUpdates[1][4] = -0.088372561575671;
        $expectedBiasWeightUpdates[1][5] = -0.08727740750161;

        $this->assertEquals($ann->getBiasWeightUpdates(), $expectedBiasWeightUpdates);
    }

    public function testApplyWeightChangesWithTwoOutputs()
    {
        $nodePerLayer = [2, 2, 2];
        $activation = new Sigmoid();
        $network = new FeedForward($nodePerLayer, $activation);
        $ann = new Backpropagation($network, 0.7, 0.3,0.005,1);
        $ann->initialise();
        $this->initialiseNetworkWithTwoOutputs($network);

        $trainingSet = [0,0,0,0];

        $network->activate($trainingSet);
        $ann->calculateNodeDeltas($trainingSet);
        $ann->calculateGradients();
        $ann->calculateWeightUpdates();

        $expectedWeights = [];
        $expectedWeights[0][2] = 0.01;
        $expectedWeights[0][3] = -0.01;
        $expectedWeights[1][2] = 0.04;
        $expectedWeights[1][3] = 0.04;
        $expectedWeights[2][4] = 0;
        $expectedWeights[2][5] = -0.02;
        $expectedWeights[3][4] = -0.02;
        $expectedWeights[3][5] = 0.03;

        $this->assertEquals($network->getWeights(), $expectedWeights);

    }

    protected function initialiseNetworkWithTwoOutputs($network)
    {
        $network->initialise();
        
        $weights = $this->getWeightsForTwoOutputs();
        $biasWeights = $this->getBiasWeightsForTwoOutputs();

        $network->setWeights($weights);
        $network->setBiasWeights($biasWeights);
        return [
            'weights' => $weights,
            'biasWeights' => $biasWeights
        ];
    }

    protected function getWeightsForTwoOutputs()
    {
        $weights[0][2] = 0.01;
        $weights[0][3] = -0.01;
        $weights[1][2] = 0.04;
        $weights[1][3] = 0.04;
        $weights[2][4] = 0;
        $weights[2][5] = -0.02;
        $weights[3][4] = -0.02;
        $weights[3][5] = 0.03;
        
        return $weights; 
    }

    protected function getBiasWeightsForTwoOutputs()
    {
        $biasWeights[0][2] = -0.03;
        $biasWeights[0][3] = -0.03;
        $biasWeights[1][4] = 0.03;
        $biasWeights[1][5] = -0.01;
        
        return $biasWeights;
    }

    /**
     * Test it learns the OR function
     */
    public function testItLearnsORFunction()
    {
        $network = new FeedForward([2, 2, 1], new Sigmoid());
        $ann = new Backpropagation($network, 0.7, 0.3, 0.001);
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
     * Test it learns the OR function with two outputs
     */
    public function testItLearnsORFunctionWithTwoOutputs()
    {
        $network = new FeedForward([2, 2, 2], new Sigmoid());
        $ann = new Backpropagation($network, 0.7, 0.3, 0.001);
        $trainingSet = [
                            [0,0,0,0],
                            [0,1,0,1],
                            [1,0,1,0],
                            [1,1,1,1]
                        ];
        do {
            $ann->initialise();
            $result = $ann->train($trainingSet);
        } while (!$result);
        
        $network->activate([0, 0]);
        $outputs = $network->getOutputs();
        $this->assertTrue($outputs[0] < 0.1);
        $this->assertTrue($outputs[1] < 0.1);
        
        $network->activate([0, 1]);
        $outputs = $network->getOutputs();
        $this->assertTrue($outputs[0] < 0.1);
        $this->assertTrue($outputs[1] > 0.9);
         
        $network->activate([1, 0]);
        $outputs = $network->getOutputs();
        $this->assertTrue($outputs[0] > 0.9);
        $this->assertTrue($outputs[1] < 0.1);
        
        $network->activate([1, 1]);
        $outputs = $network->getOutputs();
        $this->assertTrue($outputs[0] > 0.9);
        $this->assertTrue($outputs[1] > 0.9);
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
     * Test it learns the XOR function with two hidden neurons
     */
    public function testItLearnsXORFunctionWithTwoOutputs()
    {
        $network = new FeedForward([2, 2, 2], new Sigmoid());
        $ann = new Backpropagation($network, 0.7, 0.3);
        $trainingSet = [
                            [0,0,0,0],
                            [0,1,0,1],
                            [1,0,1,0],
                            [1,1,0,0]
                        ];

        do {
            $ann->initialise();
            $result = $ann->train($trainingSet);
        } while (!$result);
        
        $network->activate([0, 0]);
        $outputs = $network->getOutputs();
        $this->assertTrue($outputs[0] < 0.1);
        $this->assertTrue($outputs[1] < 0.1);
        
        $network->activate([0, 1]);
        $outputs = $network->getOutputs();
        $this->assertTrue($outputs[0] < 0.1);
        $this->assertTrue($outputs[1] > 0.9);
         
        $network->activate([1, 0]);
        $outputs = $network->getOutputs();
        $this->assertTrue($outputs[0] > 0.9);
        $this->assertTrue($outputs[1] < 0.1);
        
        $network->activate([1, 1]);
        $outputs = $network->getOutputs();
        $this->assertTrue($outputs[0] < 0.1);
        $this->assertTrue($outputs[1] < 0.1);
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
        $ann = new Backpropagation($network, 0.7, 0.3, 0.0005, 5000);
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
    
    /**
     * Test it learns the OR function
     */
    public function testItLearnsORFunctionUsingHyperbolicTangent()
    {
        $network = new FeedForward([2, 2, 1], new HyperbolicTangent());
        $ann = new Backpropagation($network, 0.7, 0.3, 0.001);
        $trainingSet = [
                            [-1,-1,-1],
                            [-1,1,1],
                            [1,-1,1],
                            [1,1,1]
                        ];
        do {
            $ann->initialise();
            $result = $ann->train($trainingSet);
        } while (!$result);
        
        $network->activate([-1, -1]);
        $outputs = $network->getOutputs();
        $this->assertTrue($outputs[0] < -0.9);
        
        $network->activate([-1, 1]);
        $outputs = $network->getOutputs();
        $this->assertTrue($outputs[0] > 0.9);
         
        $network->activate([1, -1]);
        $outputs = $network->getOutputs();
        $this->assertTrue($outputs[0] > 0.9);
        
        $network->activate([1, 1]);
        $outputs = $network->getOutputs();
        $this->assertTrue($outputs[0] > 0.9);
    }
    
    /**
     * Test it learns the AND function using hyperbolic tangent
     */
    public function testItLearnsANDFunctionUsingHyperbolicTangent()
    {
        $network = new FeedForward([2, 2, 1], new HyperbolicTangent());
        $ann = new Backpropagation($network, 0.9, 0.3, 0.001);
        $trainingSet = [
                            [-1,-1,1],
                            [-1,1,-1],
                            [1,-1,-1],
                            [1,1,1]
                        ];
        do {
            $ann->initialise();
            $result = $ann->train($trainingSet);
        } while (!$result);
        
        $network->activate([-1, -1]);
        $outputs = $network->getOutputs();
        $this->assertTrue($outputs[0] > 0.9);
        
        
        $network->activate([-1, 1]);
        $outputs = $network->getOutputs();
        $this->assertTrue($outputs[0] < -0.9);
        
        $network->activate([1, -1]);
        $outputs = $network->getOutputs();
        $this->assertTrue($outputs[0] < -0.9);
        
        $network->activate([1, 1]);
        $outputs = $network->getOutputs();
        $this->assertTrue($outputs[0] > 0.9);
    }
    
    /**
     * Test it learns the XOR function with two hidden neurons
     */
    public function testItLearnsXORFunctionWithTwoHiddenUnitsUsingHyperbolicTangent()
    {
        $network = new FeedForward([2, 2, 1], new HyperbolicTangent());
        $ann = new Backpropagation($network, 0.7, 0.3, 0.001);
        $trainingSet = [
                            [-1,-1,-1],
                            [-1,1,1],
                            [1,-1,1],
                            [1,1,-1]
                        ];

        do {
            $ann->initialise();
            $result = $ann->train($trainingSet);
        } while (!$result);
        
        $network->activate([-1, -1]);
        $outputs = $network->getOutputs();
        $this->assertTrue($outputs[0] < -0.9);
        
        $network->activate([-1, 1]);
        $outputs = $network->getOutputs();
        $this->assertTrue($outputs[0] > 0.9);
         
        $network->activate([1, -1]);
        $outputs = $network->getOutputs();
        $this->assertTrue($outputs[0] > 0.9);
        
        $network->activate([1, 1]);
        $outputs = $network->getOutputs();
        $this->assertTrue($outputs[0] < -0.9);
    }
    
    /**
     * Test it learns the XOR function with three hidden neurons
     */
    public function testItLearnsXORFunctionWithThreeHiddenUnitsUsingHyperbolicTangent()
    {
        $network = new FeedForward([2, 3, 1], new HyperbolicTangent());
        $ann = new Backpropagation($network, 0.7, 0.3);
        $trainingSet = [
                            [-1,-1,-1],
                            [-1,1,1],
                            [1,-1,1],
                            [1,1,-1]
                        ];

        do {
            $ann->initialise();
            $result = $ann->train($trainingSet);
        } while (!$result);
        
        $network->activate([-1, -1]);
        $output = $network->getOutputs();
        $this->assertTrue($output[0] < -0.9);
        
        $network->activate([-1, 1]);
        $output = $network->getOutputs();
        $this->assertTrue($output[0] > 0.9);
         
        $network->activate([1, -1]);
        $output = $network->getOutputs();
        $this->assertTrue($output[0] > 0.9);
        
        $network->activate([1, 1]);
        $output = $network->getOutputs();
        $this->assertTrue($output[0] < -0.9);
    }
    
    /**
     * Test it learns the XOR function with four hidden neurons
     */
    public function testItLearnsXORFunctionWithFourHiddenUnitsUsingHyperbolicTangent()
    {
        $network = new FeedForward([2, 4, 1], new HyperbolicTangent());
        $ann = new Backpropagation($network, 0.7, 0.3);
        $trainingSet = [
                            [-1,-1,-1],
                            [-1,1,1],
                            [1,-1,1],
                            [1,1,-1]
                        ];

        do {
            $ann->initialise();
            $result = $ann->train($trainingSet);
        } while (!$result);
        
        $network->activate([-1, -1]);
        $output = $network->getOutputs();
        $this->assertTrue($output[0] < -0.9);
        
        $network->activate([-1, 1]);
        $output = $network->getOutputs();
        $this->assertTrue($output[0] > 0.9);
         
        $network->activate([1, -1]);
        $output = $network->getOutputs();
        $this->assertTrue($output[0] > 0.9);
        
        $network->activate([1, 1]);
        $output = $network->getOutputs();
        $this->assertTrue($output[0] < -0.9);
    }

    /**
     * Test it learns the XOR function with two hidden neurons
     */
    public function testItSavesAndLoadsStateFromFile()
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

        $network->save('./network.txt');

        $network2 = FeedForward::load('./network.txt');

        $network2->activate([0, 0]);
        $outputs = $network2->getOutputs();
        $this->assertTrue($outputs[0] < 0.1);

        $network2->activate([0, 1]);
        $outputs = $network2->getOutputs();
        $this->assertTrue($outputs[0] > 0.9);

        $network2->activate([1, 0]);
        $outputs = $network2->getOutputs();
        $this->assertTrue($outputs[0] > 0.9);

        $network2->activate([1, 1]);
        $outputs = $network2->getOutputs();
        $this->assertTrue($outputs[0] < 0.1);
    }
}
