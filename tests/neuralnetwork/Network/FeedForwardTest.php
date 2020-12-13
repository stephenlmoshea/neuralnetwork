<?php
namespace neuralnetwork\Test;

use neuralnetwork\Network\FeedForward;
use neuralnetwork\Train\Backpropagation;
use neuralnetwork\Activation\Sigmoid;
use neuralnetwork\Activation\HyperbolicTangent;

class FeedForwardTest extends \PHPUnit_Framework_TestCase
{
    protected $network;
    protected $nodePerLayer;
    protected $activation;
    protected $layers;
    protected $weights;
    protected $biasWeights;
    
    public function setup()
    {
        $this->nodePerLayer = [2, 2, 1];
        $this->layers = $this->getLayers();
        $this->activation = new Sigmoid();
        $this->network = new FeedForward($this->nodePerLayer, new Sigmoid());
    }
    
    public function testItCreatesANetworkWithValidParams()
    {
        $this->assertArraySubset($this->layers, $this->network->getNetworkLayers());
        $this->assertEquals($this->activation, $this->network->getActivation());
    }
    
    public function testNetworkWeightsAreInitialisedToRandomValuesInCorrectRange()
    {
        $this->network->initialise();
        $weights = $this->network->getWeights();
        foreach($weights as $i => $connections){
            foreach($connections as $j => $value){
                $this->assertLessThanOrEqual(0.05, abs($value));
            }
        }
    }
    
    public function testNetworkNodeValuesAreInitialised()
    {
        $this->network->initialise();
        $values = $this->network->getValues();
        foreach($values as $value){
            $this->assertLessThanOrEqual(0, $value);
        }
    }
    
    public function testNetworkLayersAreFullyConnected()
    {
        $this->network->initialise();
        $weights = $this->network->getWeights();
        $this->assertTrue(isset($weights[0][2]));
        $this->assertTrue(isset($weights[0][3]));
        $this->assertTrue(isset($weights[1][2]));
        $this->assertTrue(isset($weights[1][3]));
        $this->assertTrue(isset($weights[2][4]));
        $this->assertTrue(isset($weights[3][4]));
    }
    
    public function testActivatingNetworkWithValidInputsProducesValidOutput()
    {
        $this->initialiseNetwork();
        $this->network->activate([0,1]);
        $output = $this->network->getOutputs()[0];
        $this->assertEquals(round($output, 2), 0.49);
    }

    public function testActivatingNetworkWithValidInputsProducesValidOutputs()
    {
        $this->nodePerLayer = [2, 2, 2];
        $this->layers = $this->getLayersWithTwoOutputs();
        $this->activation = new Sigmoid();
        $this->network = new FeedForward($this->nodePerLayer, new Sigmoid());

        $this->initialiseNetworkWithTwoOutputs();
        $this->network->activate([0,1]);
        $outputs = $this->network->getOutputs();
        $this->assertEquals(round($outputs[0], 3), 0.505);
        $this->assertEquals(round($outputs[1], 3), 0.499);
    }

    public function testItLearnsWeightsAndOutputsForXORFunctionWithTwoOutputs()
    {
        $this->nodePerLayer = [2, 2, 2];
        $this->layers = $this->getLayersWithTwoOutputs();
        $this->activation = new Sigmoid();
        $this->network = new FeedForward($this->nodePerLayer, new Sigmoid());
        $ann = new Backpropagation($this->network, 0.7, 0.3);
        $this->initialiseNetworkWithTwoOutputs();

        // $this->network->activate([0,1]);

        // var_dump($this->network->getWeights());
        // var_dump($this->network->getBiasWeights());
        // var_dump($this->network->getOutputs());
        

        $trainingSet = [
                            [0,0,0,0],
                            [0,1,0,1],
                            [1,0,1,0],
                            [1,1,0,0]
                        ];

        do {
            $result = $ann->train($trainingSet);
        } while (!$result);

        $this->network->activate([1,1]);

        // var_dump('--------After Training-------');
        // var_dump('Weights:');
        // var_dump($this->network->getWeights());
        // var_dump('Bias Weights:');
        // var_dump($this->network->getBiasWeights());
        // var_dump('Outputs:');
        // var_dump($this->network->getOutputs());
        // die();

        $this->network->activate([0, 0]);
        $outputs = $this->network->getOutputs();
        
        $this->assertTrue((string)$outputs[0] == (string)0.073974751048076);
        $this->assertTrue((string)$outputs[1] == (string)0.076405873198382);
        
        $this->network->activate([0, 1]);
        $outputs = $this->network->getOutputs();
        
        $this->assertTrue((string)$outputs[0] == (string)0.0011872968318554);
        $this->assertTrue((string)$outputs[1] == (string)0.90067060908902);

        $this->network->activate([1, 0]);
        $outputs = $this->network->getOutputs();
        
        $this->assertTrue((string)$outputs[0] == (string)0.90222312526496);
        $this->assertTrue((string)$outputs[1] == (string)0.00080085411873496);

        $this->network->activate([1, 1]);
        $outputs = $this->network->getOutputs();
        
        $this->assertTrue((string)$outputs[0] == (string)0.063898658496818);
        $this->assertTrue((string)$outputs[1] == (string)0.06729508546056);
 
        $expectedWeights = [
            '0' => [
                '2' => 3.0472441618378,
                '3' => -3.7054643380452
            ],
            '1' => [
             '2' => -2.5961449172696,
             '3' => 3.951078577457
            ],
            '2' => [
             '4' => 2.6105699180982,
             '5' => -4.730017947296
            ],
            '3' => [
             '4' => -7.0441994420989,
             '5' => 5.5300351551941
            ],
         ];
 
         $this->assertEquals($expectedWeights, $this->network->getWeights());
 
         $expectedBiasWeights = [
             '0' => [
                 '2' => 0.55162749854201,
                 '3' => 0.32977328382385
             ],
             '1' => [
              '4' => -0.085977993520736,
              '5' => -2.7077995410221
             ]
          ];
  
         $this->assertEquals($expectedBiasWeights, $this->network->getBiasWeights());
    }
    
    public function testGetValue(){
        $this->initialiseNetwork();
        $this->network->activate([0,1]);
        $this->assertEquals(0.5, $this->network->getValue(3));
    }
    
    public function testGetActivation(){
        $this->network->initialise();
        $activation = $this->network->getActivation();
        $this->assertInstanceOf(Sigmoid::class, $activation);
    }
    
    public function testGetNet(){
        $this->initialiseNetwork();        
        $this->network->activate([0,1]);
        $this->assertEquals(-0.06, $this->network->getNet(4));
    }
    
    public function testGetWeight(){
        $connections = $this->initialiseNetwork();        
        $this->assertEquals($connections['weights'][0], $this->network->getWeight(0));
    }
    
    public function testUpdatingWeight(){
        $connections = $this->initialiseNetwork();
        $weightUpdate = 0.02;
        $this->network->updateWeight(0, 2, $weightUpdate);
        $newWeight = $connections['weights'][0][2] + $weightUpdate;
        
        $this->assertEquals($newWeight, $this->network->getWeight(0)[2]);
    }
    
    public function testUpdatingBiasWeight(){
        $connections = $this->initialiseNetwork();
        $biasWeightUpdate = 0.02;
        $this->network->updateBiasWeight(0, 2, $biasWeightUpdate);
        $newWeight = $connections['biasWeights'][0][2] + $biasWeightUpdate;
        
        $this->assertEquals($newWeight, $this->network->getBiasWeight(0)[2]);
    }
    
    protected function getWeights()
    {
        $weights[0][2] = -0.04;
        $weights[0][3] = 0.03;
        $weights[1][2] = -0.04;
        $weights[1][3] = -0.02;
        $weights[2][4] = -0.05;
        $weights[3][4] = -0.03;
        
        return $weights; 
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
    
    protected function getBiasWeights()
    {
        $biasWeights[0][2] = 0.04;
        $biasWeights[0][3] = 0.02;
        $biasWeights[1][4] = -0.02;
        
        return $biasWeights;
    }

    protected function getBiasWeightsForTwoOutputs()
    {
        $biasWeights[0][2] = -0.03;
        $biasWeights[0][3] = -0.03;
        $biasWeights[1][4] = 0.03;
        $biasWeights[1][5] = -0.01;
        
        return $biasWeights;
    }
    
    protected function getLayers()
    {
        return [
                    [
                        'num_nodes' => 2,
                        'start_node' => 0,
                        'end_node' => 1
                    ],
                    [
                        'num_nodes' => 2,
                        'start_node' => 2,
                        'end_node' => 3
                    ],
                    [
                        'num_nodes' => 1,
                        'start_node' => 4,
                        'end_node' => 4
                    ]
                ];
    }

    protected function getLayersWithTwoOutputs()
    {
        return [
                    [
                        'num_nodes' => 2,
                        'start_node' => 0,
                        'end_node' => 1
                    ],
                    [
                        'num_nodes' => 2,
                        'start_node' => 2,
                        'end_node' => 3
                    ],
                    [
                        'num_nodes' => 2,
                        'start_node' => 4,
                        'end_node' => 5
                    ]
                ];
    }
    
    protected function initialiseNetwork()
    {
        $this->network->initialise();
        
        $weights = $this->getWeights();
        $biasWeights = $this->getBiasWeights();

        $this->network->setWeights($weights);
        $this->network->setBiasWeights($biasWeights);
        return [
            'weights' => $weights,
            'biasWeights' => $biasWeights
        ];
    }

    protected function initialiseNetworkWithTwoOutputs()
    {
        $this->network->initialise();
        
        $weights = $this->getWeightsForTwoOutputs();
        $biasWeights = $this->getBiasWeightsForTwoOutputs();

        $this->network->setWeights($weights);
        $this->network->setBiasWeights($biasWeights);
        return [
            'weights' => $weights,
            'biasWeights' => $biasWeights
        ];
    }
}
