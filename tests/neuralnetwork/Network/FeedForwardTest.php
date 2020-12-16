<?php
namespace neuralnetwork\Test;

use neuralnetwork\Test\BaseTest;
use neuralnetwork\Activation\Sigmoid;
use neuralnetwork\Network\FeedForward;
use neuralnetwork\Train\Backpropagation;
use neuralnetwork\Activation\HyperbolicTangent;

class FeedForwardTest extends BaseTest
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
        $nodePerLayer = [2, 2, 2];
        $layers = $this->getLayersWithTwoOutputs();
        $activation = new Sigmoid();
        $network = new FeedForward($nodePerLayer, $activation);

        $this->initialiseNetworkWithTwoOutputs($network);
        $network->activate([0,1]);
        $outputs = $network->getOutputs();
        $this->assertEquals(round($outputs[0], 3), 0.505);
        $this->assertEquals(round($outputs[1], 3), 0.499);
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
}
