<?php namespace ANN\Network;

use ANN\Activation\Sigmoid;
use ANN\Activation;
use ANN\Network;

class FeedForward implements Network
{
    protected $networkLayers = array();
    protected $net = array();
    protected $weights = array();
    protected $biasWeights = array();
    protected $values = array();
    protected $activation;
    protected $totalNumNodes;
    
    /**
     * Construct network
     * 
     * @param array $networkLayers
     * @param Activation $activation
     */
    public function __construct(array $networkLayers, Activation $activation)
    {
        $this->networkLayers = [];
        $startNode = 0;
        $endNode = 0;
        foreach ($networkLayers as $layer => $numNodes) {
            if ($layer > 0) {
                $startNode += $networkLayers[$layer - 1];
            }
            $endNode += $numNodes;
            $this->networkLayers[] = [
                'num_nodes' => $numNodes,
                'start_node' => $startNode,
                'end_node' => $endNode - 1,
            ];
        }
        $this->totalNumNodes = array_sum($networkLayers);
        $this->activation = $activation;
    }
    
    /**
     * Initialises the nodes outputs to zero
     * and interconnection strengths to random values
     * between -0.05 and +0.05
     */
    public function initialise()
    {
        $this->net = array();
        $this->weights = array();
        $this->biasWeights = array();
        $this->values = array();
        $this->initialiseValues();
        $this->initialiseWeights();
    }
    
    /**
     * Activate the neural network by passing
     * the values to the input layer.
     * 
     * The output of each layer to fed forward
     * to the output layer
     * 
     * @param array $inputs
     */
    public function activate(array $inputs)
    {
        for ($z = 0; $z < $this->networkLayers[0]['num_nodes']; ++$z) {
            $this->values[$z] = $inputs[$z];
        }
        foreach ($this->networkLayers as $num => $layer) {
            if ($num > 0) {
                for ($j = $layer['start_node']; $j <= $layer['end_node']; ++$j) {
                    $net = 0;
                    for ($i = $this->networkLayers[$num - 1]['start_node']; $i <= $this->networkLayers[$num - 1]['end_node']; ++$i) {
                        $net += $this->values[$i] * $this->weights[$i][$j];
                    }
                    $net += $this->biasWeights[$num - 1][$j];
                    $this->net[$j] = $net;
                    $this->values[$j] = $this->activation->getActivation($net);
                }
            }
        }
    }
    
    /**
     * Gets the values from the output layer
     * 
     * @return array
     */
    public function getOutputs()
    {
        $startNode = $this->networkLayers[count($this->networkLayers) - 1]['start_node'];
        $endNode = $this->networkLayers[count($this->networkLayers) - 1]['end_node'];
        return array_slice($this->values, $startNode, ($endNode+1) - $startNode);
    }
    
    /**
     * Gets the network layers
     * 
     * @return array
     */
    public function getNetworkLayers()
    {
        return $this->networkLayers;
    }
    
    /**
     * Gets the output value at a particular
     * node at index
     * 
     * @param int $index
     * @return float
     */
    public function getValue($index)
    {
        return $this->values[$index];
    }
    
    /**
     * Gets all values
     * 
     * @return array
     */
    public function getValues()
    {
        return $this->values;
    }
    
    /**
     * Gets the activation function being
     * used by the network
     * 
     * @return ANN\Activation
     */
    public function getActivation()
    {
        return $this->activation;
    }
    
    /**
     * Gets the net value of particular node
     * at index
     * 
     * @param int $index
     * @return float
     */
    public function getNet($index)
    {
        return $this->net[$index];
    }
    
    /**
     * Gets the interconnection weights
     * for a particular node
     * 
     * @param index $index
     * @return array
     */
    public function getWeight($index)
    {
        return $this->weights[$index];
    }
    
    /**
     * Sets the weights
     * @param array $weights
     */
    public function setWeights(array $weights)
    {
        $this->weights = $weights;
    }
    
    /**
     * Get all non bias weights
     * 
     * @return array
     */
    public function getWeights()
    {
        return $this->weights;
    }
    
    /**
     * Get all bias weights
     * 
     * @return array
     */
    public function getBiasWeights()
    {
        return $this->biasWeights;
    }
    
    /**
     * Gets the interconnection weights
     * for a particular bias node
     * 
     * @param index $index
     * @return array
     */
    public function getBiasWeight($index)
    {
        return $this->biasWeights[$index];
    }
    
    /**
     * Sets the bias weights
     * @param array $weights
     */
    public function setBiasWeights(array $biasWeights)
    {
        $this->biasWeights = $biasWeights;
    }
    
    /**
     * Updates the weight between node $i
     * and $j with given weight value
     * 
     * @param int $i
     * @param int $j
     * @param float $weight
     */
    public function updateWeight($i, $j, $weight)
    {
        $this->weights[$i][$j] += $weight;
    }
    
    /**
     * Updates the bias weight between node $i
     * and $j with given weight value
     * 
     * @param int $i
     * @param int $j
     * @param float $weight
     */
    public function updateBiasWeight($i, $j, $weight)
    {
        $this->biasWeights[$i][$j] += $weight;
    }
    
    /**
     * Gets the total number of nodes in the 
     * neural network
     * 
     * @return int
     */
    public function getTotalNumNodes()
    {
        return $this->totalNumNodes;
    }
    
    /**
     * Initialises the nodes outputs to zero
     */
    protected function initialiseValues()
    {
        $this->values = array_fill(0, $this->totalNumNodes, 0.0);
        $this->net = array_fill(0, $this->totalNumNodes, 0.0);
    }

    /**
     * Initialises interconnection strengths to random values
     * between -0.05 and +0.05
     */
    protected function initialiseWeights()
    {
        foreach ($this->networkLayers as $num => $layer) {
            if ($num < count($this->networkLayers) - 1) {
                //Calculate non bias weights
                for ($i = $layer['start_node']; $i <= $layer['end_node']; ++$i) {
                    for ($j = $this->networkLayers[$num + 1]['start_node']; $j <= $this->networkLayers[$num + 1]['end_node']; ++$j) {
                        $this->weights[$i][$j] = rand(-5, 5) / 100;
                    }
                }
                //Calculate bias weights
                for ($b = $this->networkLayers[$num + 1]['start_node']; $b <= $this->networkLayers[$num + 1]['end_node']; ++$b) {
                    $this->biasWeights[$num][$b] = rand(-5, 5) / 100;
                }
            }
        }
    }
}
