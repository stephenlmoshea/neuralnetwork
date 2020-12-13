<?php namespace neuralnetwork\Train;

use Monolog\Logger;
use Monolog\Handler\StreamHandler;
use neuralnetwork\Network;
use neuralnetwork\Train;

class Backpropagation implements Train
{
    protected $nodeDeltas = array();
    protected $gradients = array();
    protected $biasGradients = array();
    protected $learningRate;
    protected $momentum;
    protected $weightUpdates = array();
    protected $biasWeightUpdates = array();
    protected $log;
    protected $minimumError;
    protected $maxNumEpochs;
    protected $numEpochs;
    
    /**
     * Initialise training algorithm with the
     * neural network and training parameters
     * 
     * @param Network $network
     * @param float $learningRate
     * @param float $momentum
     * @param float $minimumError
     * @param int $maxNumEpochs
     */
    public function __construct(Network &$network, $learningRate, $momentum, $minimumError = 0.005, $maxNumEpochs = 2000)
    {
        $this->network = $network;
        $this->learningRate = $learningRate;
        $this->momentum = $momentum;
        $this->minimumError = $minimumError;
        $this->maxNumEpochs = $maxNumEpochs;
        $this->initialise();
    }
    
    /**
     * Initialise nodeDeltas and weight updates
     */
    public function initialise()
    {
        $this->network->initialise();
        $this->nodeDeltas = array();
        $this->gradients = array();
        $this->biasGradients = array();
        $this->weightUpdates = array();
        $this->biasWeightUpdates = array();
        $this->initialiseValues();
        $this->initialiseWeights();
    }
    
    /**
     * Training network on the provided
     * training set
     * 
     * @param array $trainingSets
     * @return boolean
     */
    public function train(array $trainingSets)
    {
        $this->numEpochs=1;
        $log = new Logger('debug');

        $log->pushHandler(new StreamHandler('./training.log', Logger::INFO));

        do {
            if ($this->numEpochs > $this->maxNumEpochs) {
                return false;
            }
            
            $sumNetworkError=0;
            foreach ($trainingSets as $trainingSet) {
                $this->network->activate($trainingSet);
                $outputs = $this->network->getOutputs();
                $this->calculateNodeDeltas($trainingSet);
                $this->calculateGradients();
                $this->calculateWeightUpdates();
                $this->applyWeightChanges();
                $sumNetworkError += $this->calculateNetworkError($trainingSet);

                // add records to the log
                $log->info('Epoch:'.$this->numEpochs,['inputs' => $trainingSet, 'outputs' => $outputs, 'error' => $sumNetworkError]);
            }
            
            $globalError = $sumNetworkError/count($trainingSets);

            $log->info('Num Epochs: '.$this->numEpochs);
            $log->info('Global Error: '.$globalError);
            
            $this->numEpochs++;
        } while ($globalError > $this->minimumError);
        
        return true;
    }

    /**
     * Initialise node deltas to zero
     */
    protected function initialiseValues()
    {
        $this->nodeDeltas = array_fill(0, $this->network->getTotalNumNodes(), 0.0);
    }
    
    /**
     * Initialise weight updates to zero
     */
    protected function initialiseWeights()
    {
        $networkLayers = $this->network->getNetworkLayers();
        foreach ($networkLayers as $num => $layer) {
            if ($num < count($networkLayers) - 1) {
                //Calculate non bias weights
                for ($i = $layer['start_node']; $i <= $layer['end_node']; ++$i) {
                    for ($j = $networkLayers[$num + 1]['start_node']; $j <= $networkLayers[$num + 1]['end_node']; ++$j) {
                        $this->weightUpdates[$i][$j] = 0.0;
                    }
                }
                //Calculate bias weights
                for ($b = $networkLayers[$num + 1]['start_node']; $b <= $networkLayers[$num + 1]['end_node']; ++$b) {
                    $this->biasWeightUpdates[$num][$b] = 0.0;
                }
            }
        }
    }
    
    /**
     * Calculate error and propagate back
     * calculating node deltas for output nodes
     * and hidden layers
     * 
     * @param array $trainingSet
     */
    public function calculateNodeDeltas(array $trainingSet)
    {
        $networkLayers = $this->network->getNetworkLayers();
        $idealOutputs = array_slice($trainingSet, -1 * $networkLayers[count($networkLayers) - 1]['num_nodes']);
        $startNode = $networkLayers[count($networkLayers) - 1]['start_node'];
        $endNode = $networkLayers[count($networkLayers) - 1]['end_node'];
        $activation = $this->network->getActivation();
        
        //Calculate node delta for output nodes
        $j = 0;
        for ($i = $startNode; $i <= $endNode; ++$i) {
            $error = $this->network->getValue($i) - $idealOutputs[$j];
            $this->nodeDeltas[$i] = (-1 * $error) * $activation->getDerivative($this->network->getNet($i));
            ++$j;
        }
        //Calculate node delta for hidden nodes
        for ($k = count($networkLayers) - 2; $k > 0; --$k) {
            $startNode = $networkLayers[$k]['start_node'];
            $endNode = $networkLayers[$k]['end_node'];
            for ($z = $startNode; $z <= $endNode; ++$z) {
                $sum = 0;
                foreach ($this->network->getWeight($z) as $connectedNode => $weight) {
                    $sum += $weight * $this->nodeDeltas[$connectedNode];
                }
                $this->nodeDeltas[$z] = $activation->getDerivative($this->network->getNet($z)) * $sum;
            }
        }
    }
    
    /**
     * Calculate gradients for bias and non bias weights
     */
    public function calculateGradients()
    {
        $networkLayers = $this->network->getNetworkLayers();
        foreach ($networkLayers as $num => $layer) {
            if ($num < count($networkLayers) - 1) {
                //Calculate gradients for non bias weights
                for ($i = $layer['start_node']; $i <= $layer['end_node']; ++$i) {
                    for ($j = $networkLayers[$num + 1]['start_node']; $j <= $networkLayers[$num + 1]['end_node']; ++$j) {
                        $this->gradients[$i][$j] = $this->network->getValue($i) * $this->nodeDeltas[$j];
                    }
                }
                //Calculate gradents for bias weights
                for ($b = $networkLayers[$num + 1]['start_node']; $b <= $networkLayers[$num + 1]['end_node']; ++$b) {
                    $this->biasGradients[$num][$b] = $this->nodeDeltas[$b];
                }
            }
        }
    }
    
    /**
     * Calculate weight updates using gradients and momentum
     * for bias and non bias weights
     */
    public function calculateWeightUpdates()
    {
        $networkLayers = $this->network->getNetworkLayers();
        foreach ($networkLayers as $num => $layer) {
            if ($num < count($networkLayers) - 1) {
                //Calculate weight changes for non bias weights
                for ($i = $layer['start_node']; $i <= $layer['end_node']; ++$i) {
                    for ($j = $networkLayers[$num + 1]['start_node']; $j <= $networkLayers[$num + 1]['end_node']; ++$j) {
                        $this->weightUpdates[$i][$j] = ($this->learningRate * $this->gradients[$i][$j]) + ($this->momentum * $this->weightUpdates[$i][$j]);
                    }
                }
                //Calculate weight changes for bias weights
                for ($b = $networkLayers[$num + 1]['start_node']; $b <= $networkLayers[$num + 1]['end_node']; ++$b) {
                    $this->biasWeightUpdates[$num][$b] = ($this->learningRate * $this->biasGradients[$num][$b]) + ($this->momentum * $this->biasWeightUpdates[$num][$b]);
                }
            }
        }
    }

    /**
     * Apply weight changes to neural network
     */
    protected function applyWeightChanges()
    {
        $networkLayers = $this->network->getNetworkLayers();
        foreach ($networkLayers as $num => $layer) {
            if ($num < count($networkLayers) - 1) {
                //Calculate weight changes for non bias weights
                for ($i = $layer['start_node']; $i <= $layer['end_node']; ++$i) {
                    for ($j = $networkLayers[$num + 1]['start_node']; $j <= $networkLayers[$num + 1]['end_node']; ++$j) {
                        $this->network->updateWeight($i, $j, $this->weightUpdates[$i][$j]);
                    }
                }
                //Calculate weight changes for bias weights
                for ($b = $networkLayers[$num + 1]['start_node']; $b <= $networkLayers[$num + 1]['end_node']; ++$b) {
                    $this->network->updateBiasWeight($num, $b, $this->biasWeightUpdates[$num][$b]);
                }
            }
        }
    }
    
    /**
     * Calculate network error
     * 
     * @param array $trainingSet
     * @return float
     */
    protected function calculateNetworkError(array $trainingSet)
    {
        $networkLayers = $this->network->getNetworkLayers();
        $idealOutputs = array_slice($trainingSet, -1 * $networkLayers[count($networkLayers) - 1]['num_nodes']);
        $startNode = $networkLayers[count($networkLayers) - 1]['start_node'];
        $endNode = $networkLayers[count($networkLayers) - 1]['end_node'];
        $numNodes = $networkLayers[count($networkLayers) - 1]['num_nodes'];
        $j = 0;
        $sum = 0;
        for ($i = $startNode; $i <= $endNode; ++$i) {
            $error = $idealOutputs[$j] - $this->network->getValue($i);
            $sum += $error * $error;
            ++$j;
        }
        $globalError = (1 / $numNodes) * $sum;
        return $globalError;
    }

    public function getNodeDeltas()
    {
        return $this->nodeDeltas;
    }

    public function getGradients()
    {
        return $this->gradients;
    }

    public function getBiasGradients()
    {
        return $this->biasGradients;
    }

    public function getWeightUpdates()
    {
        return $this->weightUpdates;
    }

    public function getBiasWeightUpdates()
    {
        return $this->biasWeightUpdates;
    }
}
