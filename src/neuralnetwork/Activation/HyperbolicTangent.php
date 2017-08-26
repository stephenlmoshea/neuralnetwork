<?php namespace neuralnetwork\Activation;

use neuralnetwork\Activation;

class HyperbolicTangent implements Activation
{
    public function getActivation($net)
    {
        return tanh($net);
    }
    
    public function getDerivative($net)
    {
        return $this->getActivation($net) * (1 - $this->getActivation($net));
    }
}
