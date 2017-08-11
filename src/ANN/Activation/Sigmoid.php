<?php namespace ANN\Activation;

use ANN\Activation;

class Sigmoid implements Activation
{
    public function getActivation($net)
    {
        return 1 / (1 + exp(-$net));
    }
    
    public function getDerivative($net)
    {
        return $this->getActivation($net) * (1 - $this->getActivation($net));
    }
}
