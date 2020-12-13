<?php

namespace neuralnetwork\Test;

abstract class BaseTest extends \PHPUnit_Framework_TestCase
{
    protected function initialiseNetworkWithTwoOutputs(&$network)
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
}