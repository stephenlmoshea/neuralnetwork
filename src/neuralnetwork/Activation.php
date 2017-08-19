<?php namespace neuralnetwork;

interface Activation
{
    public function getActivation($net);
    public function getDerivative($net);
}
