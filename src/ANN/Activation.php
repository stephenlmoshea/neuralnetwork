<?php namespace ANN;

interface Activation
{
    public function getActivation($net);
    public function getDerivative($net);
}
