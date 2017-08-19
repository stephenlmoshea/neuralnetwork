<?php namespace neuralnetwork;

use neuralnetwork\Network;

interface Train
{
    public function train(array $trainingSets);
}
