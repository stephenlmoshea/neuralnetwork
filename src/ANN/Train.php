<?php namespace ANN;

use ANN\Network;

interface Train
{
    public function train(array $trainingSets);
}
