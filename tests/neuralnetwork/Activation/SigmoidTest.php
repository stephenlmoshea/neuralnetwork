<?php
namespace neuralnetwork\Test;

use neuralnetwork\Activation\Sigmoid;

class SigmoidTest extends \PHPUnit_Framework_TestCase
{    
    protected $activation;
    
    public function setup()
    {
        $this->activation = new Sigmoid();
    }
    
    public function testGetActivationWithValidNetValue()
    {
        $this->assertEquals(0.67, round($this->activation->getActivation(0.7),2));
    }
    
    public function testGetDerivativeWithValidNetValue()
    {
        $this->assertEquals(0.22, round($this->activation->getDerivative(0.7),2));
    }
}
