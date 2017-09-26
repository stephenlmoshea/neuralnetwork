<?php namespace neuralnetwork;

interface Network
{
    public function activate(array $inputs);
    public function save($filename);
    public static function load($filename);
}
