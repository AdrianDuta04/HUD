<?php

namespace App\Entity;
class Model
{
  public function getLastTrainedNetwork():string
  {
      $jsonData = file_get_contents('../src/Entity/version.json');
      $array = json_decode($jsonData, true);
      return $array['version'];
  }
  public function getModel():array {
      $filePath=scandir("../src/NN/trained_models");
      $fileName= end($filePath);
      $file="../src/NN/trained_models/".$fileName;
      return array($file,$fileName);
  }
  
  
}
