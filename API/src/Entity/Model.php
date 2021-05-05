<?php

namespace App\Entity;
class Model
{
  public function getLastTrainedNetwork():string
  {
      $filePath=scandir("../src/NN/trained_models");
      $filePath=substr(end($filePath),27,10);
      return $filePath;
  }
  public function getModel():array {
      $filePath=scandir("../src/NN/trained_models");
      $fileName= end($filePath);
      $file="../src/NN/trained_models/".$fileName;
      return array($file,$fileName);
  }
  
  
}
