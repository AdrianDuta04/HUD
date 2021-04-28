<?php


class Model
{
  public function getTrainedNetwork():int
  {

  }
  public function getModel():string {
      $nnFile=fopen("filePath",'r');
      return base64_encode(fread($nnFile,filesize("filepath")));
  }
}