<?php


namespace App\Controller;

use Model;
use Symfony\Component\HttpFoundation\JsonResponse;
use Symfony\Component\HttpFoundation\Response;
use Symfony\Component\Routing\Annotation\Route;

class UpdateConctroller
{
        /**
         * @Route("/update/check",name="check")
         */
        function check_update():Response
        {
            $model = new Model();
            return new Response($model->getTrainedNetwork(), 200, [ 'Content-type' => 'text/plain' ]);
        }

    /**
     * @Route("/update/download",name="download")
     */
    function make_update():Response
    {
        $model = new Model();
        $downloadFile=$model->getModel();
        $jsonResp= new JsonResponse(['name'=>$model->getTrainedNetwork(),'compiled_file'=>$downloadFile],200);
        $jsonResp->headers->set('Content-type','multipart/form-data');
        $jsonResp->headers->set('Content-Disposition','form-data;filename='.$model->getTrainedNetwork().'h5');
        return $jsonResp;

    }
}