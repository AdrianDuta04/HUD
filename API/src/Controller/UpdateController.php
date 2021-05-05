<?php


namespace App\Controller;

use App\Entity\Model;
use Symfony\Component\HttpFoundation\BinaryFileResponse;
use Symfony\Component\HttpFoundation\Response;
use Symfony\Component\HttpFoundation\ResponseHeaderBag;
use Symfony\Component\Routing\Annotation\Route;
use Symfony\Bundle\FrameworkBundle\Controller\AbstractController;

class UpdateController extends AbstractController
{
    /**
     * @Route("/check")
     */
    public function check(): Response
    {
        $n = new Model();
        $arr=$n->getLastTrainedNetwork();

        return new Response(
            $arr,200
        );
    }
    /**
     * @Route("/download")
     */
    public function download(): BinaryFileResponse
    {
        $n = new Model();
        $fileArr=$n->getModel();

        $response= new BinaryFileResponse($fileArr[0]);
        $response->headers->set('Content-Type', 'text/plain');
        $response->setContentDisposition(
            ResponseHeaderBag::DISPOSITION_ATTACHMENT,
            $fileArr[1]
        );
        return $response;
    }
}