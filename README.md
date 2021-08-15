# Deep Learning based coronary plaque segmentation

Open access deep learning algortihm for atherosclerotic coronary plaque segmentation on coronary CT angoiography. This repository contains the initial fitted model traned on our internal dataset, along with the model obtained by applying transfer learning on the external validation dataset. 

The preprocessing requires MevisLab, which can be downloaded at https://www.mevislab.de/. It was used on plaques exported from a dedicated software (QAngioCT software v3.1.3.13; Medis Medical Imaging Systems, Leiden, The Netherlands).


This model was trained, validated (tuned) and tested using Python 3.7.6, on Ubuntu 20.04.1 with CUDA 11.2, all modules required can be found in the enviroment.yml

## Table of Contents
| *File* |Description|Input|
| ----------- | ----------- |----------- |
| *CSOtoDCM.mlab* | MevisLab file, to create dicom images from the Contour files exported form Mevis |CSO and DCM files form Medis |
| *enviroment.yml* | Python enviroment ||
| *model_external_dataset.h5* | Model weights transfer learned on the exteranl dataset, which contained large plaques. ||
| *model_internal_dataset.h5* | Model weights learned on the initail dataset ||
| *prediction.ipynb* | notebook on the model evaulation |Model weights to load, output DCM files from CSOtoDCM.mlab|

## Evaulation Process
Use the CSOtoDCM.mlab to create dicom files from the contours, than create a conda enviroment from the envrioment.yml file, and run prediciton.ipyn

