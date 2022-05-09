# Deep learning–based atherosclerotic coronary plaque segmentation on coronary CT angiography
Natasa Jávorszky; Bálint Homonnay; Gary Gerstenblith, MD; David Bluemke, MD; Péter Kiss; Mihály Török; David Celentano, ScD; Hong Lai, PhD; Shenghan Lai, MPH; Márton Kolossváry, MD, PhD

Published in: European Radiology (2022), doi: [10.1007/s00330-022-08801-8](<https://link.springer.com/article/10.1007/s00330-022-08801-8>)

![Figure 3](/Figure_3.png)

The DL based segmentation resulted in lower bias and variance compared to the MCA algorithm. 
DL: deep learning; 95% LOA: 95% limit of agreement; MCA: minimum cost approach


## Introduction
Volumetric evaluation of coronary artery disease (CAD) allows better prediction of cardiac events. However, CAD segmentation is labor intensive. Therefore, we created an open-source deep learning (DL) model to segment coronary plaques on coronary CT angiography (CCTA).

This repository contains the initial fitted model trained on our internal dataset, along with the model obtained by applying transfer learning on the external validation dataset. 

The preprocessing requires MevisLab, which can be downloaded at https://www.mevislab.de/. It was used on plaques saved in a dedicated software solution (QAngioCT software v3.1.3.13; Medis Medical Imaging Systems, Leiden, The Netherlands).

The model was trained, validated (tuned) and tested using Python 3.7.6, on Ubuntu 20.04.1 with CUDA 11.2, all modules required can be found in the environment.yml

## Table of Contents
| *File* |Description|Input|
| ----------- | ----------- |----------- |
| *CSOtoDCM.mlab* | MevisLab file, to create DICOM images from the Contour files exported form Mevis |CSO and DCM files form Medis |
| *environment.yml* | Python environment ||
| *model_external_dataset.h5* | Model weights transfer learned on the external dataset, which contained large vulnerable plaques. ||
| *model_internal_dataset.h5* | Model weights learned on the initial dataset ||
| *Dockerfile* | A dockerfile containing all required libraries for the notebooks, the base image is from https://hub.docker.com/r/datamachines/cudnn_tensorflow_opencv ||
| *docker-compose.yml* | A docker compose file containing initialization information for the Dockerfile ||
| *prediction.ipynb* | Notebook on the model evaluation |Model weights to load, output DCM files from CSOtoDCM.mlab|
| *transfer_learning.ipynb* | Notebook on creating a transfer learning for a new dataset |Model weights to load, output DCM files from CSOtoDCM.mlab|
| *script_libr.py* | Contains functions used in the notebooks ||

## Evaluation Process
Use the CSOtoDCM.mlab to create DICOM files from the contours, then create a conda environment from the environment.yml file, or use the docker file provided and run prediction.ipynb

## Preparations to use the docker file
Run the following commands to install dockers engine:
```
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containered.io
```
And to install nvidia drivers:
```
sudo apt install nvidia-cuda-toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```
## Using the dockerfile
In the folder containing everything downloaded from the current repository run: 
```
sudo docker-compose up
```
The jupyter notebooks will be visible in a browser at localhost:8000. The docker will only have access to files and folders in the same folder as where the docker-compose.yml file is.
