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
| *model_internal_dataset.h5* | Model weights learned on the initial dataset ||
| *Dockerfile* | A dockerfile containing all required libraries for the notebooks, the base image is from https://hub.docker.com/r/datamachines/cudnn_tensorflow_opencv ||
| *docker-compose.yml* | A docker compose file containg startup information for the Dockerfile ||
| *prediction.ipynb* | notebook on the model evaulation |Model weights to load, output DCM files from CSOtoDCM.mlab|
| *transfer_learning.ipynb* | notebook on creating a transfer learning for a ne dataset |Model weights to load, output DCM files from CSOtoDCM.mlab|
| *script_libr.py* | Contains functions used in the notebooks ||



## Evaulation Process
Use the CSOtoDCM.mlab to create dicom files from the contours, than create a conda enviroment from the envrioment.yml file, or use the docker file provided and run prediciton.ipyn

## Prepariations to use the docker file
Run the following commands to install dockers engine:
```
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
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
In the folder containing everything downloaded from here run 
```
sudo docker-compose up
```
Than, in a browser at localhost:8000, the jupyter notebooks will be visable and usable. The docker will only have access to files and folders in the same folder as where the docker-compose.yml file is.
