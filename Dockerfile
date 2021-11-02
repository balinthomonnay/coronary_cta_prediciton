FROM datamachines/cudnn_tensorflow_opencv:10.1_2.1.0_4.3.0-20200423
RUN pip install altair
RUN pip install pydicom
RUN pip install hyperopt
