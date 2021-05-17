FROM  pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install opencv-python matplotlib tensorboardX



