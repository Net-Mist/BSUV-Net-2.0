FROM  pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install opencv-python matplotlib tensorboardX



