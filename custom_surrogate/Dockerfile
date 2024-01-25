# official Ubuntu base image
FROM ubuntu:latest

# update package list and install necessary dependencies
RUN apt update && apt install -y python3 python3-pip git
RUN pip install --upgrade pip
RUN pip3 install umbridge
RUN pip3 install torch
RUN pip3 install botorch
#RUN pip3 install -e git+https://github.com/facebook/Ax.git#egg=ax 
RUN pip3 install GPy GPyOpt

# copy local application code
COPY . /server

USER root

# expose a port
#EXPOSE
#ENV

# working directory
WORKDIR /server

# run application
CMD python3 parallel_surrogate.py
