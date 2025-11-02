FROM pytorch/pytorch:2.9.0-cuda12.6-cudnn9-runtime

# Ignore all interactive dialog during apt-get update
ENV DEBIAN_FRONTEND=noninteractive

# Install linux package
RUN apt-get update && apt-get upgrade -y; \
    apt-get install -y \
    curl git htop sudo vim wget build-essential ffmpeg jq

# Install conda package
COPY docker/conda_requirements.txt /tmp/
RUN conda update -y conda;

# Install python package
COPY docker/requirements.txt /tmp/
RUN /opt/conda/bin/python3 -m pip install --upgrade pip; \
    /opt/conda/bin/python3 -m pip --no-cache-dir install -r /tmp/requirements.txt;

ENV HOME=/workspace