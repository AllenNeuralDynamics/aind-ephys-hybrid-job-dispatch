# hash:sha256:998bd7be14456412f1bc50f6ccee281a2651d38a045b1f5d44d2c3299b4952ea
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN pip install -U --no-cache-dir \
    spikeinterface[full]==0.103.2
