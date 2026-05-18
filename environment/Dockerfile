# hash:sha256:62101bab4c9887e809528cf828ad358293e2bd476ed124f8d2a8bb2df0464647
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN conda install -y \
        jupyterlab==4.5.7 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    boto3==1.43.9 \
    spikeinterface==0.103.2
