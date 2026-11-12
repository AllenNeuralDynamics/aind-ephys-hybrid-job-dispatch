# hash:sha256:b803255c69b447e2e3cdff649156d5d145d0a82b847859fa5f230f6dd287f700
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN pip install -U --no-cache-dir \
    boto3==1.43.9 \
    log-schema==0.2.10.dev1 \
    s3fs==0.4.2 \
    spikeinterface==0.103.2
