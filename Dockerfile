FROM python:3.7.10-slim-buster


# Basic setup
RUN apt update
RUN apt install -y \
    build-essential \
    git \
    curl \
    ca-certificates \
    wget \
    && rm -rf /var/lib/apt/lists

ENV GRADIO_SERVER_PORT 8080

EXPOSE 8080

WORKDIR /opt/src

ADD src/utils utils

ADD configs configs

COPY ["*.toml","src/demo.py","requirements.txt" ,"requirements.txt" ,"set_env.sh","./"]

RUN  pip install --no-cache-dir -r requirements.txt \
  && rm requirements.txt \
  && rm -rf /root/.cache/pip \
  && rm -rf /tmp/*

ENTRYPOINT bash set_env.sh