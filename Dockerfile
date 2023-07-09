FROM python:3.8-slim-buster

LABEL maintainer="Salil Gautam <salil.gtm@gmail.com>"
LABEL description="Dockerfile for Assignment 6 of EMLOv3."

WORKDIR /workspace

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install -e .
