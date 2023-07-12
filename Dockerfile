FROM python:3.10-slim-buster

LABEL maintainer="Salil Gautam <salil.gtm@gmail.com>"
LABEL description="Dockerfile for Assignment 7 of EMLOv3."

WORKDIR /workspace

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install -e .
