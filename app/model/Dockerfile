FROM python:3.7-slim
RUN apt-get update && apt-get install -y libgomp1 && \
    pip install Flask pandas lightgbm numpy
COPY . /model
WORKDIR /model