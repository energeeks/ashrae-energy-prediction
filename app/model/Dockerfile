FROM python:3.7-slim
COPY . /model
WORKDIR /model
RUN apt-get update
RUN apt-get install -y libgomp1
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
