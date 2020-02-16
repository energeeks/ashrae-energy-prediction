FROM tiangolo/uwsgi-nginx-flask:python3.7
COPY . /app
WORKDIR /app
RUN apt-get update
RUN apt-get install -y --no-install-recommends cmake
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
