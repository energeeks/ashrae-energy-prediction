FROM tiangolo/uwsgi-nginx-flask:python3.7

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        cmake && \
    pip install --upgrade pip && \
    pip install \
        flask-sqlalchemy \
        psycopg2 \
        flask-login \
        flask-bootstrap \
        flask-wtf \
        wtforms \
        flask-admin \
        requests \
	requests-cache \
        pandas \
        lightgbm \
        plotly \
        meteocalc

COPY . /app
