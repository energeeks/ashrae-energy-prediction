FROM tiangolo/uwsgi-nginx-flask:python3.7

RUN pip install --upgrade pip && \
    pip install flask-sqlalchemy flask-login wtforms

COPY ./app /app
