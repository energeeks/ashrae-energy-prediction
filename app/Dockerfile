FROM tiangolo/uwsgi-nginx-flask:python3.7

RUN pip install --upgrade pip && \
    pip install flask-sqlalchemy flask-login \
    flask-bootstrap flask-wtf wtforms flask-admin \
    requests pandas

COPY ./app /app
