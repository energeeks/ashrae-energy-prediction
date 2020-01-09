from flask import render_template, request
from energApp import app


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/something')
def do_something():
    return render_template('something.html')