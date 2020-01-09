from flask import Flask

app = Flask(__name__)

from energApp import routes
