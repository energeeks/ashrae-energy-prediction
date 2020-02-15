import pandas as pd
import numpy as np
from lightgbm import Booster
from flask import Flask, jsonify, request, current_app

app = Flask(__name__)
app.config['MODEL'] = Booster(model_file="model.txt")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    Takes a request with input data as payload and returns the predicted values
    as json.
    :return: predicted energy consumption
    """
    data = {"success": False}

    df = request.json
    if df is not None:

        df = pd.read_json(df)
        df["primary_use"] = pd.Categorical(df["primary_use"])
        df["meter"] = pd.Categorical(df["meter"])
        df["hour"] = pd.Categorical(df["hour"])
        df["weekday"] = pd.Categorical(df["weekday"])

        model = current_app.config['MODEL']
        data = {"success": True,
                "prediction": np.expm1(model.predict(df)).tolist()}

    return jsonify(data)
