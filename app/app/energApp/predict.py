import pandas as pd


def predict_energy_consumption(model, buildings, forecasts):
    return prepare_data(buildings, forecasts)


def prepare_data(buildings, forecasts):
    buildings["tmp"] = range(len(forecasts))
    for i in range(len(forecasts)):
        forecasts[i]["tmp"] = i
        forecasts[i] = pd.merge(forecasts[i], buildings, how="left", on=["tmp"])
        forecasts[i] = forecasts[i].drop("tmp", axis=1)

    return forecasts[0]
