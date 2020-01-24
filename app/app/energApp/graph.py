import plotly
import plotly.graph_objs as go
import json


def create_plot(meters, prediction):
    prediction = prediction.loc[prediction["building_id"] == 1, ]
    reading_0 = prediction["reading"].loc[prediction["meter"] == 0]
    reading_1 = prediction["reading"].loc[prediction["meter"] == 1]
    reading_2 = prediction["reading"].loc[prediction["meter"] == 2]
    reading_3 = prediction["reading"].loc[prediction["meter"] == 3]
    timestamp = prediction["timestamp"]

    data = []

    if meters[0]:
        data.append(go.Scatter(x=timestamp, y=reading_0,
                               mode='lines+markers',
                               name='Electricity',
                               line=dict(color='darkolivegreen'),
                               showlegend=False))
    if meters[1]:
        data.append(go.Scatter(x=timestamp, y=reading_1,
                               mode='lines+markers',
                               name='Chilled Water',
                               line=dict(color='aqua'),
                               showlegend=False))
    if meters[2]:
        data.append(go.Scatter(x=timestamp, y=reading_2,
                               mode='lines+markers',
                               name='Steam',
                               line=dict(color='aquamarine'),
                               showlegend=False))
    if meters[3]:
        data.append(go.Scatter(x=timestamp, y=reading_3,
                               mode='lines+markers',
                               name='Hot Water',
                               line=dict(color='darkturquoise'),
                               showlegend=False))
    layout = {
        "yaxis": {
            "title": "Energy Consumption"
        },
        "xaxis": {
            "tickformat": "%H:00 - %b %d",
            "tickmode": "auto",
            "dtick": "H3"
        }
    }
    return json.dumps({"data": data, "layout": layout}, cls=plotly.utils.PlotlyJSONEncoder)
