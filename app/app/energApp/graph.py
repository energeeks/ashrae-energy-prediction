import plotly
import plotly.graph_objs as go
import json


def create_plot(elements, prediction):
    """
    Creates a plot with the provided prediction data and returns it as json
    for further processing.
    :param elements: List containing boolean values, whether some graph
    elements should be displayed. [0-3} = meters, [4] = air temperature
    :param prediction: Data Frame containing the predicted readings.
    :return: Plotly plot as json.
    """
    air_temperature = prediction["air_temperature"].loc[prediction["meter"] == 0]
    reading_0 = prediction["reading"].loc[prediction["meter"] == 0]
    reading_1 = prediction["reading"].loc[prediction["meter"] == 1]
    reading_2 = prediction["reading"].loc[prediction["meter"] == 2]
    reading_3 = prediction["reading"].loc[prediction["meter"] == 3]
    timestamp = prediction["timestamp"].loc[prediction["meter"] == 0]

    data = []

    if elements[0]:
        data.append(go.Scatter(x=timestamp, y=reading_0,
                               mode='lines+markers',
                               name='Electricity',
                               line=dict(color='darkolivegreen'),
                               showlegend=False))
    if elements[1]:
        data.append(go.Scatter(x=timestamp, y=reading_1,
                               mode='lines+markers',
                               name='Chilled Water',
                               line=dict(color='aqua'),
                               showlegend=False))
    if elements[2]:
        data.append(go.Scatter(x=timestamp, y=reading_2,
                               mode='lines+markers',
                               name='Steam',
                               line=dict(color='aquamarine'),
                               showlegend=False))
    if elements[3]:
        data.append(go.Scatter(x=timestamp, y=reading_3,
                               mode='lines+markers',
                               name='Hot Water',
                               line=dict(color='darkturquoise'),
                               showlegend=False))

    if elements[4]:
        data.append(go.Scatter(x=timestamp, y=air_temperature,
                               mode='lines',
                               name='Air Temperature',
                               opacity=0.3,
                               yaxis='y2',
                               line=dict(color='grey'),
                               showlegend=False))
    layout = {
        "height": 230,
        "margin": go.layout.Margin(
            t=20,
            b=30
        ),
        "paper_bgcolor": "transparent",
        "yaxis": {
            "linecolor": "#001f07",
            "mirror": True,
            "title": "Energy Consumption"
        },
        "yaxis2": {
            "title": "Air Temperature",
            "overlaying": "y",
            "side": "right"
        },
        "xaxis": {
            "linecolor": "#001f07",
            "mirror": True,
            "tickformat": "%H:00 - %b %d",
            "tickmode": "auto",
            "dtick": "H3"
        }
    }
    return json.dumps({"data": data, "layout": layout}, cls=plotly.utils.PlotlyJSONEncoder)
