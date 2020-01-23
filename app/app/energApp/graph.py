import plotly
import plotly.graph_objs as go

import pandas as pd
import numpy as np
import json


def create_plot(meter0, meter1, meter2, meter3):
    N = 100
    random_x = np.linspace(0, 1, N)
    random_y0 = np.random.randn(N) + 5
    random_y1 = np.random.randn(N)
    random_y2 = np.random.randn(N) - 5

    data = []

    if meter0:
        data.append(go.Scatter(x=random_x, y=random_y0,
                               mode='lines+markers',
                               name='Electricity',
                               line=dict(color='darkolivegreen'),
                               showlegend=False))
    if meter1:
        data.append(go.Scatter(x=random_x, y=random_y1,
                               mode='lines+markers',
                               name='Chilled Water',
                               line=dict(color='aqua'),
                               showlegend=False))
    if meter2:
        data.append(go.Scatter(x=random_x, y=random_y2,
                               mode='lines+markers',
                               name='Steam',
                               line=dict(color='aquamarine'),
                               showlegend=False))
    if meter3:
        data.append(go.Scatter(x=random_x, y=random_y0 + 7,
                               mode='lines+markers',
                               name='Hot Water',
                               line=dict(color='darkturquoise'),
                               showlegend=False))

    return json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
