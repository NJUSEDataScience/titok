import plotly.plotly as plt
import plotly.offline as pltoff
from plotly.graph_objs import *

def line_plots(name):
    dataset = {'x': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    'y': [5, 4, 1, 3, 11, 2, 6, 7, 19, 20],
    'z': [12, 9, 0, 0, 3, 25, 8, 17, 22, 5]}

    data_g = []

    tr_x = Scatter(
        x=dataset['x'],
        y=dataset['y'],
        name='y'
    )
    data_g.append(tr_x)

    tr_z = Scatter(
        x=dataset['x'],
        y=dataset['z'],
        name='z'
    )
    data_g.append(tr_z)

    layout = Layout(title="line plots", xaxis={'title': 'x'}, yaxis={'title': 'value'})
    fig = Figure(data=data_g, layout=layout)
    pltoff.plot(fig, filename=name)

line_plots('test')
