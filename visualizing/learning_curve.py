import logging
import numpy as np

from matplotlib import pyplot as plt
import plotly.graph_objs as go
from chart_studio import plotly as py
from sklearn.model_selection import train_test_split

from training.cross_validation import evaluate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_learning_curve(x_train, y_train, subsets=20, mmr=None, cv=5, tool='matplotlib'):
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)

    subset_sizes = np.exp(np.linspace(3, np.log(len(y_train)), subsets)).astype(int)

    results_list = [[], []]

    for subset_size in subset_sizes:
        logger.info('Performing cross validation on subset_size %d', subset_size)
        _, _, cv_score, roc_auc, _ = evaluate([x_train[:subset_size], y_train[:subset_size]],
                                              [x_test, y_test], cv=cv)

        results_list[0].append(1 - cv_score)
        results_list[1].append(1 - roc_auc)

    if tool == 'matplotlib':
        _plot_matplotlib(subset_sizes, results_list, mmr)
    else:
        _plot_plotly(subset_sizes, results_list, mmr)


def _plot_matplotlib(subset_sizes, data_list, mmr):
    """ Plots learning curve using matplotlib backend.
    Args:
        subset_sizes: list of dataset sizes on which the evaluation was done
        data_list: list of ROC AUC scores corresponding to subset_sizes
        mmr: what MMR the data is taken from
    """
    plt.plot(subset_sizes, data_list[0], lw=2)
    plt.plot(subset_sizes, data_list[1], lw=2)

    plt.legend(['Cross validation error', 'Test error'])
    plt.xscale('log')
    plt.xlabel('Размер набора данных')
    plt.ylabel('Ошибки')

    if mmr:
        plt.title('График кривая обучения для %d MMR' % mmr)
    else:
        plt.title('График кривая обучения')

    plt.show()


def _plot_plotly(subset_sizes, data_list, mmr):
    """ Plots learning curve using plotly backend.
    Args:
        subset_sizes: list of dataset sizes on which the evaluation was done
        data_list: list of ROC AUC scores corresponding to subset_sizes
        mmr: what MMR the data is taken from
    """
    if mmr:
        title = 'Кривая обучаемости для %d MMR' % mmr
    else:
        title = 'Кривая обучаемости'

    trace0 = go.Scatter(
        x=subset_sizes,
        y=data_list[0],
        name='Ошибки перекрестной проверки'
    )
    trace1 = go.Scatter(
        x=subset_sizes,
        y=data_list[1],
        name='Ошибки теста'
    )
    data = go.Data([trace0, trace1])

    layout = go.Layout(
        title=title,

        xaxis=dict(
            title='Размер набора данных (logspace)',
            type='log',
            autorange=True,
            titlefont=dict(
                family='Courier New, monospace',
                size=15,
                color='#7f7f7f'
            )
        ),
        yaxis=dict(
            title='Ошибки',
            titlefont=dict(
                family='Courier New, monospace',
                size=15,
                color='#7f7f7f'
            )
        )
    )
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='learning_curve_%dMMR' % mmr)