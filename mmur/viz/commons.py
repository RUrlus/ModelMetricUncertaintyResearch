import matplotlib.pyplot as plt


def _set_plot_style():
    plt.style.use('ggplot')
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['figure.max_open_warning'] = 0
    return [i['color'] for i in plt.rcParams['axes.prop_cycle']]  # type: ignore
