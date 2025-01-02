import matplotlib.pyplot as plt
import numpy as np

def list_plot(datasets, aspect_ratio=1):
    plt.figure()
    if hasattr(datasets[0][0], '__len__'):
        for data in datasets:
            x, y  = zip(*data)
            plt.plot(x, y, marker=None)
    else:
        x, y  = zip(*datasets)
        plt.plot(x, y, marker=None)
    plt.gca().set_aspect(aspect_ratio)
    plt.show()
    plt.close()