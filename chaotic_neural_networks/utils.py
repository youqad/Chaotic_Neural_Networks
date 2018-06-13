import numpy as np
from scipy import signal, sparse

import matplotlib.pyplot as plt
from matplotlib import animation, rc
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec

#------------------------------------------------------------
# Target functions

def periodic(t, amp=3., freq=1/300):
    """Generates a periodic function which a sum of 4 sinusoids.
    """
    return amp*np.sin(np.pi*freq*t) + (amp/2) * np.sin(2*np.pi*freq*t) + (amp/3) * np.sin(3*np.pi*freq*t) + (amp/4) * np.sin(4*np.pi*freq*t)
periodic = np.vectorize(periodic)

def triangle(t, freq=1/600, amp=3):
    """Generates a triangle-wave function.
    """
    return amp*signal.sawtooth(2*np.pi*freq*t, 0.5)
triangle = np.vectorize(triangle)


#------------------------------------------------------------
# General utility functions

def add_collection_curves(ax, ts, data, labels=None, color='indigo',
                         y_lim=None, starting_points=None, Δ=None):
    """
    Adds a collection of curves a matplotlib ax.
    """
    # the plot limits need to be set (no autoscale!)
    ax.set_xlim(np.min(ts), np.max(ts))
    min_data, max_data = data.min(), data.max()
    
    if Δ is None:
        Δ = 0.7*(max_data - min_data)
    
    if y_lim is None:
        ax.set_ylim(min_data, max_data+Δ*(len(data)-1))
    else:
        ax.set_ylim(y_lim[0], y_lim[1]+Δ*(len(data)-1))
        
    curves = [np.column_stack((ts, curve)) for curve in data]
    ticks_positions = Δ*np.arange(len(data))
        
    offsets = np.column_stack((np.zeros(len(data)), ticks_positions))

    ax.add_collection(LineCollection(curves, offsets=offsets, colors=color))
    
    if labels is not None:
        ax.set_yticks(ticks_positions+data[:,0])
        ax.set_yticklabels(labels)
        ax.tick_params(axis='y', colors=color)
    
def draw_axis_lines(ax, positions):
    if 'right' in positions or 'left' in positions:
        ax.yaxis.set_ticks_position('left') if 'left' in positions else ax.yaxis.set_ticks_position('right')
    else:
        ax.yaxis.set_ticks([])
        
    ax.xaxis.set_ticks_position('bottom') if 'bottom' in positions else ax.xaxis.set_ticks([])
    
    for pos in ax.spines.keys():
        ax.spines[pos].set_position(('outward',7)) if pos in positions else ax.spines[pos].set_color('none')