import plotly
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go

import matplotlib.pyplot as plt
from matplotlib import animation, rc

from chaotic_neural_networks import utils, networkA

t_max = 500
seed = 2

network = networkA.NetworkA(seed=seed, f=utils.periodic)

network.FORCE_sequence(2400*3)

plt.show()