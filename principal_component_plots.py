from chaotic_neural_networks import utils, networkA
import matplotlib.pyplot as plt
import numpy as np

t_max = 2400

# Target function: Sum of sinusoids
network = networkA.NetworkA()

for _ in np.arange(0, t_max, network.dt):
    network.step(train_test='train')

network.principal_components(t_max)
plt.show()