from chaotic_neural_networks import utils, networkA
import matplotlib.pyplot as plt
import numpy as np

t_max = 2400

# Target function: Triangle-wave
#network = networkA.NetworkA()

# Target function: Sum of sinusoids
#network = networkA.NetworkA(f=utils.periodic)

# Target functions: sum of sinusoids AND triangle-wave
#network = networkA.NetworkA(nb_outputs=2, f=utils.per_tri)

# Target functions: sum of sinusoids AND triangle-wave AND cosine
network = networkA.NetworkA(nb_outputs=3, f=utils.per_tri_cos)

for _ in np.arange(0, t_max, network.dt):
    network.step(train_test='train')

network.principal_components(t_max)
plt.show()