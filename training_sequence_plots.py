from chaotic_neural_networks import utils, networkA
import matplotlib.pyplot as plt

t_max = 2400
seed=2

# Target function: Sum of sinusoids
# network = networkA.NetworkA(f=utils.periodic)

# Target function: triangle wave
#network = networkA.NetworkA()

# Target function: complicated sum of sinusoids
#t_max = 12000
#network = networkA.NetworkA(f=utils.complicated_periodic)

# Target functions: sum of sinusoids AND triangle-wave
#network = networkA.NetworkA(nb_outputs=2, f=utils.per_tri)

# Target functions: sum of sinusoids AND triangle-wave AND cosine
network = networkA.NetworkA(nb_outputs=3, f=utils.per_tri_cos)

network.FORCE_sequence(t_max*3)

plt.show()