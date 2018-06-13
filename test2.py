from chaotic_neural_networks import utils, networkA
import matplotlib.pyplot as plt

t_max = 7200
seed=2

# Target function: Sum of sinusoids
# network = networkA.NetworkA(f=utils.periodic)

# Target function: triangle wave
#network = networkA.NetworkA()

# Target function: complicated sum of sinusoids
network = networkA.NetworkA(f=utils.complicated_periodic)

network.FORCE_sequence(t_max*3)

plt.show()