from chaotic_neural_networks import utils, networkA

t_max = 500
seed = 2

# Target function: Sum of sinusoids
network1 = networkA.NetworkA(f=utils.periodic)
network1.FORCE_sequence(2400*3)

# Target function: triangle wave
network2 = networkA.NetworkA()
network2.FORCE_sequence(2400*3)

plt.show()