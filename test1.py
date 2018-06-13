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

ts_train, ts_test = np.arange(0, t_max, network.dt), np.arange(t_max, 2*t_max, network.dt)
lw_f, lw_z = 3, 1.5

# TRAIN Phase
f_train = network.f(ts_train)

plt.figure(figsize=(15, 3*4))
j = 1

for i, t in enumerate(ts_train):
    
    network.step()

    if (i+1) % int(t_max//(4*network.dt)) == 0:
        #display(Markdown('## Time: {:.1f} ms'.format(t)))
        print('## Time: {:.1f} ms'.format(t))
                
        plt.subplot('42{}'.format(j))
        plt.plot(ts_train, f_train, lw=lw_f, color='green')
        
        plt.plot(*zip(*network.z_list['train']), lw=lw_z, color='red')
        plt.title('Training')
        plt.legend(['$f$', '$z$'])
        plt.ylim((-3.5, 3.5))
        
        plt.xlabel('Time (ms)')
        plt.ylabel('$f$ and $z$')

        plt.subplot('42{}'.format(j+1))
        plt.plot(*zip(*network.w_dot_list))
        plt.xlabel('Time (ms)')
        plt.ylabel('$|\dot{w}|$')
        plt.legend(['$|\dot{w}|$'])
        
        #plt.show()
        
        #display(Markdown('__________________'))
        print('__________________')
        j+=2


#display(Markdown('> **Training Average Error:** {}'.format(network.error())))
print('> **Training Average Error:** {}'.format(network.error()))

plt.tight_layout()
plt.show()

# TEST phase

f_test = network.f(ts_test)

for t in ts_test:
    network.step(train_test='test')

#display(Markdown('> **Testing Average Error:** {}'.format(network.error(train_test='test'))))
print('> **Testing Average Error:** {}'.format(network.error(train_test='test')))


plt.figure(figsize=(17, 5))
plt.subplot('121')

plt.plot(ts_train, f_train, lw=lw_f, color='green')
plt.plot(*zip(*network.z_list['train']), lw=lw_z, color='red')

plt.title('Training phase')
plt.xlabel('Time (ms)')
plt.ylabel('$f$ and $z$')
plt.legend(['$f$', '$z$'])


plt.subplot('122')
plt.plot(ts_test, f_test, lw=lw_f, color='green')
plt.plot(*zip(*network.z_list['test']), lw=lw_z, color='red')

plt.title('Testing phase')
plt.xlabel('Time (ms)')
plt.ylabel('$f$ and $z$')
plt.legend(['$f$', '$z$'])

plt.show()