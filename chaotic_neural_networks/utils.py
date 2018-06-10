import numpy as np
from scipy import signal

#------------------------------------------------------------
# Target functions

def periodic(t, amp=3., freq=1/600):
    """
    Generates a periodic function
    """
    return amp*np.sin(np.pi*freq*t) + (amp/2) * np.sin(2*np.pi*freq*t) +\
    (amp/3) * np.sin(3*np.pi*freq*t) + (amp/4) * np.sin(4*np.pi*freq*t)
periodic = np.vectorize(periodic)

def triangle(t, freq=1/600, amp=3):
    """
    Generates a triangle-wave function
    """
    return amp*signal.sawtooth(2*np.pi*freq*t, 0.5)
triangle = np.vectorize(triangle)