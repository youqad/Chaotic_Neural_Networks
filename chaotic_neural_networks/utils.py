import numpy as np
from scipy import signal, sparse

import matplotlib.pyplot as plt
from matplotlib import animation, rc
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec

from sklearn import preprocessing
from scipy.spatial import distance

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

def cos_fun(t, amp=3., freq=1/300):
    """Generates a cos function.
    """
    return amp*np.cos(np.pi*freq*t)
cos_fun = np.vectorize(cos_fun)

def complicated_periodic(t, amp=1., freq=1/300, seed=1):
    """Generates a complicated periodic function which a sum of 10 sinusoids.
    """
    np.random.seed(seed)
    amps = np.random.randint(1, 5, size=(6,))
    freqs = np.random.randint(1, 10, size=(6,))
    return sum(am*amp*np.sin(fr*np.pi*freq*t) for am, fr in zip(amps, freqs))
complicated_periodic = np.vectorize(complicated_periodic)


def both(f, g):
    """Generates  the function \\\(t ⟼ (f(t), g(t))\\\)
    """
    return (lambda t: np.array([f(t), g(t)]) if isinstance(t, float) else np.array(list(zip(f(t), g(t)))))

per_tri = both(periodic, triangle)

def triple(f, g, h):
    """Generates  the function \\\(t ⟼ (f(t), g(t), h(t))\\\)
    """
    return (lambda t: np.array([f(t), g(t), h(t)]) if isinstance(t, float) else np.array(list(zip(f(t), g(t), h(t)))))

per_tri_cos = triple(periodic, triangle, cos_fun)

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

#------------------------------------------------------------
# Dimension reduction functions


#------------------------------------------------------------
# PCA to compute the degrees of freedom

def PCA(data, return_matrix=False, return_eigenvalues=False):
    """                                                                                       
    Principal Component Analysis (PCA) to compute the leading principal components.

    Parameters                                                                                
    ----------                                                                                
    data : (n, k) array                                                                          
        Data points matrix (data points = row vectors in the matrix)
    return_matrix : bool
        If True, returns the matrix of the data points projection on the eigenvectors
    return_eigenvalues : bool, optional
        Returns the eigenvalues.                                                       
                                                                                               
    Returns                                   
    -------                                                                                
    nb : int                                                                        
        Number of non-zero eigenvalues of the covariance matrix,
        thought of as the number of leading principal components.
        The eigenvalues thereof into two classes ("non-zero" and "zero" eigenvalues) \\\(C_1\\\) and \\\(C_2\\\),
        distinguished as follows: 
        > each \\\(λ ∈ C_1\\\) has a size closer to other the \\\(λ\\\)'s of \\\(C_1\\\) 
        than the ones of \\\(C_2\\\), and conversely. 
        The boundary between \\\(C_1\\\) and \\\(C_2\\\) corresponds to the largest ratio \\\(λ_{i+1}/λ_i\\\),
        where the \\\(λ_i\\\) are in decreasing order.
    Proj : (N_G, t_max) array                                                                          
        If return_matrix == True: Projection of the data points on the eigenvectors                                                     
    """

    # Covariance matrix
    cov_matrix = np.cov(preprocessing.scale(data))

    # Diagonalization of the covariance matrix
    eig_val, eig_vec = np.linalg.eigh(cov_matrix)
    
    # Compute the boundary index separating zero eigenvalues to non-zero ones
    ratios = [eig_val[i+1]/eig_val[i] for i in range(len(eig_val)-1)]
    max_ratio = np.argmax(ratios)
    
    if return_matrix or return_eigenvalues:
        if return_matrix:
            # Projection of the data points over the eigenvectors 
            Proj = data.dot(eig_vec[:,max_ratio+1:])
        if return_matrix and return_eigenvalues:
            return eig_vec[max_ratio+1:], Proj, eig_val
        elif return_matrix:
            return eig_vec[max_ratio+1:], Proj
        else:
            return eig_vec[max_ratio+1:], eig_val

    return eig_vec[max_ratio+1:]

#------------------------------------------------------------
# Classical multidimensional scaling (MDS)

def MDS(data, return_matrix=False, return_eigenvalues=False):
    """                                                                                       
    Classical multidimensional scaling (MDS) to compute the leading principal components
    cf. https://en.wikipedia.org/wiki/Multidimensional_scaling#Classical_multidimensional_scaling
                                                                                               
    Parameters                                                                                
    ----------                                                                                
    data : (n, k) array                                                                          
        Data points matrix (data points = row vectors in the matrix)
    return_matrix : bool
        If True, returns the coordinate matrix
    return_eigenvalues : bool, optional
        Returns the eigenvalues (``False`` by default).                                                
                                                                                               
    Returns                                   
    -------                                                                               
    nb : int                                                                        
        Number of "non-zero" eigenvalues of $$B = X X^T$$ (where \\\(X\\\) is the coordinate matrix), 
        thought of as the number of leading principal components.
        The eigenvalues fall into two classes (non-zero and zero eigenvalues): cf. ``utils.PCA`` to see how they are distinguished.
    X : (n, dim_rigid_group) array                                                                          
        If return_matrix == True: Coordinate matrix.                                                    
    """
    
    # Number of points                                                                        
    n = len(data)

    # 1. Squared Symmetric pairwise distance matrix.
    D_sq = distance.cdist(data, data, 'euclidean')**2
 
    # Centering matrix                                                                        
    J = np.eye(n)-np.ones((n, n))/n
 
    # 2. Double centering: B = X X^T                                                                                    
    B = -J.dot(D_sq).dot(J)/2
 
    # 3. Diagonalize
    eig_val, eig_vec = np.linalg.eigh(B)

    # Compute the boundary index separating zero eigenvalues to non-zero ones
    ratios = [eig_val[i+1]/eig_val[i] for i in range(len(eig_val)-1)]
    max_ratio = np.argmax(ratios)
    
    if return_matrix or return_eigenvalues:
        if return_matrix:
            # Compute the coordinate matrix
            Λ_sqrt  = np.diag(np.sqrt(eig_val[max_ratio+1:]))
            E  = eig_vec[max_ratio+1:]
            X  = E.dot(Λ_sqrt)
        if return_matrix and return_eigenvalues:
            return eig_vec[max_ratio+1:], X, eig_val
        elif return_matrix:
            return eig_vec[max_ratio+1:], X
        else:
            return eig_vec[max_ratio+1:], eig_val

    return eig_vec[max_ratio+1:]

#------------------------------------------------------------
# Dictionary to access the dimension reduction functions in the classes later

dim_reduction_dict = {'PCA': PCA, 'MDS': MDS}