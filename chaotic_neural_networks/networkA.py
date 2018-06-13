from .utils import *

#------------------------------------------------------------
# Parameters: default values

N_G = 1000
"""
Generator Network: Number of neurons
"""

p_GG = 0.1
"""
Generator Network: **sparseness parameter** of the connection matrix.
Each coefficient thereof is set to \\\(0\\\) with probability \\\(1-p_{GG}\\\).
"""

p_z = 1.
"""
Sparseness parameter of the readout: a random fraction \\\(1-p_z\\\) of the components
 of \\\(\\\mathbf{w}\\\) are held to \\\(0\\\).
"""

g_Gz = 1.
"""
Scaling factor of  the feedback loop:
Increasing the feedback connections result in the network chaotic activity allowing the learning process.
"""

α = 1.
"""
Inverse Learning rate parameter: \\\(P\\\), the estimate of the inverse of the network rates correlation matrix plus a regularization term,
is initialized as $$P(0) = \\\\frac 1 α \\\mathbf{I}$$

So a sensible value of \\\(α\\\)
- depends on the target function
- ought to be chosen such that \\\(α << N\\\)

If 
- \\\(α\\\) is too small ⟹ the learning is so fast it can cause unstability issues.
- \\\(α\\\) is too large ⟹ the learning is so slow it may fail
"""


g_GG = 1.5 # g_GG > 1 ⟹ chaos
"""
Scaling factor of the connection synaptic strength matrix of the generator network.
$$g_{GG} > 1 ⟹ \\\text{chaotic behavior}$$
"""

τ = 10.
"""
Time constant of the units dynamics.
"""

dt = 0.1
"""
Network integration time step.
"""

Δt = 10*dt
"""
Time span between modifications of the readout weights: \\\(Δt ≥ dt\\\)
"""

class NetworkA:
    """
    Neural Architecture A:

        .. image:: http://younesse.net/images/Neuromodeling/networkA.png
          :width: 50%
          :align: center

        - A recurrent generator network with firing rates \\\(\\\mathbf{r}\\\) driving a linear readout unit with output \\\(z\\\) through weights \\\(\\\mathbf{r}\\\) that are modified during training.
        - Feedback to the generator network is provided by the readout unit.
    """
    def __init__(self, N_G=N_G, p_GG=p_GG, g_GG=g_GG, g_Gz=g_Gz,
                f=triangle, dt=dt, Δt=Δt, α=α, τ=τ, seed=1, nb_outputs=1):
        
        self.N_G, self.p_GG, self.g_GG, self.g_Gz = N_G, p_GG, g_GG, g_Gz
        self.f, self.dt, self.Δt, self.α, self.τ = f, dt, Δt, α, τ
        
        self.nb_train_steps, self.time_elapsed = 0, {}
        self.time_elapsed['train'], self.time_elapsed['test'] = 0, 0
        
        self.seed = seed
        np.random.seed(seed)

        std = 1./np.sqrt(p_GG*N_G)
        self.J_GG = std*sparse.random(N_G, N_G, density=p_GG, random_state=seed,
                                            data_rvs=np.random.randn).toarray()
        
        self.J_Gz = 2*np.random.rand(N_G, nb_outputs) - 1
        
        self.w = np.zeros((N_G, nb_outputs))
        self.x = 0.5 * np.random.randn(N_G)
        self.r = np.tanh(self.x)
        self.z = 0.5 * np.random.randn(nb_outputs)
        self.P = np.eye(N_G)/α
        
        # Storing the values of w and its time-derivative
        self.w_list = []
        self.w_dot_list = []
        
        self.z_list = {}
        self.z_list['train'], self.z_list['test'] = [], []

    def error(self, train_test='train'):
        """Compute the average training/testing error.
        
        Parameters
        ----------
        train_test : {'PCA', 'MDA'}, optional
            Choice of the error to compute: train or test.

        Returns
        -------
        (len(self.z_list),) array
            Train of test error, depending on `train_test`
        """  
        z_time, z_val = map(np.array, zip(*self.z_list[train_test]))
        f_time = self.f(z_time)

        if len(z_val.shape) > len(f_time.shape)==1:
            f_time = f_time.reshape([-1, 1])

        return np.mean(np.abs(z_val-f_time), axis=0)

    def step(self, train_test='train'):
        """Execute one time step of length ``dt`` of the network dynamics.
        
        Parameters
        ----------
        train_test : {'PCA', 'MDA'}, optional
            Learning phase (when \\\(P\\\) and the readout unit are updated) or test phase
            (no such update)

        Examples
        --------
        >>> from chaotic_neural_networks import networkA; net = networkA.NetworkA()
        >>> for _ in np.arange(0, 1200, net.dt):
        ...     net.step()
        >>> net.error()
        0.015584795078446064
        """
        
        dt, Δt, τ, g_GG, g_Gz = self.dt, self.Δt, self.τ, self.g_GG, self.g_Gz
        
        self.x = (1 - dt/τ)*self.x + g_GG*self.J_GG.dot(self.r)*dt/τ + g_Gz*self.J_Gz.dot(self.z)*dt/τ
        self.r = np.tanh(self.x)
        self.z = self.w.T.dot(self.r).flatten()
        self.time_elapsed[train_test] += dt
        current_time = int(train_test!='train')*self.time_elapsed['train']+self.time_elapsed[train_test]
        
        if train_test == 'train':
            self.nb_train_steps+=1
            if self.nb_train_steps%(Δt//dt) == 0:
                # P update
                Pr = self.P.dot(self.r)
                self.P -= np.outer(Pr, self.r).dot(self.P)/(1+self.r.dot(Pr))

                # prediction error update
                self.e_minus = self.z - self.f(current_time)

                # output weights update
                Δw = np.outer(self.P.dot(self.r), self.e_minus)
                self.w -= Δw

                # Store w and \dot{w}'s norm
                self.w_list.append((current_time, np.linalg.norm(self.w)))
                self.w_dot_list.append((current_time, np.linalg.norm(Δw/Δt)))
            
        # Store z
        self.z_list[train_test].append((current_time, self.z))