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
        self.f, self.dt, self.Δt, self.α, self.τ, self.nb_outputs = f, dt, Δt, α, τ, nb_outputs
        
        self.seed = seed
        np.random.seed(seed)

        std = 1./np.sqrt(p_GG*N_G)
        self.J_GG = std*sparse.random(N_G, N_G, density=p_GG, random_state=seed,
                                            data_rvs=np.random.randn).toarray()
        
        self.J_Gz = 2*np.random.rand(N_G, nb_outputs) - 1
        self._init_variables()

    def _init_variables(self):
        self.nb_train_steps = 0
        self.time_elapsed = 0
        
        self.w = np.zeros((self.N_G, self.nb_outputs))
        self.x = 0.5 * np.random.randn(self.N_G)
        self.r = np.tanh(self.x)
        self.z = 0.5 * np.random.randn(self.nb_outputs)
        self.P = np.eye(self.N_G)/self.α
        
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

    def step(self, train_test='train', store=True):
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
        self.time_elapsed += dt
        #current_time = int(train_test!='train')*self.time_elapsed['train']+self.time_elapsed[train_test]
        Δw = np.zeros(self.w.shape)

        if train_test == 'train':
            self.nb_train_steps+=1
            if self.nb_train_steps%(Δt//dt) == 0:
                # P update
                Pr = self.P.dot(self.r)
                self.P -= np.outer(Pr, self.r).dot(self.P)/(1+self.r.dot(Pr))

                # prediction error update
                self.e_minus = self.z - self.f(self.time_elapsed)

                # output weights update
                Δw = np.outer(self.P.dot(self.r), self.e_minus)
                self.w -= Δw
        
        if store:
            # Store w and \dot{w}'s norm
            self.w_list.append((self.time_elapsed, np.linalg.norm(self.w, axis=0)))
            self.w_dot_list.append((self.time_elapsed, np.linalg.norm(Δw/Δt, axis=0)))

            # Store z
            self.z_list[train_test].append((self.time_elapsed, self.z))
        
    def FORCE_figure(self, ts, fs, zs, xs, ws, neuron_indexes=None, already_split=True):
        lw_f, lw_z = 3.5, 1.5
        nb_split = 3 # 3 phases: pre-training, training, testing (post-training)
        if neuron_indexes is None:
            neuron_indexes = np.arange(len(xs))

        fig = plt.figure(figsize=(17, 13))
        gs = GridSpec(3, 3)

        if not already_split:
            fs_list, zs_list, xs_list, ws_list, ts_list = [np.array_split(y, nb_split, axis=len(y.shape)-1) 
                                                        for i,y in enumerate([fs, zs, xs, ws, ts])]
        else:
            fs_list, zs_list, xs_list, ws_list, ts_list = fs, zs, xs, ws, ts

        f_lim, x_lim, w_lim = [(min([i.min() for i in L]), max([i.max() for i in L])) for L in [fs, xs, ws]]

        Δ = 1.2*(x_lim[1] - x_lim[0])

        for i, (fs_sub, zs_sub, xs_sub, ws_sub, ts_sub, title) in enumerate(zip(fs_list, zs_list, xs_list, ws_list, ts_list, 
                                                                                ['Spontaneous Activity', 'Learning', 'Testing'])):
            if len(fs_sub.shape)==1:
                fs_sub = [fs_sub]
            if len(zs_sub.shape)==1:
                zs_sub = [zs_sub]
            if len(ws_sub.shape)==1:
                ws_sub = [ws_sub]
                
            # Plotting f and z
            ax_fz = fig.add_subplot(gs[0,i])
            ax_fz.set_title(title).set_fontsize('x-large')
            
            highest_color = .9 if len(fs_sub)>1 else .6
            ax_fz.set_prop_cycle(plt.cycler('color', plt.cm.Greens(np.linspace(0, highest_color, len(fs_sub)+1)[1:]))) 
            for j, f in enumerate(fs_sub):
                ax_fz.plot(ts_sub, f, lw=lw_f, label='$f_{'+str(j+1)+'}$' if len(fs_sub)>1 else '$f$')
            
            ax_fz.set_prop_cycle(plt.cycler('color', plt.cm.Reds(np.linspace(0, highest_color, len(zs_sub)+1)[1:])))
            for j, z in enumerate(zs_sub):
                ax_fz.plot(ts_sub, z, lw=lw_z, label='$z_{'+str(j+1)+'}$' if len(zs_sub)>1 else '$z$')
            
            ax_fz.legend(loc='best', fancybox=True, framealpha=0.7)
            ax_fz.set_ylim((f_lim[0]-.5, f_lim[1]+.5))
            pos = [['left'], [], ['right']]
            draw_axis_lines(ax_fz, pos[i])

            
            # Plotting the firing rates of sample neurons
            ax_x = fig.add_subplot(gs[1,i])
            draw_axis_lines(ax_x, [])
            add_collection_curves(ax_x, ts_sub, xs_sub.T, y_lim=(x_lim[0]-.1, x_lim[1]+.1), Δ=Δ,
                                labels=['Neuron ${}$'.format(i) for i in neuron_indexes] if i==0 else None)
            
            # Plotting the time-derivative of the readout weight vector
            ax_w = fig.add_subplot(gs[2,i])
            
            highest_color = .9 if len(ws_sub)>1 else .6
            ax_w.set_prop_cycle(plt.cycler('color', plt.cm.Oranges(np.linspace(0, highest_color, len(ws_sub)+1)[1:])))
            
            for j, w in enumerate(ws_sub):
                ax_w.plot(ts_sub, w, label='$|\dot{w_{'+str(j+1)+'}}|$' if len(ws_sub)>1 else '$|\dot{w}|$')
            
            ax_w.legend(loc='best', fancybox=True, framealpha=0.7)
            ax_w.set_ylim(*w_lim)
            draw_axis_lines(ax_w,['bottom'])
            ax_w.set_xlabel('Time (ms)')

        fig.suptitle("FORCE Training Sequence").set_fontsize('xx-large')
        self.fig = fig
        return fig
        
    def FORCE_sequence(self, t_max, number_neurons=5):
        """
        Returns a matplotlib figure of a full FORCE training sequence, showing the evolution of:
        
        - network ouput(s)
        - ``number_neurons`` neurons membrane potential
        - and the time-derivative of the readout vector \\\(\\\dot{\\\\textbf{w}}\\\) 
        
        before training (spontaneous activity), throughout training, and after training (test phase): each one of these phases lasts ``t_max/3``.

        See ``training_sequence_plots.py`` in the github repository for further examples.

        Examples
        --------
        >> network = networkA.NetworkA(f=utils.periodic); network.FORCE_sequence(600)
        Pre-training / Spontaneous activity...
        Training...
        > **Average Train Error:** [ 0.02805716]
        Testing...
        > **Average Test Error:** [ 2.50709125]
        """
        assert t_max%3==0
        # Reinitialization of the network
        self._init_variables()
        ts_list = np.array_split(np.arange(0, t_max, self.dt), 3)
        ts_pretrain, ts_train, ts_test = ts_list
        
        fs_list, zs_list, xs_list, ws_list = [], [], [], []

        # Random neuron indices: the neurons we will plot
        mask_random = np.arange(self.N_G)
        np.random.shuffle(mask_random)
        mask_random = mask_random[:number_neurons]

        #------------------------------------------------------------
        # Pre-training / Spontaneous activity
        print('Pre-training / Spontaneous activity...')
        xs_sublist, zs_sublist = [], []

        for _ in ts_pretrain:
            self.step(train_test='test', store=False)
            xs_sublist.append(self.r[mask_random])
            zs_sublist.append(self.z)

        xs_list.append(np.array(xs_sublist))
        zs_list.append(np.array(list(zip(*zs_sublist))))
        ws_list.append(np.zeros((self.nb_outputs, len(ts_pretrain))))
        f_time = self.f(ts_pretrain)
        if len(f_time.shape)==1:
            f_time = f_time.reshape([-1, 1])
        fs_list.append(np.array(list(zip(*f_time))))
     

        #------------------------------------------------------------
        # TRAINING Phase
        print('Training...')

        xs_sublist, zs_sublist = [], []

        for _ in ts_train:
            self.step()
            xs_sublist.append(self.r[mask_random])
        
        xs_list.append(np.array(xs_sublist))

        _, zs_sublist = zip(*self.z_list['train'])
        _, ws_sublist = zip(*self.w_dot_list)
        
        zs_list.append(np.array(list(zip(*zs_sublist))))
        ws_list.append(np.array(list(zip(*ws_sublist))))
        f_time = self.f(ts_train)
        if len(f_time.shape)==1:
            f_time = f_time.reshape([-1, 1])
        fs_list.append(np.array(list(zip(*f_time))))

        print('> **Average Train Error:** {}'.format(self.error()))

        #------------------------------------------------------------
        # TESTING phase
        print('Testing...')

        xs_sublist, zs_sublist = [], []

        for _ in ts_train:
            self.step(train_test='test')
            xs_sublist.append(self.r[mask_random])
        
        xs_list.append(np.array(xs_sublist))

        _, zs_sublist = zip(*self.z_list['test'])

        zs_list.append(np.array(list(zip(*zs_sublist))))
        ws_list.append(np.array(list(zip(*[ws_sublist[-1]]*len(ts_test)))))
        f_time = self.f(ts_test)
        if len(f_time.shape)==1:
            f_time = f_time.reshape([-1, 1])
        fs_list.append(np.array(list(zip(*f_time))))

        print('> **Average Test Error:** {}'.format(self.error(train_test='test')))

        self.FORCE_figure(ts_list, fs_list, zs_list, xs_list, ws_list,
                             neuron_indexes=mask_random).show()

    def _principal_components_figure(self, ts, fs_list, zs_list, xs_list, eigvals):
        lw_f, lw_z = 3.5, 1.5
        fig = plt.figure(figsize=(17, 13))
        gs = GridSpec(3, 1)

        f_lim, x_lim = [(i.min(), i.max()) for i in [fs_list, xs_list]]

        Δ = 1.2*(x_lim[1] - x_lim[0])

        if len(fs_list.shape)==1:
            fs_list = [fs_list]
        if len(zs_list.shape)==1:
            zs_list = [zs_list]
            
        # Plotting f and z_eig (projection on leading components)
        ax_fz = fig.add_subplot(gs[0])
        ax_fz.set_title('Projection onto the {} leading principal components'.format(len(xs_list))).set_fontsize('x-large')
        
        highest_color = .9 if len(fs_list)>1 else .6
        ax_fz.set_prop_cycle(plt.cycler('color', plt.cm.Greens(np.linspace(0, highest_color, len(fs_list)+1)[1:]))) 
        for j, f in enumerate(fs_list):
            ax_fz.plot(ts, f, lw=lw_f, label='$f_{'+str(j+1)+'}$' if len(fs_list)>1 else '$f$')
        
        ax_fz.set_prop_cycle(plt.cycler('color', plt.cm.Reds(np.linspace(0, highest_color, len(zs_list)+1)[1:])))
        for j, z in enumerate(zs_list):
            ax_fz.plot(ts, z, lw=lw_z, label='$z_{'+str(j+1)+'}$' if len(zs_list)>1 else '$z$')
        
        ax_fz.legend(loc='best', fancybox=True, framealpha=0.7)
        ax_fz.set_ylim((f_lim[0]-.5, f_lim[1]+.5))
        draw_axis_lines(ax_fz, ['left'])

        # Plotting the firing rates of sample neurons
        ax_x = fig.add_subplot(gs[1])
        draw_axis_lines(ax_x, ['bottom'])
        ax_x.set_xlabel('Time (ms)')
        ax_x.xaxis.set_label_coords(1.05, -0.025)
        add_collection_curves(ax_x, ts, xs_list, y_lim=(x_lim[0]-.1, x_lim[1]+.1), Δ=Δ,
                            labels=['PC ${}$'.format(len(xs_list)-i) for i in range(len(xs_list))])
        
        # Plotting the time-derivative of the readout weight vector
        ax_w = fig.add_subplot(gs[2])
        
        ax_w.semilogy(eigvals, color='blue', label='eigenvalues (logscale)')
        
        ax_w.legend(loc='best', fancybox=True, framealpha=0.7)
        draw_axis_lines(ax_w,['left', 'bottom'])
        ax_w.set_xlabel('Index')

        fig.suptitle("Principal Component Analysis of Network Activity").set_fontsize('xx-large')
        self.fig_eig = fig
        return fig


    def principal_components(self, t_max, nb_eig=8):
        #------------------------------------------------------------
        # TESTING phase
        print('Testing...')
        
        ts = np.arange(self.time_elapsed, self.time_elapsed+t_max, self.dt)

        xs_list = []

        for _ in ts:
            self.step(train_test='test')
            xs_list.append(self.x)

        xs_list = np.array(xs_list)
        self.eig_vec, self.proj, self.eig_val = PCA(xs_list, nb_eig=nb_eig, return_matrix=True, return_eigenvalues=True)

        f_time = self.f(ts)

        if len(f_time.shape)==1:
            f_time = f_time.reshape([-1, 1])
        fs_list = np.array(list(zip(*f_time)))
        
        # Projection over the leading principal components
        proj_w = self.w.T.dot(self.eig_vec)
        z_eig = np.tanh(self.proj).dot(proj_w.T)
        if len(z_eig.shape)==1:
            z_eig = z_eig.reshape([-1, 1])
        z_eig_list = np.array(list(zip(*z_eig)))

        self.eig_val = self.eig_val[::-1]

        self._principal_components_figure(ts, fs_list, z_eig_list, self.proj.T, self.eig_val).show()