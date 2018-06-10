from .utils import *

#------------------------------------------------------------
# Parameters

N_G = 1000
p_GG = 0.1
p_z = 1.
g_Gz = 1.
α = 1.
g_GG = 1.5 # g_GG > 1 ⟹ chaos
τ = 10.

t_max = 2400

dt = 0.1
Δt = 10*dt

class NetworkA:
  """
  **Neural Architecture A (cf. figure 1):**

      - A recurrent generator network with firing rates \\\(\\\mathbf{r}\\\) driving a linear readout unit with output \\\(z\\\) through weights \\\(\\\mathbf{r}\\\) that are modified during training.
      - Feedback to the generator network is provided by the readout unit.

  """
  def __init__(self, N_G=N_G, p_GG=p_GG, g_GG=g_GG, g_Gz=g_Gz,
                f=triangle, dt=dt, Δt=Δt, α=α, τ=τ, seed=1):
      
      self.N_G, self.p_GG, self.g_GG, self.g_Gz = N_G, p_GG, g_GG, g_Gz
      self.f, self.dt, self.Δt, self.α, self.τ = f, dt, Δt, α, τ
      
      self.nb_train_steps, self.time_elapsed = 0, {}
      self.time_elapsed['train'], self.time_elapsed['test'] = 0, 0
      
      self.seed = seed
      np.random.seed(seed)

      std = 1./np.sqrt(p_GG*N_G)
      self.J_GG = std*scipy.sparse.random(N_G, N_G, density=p_GG, random_state=seed,
                                          data_rvs=np.random.randn).toarray()
      
      self.J_Gz = 2*np.random.rand(N_G) - 1
      
      self.w = np.zeros(N_G)
      self.x = 0.5 * np.random.randn(N_G)
      self.r = np.tanh(self.x)
      self.z = 0.5 * np.random.randn()
      self.P = np.eye(N_G)/α
      
      self.w_list = []
      self.w_dot_list = []
      
      self.z_list = {}
      self.z_list['train'], self.z_list['test'] = [], []

  def error(self, train_test='train'):
      """Compute the average training/testing error"""  
      z_arr = np.array(self.z_list[train_test])
      return np.mean(np.abs(z_arr[:,1] - self.f(z_arr[:,0])))

  def step(self, train_test='train'):
      """Execute one time step of length ``dt``"""
      
      dt, Δt, τ, g_GG, g_Gz = self.dt, self.Δt, self.τ, self.g_GG, self.g_Gz
      
      self.x = (1 - dt/τ)*self.x + g_GG*self.J_GG.dot(self.r)*dt/τ + g_Gz*self.J_Gz.dot(self.z)*dt/τ
      self.r = np.tanh(self.x)
      self.z = self.w.dot(self.r)
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
              Δw = self.e_minus*self.P.dot(self.r)
              self.w -= Δw

              # Store w and \dot{w}'s norm
              self.w_list.append((current_time, np.linalg.norm(self.w)))
              self.w_dot_list.append((current_time, np.linalg.norm(Δw/Δt)))
          
      # Store z
      self.z_list[train_test].append((current_time, self.z))