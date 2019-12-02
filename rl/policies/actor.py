import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.distributions import DiagonalGaussian

from torch import sqrt
from torch.distributions import Normal

from rl.policies.base import Net

class Actor(Net):
  def __init__(self):
    super(Actor, self).__init__()

  def forward(self):
    raise NotImplementedError

  def get_action(self):
    raise NotImplementedError

class Linear_Actor(Actor):
  def __init__(self, state_dim, action_dim, hidden_size=32):
    super(Linear_Actor, self).__init__()

    self.l1 = nn.Linear(state_dim, hidden_size)
    self.l2 = nn.Linear(hidden_size, action_dim)

    self.action_dim = action_dim

    for p in self.parameters():
      p.data = torch.zeros(p.shape)

  def forward(self, state):
    a = self.l1(state)
    a = self.l2(a)
    self.action = a
    return a

  def get_action(self):
    return self.action

# Actor network for gaussian mlp
class GaussianMLP_Actor(Actor):
  def __init__(self, state_dim, action_dim, hidden_size=256, hidden_layers=2, env_name='NOT SET', nonlinearity=torch.tanh, init_std=1, learn_std=True, bounded=False, normc_init=True, obs_std=None, obs_mean=None):
    super(GaussianMLP_Actor, self).__init__()

    self.actor_layers = nn.ModuleList()
    self.actor_layers += [nn.Linear(state_dim, hidden_size)]
    for _ in range(hidden_layers-1):
        self.actor_layers += [nn.Linear(hidden_size, hidden_size)]
    self.network_out = nn.Linear(hidden_size, action_dim)

    self.dist = DiagonalGaussian(action_dim, init_std, learn_std)

    self.action = None
    self.action_dim = action_dim
    self.env_name = env_name
    self.nonlinearity = nonlinearity

    self.obs_std = obs_std
    self.obs_mean = obs_mean

    # weight initialization scheme used in PPO paper experiments
    self.normc_init = normc_init

    self.bounded = bounded

    self.init_parameters()
    self.train()

  def init_parameters(self):
    if self.normc_init:
        self.apply(normc_fn)

        if self.dist.__class__.__name__ == "DiagGaussian":
            self.network_out.weight.data.mul_(0.01)

  def forward(self, inputs):
    if self.training == False:
        inputs = (inputs - self.obs_mean) / self.obs_std

    x = inputs
    for l in self.actor_layers:
        x = self.nonlinearity(l(x))
    x = self.network_out(x)

    if self.bounded:
        mean = torch.tanh(x) 
    else:
        mean = x

    self.action = mean
    return mean

  def get_action(self):
    return self.action

  def act(self, inputs, deterministic=False):
    action = self.dist.sample(self(inputs), deterministic=deterministic)
    return action.detach()

  def evaluate(self, inputs):
    x = self(inputs)
    return self.dist.evaluate(x)

# TODO: make dedicated method for computing deterministic action with minimal other work
# TODO: remove redundancies with this class and GaussianMLP, factor out the network head to rl.policies.distributions
# Modification of above for Soft Actor Critic
LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPSILON = 1e-6
# Actor network for soft actor critic
class SAC_Gaussian_Actor(Actor):
  def __init__(self, state_dim, action_dim, hidden_size=256, hidden_layers=2, env_name='NOT SET', nonlinearity=F.relu, normc_init=True, action_space=None):
    super(SAC_Gaussian_Actor, self).__init__()

    self.actor_layers = nn.ModuleList()
    self.actor_layers += [nn.Linear(state_dim, hidden_size)]
    for _ in range(hidden_layers-1):
        self.actor_layers += [nn.Linear(hidden_size, hidden_size)]

    # 2 network heads also computed as fully connected layers
    self.mean_head = nn.Linear(hidden_size, action_dim)
    self.logstd_head = nn.Linear(hidden_size, action_dim)

    self.logstd_min = LOG_STD_MIN
    self.logstd_max = LOG_STD_MAX
    self.epsilon = EPSILON

    self.action = None
    self.action_dim = action_dim
    self.env_name = env_name
    self.nonlinearity = nonlinearity

    # rescaling the action (from pranz24 implementation of SAC)
    if action_space is None:
        self.action_scale = torch.tensor(1.)
        self.action_bias = torch.tensor(0.)
    else:
        self.action_scale = torch.FloatTensor(
            (action_space.high - action_space.low) / 2.)
        self.action_bias = torch.FloatTensor(
            (action_space.high + action_space.low) / 2.)

    # weight initialization scheme used in PPO paper experiments
    self.normc_init = normc_init

    self.init_parameters()
    self.train()

  # from pranz24's implementation of SAC
  def to(self, device):
    self.action_scale = self.action_scale.to(device)
    self.action_bias = self.action_bias.to(device)
    return super(SAC_Gaussian_Actor, self).to(device)

  def init_parameters(self):
    if self.normc_init:
        self.apply(normc_fn)

  def forward(self, inputs):

    x = inputs
    for l in self.actor_layers:
        x = self.nonlinearity(l(x))

    mean = self.mean_head(x)
    logstd = self.logstd_head(x)

    logstd = torch.clamp(logstd, self.logstd_min, self.logstd_max)

    self.action = mean

    return mean, logstd

  # most of this code is copied from pranz24's implementation of SAC on github. exception is reparametrization trick
  def sample(self, inputs):
    mean, logstd = self.forward(inputs)
    std = logstd.exp()

    # Behold the reparametrization trick
    normal = Normal(0, 1)
    z = normal.sample()
    x_t = mean + std * z   # does z need .to(device)?
    y_t = torch.tanh(x_t)

    # rescale the action
    action = y_t * self.action_scale + self.action_bias

    # compute log probabilities and enforce action bound
    log_prob = normal.log_prob(x_t) - torch.log(self.action_scale * (1 - y_t.pow(2)) + self.epsilon)
    log_prob = log_prob.sum(1, keepdim=True)
    mean = torch.tanh(mean) * self.action_scale + self.action_bias

    return action, log_prob, mean

  def get_action(self):
    return self.action

  def act(self, inputs, deterministic=False):
    if deterministic is False:
      self.action, _, _ = self.sample(inputs)
    else:
      _, _, self.action = self.sample(inputs)

    return action.detach()

  def evaluate(self, inputs):
    mean, logstd = self(inputs)
    return torch.distributions.Normal(mean, logstd.exp())



class FF_Actor(Actor):
  def __init__(self, state_dim, action_dim, hidden_size=256, hidden_layers=2, env_name='NOT SET', nonlinearity=F.relu, max_action=1):
    super(FF_Actor, self).__init__()

    self.actor_layers = nn.ModuleList()
    self.actor_layers += [nn.Linear(state_dim, hidden_size)]
    for _ in range(hidden_layers-1):
        self.actor_layers += [nn.Linear(hidden_size, hidden_size)]
    self.network_out = nn.Linear(hidden_size, action_dim)

    self.action = None
    self.action_dim = action_dim
    self.env_name = env_name
    self.nonlinearity = nonlinearity

    self.initialize_parameters()

    self.max_action = max_action

  def forward(self, state):
    x = state
    for idx, layer in enumerate(self.actor_layers):
      x = self.nonlinearity(layer(x))

    self.action = torch.tanh(self.network_out(x))
    return self.action * self.max_action

  def get_action(self):
    return self.action

# identical to FF_Actor but allows output to scale to max_action
class Scaled_FF_Actor(Actor):
  def __init__(self, state_dim, action_dim, max_action, hidden_size=256, hidden_layers=2, env_name='NOT SET', nonlinearity=F.relu):
    super(Scaled_FF_Actor, self).__init__()

    self.max_action = max_action

    self.actor_layers = nn.ModuleList()
    self.actor_layers += [nn.Linear(state_dim, hidden_size)]
    for _ in range(hidden_layers-1):
        self.actor_layers += [nn.Linear(hidden_size, hidden_size)]
    self.network_out = nn.Linear(hidden_size, action_dim)

    self.action = None
    self.action_dim = action_dim
    self.env_name = env_name
    self.nonlinearity = nonlinearity

  def forward(self, state):
    x = state
    #print(x.size())
    for idx, layer in enumerate(self.actor_layers):
      x = self.nonlinearity(layer(x))

    self.action = self.max_action * torch.tanh(self.network_out(x))
    #print(self.action)
    #exit(1)
    return self.action

  def get_action(self):
    return self.action

class LSTM_Actor(Actor):
  def __init__(self, input_dim, action_dim, hidden_size=64, hidden_layers=1, env_name='NOT SET', nonlinearity=torch.tanh, max_action=1):
    super(LSTM_Actor, self).__init__()

    self.actor_layers = nn.ModuleList()
    self.actor_layers += [nn.LSTMCell(input_dim, hidden_size)]
    for _ in range(hidden_layers-1):
        self.actor_layers += [nn.LSTMCell(hidden_size, hidden_size)]
    self.network_out = nn.Linear(hidden_size, action_dim)

    self.action = None
    self.action_dim = action_dim
    self.init_hidden_state()
    self.env_name = env_name
    self.nonlinearity = nonlinearity
    
    self.is_recurrent = True

    self.max_action = max_action

  def get_hidden_state(self):
    return self.hidden, self.cells

  def set_hidden_state(self, data):
    if len(data) != 2:
      print("Got invalid hidden state data.")
      exit(1)

    self.hidden, self.cells = data
    
  def init_hidden_state(self, batch_size=1):
    self.hidden = [torch.zeros(batch_size, l.hidden_size) for l in self.actor_layers]
    self.cells  = [torch.zeros(batch_size, l.hidden_size) for l in self.actor_layers]

  def forward(self, x):

    if len(x.size()) == 3: # if we get a batch of trajectories
      self.init_hidden_state(batch_size=x.size(1))
      action = []
      for t, x_t in enumerate(x):

        for idx, layer in enumerate(self.actor_layers):
          c, h = self.cells[idx], self.hidden[idx]
          self.hidden[idx], self.cells[idx] = layer(x_t, (h, c))
          x_t = self.hidden[idx]
        x_t = self.network_out(x_t)
        action.append(x_t)

      x = torch.stack([a.float() for a in action])

    elif len(x.size()) == 2: # if we get a whole trajectory
      self.init_hidden_state()

      self.action = []
      for t, x_t in enumerate(x):
        x_t = x_t.view(1, -1)

        for idx, layer in enumerate(self.actor_layers):
          h, c = self.hidden[idx], self.cells[idx]
          self.hidden[idx], self.cells[idx] = layer(x_t, (h, c))
          x_t = self.hidden[idx]
        x_t = self.nonlinearity(self.network_out(x_t))
        self.action.append(x_t)

      x = torch.cat([a.float() for a in self.action])

    elif len(x.size()) == 1: # if we get a single timestep
      x = x.view(1, -1)

      for idx, layer in enumerate(self.actor_layers):
        h, c = self.hidden[idx], self.cells[idx]
        self.hidden[idx], self.cells[idx] = layer(x, (h, c))
        x = self.hidden[idx]
      x = self.nonlinearity(self.network_out(x))[0]

    else:
      print("Invalid input dimensions.")
      exit(1)

    self.action = x * self.max_action
    return x
  
  def get_action(self):
    return self.action

## Initialization scheme for gaussian mlp (from ppo paper)
# NOTE: the fact that this has the same name as a parameter caused a NASTY bug
# apparently "if <function_name>" evaluates to True in python...
def normc_fn(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)
