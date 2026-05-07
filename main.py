import torch.nn as nn
import torch
import torch.functional as F
import numpy as np
import gymnasium as gym
import copy

class PPOAgent():
    # has network and device
    # callable to take in states and return actions
    def __init__(self, net, device):
        self.net = net
        self.device = device
    
    def __call__(self, states):
        pass


class D4PGCritic(nn.Module):
    def __init__(self, obs_size, act_size, n_atoms, v_min, v_max, use_target):
        super(D4PGCritic, self).__init__()

        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU()
        )

        self.out_net = nn.Sequential(
            nn.Linear(400 + act_size, 300),
            nn.ReLU(),
            nn.Linear(300, n_atoms)
        )

        delta = (v_max - v_min) / (n_atoms - 1)
        self.register_buffer("supports", torch.arange(v_min, v_max + delta,
                                                      delta))
        
        self.use_target = use_target
        if self.use_target:
            self.target = DDPGActor(obs_size, act_size, n_atoms, v_min, v_max, 
                                    use_target, False)

    def forward(self, x: torch.Tensor, a: torch.Tensor):
        obs = self.obs_net(x)
        return self.out_net(torch.cat([obs, a], dim=1))
    
    def distr_to_q(self, distr: torch.Tensor):
        # distr converted into probability distribution
        # weights are expected values, summed together into res
        weights = F.softmax(distr, dim=1) * self.supports
        res = weights.sum(dim=1)
        return res.unsqueeze(dim=-1)
    
class D4PGAgent():
    def __init__(self, actor, device, epsilon):
        self.actor = actor
        self.device = device
        self.epsilon = epsilon

    def __call__(self, states):
        # take state and return action
        # states need to be float32 type
        states_tensor = torch.as_tensor(np.array(states, dtype=np.float32))
        states_tensor = states_tensor.to(self.device)

        # get actions from network, extract to cpu, add noise, clip, return
        actions = self.actor(states_tensor)
        actions = actions.detach().cpu().numpy()
        actions += self.epsilon * np.random.normal(size=actions.shape)
        actions = np.clip(actions, -1, 1)

        return actions
    
class DDPGActor(nn.Module):
    def __init__(self, obs_size, act_size, use_target):
        super(DDPGActor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, act_size),
            nn.Tanh() # restrict to [-1, 1]
        )

        self.use_target = use_target
        if self.use_target:
            self.target = DDPGActor(obs_size, act_size, False)

    def forward(self, x):
        return self.net(x)
        
class D4PGActor(DDPGActor):
    pass



if __name__ == "__main__":
    GAMMA = 0.99
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    REPLAY_SIZE = 100000
    REPLAY_INITIAL = 10000
    REWARD_STEPS = 5
    TEST_ITERS = 1000
    Vmax = 10
    Vmin = -10
    N_ATOMS = 51 # atoms for Q distribution
    EPSILON = 0.3 # noise
    DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)

    env = gym.make('Ant-v5', render_mode='human')
    critic = D4PGCritic(env.observation_space.shape[0],
                        env.action_space.shape[0], N_ATOMS, Vmin, Vmax, True)
    
    actor = D4PGActor(env.observation_space.shape[0],
                      env.action_space.shape[0], True)
    
    agent = D4PGAgent(actor, torch.device('cuda' if torch.cuda.is_available() else 'cpu'), EPSILON)

    ep = 0
    while True:
        
        obs, _ = env.reset()
        term, trunc = False, False

        ep_rews = []

        while not term or trunc:
            action = env.action_space.sample()

            obs, rew, term, trunc, _ = env.step(action)

            ep_rews.append(rew)

        print(f'Ep {ep}: {np.array(ep_rews).mean()}')

        batch = buffer.sample(BATCH_SIZE)
        # get all state info
        # work backwards, unroll N rewards with critic Q value summed

        ep += 1
        break

    print('Training done.')
    env.close()