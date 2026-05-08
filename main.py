import torch.nn as nn
import torch
import torch.functional as F
import numpy as np
import gymnasium as gym
from collections import deque
from dataclasses import dataclass

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

@dataclass
class Experience:
    state: np.ndarray
    action: np.ndarray
    reward: float
    done: bool

class ExperienceSource():
    def __init__(self, env, agent, steps_count:int = 5, steps_delta:int = 1):
        self.env = env
        self.steps_count = steps_count # number of steps per memory
        self.agent = agent
        self.steps_delta = steps_delta # number of steps between start states
                                       # of memories
        self.total_rews = []
        self.total_steps = []

    def __iter__(self):
        # make environment
        obs, _ = env.reset()

        history = deque(maxlen=self.steps_count)
        cur_rews = 0.0
        cur_steps = 0
        
        iters = 0
        while True:
            actions = self.agent(obs)
            obs, rew, term, trunc, _ = env.step(actions)

            cur_rews += rew
            cur_steps += 1

            # if buffer is full enough and it's time to send back obs, yield
            # if env is done, incrementally yield last few until it's empty

            history.append(Experience(state=obs, action=actions, reward=rew,
                                      done=term or trunc))
            if len(history) == self.steps_count \
               and iters % self.steps_delta == 0:
                yield tuple(history)

            if term or trunc:
                if 0 < len(history) < self.steps_count:
                    yield tuple(history)
                while len(history) > 1:
                    history.popleft()
                    yield tuple(history)

                # metrics
                self.total_rews.append(cur_rews)
                self.total_steps.append(cur_steps)

                cur_rews = 0.0
                cur_steps = 0

                obs, _ = env.reset()
                history.clear()

            iters += 1

@dataclass
class ExperienceFL:
    state: np.ndarray
    action: np.ndarray
    reward: float
    final: np.ndarray

class ExperienceSourceFL():
    def __init__(self, env, agent, steps_count, steps_delta, gamma):
        super(ExperienceSourceFL, self).__init__(env, agent, steps_count+1,
                                                 steps_delta)
        self.gamma = gamma # reward discount
        self.steps = steps_count

    def __iter__(self):
        for mem in super(ExperienceSourceFL, self).__iter__():
            # if memory is terminal, set last state to none and calculate
            # reward based on all states
            if mem[-1].done and len(mem) <= self.steps:
                last_state = None
                items = mem

            # otherwise, calculate reward based on all but last state
            # and last state is returned to be used for bootstrapping
            else:
                last_state = mem[-1]
                items = mem[:-1]

            # calculate discounted reward backward
            discounted_reward = 0.0
            for m in reversed(items):
                discounted_reward *= self.gamma
                discounted_reward += m.reward

            yield ExperienceFL(state=mem[0], action=action, reward=discounted_reward, final=last_state)

def distr_proj(next_distr, rewards, dones, gamma):
    batch_size = len(rewards)
    proj_distr = np.zeros((batch_size, N_ATOMS), dtype=np.float32)
    delta_z = (Vmax - Vmin) / (N_ATOMS - 1) # "size" of atom
    for atom in range(N_ATOMS):
        # for each atom, add reward and scale by gamma
        v = rewards + (Vmin + atom * delta_z) * gamma

        # set atom to be between Vmax and Vmin range 
        rew_clipped = np.minimum(Vmax, np.maximum(Vmin, v))
        frac_position = (rew_clipped - Vmin) / delta_z
        lower_neighbor = np.floor(frac_position).astype(np.int64)
        upper_neighbor = np.ceil(frac_position).astype(np.int64)

        # if fractional position of reward is right on an atom, then simply
        # add to that atom
        eq_mask = upper_neighbor == lower_neighbor
        proj_distr[eq_mask, lower_neighbor[eq_mask]] += next_distr[eq_mask, atom]

        # more likely it will not land directly on an atom
        # split and spread by weight to each neighbor
        ne_mask = upper_neighbor != lower_neighbor

        proj_distr[ne_mask, lower_neighbor[ne_mask]] += next_distr[ne_mask, atom] * (upper_neighbor - frac_position)[ne_mask]
        
        proj_distr[ne_mask, upper_neighbor[ne_mask]] += next_distr[ne_mask, atom] * (frac_position - lower_neighbor)[ne_mask]

    if dones.any():
        proj_distr[dones] = 0.0 # set terminal states to 0 across distribution
        rew_clipped = np.minimum(Vmax, np.maximum(Vmin, rewards[dones]))
        frac_position = (rew_clipped - Vmin) / delta_z
        lower_neighbor = np.floor(frac_position).astype(np.int64)
        upper_neighbor = np.ceil(frac_position).astype(np.int64)

        # if fractional position right on atom
        eq_mask = lower_neighbor == upper_neighbor
        eq_dones = dones.copy()
        eq_dones[dones] = eq_mask
        if eq_dones.any():
            proj_distr[eq_dones, lower_neighbor[eq_mask]] = 1.0

        ne_mask = lower_neighbor != upper_neighbor
        ne_dones = dones.copy()
        ne_dones[dones] = ne_mask
        if ne_dones.any():
            proj_distr[ne_dones, lower_neighbor[ne_mask]] = (upper_neighbor - frac_position)[ne_mask]

            proj_distr[ne_dones, upper_neighbor[ne_mask]] = (frac_position - lower_neighbor)[ne_mask]

    return proj_distr

if __name__ == "__main__":
    GAMMA = 0.99 # reward discounting
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

        steps = 0
        reward_group = []
        while not term or trunc:
            action = env.action_space.sample()

            obs, rew, term, trunc, _ = env.step(action)

            ep_rews.append(rew)

            reward_group.append([obs, rew, term or trunc])
            if steps % REWARD_STEPS == 0:
                # work backwards to get rewards
                discounted_reward = 0
                done_states = []
                for i,item in enumerate(reversed(reward_group)):
                    discounted_reward *= GAMMA
                    discounted_reward += reward_group[1]

                # mask by done states and add + scale to critic output

            steps += 1

        print(f'Ep {ep}: {np.array(ep_rews).mean()}')

        batch = buffer.sample(BATCH_SIZE)
        # get all state info
        # work backwards, unroll N rewards with critic Q value summed

        ep += 1
        break

    print('Training done.')
    env.close()