import torch.nn as nn
import torch
from torch import optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from collections import deque
from dataclasses import dataclass
import random
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
    def __init__(self, obs_size, act_size, n_atoms, v_min, v_max):
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
    def __init__(self, obs_size, act_size):
        super(DDPGActor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, act_size),
            nn.Tanh() # restrict to [-1, 1]
        )

    def forward(self, x):
        return self.net(x)
    
class TargetNetwork(nn.Module):
    def __init__(self, model):
        super(TargetNetwork, self).__init__()

        self.model = model
        # deep copy for target network to separate gradients
        self.target = copy.deepcopy(model)

    def sync(self, a):
        state = self.model.state_dict()
        target_state = self.target.state_dict()
        for k, v in state.items():
            target_state[k] = target_state[k] * a + (1-a) * v

        self.target.load_state_dict(target_state)
        
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
                print(sum(self.total_rews) / len(self.total_rews))
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

class ExperienceSourceFL(ExperienceSource):
    def __init__(self, env, agent, steps_count, gamma):
        super(ExperienceSourceFL, self).__init__(env, agent, steps_count+1)
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
                last_state = mem[-1].state
                items = mem[:-1]

            # calculate discounted reward backward
            discounted_reward = 0.0
            for m in reversed(items):
                discounted_reward *= self.gamma
                discounted_reward += m.reward

            yield ExperienceFL(state=mem[0].state, action=mem[0].action, reward=discounted_reward, final=last_state)

class ExperienceReplayBuffer():
    def __init__(self, exp_source, buffer_size):
        self.exp_iterative = iter(exp_source)
        self.buffer = []
        self.capacity = buffer_size
        self.idx = 0

    def __len__(self):
        return len(self.buffer)
    
    def __iter__(self):
        return iter(self.buffer)
    
    def sample(self, batch_size):
        if len(self.buffer) <= batch_size:
            return self.buffer
        
        idxs = np.random.choice(len(self.buffer), batch_size, replace=True)
        return [self.buffer[idx] for idx in idxs]
    
    def _add(self, sample):
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            self.buffer[self.idx] = sample

        self.idx = (self.idx + 1) % self.capacity

    def populate(self, samples):
        for _ in range(samples):
            exp = next(self.exp_iterative)
            self._add(exp)

def unpack_batch(batch, device="cpu"):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.final is None)
        if exp.final is None:
            last_states.append(exp.state)
        else:
            last_states.append(exp.final)
    states_v = torch.as_tensor(np.array(states, dtype=np.float32)).to(device)
    actions_v = torch.as_tensor(np.array(actions, dtype=np.float32)).to(device)
    rewards_v = torch.as_tensor(np.array(rewards, dtype=np.float32)).to(device)
    last_states_v = torch.as_tensor(np.array(last_states, dtype=np.float32)).to(device)
    dones_t = torch.BoolTensor(dones).to(device)
    return states_v, actions_v, rewards_v, dones_t, last_states_v


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

GAMMA = 0.99 # reward discounting
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
REPLAY_SIZE = 100000
REPLAY_INITIAL = 100 # 10_000
REWARD_STEPS = 5
TEST_ITERS = 1000
Vmax = 10
Vmin = -10
N_ATOMS = 51 # atoms for Q distribution
EPSILON = 0.3 # noise
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)

if __name__ == "__main__":

    env = gym.make('Ant-v5')
    critic = D4PGCritic(env.observation_space.shape[0],
                        env.action_space.shape[0], N_ATOMS, Vmin, Vmax)
    critic_target = TargetNetwork(critic)
    
    actor = D4PGActor(env.observation_space.shape[0],
                      env.action_space.shape[0])
    actor_target = TargetNetwork(actor)
    
    agent = D4PGAgent(actor, torch.device('cuda' if torch.cuda.is_available() else 'cpu'), EPSILON)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    actor.to(device)
    critic.to(device)
    actor_target.to(device)
    critic_target.to(device)

    exp_source = ExperienceSourceFL(env, agent, steps_count=REWARD_STEPS, gamma=GAMMA)
    buffer = ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)

    actor_optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
    critic_optimizer = optim.Adam(critic.parameters(), lr=LEARNING_RATE)

    step = 0
    best_rew = None
    while True:
        step += 1
        buffer.populate(1)

        # keep going until buffer is full
        if len(buffer) < REPLAY_INITIAL:
            continue

        batch = buffer.sample(BATCH_SIZE)

        states, actions, rews, dones, last_states = unpack_batch(batch, device)

        # critic > add discounted critic output for last state
        critic_optimizer.zero_grad()
        critic_out = critic(states, actions)
        
        last_actions = actor_target.target(last_states)
        last_distr = F.softmax(critic_target.target(last_states, last_actions), dim=1)

        # target distribution
        proj_distr = distr_proj(last_distr.detach().cpu().numpy(),
                                rews.detach().cpu().numpy(),
                                dones.detach().cpu().numpy(),
                                GAMMA**REWARD_STEPS)
        proj_distr_tensor = torch.tensor(proj_distr).to(device)

        # log_softmax critic output to form prrobability distribution
        # mulitply by target distribution to get expected reward
        prob_distr_tensor = -F.log_softmax(critic_out, dim=1) * proj_distr_tensor
        critic_loss_tensor = prob_distr_tensor.sum(dim=1).mean()
        critic_loss_tensor.backward()
        critic_optimizer.step()

        # actor > optimize negative critic
        actor_optimizer.zero_grad()
        
        actor_actions = actor(states)
        critic_out = critic(states, actor_actions)

        # collapse distribution to one q value
        actor_loss = -critic.distr_to_q(critic_out).mean()
        actor_loss.backward()
        actor_optimizer.step()

        # update target networks with alpha=1-1e-3
        actor_target.sync(1 - 1e-3)
        critic_target.sync(1 - 1e-3)

        if step % TEST_ITERS == 0:
            print(f'Current rewards [{step//1000}]: {sum(exp_source.total_rews[:-1000]) / 1000}')

        if step % (TEST_ITERS * 10) == 0:
            print('Saving...')
            torch.save(actor.state_dict(), f'actor_{step}.pt')
            torch.save(critic.state_dict(), f'critic_{step}.pt')


    print('Training done.')
    env.close()