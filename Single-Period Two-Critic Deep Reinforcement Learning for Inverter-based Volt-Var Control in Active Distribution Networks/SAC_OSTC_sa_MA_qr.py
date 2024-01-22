'''
@Author: Qiong Liu, Ye Guo, Lirong Deng, Haotian Liu, Dongyu Li, Hongbin Sun, and Wenqi Huang
@Email: liuqiong_yl@outlook.com
@Description:
# code for paper
# @article{liu2022reducing,
#   title={Reducing Learning Difficulties: One-Step Two-Critic Deep Reinforcement Learning for Inverter-based Volt-Var Control},
#   author={Liu, Qiong and Guo, Ye and Deng, Lirong and Liu, Haotian and Li, Dongyu and Sun, Hongbin and Huang, Wenqi},
#   journal={arXiv preprint arXiv:2203.16289},
#   year={2022}
# }
'''
import copy
import random
from typing import Dict, List, Tuple
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandapower as pp
from pandas.core.frame import DataFrame
from torch.distributions import Normal
import pandas as pd
import Env

from IPython.display import clear_output

if torch.backends.cudnn.enabled:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed = 777
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def Relu(x: np.ndarray):
    return np.maximum(0, x)


class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: int, action_dim: int, size: int, batch_size: int = 32):
        """Initializate."""
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, action_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size,2], dtype=np.float32)
        self.done_buf = np.zeros([size], dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def store(
            self,
            obs: np.ndarray,
            act: np.ndarray,
            rew: float,
            next_obs: np.ndarray,
            done: bool,
    ):
        """Store the transition in buffer."""
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        """Randomly sample a batch of experiences from memory."""
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def __len__(self) -> int:
        return self.size

class GaussianNoise:
    """Gaussian Noise.
    Taken from https://github.com/vitchyr/rlkit
    """

    def __init__(
        self,
        action_dim: int,
        min_sigma: float = 1.0,
        max_sigma: float = 1.0,
        decay_period: int = 1000000,
    ):
        """Initialize."""
        self.action_dim = action_dim
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period

    def sample(self, t: int = 0) -> float:
        """Get an action with gaussian noise."""
        sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(
            1.0, t / self.decay_period
        )
        return np.random.normal(0, sigma, size=self.action_dim)


def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    """Init uniform parameters on the single layer."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)

    return layer


class Actor(nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            log_std_min: float = -20,
            log_std_max: float = 2,
            n_layer = [512, 512],
    ):
        """Initialize."""
        super(Actor, self).__init__()

        # set the log std range
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # set the hidden layers
        self.hidden1 = nn.Linear(in_dim, n_layer[0])
        self.hidden2 = nn.Linear(n_layer[0], n_layer[1])
        self.hidden1e = nn.Linear(in_dim, n_layer[0])
        self.hidden2e = nn.Linear(n_layer[0], n_layer[1])
        # self.hidden3 = nn.Linear(n_layer[1], n_layer[2])

        # set log_std layer
        self.log_std_layer = nn.Linear(n_layer[1], out_dim)
        self.log_std_layer = init_layer_uniform(self.log_std_layer)

        # set mean layer
        self.mu_layer = nn.Linear(n_layer[1], out_dim)
        self.mu_layer = init_layer_uniform(self.mu_layer)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        xe = F.relu(self.hidden1e(state))
        xe = F.relu(self.hidden2e(xe))
        # x = F.relu(self.hidden3(x))

        # get mean
        mu = self.mu_layer(x)
        # .tanh()

        # get std
        log_std = self.log_std_layer(xe).tanh()
        log_std = self.log_std_min + 0.5 * (
                self.log_std_max - self.log_std_min
        ) * (log_std + 1)
        std = torch.exp(log_std)

        # sample actions
        dist = Normal(mu, std)
        z = dist.rsample()

        # normalize action and log_prob
        # see appendix C of [2]
        action = z.tanh()
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob, mu.tanh()


class CriticQ(nn.Module):
    def __init__(self,
                 in_dim: int,
                 n_layer = [512, 512]):
        """Initialize."""
        super(CriticQ, self).__init__()

        self.hidden1 = nn.Linear(in_dim, n_layer[0])
        self.hidden2 = nn.Linear(n_layer[0], n_layer[1])
        # self.hidden3 = nn.Linear(n_layer[1], n_layer[2])
        self.out = nn.Linear(n_layer[1], 1)
        self.out = init_layer_uniform(self.out)

    def forward(
            self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Forward method implementation."""
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        # x = F.relu(self.hidden3(x))
        value = self.out(x)

        return value

class SACAgent:

    def __init__(
            self,
            env,
            memory_size: int,
            batch_size: int,
            gamma: float = 0.9,
            tau: float = 5e-3,
            initial_random_steps: int = 1e4,
            policy_update_freq: int = 1,
            seed: int = 777,
            env_name=33,
            is_local = False
    ):
        """Initialize."""
        obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.env = env
        self.memory = ReplayBuffer(obs_dim, self.action_dim, memory_size, batch_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.initial_random_steps = initial_random_steps
        self.policy_update_freq = policy_update_freq
        self.seed = seed
        self.env.name = env_name
        self.is_local = is_local

        if self.is_local:
            if self.env.name == 33:
                self.id_multi_bus = [[17], [21], [24], [32]]
                self.id_multi_actor = [[0], [1], [2], [3]]
            if self.env.name == 69:
                self.id_multi_bus = [[5], [23], [44], [57], [13]]
                self.id_multi_actor = [[0], [1], [2], [3], [4]]
            if self.env.name == 118:
                self.id_multi_bus = [[33],  [50], [53], [68], [74], [97], [107], [111], [44], [104]]
                self.id_multi_actor = [[0], [1],  [2],  [3],  [4],  [5],  [6],   [7],   [8],  [9]]
        else:
            if self.env.name == 33:
                self.id_multi_bus = [list(range(6,18,1)), list(range(18,22,1)), list(range(22,25,1)),list(range(25,33,1))]
                self.id_multi_actor = [[0], [1], [2], [3]]
            if self.env.name == 69:
                self.id_multi_bus = [list(range(1,10,1)), list(range(10,26,1)), list(range(35,45,1)),list(range(52,64,1))]
                self.id_multi_actor = [[0], [1,4], [2], [3]]
            if self.env.name == 118:
                self.id_multi_bus = [list(range(1,60,1)), list(range(62,98,1)), list(range(99, 117,1))]
                self.id_multi_actor = [[0, 1, 2, 8], [3, 4, 5], [6, 7, 9]]


        self.multi_state_dim = []
        self.multi_action_dim = []
        self.id_multi_state = []
        for i in range(len(self.id_multi_bus)):
            self.id_multi_state.append(self.id_multi_bus[i] + [j + self.env.n_bus for j in self.id_multi_bus[i]] + [j+self.env.n_bus*2 for j in self.id_multi_bus[i]] \
                                + [j + self.env.n_bus*3 for j in self.id_multi_actor[i]])
            self.multi_state_dim.append(len(self.id_multi_state[i]))
            self.multi_action_dim.append(len(self.id_multi_actor[i]))

        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)


        # automatic entropy tuning
        self.target_entropy = -np.prod((self.action_dim,)).item()  # heuristic
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)

        # actor
        self.actor = []
        for i in range(len(self.multi_state_dim)):
            self.actor.append(Actor(self.multi_state_dim[i], self.multi_action_dim[i]).to(self.device))
        # self.actor = Actor(obs_dim, self.action_dim).to(self.device)

        # q function
        self.qf_1 = CriticQ(obs_dim + self.action_dim).to(self.device)
        self.qf_target1 = CriticQ(obs_dim + self.action_dim).to(self.device)
        self.qf_target1.load_state_dict(self.qf_1.state_dict())

        self.qf_2 = CriticQ(obs_dim + self.action_dim).to(self.device)
        self.qf_target2 = CriticQ(obs_dim + self.action_dim).to(self.device)
        self.qf_target2.load_state_dict(self.qf_2.state_dict())

        # actor_parameters = list(self.actor[0].parameters()) + list(self.actor[1].parameters()) + list(self.actor[2].parameters()) + list(self.actor[3].parameters())
        actor_parameters = []
        for i in self.actor:
            actor_parameters = actor_parameters + list(i.parameters())

        # actor_parameters1 = [i.parameters() for i in self.actor]
        # optimizers
        # self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.actor_optimizer = optim.Adam(actor_parameters, lr=1e-4)
        self.qf_1_optimizer = optim.Adam(self.qf_1.parameters(), lr=3e-4)
        self.qf_2_optimizer = optim.Adam(self.qf_2.parameters(), lr=3e-4)

        # transition to store in memory
        self.transition = list()

        # total steps count
        self.total_step = 0

        # mode: train / test
        self.is_test = False

        self.actor_loss = torch.zeros(1)

        self.state = self.env.reset()

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # if initial random action should be conducted

        if self.total_step < self.initial_random_steps and not self.is_test:
            # selected_action = self.env.action_space.sample()
            selected_action0 = np.random.randn(self.action_dim)
            selected_action = np.random.randn(self.action_dim)
        else:
            selected_action = 0*np.random.randn(self.action_dim)
            selected_action0 = 0 * np.random.randn(self.action_dim)
            for i in range(len(self.multi_state_dim)):
                selected_action[self.id_multi_actor[i]] = self.actor[i](torch.FloatTensor(state[self.id_multi_state[i]]).to(self.device))[0].detach().cpu().numpy()
                selected_action0[self.id_multi_actor[i]] = self.actor[i](torch.FloatTensor(state[self.id_multi_state[i]]).to(self.device))[2].detach().cpu().numpy()

        if self.is_test:
            selected_action = selected_action0
        return selected_action, selected_action0

    def step_model(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, done, violation, violation_M, violation_N, voltage_M, voltage_N, grid_loss, new_state = self.env.step_model(action)
        return next_state, reward, done, violation, violation_M, violation_N, voltage_M, voltage_N, grid_loss, new_state


    def update_model(self) -> Tuple[torch.Tensor, ...]:
        """Update the model by gradient descent."""
        device = self.device  # for shortening the following lines

        samples = self.memory.sample_batch()
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.FloatTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"]).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        # next_action, log_prob, _ = self.actor(next_state)
        log_prob = torch.zeros_like(action)
        for i in range(len(self.multi_state_dim)):
            multi_state = torch.FloatTensor(samples["obs"][:, self.id_multi_state[i]]).to(device)
            log_prob[:, self.id_multi_actor[i]] = self.actor[i](multi_state)[1]

        # train alpha (dual problem)
        alpha_loss = (
                -self.log_alpha.exp() * (log_prob + self.target_entropy).detach()
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        alpha = self.log_alpha.exp()  # used for the actor loss calculation

        # q function loss
        mask = 1 - done
        q_1_pred = self.qf_1(state, action)
        q_2_pred = self.qf_2(state, action)
        # v_target = self.vf_target(next_state)

        # q_pred = torch.min(self.qf_1(next_state, next_action), self.qf_2(next_state, next_action))

        q1_target = reward[:, 0].reshape(-1, 1)
        q2_target = 50*reward[:, 1].reshape(-1, 1)

        qf_1_loss = F.mse_loss(q_1_pred, q1_target.detach())
        # qf_2_loss = F.mse_loss(q_2_pred, q2_target.detach())
        errors = q2_target.detach() - q_2_pred
        qf_2_loss = torch.max((0.05 - 1) * errors, 0.05 * errors).mean()

        self.qf_1_optimizer.zero_grad()
        qf_1_loss.backward()
        self.qf_1_optimizer.step()

        self.qf_2_optimizer.zero_grad()
        qf_2_loss.backward()
        self.qf_2_optimizer.step()

        qf_loss = qf_1_loss + qf_2_loss

        if self.total_step % self.policy_update_freq == 0:
            # actor loss
            # advantage = self.qf_1(state, self.actor(state)[0]) + self.qf_2(state, self.actor(state)[0])

            action = torch.zeros_like(action)
            for i in range(len(self.multi_state_dim)):
                multi_state = torch.FloatTensor(samples["obs"][:, self.id_multi_state[i]]).to(device)
                action[:, self.id_multi_actor[i]] = self.actor[i](multi_state)[0]

            advantage = self.qf_1(state, action) + self.qf_2(state, action)

            actor_loss = (alpha * log_prob - advantage).mean()

            # train actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # target update (vf)
            # self._target_soft_update()
        else:
            actor_loss = torch.zeros(1)

        # train Q functions


        return actor_loss.detach().cpu().numpy(), qf_loss.detach().cpu().numpy(),  alpha_loss.detach().cpu().numpy()

    def train(self, num_frames: int, plotting_interval: int = 400):
        """Train the agent."""
        self.is_test = False
        actor_losses, qf_losses,  alpha_losses = [], [], []
        scores = []
        violation_sum_s = []
        violation_sum_M_s = []
        violation_sum_N_s = []
        grid_loss_sum_s = []
        score = 0
        violation_sum = 0
        violation_sum_M = 0
        violation_sum_N = 0
        grid_loss_sum = 0

        scorest = []
        violation_sum_st = []
        violation_sum_M_st = []
        violation_sum_N_st = []
        grid_loss_sum_st = []
        scoret = 0
        violation_sumt = 0
        violation_sum_Mt = 0
        violation_sum_Nt = 0
        grid_loss_sumt = 0

        # virtual interaction
        violation_sum_s_virtual = []
        violation_sum_M_s_virtual = []
        violation_sum_N_s_virtual = []
        grid_loss_sum_s_virtual = []
        score_virtual = 0
        violation_sum_virtual = 0
        violation_sum_M_virtual = 0
        violation_sum_N_virtual = 0
        grid_loss_sum_virtual = 0

        for self.total_step in range(1, num_frames + 1):
            action, action0 = self.select_action(self.state)

            next_state, reward, done, violation, violation_M, violation_N, voltage_M, voltage_N, grid_loss, new_state = self.step_model(action)
            self.env.step_n = self.env.step_n - 1
            next_state1, reward1, done1, violation1, violation_M1, violation_N1, voltage_M, voltage_N, grid_loss1, new_state1 = self.step_model(action0)
            self.transition = [self.state, action, reward, next_state, done]
            self.memory.store(*self.transition)

            if self.env.step_n > 96*390:
                self.env.step_n = 0
            # state = next_state
            self.state = new_state
            score += reward.sum()
            violation_sum = violation_sum + violation
            violation_sum_M = violation_sum_M + violation_M
            violation_sum_N = violation_sum_N + violation_N
            grid_loss_sum = grid_loss_sum + grid_loss

            scoret += reward1.sum()
            violation_sumt = violation_sumt + violation1
            violation_sum_Mt = violation_sum_Mt + violation_M1
            violation_sum_Nt = violation_sum_Nt + violation_N1
            grid_loss_sumt = grid_loss_sumt + grid_loss1

            if self.total_step % 96 == 0:
                # state = env.reset()
                scores.append(score)
                violation_sum_s.append(violation_sum)
                violation_sum_M_s.append(violation_sum_M)
                violation_sum_N_s.append(violation_sum_N)
                grid_loss_sum_s.append(grid_loss_sum)
                score = 0
                violation_sum = 0
                violation_sum_M = 0
                violation_sum_N = 0
                grid_loss_sum = 0

                scorest.append(scoret)
                violation_sum_st.append(violation_sumt)
                violation_sum_M_st.append(violation_sum_Mt)
                violation_sum_N_st.append(violation_sum_Nt)
                grid_loss_sum_st.append(grid_loss_sumt)
                scoret = 0
                violation_sumt = 0
                violation_sum_Mt = 0
                violation_sum_Nt = 0
                grid_loss_sumt = 0

            # if training is ready
            for i in range(4):
                if (
                        len(self.memory) >= self.batch_size
                        and self.total_step > self.initial_random_steps
                ):
                    losses = self.update_model()
            if 'losses' in locals():
                actor_losses.append(losses[0])
                qf_losses.append(losses[1])
                alpha_losses.append(losses[2])

            # plotting
            if self.total_step % plotting_interval == 0:
                self._plot(
                    self.total_step,
                    scores,
                    violation_sum_s,
                    violation_sum_M_s,
                    violation_sum_N_s,
                    grid_loss_sum_s,
                    actor_losses,
                    qf_losses,
                    alpha_losses
                )

        data_tr = {"scores": scores,
                   "violation_sum_s": violation_sum_s,
                   "violation_sum_M_s": violation_sum_M_s,
                   "violation_sum_N_s": violation_sum_N_s,
                   "grid_loss_sum_s": grid_loss_sum_s
                   }
        data_train = DataFrame(data_tr)
        data_train.to_csv('trainsac' + str(self.env.name) + str(self.seed) + 'tootsa'+'local'+str(self.is_local)+'qr.csv')

        data_tr_test = {"scorest": scorest,
                        "violation_sum_st": violation_sum_st,
                        "violation_sum_M_st": violation_sum_M_st,
                        "violation_sum_N_st": violation_sum_N_st,
                        "grid_loss_sum_st": grid_loss_sum_st
                        }
        data_train_test = DataFrame(data_tr_test)
        data_train_test.to_csv('traintestsac' + str(self.env.name) + str(self.seed) + 'tootsa'+'local'+str(self.is_local)+'qr.csv')

        data_tr_loss = {"actor_losses": actor_losses,
                        "qf_losses": qf_losses,
                        "alpha_losses": alpha_losses
                        }
        data_train_loss = DataFrame(data_tr_loss)
        data_train_loss.to_csv('trainlosssac' + str(self.env.name) + str(self.seed) + 'tootsa'+'local'+str(self.is_local)+'qr.csv')
        #
        # torch.save(self.log_alpha, 'log_alpha_params' + 'sac'+'.pth')
        # torch.save(self.actor.state_dict(), 'actor_p_params' + 'sac' + '.pth')
        # torch.save(self.qf_1.state_dict(), 'critic1_p_params' + 'sac' + '.pth')
        # torch.save(self.qf_2.state_dict(), 'critic2_p_params' + 'sac' + '.pth')
        # torch.save(self.qf_target1.state_dict(), 'critic_target1_p_params' + 'sac' + '.pth')
        # torch.save(self.qf_target2.state_dict(), 'critic_target2_p_params' + 'sac' + '.pth')

    def test(self, test_frams):
        """Test the agent."""
        self.is_test = True
        self.env.step_n = test_frams-1
        # state = self.env.reset()
        state = self.state

        # # initial state for real model
        action, _ = self.select_action(state)
        next_state, reward, done, violation, violation_M, violation_N, voltage_M, voltage_N, grid_loss, new_state = self.step_model(action)

        state = new_state

        device = self.device  # for shortening the following lines

        scores_t = []
        violation_sum_s_t = []
        violation_sum_M_s_t = []
        violation_sum_N_s_t = []
        grid_loss_sum_s_t = []
        score = 0
        violation_sum = 0
        violation_sum_M = 0
        violation_sum_N = 0
        grid_loss_sum = 0

        violation_s_t = []
        violations_M_t = []
        violations_N_t = []
        grid_loses_t = []
        actions= []


        while self.env.step_n < 96*390:
            action, _ = self.select_action(state)
            next_state, reward, done, violation, violation_M, violation_N, voltage_M, voltage_N, grid_loss,  new_state = self.step_model(action)

            violation_s_t.append(violation)
            violations_M_t.append(violation_M)
            violations_N_t.append(violation_N)
            grid_loses_t.append(grid_loss)
            actions.append(action)

            # state = next_state
            state = new_state
            score += reward.sum()
            violation_sum = violation_sum + violation
            violation_sum_M = violation_sum_M + violation_M
            violation_sum_N = violation_sum_N + violation_N
            grid_loss_sum = grid_loss_sum + grid_loss

            if self.env.step_n % 96 == 0:
                scores_t.append(score)
                violation_sum_s_t.append(violation_sum)
                violation_sum_M_s_t.append(violation_sum_M)
                violation_sum_N_s_t.append(violation_sum_N)
                grid_loss_sum_s_t.append(grid_loss_sum)
                score = 0
                violation_sum = 0
                violation_sum_M = 0
                violation_sum_N = 0
                grid_loss_sum = 0

        def subplot(loc: int, title: str, values: List[float]):
            plt.subplot(loc)
            plt.title(title)
            plt.plot(values)

        subplot_params = [
            (151, f"scores_t", scores_t),
            (152, f"violation_sum_s_t", violation_sum_s_t),
            (153, f"violation_sum_M_s_t", violation_sum_M_s_t),
            (154, f"violation_sum_N_s_t", violation_sum_N_s_t),
            (155, "grid_loss_sum_s_t", grid_loss_sum_s_t)
        ]

        clear_output(True)
        plt.figure(figsize=(30, 6))
        for loc, title, values in subplot_params:
            subplot(loc, title, values)
        plt.show()

        data_te = {"scores_t": scores_t,
                   "violation_sum_s_t": violation_sum_s_t,
                   "violation_sum_M_s_t": violation_sum_M_s_t,
                   "violation_sum_N_s_t": violation_sum_N_s_t,
                   "grid_loss_sum_s_t": grid_loss_sum_s_t,
                   }

        data_test = DataFrame(data_te)
        data_test.to_csv('testsac'+str(self.env.name)+str(self.seed)+ 'tootsa'+'local'+str(self.is_local)+'.csv')

        data_te_step = {"violation_s_t": violation_s_t,
                   "violations_M_t": violations_M_t,
                   "violations_N_t": violations_N_t,
                   "grid_loses_t": grid_loses_t,
                   }
        data_test_step = DataFrame(data_te_step)
        data_test_step.to_csv('testsac_step' +str(self.env.name) + str(self.seed) + 'tootsa'+'local'+str(self.is_local)+'.csv')
        data_test_step_action = DataFrame(actions)
        data_test_step_action.to_csv(
            'testsac_step_action' + str(self.env.name) + str(self.seed) + 'tootsa'+'local'+str(self.is_local)+'.csv')

    def _target_soft_update(self):
        """Soft-update: target = tau*local + (1-tau)*target."""
        tau = self.tau

        # for t_param, l_param in zip(
        #         self.vf_target.parameters(), self.vf.parameters()
        # ):
        #     t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)
        for t_param, l_param in zip(
                self.qf_target1.parameters(), self.qf_1.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)
        for t_param, l_param in zip(
                self.qf_target2.parameters(), self.qf_2.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

    def _plot(
            self,
            frame_idx: int,
            scores: List[float],
            violation_sum_s: List[float],
            violation_sum_M_s: List[float],
            violation_sum_N_s: List[float],
            grid_loss_sum_s: List[float],
            actor_losses: List[float],
            qf_losses: List[float],
            alpha_losses: List[float],
    ):
        """Plot the training progresses."""

        def subplot(loc: int, title: str, values: List[float]):
            plt.subplot(loc)
            plt.title(title)
            plt.plot(values)

        subplot_params = [
            (241, f"frame {frame_idx}. score: {np.mean(scores[-10:])}", scores),
            (242, f"actor_loss", actor_losses),
            (243, f"qf_loss", qf_losses),
            (244, "alpha_loss ", alpha_losses),
            (245, f"violation_sum_s {self.env.name}", violation_sum_s),
            (246, "violation_sum_M_s", violation_sum_M_s),
            (247, "violation_sum_N_s", violation_sum_N_s),
            (248, f"grid_loss_sum_s {np.mean(grid_loss_sum_s[-10:])}", grid_loss_sum_s),
        ]

        clear_output(True)
        plt.figure(figsize=(20, 10))
        for loc, title, values in subplot_params:
            subplot(loc, title, values)
        plt.show()

if __name__ == "__main__":
    load_pu = np.load('load96.npy')
    gene_pu = np.load('gen96.npy')

    # parameters
    num_frames = 96*300
    test_frames = 96*360
    memory_size = 30000
    batch_size = 128
    initial_random_steps = 96*10

    for seed in [555, 777, 999]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # is_local = False
        # env_name = 69
        for env_name in [33, 69, 118]:
            for is_local in [True,False]:

                if env_name == 69:
                    # ieee_model = pc.from_mpc('case69.mat', f_hz=50, casename_mpc_file='mpc', validate_conversion=False)
                    id_iber = [5, 23, 44, 57]
                    id_svc = [13]  # , 33
                if env_name == 33:
                    # ieee_model = pc.from_mpc('case33_bw.mat', f_hz=50, casename_mpc_file='mpc', validate_conversion=False)
                    id_iber = [17, 21, 24]
                    id_svc = [32]
                if env_name == 118:
                    id_iber = [33, 50, 53, 68, 74, 97, 107, 111]
                    id_svc = [44, 104]
                #
                env = Env.grid_case(env_name, load_pu, gene_pu, id_iber, id_svc)

                agent = SACAgent(
                    env,
                    memory_size,
                    batch_size,
                    initial_random_steps=initial_random_steps,
                    seed = seed,
                    env_name = env_name,
                    is_local = is_local,
                )

                agent.train(num_frames, plotting_interval=40000)
                # agent.test(test_frames)

    # f = open('agent.pkl', 'wb')
    # pickle.dump(agent, f)
    # f.close()
    #  save buffer only eliminating voltage violation
    # f = open('agentmemory33rand.pkl', 'wb')
    # pickle.dump(agent.memory, f)
    # f.close()
    #
    # f = open('agentmemory69rand.pkl', 'wb')
    # pickle.dump(agent.memory, f)
    # f.close()
    # f = open('agentmemory'+str(env_name)+'randnoise'+str(noise)+'.pkl', 'wb')
    # pickle.dump(agent.memory, f)
    # f.close()

    # f = open('agentmemory69expertnosie.pkl', 'wb')
    # pickle.dump(agent.memory, f)
    # f.close()
    #
    # fl = open('agentmemory.pkl', 'rb')
    # memory1 = pickle.load(fl)
    # fl.close()

