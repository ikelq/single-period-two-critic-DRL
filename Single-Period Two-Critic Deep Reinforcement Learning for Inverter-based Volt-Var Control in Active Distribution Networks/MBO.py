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
import pandapower.converter as pc
from pandas.core.frame import DataFrame
from torch.distributions import Normal
import pandas as pd

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

class grid_case:

    def __init__(self,
                 env_name: int,
                 load_pu: np.ndarray,
                 gene_pu: np.ndarray,
                 id_iber: list,
                 id_svc: list):
        """Initializate."""
        self.id_iber = id_iber
        self.id_svc = id_svc
        self.load_pu = load_pu
        self.gene_pu = gene_pu
        self.action_dim = len(self.id_iber) + len(self.id_svc)
        self.step_n = 0
        self.done = False
        self.uncertain_level = 3
        self.env_name = env_name
        if self.env_name == 33:
            self.model = pc.from_mpc('case33_bw.mat', f_hz=50, casename_mpc_file='mpc', validate_conversion=False)
        if self.env_name == 69:
            self.model = pc.from_mpc('case69.mat', f_hz=50, casename_mpc_file='mpc', validate_conversion=False)
        if self.env_name == 118:
            self.model = pc.from_mpc('case1180zh.mat', f_hz=50, casename_mpc_file='mpc', validate_conversion=False)
            # pp.runpp(self.model, algorithm='bfsw')
            # self.model.load = np.max(self.model.load,0.5)

        self.n_bus = len(self.model.bus)
        self.control_n = len(self.id_iber) + len(self.id_svc)
        # 1.32
        for i in self.id_iber:
            pp.create_sgen(self.model, bus=i, p_mw=0, q_mvar=0, name='IBVR', scaling=1.0, in_service=True,
                           max_p_mw=0, min_p_mw=0, max_q_mvar=2, min_q_mvar=-2, controllable=True)
        for i in self.id_svc:
            pp.create_sgen(self.model, bus=i, p_mw=0, q_mvar=0, name='SVC', scaling=1.0, in_service=True,
                           max_p_mw=0, min_p_mw=0, max_q_mvar=2, min_q_mvar=0, controllable=True)
        for i in self.id_iber:
            pp.create_sgen(self.model, bus=i, p_mw=0, q_mvar=0, name='IBVR', scaling=1.0, in_service=True,
                           max_p_mw=4, min_p_mw=0, max_q_mvar=2, min_q_mvar=-2, controllable=False)

        pp.runpp(self.model, algorithm='bfsw')
        self.observation_space = copy.deepcopy(np.hstack(
            (np.array(self.model.res_bus.vm_pu), np.array(self.model.res_bus.p_mw), np.array(self.model.res_bus.q_mvar),
             np.zeros(self.action_dim))))

        self.id_Gp = self.id_iber
        self.id_Gp = [x-1 for x in self.id_Gp]
        self.id_Gq = self.id_iber + self.id_svc
        self.id_Gq = [x+self.n_bus-1-1 for x in self.id_Gq]

        self.id_bus_load = self.model.load.bus.values

        self.injection_pq = np.hstack((self.model.res_bus.p_mw.values[1:], self.model.res_bus.q_mvar.values[1:]))
        self.action_space = np.zeros(self.action_dim)

        self.init_load_p_mw = copy.deepcopy(self.model.load.p_mw)
        self.init_load_q_mvar = copy.deepcopy(self.model.load.q_mvar)

        self.init_line_r_ohm_per_km = copy.deepcopy(self.model.line.r_ohm_per_km)
        self.init_line_x_ohm_per_km = copy.deepcopy(self.model.line.x_ohm_per_km)
        # self.model.line.r_ohm_per_km = 1.5*self.init_line_r_ohm_per_km
        # self.model.line.x_ohm_per_km = 1.5*self.init_line_x_ohm_per_km

        self.load_pu = self.load_pu[:,np.newaxis] * np.ones_like(self.init_load_p_mw)[np.newaxis,:]
        self.load_pu[:370*96,:] = self.load_pu[:370*96,:] * np.random.uniform(low=0.8,high=1.2,size= self.load_pu[:370*96,:].shape)
        # self.load_pu[96*380:96*385,:] = 1.1 * self.load_pu[96*380:96*385,:]
        # self.load_pu[96*385:96*395, :] = 1.2 * self.load_pu[96*385:96*395, :]
        #
        self.gene_pu = self.gene_pu[:,np.newaxis] * np.ones_like(self.id_iber)[np.newaxis,:]
        self.gene_pu[:370*96,:] = self.gene_pu[:370*96,:] * np.random.uniform(low=0.8,high=1.2,size= self.gene_pu[:370*96,:].shape)
        # self.gene_pu[96*380:96*385,:] = 1.1 * self.gene_pu[96*380:96*385,:]
        # self.gene_pu[96*385:96*395, :] = 1.2 * self.gene_pu[96*385:96*395, :]
        #
        np.save('two'+str(self.n_bus)+'load', self.load_pu)
        np.save( 'two'+str(self.n_bus) + 'gen', self.gene_pu)
        self.load_pu = np.load('two'+str(self.n_bus)+'load.npy')
        self.gene_pu = np.load('two'+str(self.n_bus) + 'gen.npy')

        self.model.bus.min_vm_pu[1:] = 0.95
        self.model.bus.max_vm_pu[1:] = 1.05
        self.model.ext_grid.max_p_mw = 100
        self.model.ext_grid.max_q_mvar = 100
        self.model.ext_grid.min_p_mw = -100
        self.model.ext_grid.min_q_mvar = -100

    def action_clip(self, action: np.ndarray) -> np.ndarray:
        """Change the range (-1, 1) to (low, high)."""
        # low = 0
        # high = 2
        low = np.array(self.model.sgen.min_q_mvar)
        high = np.array(self.model.sgen.max_q_mvar)
        # low = - np.array([2.6, 2.6, 2.6, 2.6, 0, 0])
        # high =  np.array([2.6, 2.6, 2.6, 2.6, 3.5, 3.5])

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = action * scale_factor + reloc_factor
        action = np.clip(action, low, high)

        return action

    def step_model(self):
        self.step_n = self.step_n + 1
        print(self.step_n)
        # action = self.action_clip(action)
        # action = self.action_clip(action)
        self.model.load.p_mw = self.load_pu[self.step_n-1] * self.init_load_p_mw
        self.model.load.q_mvar = self.load_pu[self.step_n-1] * self.init_load_q_mvar
        self.model.sgen.p_mw[-len(env.id_iber):] = self.gene_pu[self.step_n-1]
        pp.runopp(self.model, OPF_VIOLATION=1e-10, PDIPM_COMPTOL=1e-10, PDIPM_GRADTOL=1e-10)
        # pp.runpp(self.model, algorithm='bfsw')
        action = self.model.res_sgen.q_mvar[:self.control_n]
        #
        # self.model.line.r_ohm_per_km = self.init_line_r_ohm_per_km*(1+0.2*(np.random.rand(1)-0.5))
        # self.model.line.x_ohm_per_km = self.init_line_x_ohm_per_km*(1+0.2*(np.random.rand(1)-0.5))


        violation_M = Relu(self.model.res_bus.vm_pu - 1.05).sum()
        violation_N = Relu(0.95-self.model.res_bus.vm_pu).sum()
        grid_loss = -self.model.res_line.pl_mw.sum()
        # grid_loss1 = self.model.res_bus.p_mw.sum()
        reward_p = grid_loss
        reward_v = - 50 * violation_M - 50 * violation_N
        reward = np.array((reward_p, reward_v))

        violation = 0
        if (1 * (self.model.res_bus.vm_pu > 1.05).sum() + 1 * (self.model.res_bus.vm_pu < 0.95).sum()) > 0:
            violation = 1

        next_state = np.hstack( (np.array(self.model.res_bus.vm_pu), np.array(self.model.res_bus.p_mw), np.array(self.model.res_bus.q_mvar), action))
        voltage_M = self.model.res_bus.vm_pu.max()
        voltage_N = self.model.res_bus.vm_pu.min()

        # new disturbance
        # self.model.load.p_mw = load_pu[self.step_n] * self.init_load_p_mw
        # self.model.load.q_mvar = load_pu[self.step_n] * self.init_load_q_mvar
        # # self.model.sgen.q_mvar = action
        # self.model.sgen.p_mw[:len(env.id_iber)] = self.gene_pu[self.step_n]
        # pp.runpp(self.model, algorithm='bfsw')
        #
        # new_state = np.hstack((np.array(self.model.res_bus.vm_pu), np.array(self.model.res_bus.p_mw), np.array(self.model.res_bus.q_mvar), action))
        # return next_state, reward, self.done, violation, violation_M, violation_N,voltage_M,voltage_N, grid_loss, new_state

        return next_state, reward, self.done, violation, violation_M, violation_N, voltage_M, voltage_N, grid_loss, action

    def test_model(self,
                   action: np.ndarray):
        self.step_n = self.step_n + 1
        self.model.sgen.q_mvar[:self.control_n] = action
        self.model.load.p_mw = self.load_pu[self.step_n-1] * self.init_load_p_mw
        self.model.load.q_mvar = self.load_pu[self.step_n-1] * self.init_load_q_mvar
        self.model.sgen.p_mw[-len(env.id_iber):] = self.gene_pu[self.step_n-1]

        #
        # self.model.line.r_ohm_per_km = self.init_line_r_ohm_per_km
        # self.model.line.x_ohm_per_km = self.init_line_x_ohm_per_km

        pp.runpp(self.model, algorithm='bfsw')
        violation_M = Relu(self.model.res_bus.vm_pu - 1.05).sum()
        violation_N = Relu(0.95-self.model.res_bus.vm_pu).sum()
        grid_loss = -self.model.res_line.pl_mw.sum()
        # grid_loss1 = self.model.res_bus.p_mw.sum()

        reward_p = grid_loss
        reward_v = - 50 * violation_M - 50 * violation_N
        reward = np.array((reward_p, reward_v))

        violation = 0
        if (1 * (self.model.res_bus.vm_pu > 1.05).sum() + 1 * (self.model.res_bus.vm_pu < 0.95).sum()) > 0:
            violation = 1

        next_state = np.hstack( (np.array(self.model.res_bus.vm_pu), np.array(self.model.res_bus.p_mw), np.array(self.model.res_bus.q_mvar), action))
        voltage_M = self.model.res_bus.vm_pu.max()
        voltage_N =  self.model.res_bus.vm_pu.min()

        #         new disturbance
        # self.model.load.p_mw = load_pu[self.step_n] * self.init_load_p_mw
        # self.model.load.q_mvar = load_pu[self.step_n] * self.init_load_q_mvar
        # self.model.sgen.q_mvar = action
        # self.model.sgen.p_mw[:len(env.id_iber)] = self.gene_pu[self.step_n]
        # pp.runpp(self.model, algorithm='bfsw')
        #
        # new_state = np.hstack((np.array(self.model.res_bus.vm_pu), np.array(self.model.res_bus.p_mw), np.array(self.model.res_bus.q_mvar), action))
        new_state = next_state

        # self.model.line.r_ohm_per_km = 1.5*self.init_line_r_ohm_per_km
        # self.model.line.x_ohm_per_km = 1.5*self.init_line_x_ohm_per_km

        return next_state, reward, self.done, violation, violation_M, violation_N,voltage_M,voltage_N, grid_loss, new_state

    def reset(self):
        self.done = False
        return self.observation_space

class DPGAgent:

    def __init__(
            self,
            env,
            seed: int = 777,
            env_name=33
    ):
        """Initialize."""
        obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.env = env
        self.seed = seed
        self.env.name = env_name

        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)



        # total steps count
        self.total_step = 0
        self.state = self.env.reset()

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # if initial random action should be conducted
        if self.total_step < self.initial_random_steps and not self.is_test:
            # selected_action = self.env.action_space.sample()
            selected_action0 = np.random.randn(self.action_dim)
        else:
            selected_action0 = self.actor(
                torch.FloatTensor(state).to(self.device)
            ).detach().cpu().numpy()
            selected_action = selected_action0

        if not self.is_test:
            noise = self.exploration_noise.sample()
            selected_action = np.clip(
                selected_action0 + noise, -1.0, 1.0
            )
        return selected_action, selected_action0

    def step_model(self) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, done, violation, violation_M, violation_N, voltage_M, voltage_N, grid_loss, new_state = self.env.step_model()
        return next_state, reward, done, violation, violation_M, violation_N, voltage_M, voltage_N, grid_loss, new_state

    def test_model(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, done, violation, violation_M, violation_N, voltage_M, voltage_N, grid_loss, new_state = self.env.test_model(action)
        return next_state, reward, done, violation, violation_M, violation_N, voltage_M, voltage_N, grid_loss, new_state


    def train(self, num_frames: int, plotting_interval: int = 400, safepolicy=True, virtualinteraction=True):
        """Train the agent."""
        self.is_test = False

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

        rewards = []
        violation_Ms = []
        violation_Ns = []
        grid_losses = []
        actions = []

        rewardst = []
        violation_Mst = []
        violation_Nst = []
        grid_lossest = []

        for self.total_step in range(1, num_frames + 1):

            next_state, reward, done, violation, violation_M, violation_N,voltage_M, voltage_N, grid_loss, action1 = self.step_model()
            self.env.step_n = self.env.step_n-1
            next_state1, reward1, done1, violation1, violation_M1, violation_N1, voltage_M1, voltage_N1, grid_loss1, new_state1 = self.test_model(action1)
            # self.transition = [self.state, action, reward, next_state, done]
            # self.memory.store(*self.transition)

            rewards.append(reward)
            violation_Ms.append(voltage_M)
            violation_Ns.append(voltage_N)
            grid_losses.append(grid_loss)
            actions.append(action1)

            rewardst.append(reward1)
            violation_Mst.append(violation_M1)
            violation_Nst.append(violation_N1)
            grid_lossest.append(grid_loss1)

            if self.env.step_n > 96*390:
                self.env.step_n = 0
            # state = next_state
            self.state = next_state
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

        data_tr = {"scores": scores,
                   "violation_sum_s": violation_sum_s,
                   "violation_sum_M_s": violation_sum_M_s,
                   "violation_sum_N_s": violation_sum_N_s,
                   "grid_loss_sum_s": grid_loss_sum_s
                   }
        data_train = DataFrame(data_tr)
        data_train.to_csv('train' + str(self.env.name) + str(self.seed) + 'vvo'+'.csv')

        data_tr_test = {"scorest": scorest,
                   "violation_sum_st": violation_sum_st,
                   "violation_sum_M_st": violation_sum_M_st,
                   "violation_sum_N_st": violation_sum_N_st,
                   "grid_loss_sum_st": grid_loss_sum_st
                   }
        data_train_test = DataFrame(data_tr_test)
        data_train_test.to_csv('traintest' + str(self.env.name) + str(self.seed) + 'vvo'+'.csv')



        data_tr = {"violation_Ms": violation_Ms,
                   "violation_Ns": violation_Ns,
                   "grid_losses": grid_losses,
                   }
        data_inmodel = DataFrame(data_tr)
        data_inmodel.to_csv('trainstep' + str(self.env.name) + str(self.seed) + 'vvo' + '.csv')

        data_action = np.array(actions)
        np.save('action' + str(self.env.name) + str(self.seed) + 'vvo' + '.npy', data_action)

        data_reward = np.array(rewards)
        np.save('reward' + str(self.env.name) + str(self.seed) + 'vvo' + '.npy', data_reward)

        data_st = {"violation_Ms": violation_Mst,
                   "violation_Ns": violation_Nst,
                   "grid_losses": grid_lossest,
                   }
        datatest_step = DataFrame(data_st)
        datatest_step.to_csv('trainteststep' + str(self.env.name) + str(self.seed) + 'vvo' + '.csv')



    def test(self, test_frams, real_model = False, safepolicy = False):
        """Test the agent."""
        self.is_test = True
        self.env.step_n = test_frams
        state = self.state

        # # # initial state for real model
        # action, _ = self.select_action(state)
        # next_state, reward, done, violation, violation_M, violation_N, voltage_M, voltage_N, grid_loss, new_state = self.step_model(action)
        # state = new_state
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
        voltages_M_t =[]
        voltages_N_t = []
        actions = []

        while self.env.step_n < 96*2:
            # action,_ = self.select_action(state)
            next_state, reward, done, violation, violation_M, violation_N, voltage_M, voltage_N, grid_loss, action = self.step_model(np.array([1,2]))

            violation_s_t.append(violation)
            violations_M_t.append(violation_M)
            violations_N_t.append(violation_N)
            voltages_M_t.append(voltage_M)
            voltages_N_t.append(voltage_N)
            grid_loses_t.append(grid_loss)
            actions.append(action)

            # state = next_state
            # state = new_state
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
            (151, f"scores_t,offline{self.offline}", scores_t),
            (152, f"violation_sum_s_t,id_policy{self.id_policy}", violation_sum_s_t),
            (153, f"violation_sum_M_s_t,safepolicy{safepolicy}", violation_sum_M_s_t),
            (154, f"violation_sum_N_s_t,realmodel{real_model}", violation_sum_N_s_t),
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
        data_test.to_csv('test'+str(self.env.name)+str(self.seed) +'vvo'+'.csv')

        data_te_step = {"violation_s_t": violation_s_t,
                   "violations_M_t": violations_M_t,
                   "violations_N_t": violations_N_t,
                   "grid_loses_t": grid_loses_t,
                        "voltages_M_t":voltages_M_t,
                        "voltages_N_t":voltages_N_t
                   }
        data_test_step = DataFrame(data_te_step)
        data_test_step.to_csv('test_step' + str(self.env.name) + str(self.seed) +'vvo'+ '.csv')

        data_test_step_action = DataFrame(actions)
        data_test_step_action.to_csv('test_step_action' + str(self.env.name) + str(self.seed)  +'vvo'+ '.csv')


    def _plot(
            self,
            frame_idx: int,
            scores: List[float],
            violation_sum_s: List[float],
            violation_sum_M_s: List[float],
            violation_sum_N_s: List[float],
            grid_loss_sum_s: List[float],
            actor_losses: List[float],
            qf1_losses: List[float],
            qf2_losses: List[float],
    ):
        """Plot the training progresses."""

        def subplot(loc: int, title: str, values: List[float]):
            plt.subplot(loc)
            plt.title(title)
            plt.plot(values)

        subplot_params = [
            (241, f"frame {frame_idx}. score: {np.mean(scores[-10:])}", scores),
            (242, f"actor_loss, offline:{self.offline}", actor_losses),
            (243, f"qf1_loss, id_policy:{self.id_policy}", qf1_losses),
            (244, "qf2_losses ", qf2_losses),
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

    # parameters *330
    num_frames = 96*330
    test_frames = 0

    for env_name in [118]:
        if env_name == 69:
            id_iber = [5, 22, 44, 63]
            id_svc = [13]  #, 33
            load_scale = 1
            gen_scale = 1.5
        if env_name == 33:
            id_iber = [17, 21, 24]
            id_svc = [32]
            load_scale = 1
            gen_scale = 1.5

        if env_name == 118:
            id_iber = [33, 50, 53, 68, 74, 97, 107, 111]
            id_svc = [44, 104]
            load_scale = 1
            gen_scale = 2

        env = grid_case(env_name, load_scale * load_pu, gen_scale*gene_pu, id_iber, id_svc)

        agent = DPGAgent(
            env,
            seed = seed,
            env_name = env_name
        )

        agent.train(num_frames, plotting_interval=40000)
        # agent.test(test_frames)
