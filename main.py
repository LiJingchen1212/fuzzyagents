import os

import time
import torch
import pickle
import argparse
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from gym.spaces import Box, Discrete
from arguments import parse_args
from replay_buffer import ReplayBuffer
import multiagent.scenarios as scenarios
from model import openai_actor, openai_critic
from multiagent.environment import MultiAgentEnv
from scipy import stats
import envs.mpe_scenarios as new_scenarios

def make_env(scenario_name, arglist, benchmark=False):
    """
    create the environment from script
    """
    scenario = new_scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world()
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

    num_adversaries= env.world.num_adversaries
    num_good_agents = env.world.num_good_agents
    return env, num_adversaries, num_good_agents

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)

def get_trainers(env, advers,goods, obs_shape_n, action_shape_n, arglist):
    """
    init the trainers or load the old model
    """
    num_fuzzy = advers //10
    actors_cur = [None for _ in range(env.n-advers+num_fuzzy)]
    critics_cur = [None for _ in range(env.n-advers+num_fuzzy)]
    actors_tar = [None for _ in range(env.n-advers+num_fuzzy)]
    critics_tar = [None for _ in range(env.n-advers+num_fuzzy)]
    optimizers_c = [None for _ in range(env.n-advers+num_fuzzy)]
    optimizers_a = [None for _ in range(env.n-advers+num_fuzzy)]
    # input_size_global = sum(obs_shape_n) + sum(action_shape_n)

    if arglist.restore == True:  # restore the model
        for idx in arglist.restore_idxs:
            trainers_cur[idx] = torch.load(arglist.old_model_name + 'c_{}'.format(agent_idx))
            trainers_tar[idx] = torch.load(arglist.old_model_name + 't_{}'.format(agent_idx))

    # Note: if you need load old model, there should be a procedure for juding if the trainers[idx] is None
    for i in range(num_fuzzy):
        actors_cur[i] = openai_actor(obs_shape_n[0], action_shape_n[0], arglist).to(arglist.device)
        critics_cur[i] = openai_critic(num_fuzzy*obs_shape_n[0], num_fuzzy*action_shape_n[0], arglist).to(arglist.device)
        actors_tar[i] = openai_actor(obs_shape_n[0], action_shape_n[0], arglist).to(arglist.device)
        critics_tar[i] = openai_critic(num_fuzzy * obs_shape_n[0], num_fuzzy * action_shape_n[0], arglist).to(
            arglist.device)
        optimizers_a[i] = optim.Adam(actors_cur[i].parameters(), arglist.lr_a)
        optimizers_c[i] = optim.Adam(critics_cur[i].parameters(), arglist.lr_c)
    for i in range(num_fuzzy, env.n-advers+num_fuzzy):
        actors_cur[i] = openai_actor(obs_shape_n[i+advers-num_fuzzy], action_shape_n[i+advers-num_fuzzy], arglist).to(arglist.device)
        critics_cur[i] = openai_critic(goods*obs_shape_n[i+advers-num_fuzzy], goods*action_shape_n[i+advers-num_fuzzy], arglist).to(arglist.device)
        #print (critics_cur[i])
        actors_tar[i] = openai_actor(obs_shape_n[i+advers-num_fuzzy], action_shape_n[i+advers-num_fuzzy], arglist).to(arglist.device)
        critics_tar[i] = openai_critic(goods*obs_shape_n[i+advers-num_fuzzy], goods*action_shape_n[i+advers-num_fuzzy], arglist).to(arglist.device)
        #print (critics_tar[i])
        optimizers_a[i] = optim.Adam(actors_cur[i].parameters(), arglist.lr_a)
        optimizers_c[i] = optim.Adam(critics_cur[i].parameters(), arglist.lr_c)
    actors_tar = update_trainers(actors_cur, actors_tar, 1.0)  # update the target par using the cur
    critics_tar = update_trainers(critics_cur, critics_tar, 1.0)  # update the target par using the cur
    return actors_cur, critics_cur, actors_tar, critics_tar, optimizers_a, optimizers_c


def update_trainers(agents_cur, agents_tar, tao):
    """
    update the trainers_tar par using the trainers_cur
    This way is not the same as copy_, but the result is the same
    out:
    |agents_tar: the agents with new par updated towards agents_current
    """
    for agent_c, agent_t in zip(agents_cur, agents_tar):
        key_list = list(agent_c.state_dict().keys())
        state_dict_t = agent_t.state_dict()
        state_dict_c = agent_c.state_dict()
        for key in key_list:
            state_dict_t[key] = state_dict_c[key] * tao + \
                                (1 - tao) * state_dict_t[key]
        agent_t.load_state_dict(state_dict_t)
    return agents_tar

#def FuzzyObservation(obs_n,advers,goods,membership):

def updatemembership(obs_adv,obs_fuzzy):
    sigma = 0.5
    membership= np.zeros((len(obs_adv),len(obs_fuzzy)))
    for i in range(len(obs_adv)):
        for j in range(len(obs_fuzzy)):
            temp = 1
            for k in range(len(obs_fuzzy[0])):
                #temp *= stats.norm.pdf(obs_adv[i][k],obs_fuzzy[j][k],sigma)
                temp *= np.exp(-0.2*(abs(obs_adv[i][k]-obs_fuzzy[j][k]))) #1/action_space*num_fuzzy
            membership[i,j] = temp
    return membership
def getadveraction(fuzzy_action,membership):
    temp = [[] for _ in range(int(membership.shape[0]))]
    for i in range(int(membership.shape[0])):
        for j in range(len(fuzzy_action)):
            temp[i].append(membership[i,j]*fuzzy_action[j])
    #解模糊
    advers_action = []
    all_action = fuzzy_action[0]
    for i in range(1,len(fuzzy_action)):
        all_action += fuzzy_action[i]

    for i in range(int(membership.shape[0])):
        action = temp[i][0]
        for j in range(1,len(fuzzy_action)):
            action += temp[i][j]
        advers_action.append(softmax(action/all_action))
    return advers_action

def getadverrew(rew_n,membership):
    temp = [0 for _ in range(int(membership.shape[1]))]
    for i in range(int(membership.shape[1])):
        membership_softmax = softmax(membership.T[i])
        for j in range(int(membership.shape[0])):
            temp[i]+=(membership_softmax[j]*rew_n[j])
    return temp
def getadverobs(new_obs_n,membership):
    new_obs_n = np.array(new_obs_n)
    fuzzy_obs = [None for _ in range(int(membership.shape[1]))]
    for i in range(len(fuzzy_obs)):
        temp=softmax(membership.T[i])
        fuzzy_obs[i] = temp[0]*new_obs_n[0]
        for j in range(1,membership.shape[0]):
            fuzzy_obs[i]+=(temp[j]*new_obs_n[j])
        fuzzy_obs[i].tolist
    return fuzzy_obs


def agents_train(arglist, game_step, update_cnt, memory, obs_size, action_size, \
                 actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c, num_fuzzy):
    """
    use this func to make the "main" func clean
    par:
    |input: the data for training
    |output: the data for next update
    """
    # update all trainers, if not in display or benchmark mode
    if game_step > arglist.learning_start_step and \
            (game_step - arglist.learning_start_step) % arglist.learning_fre == 0:
        if update_cnt == 0: print('\r=start training ...' + ' ' * 100)
        # update the target par using the cur
        update_cnt += 1

        # update every agent in different memory batch
        for agent_idx, (actor_c, actor_t, critic_c, critic_t, opt_a, opt_c) in \
                enumerate(zip(actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c)):
            if opt_c == None: continue  # jump to the next model update

            # sample the experience
            _obs_n_o, _action_n, _rew_n, _obs_n_n, _done_n = memory.sample( \
                arglist.batch_size, agent_idx)  # Note_The func is not the same as others

            # --use the date to update the CRITIC
            rew = torch.tensor(_rew_n, device=arglist.device, dtype=torch.float)  # set the rew to gpu
            done_n = torch.tensor(~_done_n, dtype=torch.float, device=arglist.device)  # set the rew to gpu
            action_cur_o = torch.from_numpy(_action_n).to(arglist.device, torch.float)
            obs_n_o = torch.from_numpy(_obs_n_o).to(arglist.device, torch.float)
            obs_n_n = torch.from_numpy(_obs_n_n).to(arglist.device, torch.float)
            action_tar = torch.cat([a_t(obs_n_n[:, obs_size[idx][0]:obs_size[idx][1]]).detach() \
                                    for idx, a_t in enumerate(actors_tar)], dim=1)
            if agent_idx < num_fuzzy:
                obs_n_o_p = obs_n_o[:,0:obs_size[num_fuzzy-1][1]]
                obs_n_n_p = obs_n_n[:,0:obs_size[num_fuzzy - 1][1]]
                action_cur_o_p = action_cur_o[:, 0:action_size[num_fuzzy - 1][1]]
                action_tar_p = action_tar[:,0:action_size[num_fuzzy-1][1]]
            else:
                obs_n_o_p = obs_n_o[:,obs_size[num_fuzzy][0]:]
                obs_n_n_p = obs_n_n[:,obs_size[num_fuzzy][0]:]
                action_cur_o_p = action_cur_o[:,action_size[num_fuzzy][0]:]
                action_tar_p = action_tar[:,action_size[num_fuzzy][0]:]
                #print(action_cur_o.shape)
                #print(action_size[num_fuzzy])
            q = critic_c(obs_n_o_p, action_cur_o_p).reshape(-1)  # q
            q_ = critic_t(obs_n_n_p, action_tar_p).reshape(-1)  # q_
            tar_value = q_ * arglist.gamma * done_n + rew  # q_*gamma*done + reward
            loss_c = torch.nn.MSELoss()(q, tar_value)  # bellman equation
            opt_c.zero_grad()
            loss_c.backward()
            nn.utils.clip_grad_norm_(critic_c.parameters(), arglist.max_grad_norm)
            opt_c.step()

            # --use the data to update the ACTOR
            # There is no need to cal other agent's action
            model_out, policy_c_new = actor_c( \
                obs_n_o[:, obs_size[agent_idx][0]:obs_size[agent_idx][1]], model_original_out=True)
            # update the aciton of this agent
            action_cur_o[:, action_size[agent_idx][0]:action_size[agent_idx][1]] = policy_c_new
            loss_pse = torch.mean(torch.pow(model_out, 2))
            loss_a = torch.mul(-1, torch.mean(critic_c(obs_n_o_p, action_cur_o_p)))

            opt_a.zero_grad()
            (1e-3 * loss_pse + loss_a).backward()
            nn.utils.clip_grad_norm_(actor_c.parameters(), arglist.max_grad_norm)
            opt_a.step()

        # save the model to the path_dir ---cnt by update number
        # if update_cnt > arglist.start_save_model and update_cnt % arglist.fre4save_model == 0:
        # time_now = time.strftime('%y%m_%d%H%M')
        # print('=time:{} step:{}        save'.format(time_now, game_step))
        # model_file_dir = os.path.join(arglist.save_dir, '{}_{}_{}'.format( \
        # arglist.scenario_name, time_now, game_step))
        # if not os.path.exists(model_file_dir): # make the path
        # os.mkdir(model_file_dir)
        # for agent_idx, (a_c, a_t, c_c, c_t) in \
        # enumerate(zip(actors_cur, actors_tar, critics_cur, critics_tar)):
        # torch.save(a_c, os.path.join(model_file_dir, 'a_c_{}.pt'.format(agent_idx)))
        # torch.save(a_t, os.path.join(model_file_dir, 'a_t_{}.pt'.format(agent_idx)))
        # torch.save(c_c, os.path.join(model_file_dir, 'c_c_{}.pt'.format(agent_idx)))
        # torch.save(c_t, os.path.join(model_file_dir, 'c_t_{}.pt'.format(agent_idx)))

        # update the tar par
        actors_tar = update_trainers(actors_cur, actors_tar, arglist.tao)
        critics_tar = update_trainers(critics_cur, critics_tar, arglist.tao)

    return update_cnt, actors_cur, actors_tar, critics_cur, critics_tar


def train(arglist):
    """
    init the env, agent and train the agents
    """
    """step1: create the environment """
    env,advers,goods = make_env(arglist.scenario_name, arglist, arglist.benchmark)
    num_fuzzy = advers // 10
    print('=============================')
    print('=1 Env {} is right ...'.format(arglist.scenario_name))
    print('=============================')

    """step2: create agents"""
    # a = [acsp.shape[0] if isinstance(acsp, Box) else acsp.n for acsp in env.action_space]
    # print(a)
    # print(env.action_space[0])
    obs_shape_n = [env.observation_space[i].shape[0] for i in range(advers+goods)]
    fuzzy_obs_shape_n = obs_shape_n[0:num_fuzzy]+obs_shape_n[advers:(advers+goods)]
    action_shape_n = [acsp.shape[0] if isinstance(acsp, Box) else acsp.n for acsp in env.action_space]
    fuzzy_action_shape_n = action_shape_n[0:num_fuzzy] + action_shape_n[advers:(advers + goods)]
    # action_shape_n = [env.action_space[i] for i in range(env.n)] # no need for stop bit
    #num_adversaries = min(env.n, arglist.num_adversaries)
    actors_cur, critics_cur, actors_tar, critics_tar, optimizers_a, optimizers_c = \
        get_trainers(env, advers,goods, obs_shape_n, action_shape_n, arglist) #______________________________________________________________________________________________
    # memory = Memory(num_adversaries, arglist)
    memory = ReplayBuffer(arglist.memory_size)

    print('=2 The {} agents are inited ...'.format(env.n))
    print('=============================')

    """step3: init the pars """
    obs_size = []
    action_size = []
    game_step = 0
    episode_cnt = 0
    update_cnt = 0
    t_start = time.time()
    rew_n_old = [0.0 for _ in range(env.n)]  # set the init reward
    agent_info = [[[]]]  # placeholder for benchmarking info
    episode_rewards = [0.0]  # sum of rewards for all agents
    advers_rewards = [0.0]
    agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
    head_o, head_a, end_o, end_a = 0, 0, 0, 0
    for obs_shape, action_shape in zip(fuzzy_obs_shape_n, fuzzy_action_shape_n):
        end_o = end_o + obs_shape
        end_a = end_a + action_shape
        range_o = (head_o, end_o)
        range_a = (head_a, end_a)
        obs_size.append(range_o)
        action_size.append(range_a)
        head_o = end_o
        head_a = end_a
    #print(obs_size,action_size)
    print('=3 starting iterations ...')
    print('=============================')

    obs_n = env.reset()
    #print(membership)
    #obs_n = FuzzyObservation(obs_n,advers,goods)

    total_reward = np.zeros(150000)
    for episode_gone in range(arglist.max_episode):
        # cal the reward print the debug data
        if game_step > 1 and game_step % 100 == 0:
            mean_agents_r = [round(np.mean(agent_rewards[idx][-200:-1]), 6) for idx in range(env.n)]
            mean_ad_r = round(np.mean(advers_rewards[-200:-1]), 3)
            print(" " * 43 + 'advers reward:{} agents mean reward:{}'.format(mean_ad_r, mean_agents_r), end='\r')
        print('=Training: steps:{} episode:{}'.format(game_step, episode_gone), end='\r')
        obs_fuzzy = []

        temp = []

        while len(temp) < num_fuzzy:
            i = random.randint(0, advers - 1)
            if i not in temp:
                temp.append(i)
                obs_fuzzy.append(obs_n[i])
        # obs_fuzzy = np.array(obs_fuzzy)
        membership = updatemembership(obs_n[0:advers], obs_fuzzy)
        for episode_cnt in range(arglist.per_episode_max_len):
            # get action
            fuzzy_action = [agent(torch.from_numpy(obs).to(arglist.device, torch.float)).detach().cpu().numpy() \
                        for agent, obs in zip(actors_cur[0:num_fuzzy], obs_fuzzy)]
            goods_action = [agent(torch.from_numpy(obs).to(arglist.device, torch.float)).detach().cpu().numpy() \
                        for agent, obs in zip(actors_cur[num_fuzzy:(env.n-advers+num_fuzzy)], obs_n[advers:env.n])]
            advers_action = getadveraction(fuzzy_action,membership)
            action_n = advers_action+goods_action

            #print(action_n)

            # interact with env
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)

            #env.render()
            fuzzy_rew = getadverrew(rew_n[0:advers],membership)
            new_rew_n = fuzzy_rew + rew_n[advers:env.n]
            new_obs_fuzzy = getadverobs(new_obs_n[0:advers],membership)
            # save the experience
            memory.add(obs_fuzzy+obs_n[advers:env.n], np.concatenate(fuzzy_action+goods_action), new_rew_n, new_obs_fuzzy+new_obs_n[advers:env.n], done_n[0:num_fuzzy]+done_n[advers:env.n])
            episode_rewards[-1] += np.sum(rew_n)
            advers_rewards[-1] += np.sum(rew_n[0:advers])
            for i, rew in enumerate(rew_n): agent_rewards[i][-1] += rew

            # train our agents
            update_cnt, actors_cur, actors_tar, critics_cur, critics_tar = agents_train( \
                arglist, game_step, update_cnt, memory, obs_size, action_size, \
                actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c, num_fuzzy)

            # update the obs_n
            game_step += 1
            obs_n = new_obs_n
            obs_fuzzy = new_obs_fuzzy
            membership = updatemembership(obs_n[0:advers],obs_fuzzy)
            done = all(done_n)
            terminal = (episode_cnt >= arglist.per_episode_max_len - 1)
            if done or terminal:
                total_reward[episode_gone] = advers_rewards[-1]
                # print("episode_reward:",episode_rewards[-1])
                episode_step = 0
                obs_n = env.reset()
                agent_info.append([[]])
                episode_rewards.append(0)
                advers_rewards.append(0)
                for a_r in agent_rewards:
                    a_r.append(0)
                continue
    np.savetxt("fuzzy",total_reward)


if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
