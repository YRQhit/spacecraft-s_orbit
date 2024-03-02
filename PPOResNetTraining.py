import os
import glob
import time
from datetime import datetime

import torch
import numpy as np
from typing import Optional
import gym
# import roboschool
from gym.envs.classic_control import utils
from PPOResNet import PPO
from gym import spaces

global Reward
import math
#-------------研究action达到三个维度的时候怎么改进---------------- wwwwwww
#
class Environment(gym.Env):
    def __init__(self):
        self.min_action = -1
        self.max_action = 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,),
                                            dtype=float)  #追击rv,逃逸rv
        self.action_space = spaces.Box(low = self.min_action, high = self.max_action, shape=(3,),
                                       dtype=float)  # action给出两个xy两个力量方向
        self.escape_position = None
        self.escape_speed = None
        self.chase_position = None
        self.chase_speed = None
        self.time = 1

    def update_chase(self,action):
        self.chase_speed = self.chase_speed + action
        self.chase_position = self.chase_position + self.chase_speed * self.time
    def update_escape(self,action):
        self.escape_speed = self.escape_speed + action
        self.escape_position = self.escape_position + self.escape_speed * self.time
    def reset(self):
        self._initialize_positions()
        observation = self._get_observation()
        return observation

    def _initialize_positions(self):  # 初始化航天器位置信息
        chase = [10, 10, 10, 0, 0, 0]
        escape = [0, 0, 0, 0, 0, 0]
        self.escape_position = np.array(escape[0:3])
        self.escape_speed = np.array(escape[3:])
        self.chase_position = np.array(chase[:3])
        self.chase_speed = np.array(chase[3:])

    def step(self,action):
        action = np.clip(action, -1, 1)
        self.update_chase(action)
        escape_action = [0, 0, 0]
        self.update_escape(escape_action)
        observation = self._get_observation()
        distance = np.linalg.norm(self.escape_position - self.chase_position)
        reward = -0.001 * distance
        done = self.check_termination()
        if done:
            reward = 10

        return observation,reward,done,None

    def _get_observation(self):
        chase_obervation = np.concatenate((self.chase_position, self.chase_speed), axis=0)
        escape_obervation = np.concatenate((self.escape_position, self.escape_speed), axis=0)
        observation = np.concatenate((chase_obervation, escape_obervation), axis=0)
        return observation

    def check_termination(self):
        terminate = False
        distance = np.linalg.norm(self.escape_position - self.chase_position)
        if distance <= 0.5:
            terminate = True
        return terminate
# class Environment(gym.Env):#模仿mountain
#     def __init__(self, goal_velocity=0):
#         self.min_action = -1.0
#         self.max_action = 1.0
#         self.min_position = -1.2
#         self.max_position = 0.6
#         self.max_speed = 0.07
#         self.goal_position = (
#             0.45  # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
#         )
#         self.goal_velocity = goal_velocity
#         self.power = 0.0015
#         self.low_state = np.array(
#             [self.min_position, -self.max_speed], dtype=np.float32
#         )
#         self.high_state = np.array(
#             [self.max_position, self.max_speed], dtype=np.float32
#         )
#         self.action_space = spaces.Box(
#             low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float32
#         )
#         self.observation_space = spaces.Box(
#             low=self.low_state, high=self.high_state, dtype=np.float32
#         )
#     def step(self, action):
#         position = self.state[0]
#         velocity = self.state[1]
#         force = min(max(action[0], self.min_action), self.max_action)
#
#         velocity += force * self.power - 0.0025 * math.cos(3 * position)
#         if velocity > self.max_speed:
#             velocity = self.max_speed
#         if velocity < -self.max_speed:
#             velocity = -self.max_speed
#         position += velocity
#         if position > self.max_position:
#             position = self.max_position
#         if position < self.min_position:
#             position = self.min_position
#         if position == self.min_position and velocity < 0:
#             velocity = 0
#         terminated = bool(
#             position >= self.goal_position and velocity >= self.goal_velocity
#         )
#         reward = 0
#         if terminated:
#             reward = 100.0
#         reward -= math.pow(action[0], 2) * 0.1
#
#         self.state = np.array([position, velocity], dtype=np.float32)
#         return self.state, reward, terminated, {}
#     def reset(self,*,
#         seed: Optional[int] = None,
#         return_info: bool = False,options: Optional[dict] = None):
#
#         low, high = utils.maybe_parse_reset_bounds(options, -0.6, -0.4)
#         self.state = np.array([self.np_random.uniform(low=low, high=high), 0])
#         if not return_info:
#             return np.array(self.state, dtype=np.float32)
#         else:
#             return np.array(self.state, dtype=np.float32), {}
#
#     def close(self):
#         pass

################################### Training ###################################


def train():
    print("============================================================================================")

    ####### initialize environment hyperparameters ######
    env_name = "test_simple_2"
    chase = []
    escape = []
    has_continuous_action_space = True  # continuous action space; else discrete

    max_ep_len = 30  # max timesteps in one episode
    max_training_timesteps = int(2e6)  # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 100  # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2  # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e5)  # save model frequency (in num timesteps)

    action_std = 0.6  # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2e5)  # action_std decay frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 4  # update policy every n timesteps
    K_epochs = 80  # update policy for K epochs in one PPO update

    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr_actor = 0.0003  # learning rate for actor network
    lr_critic = 0.001  # learning rate for critic network

    random_seed = 0  # set random seed if required (0 = no random seed)
    #####################################################

    print("training environment name : " + env_name)

    env = Environment()

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    ###################### logging ######################


    #### log files for multiple runs are NOT overwritten
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################
    run_num_pretrained = 0  #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)
    #####################################################

    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
    else:
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                    action_std)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name, "w+")
    log_f.write('max_ep_len{},max_training_timesteps{},action_std{},action_std_decay_rate{},min_action_std{},action_std_decay_freq{},update_timestep{}\n'
                .format(max_ep_len, max_training_timesteps, action_std, action_std_decay_rate, min_action_std, action_std_decay_freq, update_timestep))
    log_f.write('episode,timestep,reward\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0
    Reward = []
    Reward_step =[]
    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    # training loop
    while time_step <= max_training_timesteps:

        state = env.reset()
        current_ep_reward = 0#每一次收益
        Reward_step = []

        for t in range(1, max_ep_len + 1):
            # print("state = ",state)
            # select action with policy
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)
            # print("state = {},reward = {}".format(np.around(state),np.around(reward,2)))
            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward
            Reward_step.append(reward)
            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()
                # print("================PPO update ======================")
            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)
                print("ppo agent action std decay")

            # log in logging file
            # if time_step % log_freq == 0:
            #     # log average reward till last episode
            #     log_avg_reward = log_running_reward / log_running_episodes
            #     log_avg_reward = round(log_avg_reward, 4)
            #
            #     log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
            #     log_f.flush()
            #
            #     log_running_reward = 0
            #     log_running_episodes = 0

            # printing average reward

            if time_step % print_freq == 0:
                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step,
                                                                                        print_avg_reward))
                if(print_avg_reward>1000):
                    print("Reward ={},episodes = {}".format(print_running_reward,print_running_episodes))
                    print(f"Reward = {Reward}")
                    print(f"reward step ={Reward_step}")

                print_running_reward = 0
                print_running_episodes = 0
                Reward =[]
            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                print("already in the terminal")
                break
            # else:
            #     print("not int the terminal")

        # print("-------------------------------------,end of one step")
        print_running_reward += current_ep_reward#计算多次的收益
        print_running_episodes += 1
        Reward.append(current_ep_reward)
        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")

def Test():
    print("============================================================================================")

    ################## hyperparameters ##################

    # env_name = "CartPole-v1"
    # has_continuous_action_space = False
    # max_ep_len = 400
    # action_std = None

    # env_name = "LunarLander-v2"
    # has_continuous_action_space = False
    # max_ep_len = 300
    # action_std = None

    # env_name = "BipedalWalker-v2"
    # has_continuous_action_space = True
    # max_ep_len = 1500           # max timesteps in one episode
    # action_std = 0.1            # set same std for action distribution which was used while saving
    env_name = "test_simple_2"
    has_continuous_action_space = True
    max_ep_len = 1000           # max timesteps in one episode
    action_std = 0.1            # set same std for action distribution which was used while saving

    render = True              # render environment on screen
    frame_delay = 0             # if required; add delay b/w frames

    total_test_episodes = 1   # total num of testing episodes

    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic

    #####################################################

    env = Environment()

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # preTrained weights directory

    random_seed = 0             #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0      #### set this to load a particular checkpoint num

    directory = "PPO_preTrained" + '/' + env_name + '/'
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0

    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        state = env.reset()

        for t in range(1, max_ep_len+1):
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)
            ep_reward += reward

            # if render:
            #     env.render()
            #     time.sleep(frame_delay)

            if done:
                break

        # clear buffer
        ppo_agent.buffer.clear()

        test_running_reward +=  ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0

    env.close()

    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average Test reward : " + str(avg_test_reward))

    print("============================================================================================")

if __name__ == '__main__':
    train()
    # Test()







