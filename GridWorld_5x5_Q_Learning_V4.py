# -*- coding: utf-8 -*-

"""
Look-up table Gridworld environment with trophy and adversary
(Author: E. Morman)

Version 3 is an update to  Version 3 in that it suppresses the print commands 
in order speed up the run time process.

A look-up table Q-learning formulation of the Gridworld problem.
 The orignal code used Q-learning to train an epsilon-greedy agent to find 
 the shortest path between position (0, 0) to opposing corner (Ny-1, Nx-1) of 
 a 2D rectangular grid in the 2D GridWorld environment of size (Ny, Nx).  Anson
 Wong's baseline code solves for an 8x8 Gridworld '


Baseline code leveraged from :
Blog Post for Anson Wong's Original Q-learning Problem
https://towardsdatascience.com/training-an-agent-to-beat-grid-world-fac8a48109a8


Github page for Anson Wong's Original Q-Learning Problem
https://github.com/ankonzoid/LearningX/tree/master/classical_RL/gridworld


The adoption creates two objectives in the Gridworld environment:
    1) The agent must capture a trophy located in the center square: location (2,2)
    2) After capturing the trophy the agent must exit Gridworld: location (4,4)
    

Also, the agent must avoid an adversary that circles the trophy moving in either
a clockwise, counterclockwise, or random pattern around the torphy.

    

 Example optimal policy:
 
Pre-Trophy Movements     
  [[2 1 2 2 2]
   [2 0 2 2 2]
   [1 1 0 3 3]
   [0 2 2 3 0]
   [1 3 3 3 3]]
  
  
Post-Trophy Movements  
  [[1 2 1 1 3]
   [1 1 1 2 2]
   [3 1 1 2 2]
   [3 0 2 2 2]
   [2 1 1 1 0]]

  action['up'] = 0
  action['right'] = 1
  action['down'] = 2
  action['left'] = 3 

"""

import os, sys, random, operator
import numpy as np
import time
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)  # Suppresses scientific notation in Numpy arrays

tic = time.perf_counter()

class Environment:
    
    def __init__(self, Ny=5, Nx=5):
        # Define state space
        self.Ny = Ny  # y grid size, Ny is the number of rows in GridWorld
        self.Nx = Nx  # x grid size, Nx is the number of columns in GridWorld
        self.state_dim = (Ny, Nx)
        # Define action space
        # To write a tuple containing a single value you have to include a comma,
        # even though there is only one value --
        self.action_dim = (4,)  # up, right, down, left
        self.action_dict = {"up": 0, "right": 1, "down": 2, "left": 3}      
        self.movement_dict = {0: "up", 1: "right", 2: "down", 3: "left"}
        self.action_coords = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # translations
        # Define rewards table
        self.r_trophy = 200                 # reward for capturing the trophy (middle square)
        self.r_exit = 100                   # reward for exiting the GridWorld game (bottom right)  
        self.r_nongoal = -1                 # penalty for not reaching terminal state
        self.r_adversary_penalty = -1000    # penalty for adversary capturing the agent
        
        self.R = self._build_initial_rewards()  # R(s,a) agent rewards
        
        # Adversary movement directions
        
        self.adversary_movement_pattern = {"random":0, "clockwise":1, "counterclockwise":2}  
        self.clockwise_adversary = {(3,1):0, (2,1):0, (1,1):1, (1,2):1,
                    (1,3):2, (2,3):2, (3,3):3, (3,2):3} 
        self.counterclockwise_adversary = {(3,1):1, (3,2):1, (3,3):0, (2,3):0,
                    (1,3):3, (1,2):3, (1,1):2, (2,1):2}
        # Check action space consistency
        if len(self.action_dict.keys()) != len(self.action_coords):
            exit("err: inconsistent actions given")


    def reset(self):
        
        self.state_agent = (0, 0)                # Reset agent state to top-left grid corner   
        self.state_adversary = (1,1)             # Reset adversary state to underneath and to the left of the trophy
        self.flag_captured = False               # Update flag status 
        self.R = self._build_initial_rewards()   # Reset the reward matrix, R(s,a) agent rewards
        return self.state_agent


    def step(self, action_agent, action_adversary):
        
        # Evolve agent state
        state_next_agent = (self.state_agent[0] + self.action_coords[action_agent][0],
                      self.state_agent[1] + self.action_coords[action_agent][1])
        
        # Evolve adversary state
        state_next_adversary = (self.state_adversary[0] + self.action_coords[action_adversary][0],
                      self.state_adversary[1] + self.action_coords[action_adversary][1])
        
        # print()
        # print("-----------------TAKE NEXT STEP------------------------------")
        # print("The Capture Flag Status is {}".format(self.flag_captured))
        # print("The Current Agent State is {}".format(self.state_agent))
        # print("The Next Agent State is {}".format(state_next_agent))
        # print("The Current Adversary State is {}".format(self.state_adversary))
        # print("The Next Adversary State is {}".format(state_next_adversary))
              
        # Collect reward and terminate game status
        
        win = 'N/A'
        
        if state_next_agent == state_next_adversary:
            reward = self.r_adversary_penalty
            done = True
            win = 0
            # print()
            # print("******** ADVERSARY CAPTURES AGENT ********")
            # print("******** GRIDWORLD GAME IS OVER ********")
        else:    
            reward = self.R[self.state_agent + (action_agent,)]
        
            # Terminate if we reach bottom-right grid corner and the flag has been captured
            done = (state_next_agent[0] == self.Ny - 1) and (state_next_agent[1] == self.Nx - 1) and \
                self.flag_captured == True
                
            if done == True:
                win = 1
        
            # print() 
            # print("The One-Step Reward is {}".format(reward))
            # print("The One-Step Reward Matrix is:")
            # print(self.R)
            # print()
            # print("Has the Agent Completed the Game? {}".format(done)) 
        
        
        # Update instance attributes self.state_agent and self.state_adversary
              
        self.state_agent = state_next_agent
        self.state_adversary = state_next_adversary
        return state_next_agent, state_next_adversary, reward, done, win
    
    
    def allowed_actions(self):
        
        # Generate list of actions allowed by the agent depending on agent grid location
        actions_allowed = []
        y, x = self.state_agent[0], self.state_agent[1]
        if (y > 0):  # no passing top-boundary
            actions_allowed.append(self.action_dict["up"])
        if (y < self.Ny - 1):  # no passing bottom-boundary
            actions_allowed.append(self.action_dict["down"])
        if (x > 0):  # no passing left-boundary
            actions_allowed.append(self.action_dict["left"])
        if (x < self.Nx - 1):  # no passing right-boundary
            actions_allowed.append(self.action_dict["right"])
        actions_allowed = np.array(actions_allowed, dtype=int)
        return actions_allowed


    def _build_initial_rewards(self):
        # Define agent rewards R[s,a]
        R = self.r_nongoal * np.ones(self.state_dim + self.action_dim, dtype=float)  # R[s,a]
        
        # Reward for capturing the flagin the middle square
        R[((self.Ny + 1)//2)-2, ((self.Nx+1)//2)-1, self.action_dict["down"]] = self.r_trophy
        R[((self.Ny + 1)//2)-1, (self.Nx+1)//2, self.action_dict["left"]] = self.r_trophy
        R[(self.Ny + 1)//2, ((self.Nx+1)//2)-1, self.action_dict["up"]] = self.r_trophy
        R[((self.Ny + 1)//2)-1, ((self.Nx+1)//2)-2, self.action_dict["right"]] = self.r_trophy 
        return R


    def _update_rewards(self):
        
        # Updated reward for moving into the middle square
        self.R[((self.Ny + 1)//2)-2, ((self.Nx+1)//2)-1, self.action_dict["down"]] = self.r_nongoal
        self.R[((self.Ny + 1)//2)-1, (self.Nx+1)//2, self.action_dict["left"]] = self.r_nongoal
        self.R[(self.Ny + 1)//2, ((self.Nx+1)//2)-1, self.action_dict["up"]] = self.r_nongoal
        self.R[((self.Ny + 1)//2)-1, ((self.Nx+1)//2)-2, self.action_dict["right"]] = self.r_nongoal 
        
        # Updated reward for exiting GridWorld
        self.R[self.Ny - 2, self.Nx - 1, self.action_dict["down"]] = self.r_exit  # arrive from above
        self.R[self.Ny - 1, self.Nx - 2, self.action_dict["right"]] = self.r_exit  # arrive from the left
        # print()
        # print("---------------------------------------------------------------")
        # print("Updated Reward Matrix For Capturing the Flag")
        # print(self.R)
        # print("----------------------------------------------------------------")
        
        
    def _check_flag_status(self):  
        
        # print()
        # print("-------------------------------------------------------------")
        # print("Agent moves to state {}".format(self.state_agent))
        # print()
        # print("--------------------TROPHY STATUS IS-------------------------")
        if self.state_agent == (((self.Ny+1)//2)-1,((self.Nx+1)//2)-1) and self.flag_captured == False:
            # print("------------------TROPHY HAS BEEN CAPTURED---------------")
            self._update_rewards()
            self.flag_captured = True 
        # elif self.flag_captured == True:
            # print("------------------TROPHY HAS BEEN CAPTURED---------------")
        # else:
            # print("------------------TROPHY NOT CAPTURED---------------------")


class Agent:
    
    
    def __init__(self, env):
        # Store state and action dimension 
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        # Agent learning parameters
        self.epsilon = 1.0  # initial exploration probability
        # self.epsilon_decay = 8500000  # Control parameter for epsilon decay - 500 iterations
        self.epsilon_decay = 45000000  # Control parameter for epsilon decay _ 1,000 iterations
        # self.epsilon_decay = 4500000000  # Control parameter for epsilon decay _ 5,000 iterations
        # self.epsilon_decay = 120000000000   # Control parameter for epsilon decay - 10,000 iterations
        # self.epsilon_decay = 16000000000000   # Control parameer for epsilon decay - 50,000 iterations
        self.beta = 0.99  # learning rate
        self.gamma = 0.99  # reward discount factor
        # Initialize Q[s,a] table
        self.Q = np.zeros((2,) + self.state_dim + self.action_dim, dtype=float)


    def get_action(self, env, adversary_movements):
        
        # Epsilon-greedy agent policy
        # print()
        # print("-----------------------Get Action Agent----------------------------")
        explore_or_exploit = random.uniform(0,1)
        # print("Epsilon Value is {:.3f}".format(self.epsilon))
        # print("Random N~(0,1) is {:.3f}".format(explore_or_exploit))        
 
        if explore_or_exploit < self.epsilon:
            # explore
            action_agent = np.random.choice(env.allowed_actions())
            # print("Agent Action Taken is to EXPLORE --- Agent Moves {}".format(env.movement_dict[action_agent]))
        else:
            # exploit on allowed actions
            state = env.state_agent;
            actions_allowed = env.allowed_actions()
            if env.flag_captured == False:
                Q_s = self.Q[0,state[0], state[1], actions_allowed]
            else:
                Q_s = self.Q[1,state[0], state[1], actions_allowed]
            actions_greedy = actions_allowed[np.flatnonzero(Q_s == np.max(Q_s))]
            action_agent = np.random.choice(actions_greedy)
            # print("Agent Action Taken is to EXPLOIT --- Agent Moves {}".format(env.movement_dict[action_agent]))
        
        # print()             
        # print("-----------------------Get Action Adversary----------------------------")    
        # print()
        if adversary_movements == 0:
        
            random_adversary_action = random.uniform(0,1)
            # print("Adversary Takes a Random Action")
            # print("Random Number Draw for Adversary Movement is {:.3f}".format(random_adversary_action))
        
            if random_adversary_action <= 0.5:
                # print("Adversary Moves Clockwise")
                action_adversary = env.clockwise_adversary[env.state_adversary]
            
            else:
                # print("Adversary Moves Counterclockwise")
                action_adversary = env.counterclockwise_adversary[env.state_adversary]
                
        elif adversary_movements == 1:
        
             # print("Adversary Moves Clockwise")
             action_adversary = env.clockwise_adversary[env.state_adversary]
            
        elif adversary_movements == 2:
            
            # print("Adversary Moves Counterclockwise")
            action_adversary = env.counterclockwise_adversary[env.state_adversary]
                
        return action_agent, action_adversary


    def train(self, memory, env):
        # -----------------------------
        # Update:
        #
        # Q[s,a] <- Q[s,a] + beta * (R[s,a] + gamma * max(Q[s,:]) - Q[s,a])
        #
        #  R[s,a] = reward for taking action a from state s
        #  beta = learning rate
        #  gamma = discount factor
        # -----------------------------
        (state, action, state_next, reward) = memory
        # This is interesting, even though I have created a tuple that contains important objects
        # I can access the objects in this tuple by their individual name
        
        # print()
        # print("------------------ Train the Q-Factors ----------------------")
        # print("------------------ Q-Learning Training ----------------------")
        # print()
        # print("Previous State {}".format(state))
        # print("Action Taken was {}".format(env.movement_dict[action]))
        # print("Next State {}".format(state_next))
                      
        if env.flag_captured == False:
            Q_level = 0               
        else:
            Q_level = 1
        sa = (Q_level,) + state + (action,)
        
        # print("Next State Q-Value is {}".format(np.max(self.Q[Q_level,state_next])))
        # print("Previous State Q-Value is {}".format(self.Q[sa]))
        # print("Flag Capture Status is {}".format(env.flag_captured))
        # print()
        # print("Current Q-Learning Table:")
        # print(self.Q)
        # print()       
        # print("Beta * [reward + discount*[Q-Value Next_State] - [Q-Value Previous_State]]")
        # print(self.beta * (reward + self.gamma*np.max(self.Q[Q_level,state_next]) - self.Q[sa]))
        
        self.Q[sa] += self.beta * (reward + self.gamma*np.max(self.Q[Q_level,state_next]) - self.Q[sa])
        # print()
        # print("Updated previous state Q-Value is {:.3f}".format(self.Q[sa]))
        # print("Updated Q-Learning Table:")
        # print(self.Q)


    def display_greedy_policy(self, env):
        # greedy policy = argmax[a'] Q[s,a']
        greedy_policy = np.zeros((2,) + (self.state_dim[0], self.state_dim[1]), dtype=int)
        for z in range(2):
            for x in range(self.state_dim[0]):
                for y in range(self.state_dim[1]):
                    greedy_policy[z, y, x] = np.argmax(self.Q[z, y, x, :])
        print("\nGreedy policy(flag status, row, column):")
        print(greedy_policy)
        
        pre_trophy_policy = greedy_policy[0,:,:].reshape(env.Ny, env.Nx)
        buffer = [8, 8, 8, 8, 8]
        pre_trophy_policy = np.vstack((pre_trophy_policy,buffer))        
        post_trophy_policy = greedy_policy[1,:,:].reshape(env.Ny, env.Nx)        
        policy = np.vstack((pre_trophy_policy, post_trophy_policy))
        
        np.savetxt("GridWorld_Policy_Results.txt", policy , delimiter=' ; ', 
                    newline="\n" , fmt= ['%d', '%d', '%d', '%d', '%d'] )
        print()


# Settings
env = Environment(Ny=5, Nx=5)
agent = Agent(env)


# Set Adversary Movements: Either "random", "clockwise", "counterclockwise"
adversary_movements = env.adversary_movement_pattern["clockwise"]

# Train agent
print("\nTraining agent...\n")
N_episodes = 1000
N_episode_data = np.zeros((N_episodes,6))
N_epsilon_data = np.zeros(N_episodes)
row_index_episode_data = 0

for episode in range(N_episodes):
    # print()
    # print()
    # print("----------------------------------------------------------------------")    
    # print("START EPISODE {} AND ITERATION 1".format(episode + 1))
    # Generate an episode
    iter_episode, reward_episode = 0, 0
    state_agent = env.reset()               # Starting state
    # print("Starting Agent State is {}".format(env.state_agent))
    # print("Starting Adversary State is {}".format(env.state_adversary))
    # print("Flag Captured Status is {}".format(env.flag_captured))
    while True:
        action_agent, action_adversary = agent.get_action(env, adversary_movements)                     # get action
        state_next_agent, state_next_adversary, reward, done, win = env.step(action_agent, action_adversary)   # evolve state by action
        agent.train((state_agent, action_agent, state_next_agent, reward),env)                           # train agent
        env._check_flag_status()        
        iter_episode += 1
        # print()
        # print("Completed iteration number {}".format(iter_episode))
        # print("--------------------------------------------------------------")
        # print("START ITERATION NUMBER {}".format(iter_episode + 1))
        reward_episode += reward
        if done:
            # print("************************************************************")
            # print("**                                                        **")
            # print("**             GRIDWORLD GAME IS OVER                     **")
            # print("**                                                        **")
            # print("************************************************************")
            break
        state_agent = state_next_agent  # transition to next state


    # Store episode data
    
    N_episode_data[row_index_episode_data,0] = episode + 1
    N_episode_data[row_index_episode_data,1] = N_episodes
    N_episode_data[row_index_episode_data,2] = agent.epsilon
    N_episode_data[row_index_episode_data,3] = iter_episode
    N_episode_data[row_index_episode_data,4] = reward_episode
    N_episode_data[row_index_episode_data,5] = win
    row_index_episode_data += 1
    
    # if (episode == 0) or (episode + 1) % 10 == 0:
        # print("[episode {}/{}] eps = {:.3F} -> iter = {}, rew = {:.1F}, win = {}".format(
            # episode + 1, N_episodes, agent.epsilon, iter_episode, reward_episode, win))
        
    # Update epsilon-decay exploration parameter
    # agent.epsilon = max(agent.epsilon * agent.epsilon_decay, 0.01) 
    N_epsilon_data[episode] = agent.epsilon       
    agent.epsilon = agent.epsilon / (1 + (((episode+1)**2) / (agent.epsilon_decay + episode+1)))      

    # Print episode data and greedy policy
    if (episode == N_episodes - 1):
        print()
        print("*********************************************************************")
        print("**                                                                 **")
        print("**                SUMMARY TABLE OF RESULTS                         **")
        print("**                                                                 **")
        print("*********************************************************************") 
        print()
        for i in range(N_episodes):            
            if (i == 0) or (i + 1) % 10 == 0:            
                print("[episode {}/{}] eps = {:.3F} -> iter = {}, rew = {:.1F}, win = {}".format(
                    N_episode_data[i][0], N_episode_data[i][1], N_episode_data[i][2],
                    N_episode_data[i][3], N_episode_data[i][4], N_episode_data[i][5]))
        
        print()
        print("There were {} GridWorld wins over the total {} simulation episodes".format(
            N_episode_data.sum(axis=0)[5], N_episodes))      
                
        agent.display_greedy_policy(env)
        
        np.savetxt("GridWorld_Results.txt", N_episode_data, delimiter=' ; ', newline="\n", 
                   fmt= ['%d', '%d', '%.3f', '%d', '%.1f', '%d'])
        
        
        for (key, val) in sorted(env.action_dict.items(), key=operator.itemgetter(1)):
            # The sorted(env.action_dict.items(), key=operator.itemgetter(1)) sequence converts
            # the env.action_dict = {'up': 0, 'right': 1, 'down': 2, 'left':3}
            # into an ordered list of tuples, that are ordered on the "values" of the action_dict
            # [('up', 0), ('right', 1), ('down', 2), ('left', 3)]
            print(" action['{}'] = {}".format(key, val))
        print()
        
        toc = time.perf_counter()
        
        print(f"GridWorld Trainig Time Took {toc-tic:0.3f} Seconds")
        
        plt.plot(N_epsilon_data)