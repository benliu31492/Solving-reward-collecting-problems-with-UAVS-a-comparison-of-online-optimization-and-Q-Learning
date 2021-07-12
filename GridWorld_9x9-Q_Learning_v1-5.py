# -*- coding: utf-8 -*-

"""
June 9, 2021
GridWorld_9x9-Q_Learning_v1-4.py
(Author: E. Morman)

Version 1.5: Cleans up v1.4 output and print commands used in the debugging of the issue with opening the exit gate
in the GridWorld problem

Properties of the GridWorld environment:
    1. 9x9 Gridworld scenario with two trophies and two adversaries
    2. Starting point for agent is row 1 column 1
    3. Location of trophies:
        a) Trophy 1 is located in row 2 column 7
        b) Trophy 2 is located in row 8 column 2
    4. Agent must successfully capture both trophies and then exit GridWorld at location row 9 column 9

Program adapted from Anson Wong
Anson Wong's baseline code solves a no adversary game for an 8x8 Gridworld '
A look-up table Q-learning formulation of the Gridworld problem.
The original code used Q-learning to train an epsilon-greedy agent to find
the shortest path between position (0, 0) to opposing corner (Ny-1, Nx-1) of
a 2D rectangular grid in the 2D GridWorld environment of size (Ny, Nx).


Baseline code leveraged from :
Blog Post for Anson Wong's Original Q-learning Problem
https://towardsdatascience.com/training-an-agent-to-beat-grid-world-fac8a48109a8

Github page for Anson Wong's Original Q-Learning Problem
https://github.com/ankonzoid/LearningX/tree/master/classical_RL/gridworld


Example optimal policy:

action['up'] = 0
action['right'] = 1
action['down'] = 2
action['left'] = 3

No Trophies Captured
  [[2 1 1 1 1 2]
  [1 1 1 1 1 2]
  [1 1 1 1 1 2]
  [1 1 1 1 1 2]
  [1 1 1 1 1 2]
  [1 1 1 1 1 0]]

Only Upper Right Trophy Captured
  [[1 1 1 1 1 2]
  [1 2 3 3 3 2]
  [1 2 1 1 1 2]
  [1 2 1 1 1 2]
  [1 2 1 1 1 2]
  [1 1 1 1 1 0]]

Only Bottom Left Trophy Captured
  [[1 1 1 1 1 2]
  [1 1 1 1 1 2]
  [1 0 1 1 1 2]
  [1 0 1 1 1 2]
  [1 0 1 1 1 2]
  [1 1 1 1 1 0]]

Both Trophies Captured
  [[1 1 1 1 1 2]
  [1 1 1 1 2 2]
  [1 1 1 1 2 2]
  [1 1 1 1 2 2]
  [1 1 1 1 2 2]
  [1 1 1 1 1 0]]

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
    
    def __init__(self, Ny=9, Nx=9):
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
        self.r_trophy_top_right = 200       # reward for capturing top right trophy (1,7)
        self.r_trophy_bottom_left = 200     # reward for capturing bottom left trophy (7,1)    
        self.r_exit = 100                   # reward for exiting the GridWorld game (bottom right)  
        self.r_nongoal = -1                 # penalty for each step taken
        self.r_adversary_penalty = -1000    # penalty for advesary capturing the agent
        
        self.R = self._build_initial_rewards()  # R(s,a) agent rewards
        
        # Adversary movement directions
        
        self.adversary_movement_pattern = {"random":0, "clockwise":1, "counterclockwise":2}  
        self.clockwise_adversary_top_right = {(2,6):0, (1,6):0, (0,6):1, (0,7):1,
                    (0,8):2, (1,8):2, (2,8):3, (2,7):3} 
        self.clockwise_adversary_bottom_left = {(8,0):0, (7,0):0, (6,0):1, (6,1):1,
                    (6,2):2, (7,2):2, (8,2):3, (8,1):3}         
        self.counterclockwise_adversary_top_right = {(2,6):1, (2,7):1, (2,8):0, (1,8):0,
                    (0,8):3, (0,7):3, (0,6):2, (1,6):2}
        self.counterclockwise_adversary_bottom_left = {(8,0):1, (8,1):1, (8,2):0, (7,2):0,
                    (6,2):3, (6,1):3, (6,0):2, (7,0):2}
        # Check action space consistency
        if len(self.action_dict.keys()) != len(self.action_coords):
            exit("err: inconsistent actions given")


    def reset(self):
        
        self.state_agent = (0, 0)                   # Reset agent state to top-left grid corner   
        self.state_adversary_top_right = (0,6)      # Reset top right adversary state 
        self.state_adversary_bottom_left = (6,0)    # Reset bottom left adversary state   
        self.flag_captured_top_right = False        # Update top right flag status
        self.flag_captured_bottom_left = False      # Update bottom left flag status
        self.R = self._build_initial_rewards()   # Reset the reward matrix, R(s,a) agent rewards
        self.exit_open = False
        return self.state_agent


    def step(self, action_agent, action_adversary_top_right, action_adversary_bottom_left):
        
        # Evolve agent state
        state_next_agent = (self.state_agent[0] + self.action_coords[action_agent][0],
                self.state_agent[1] + self.action_coords[action_agent][1])
        
        # Evolve adversaries states
        state_next_adversary_top_right = (self.state_adversary_top_right[0] + self.action_coords[action_adversary_top_right][0],
                self.state_adversary_top_right[1] + self.action_coords[action_adversary_top_right][1])
        
        state_next_adversary_bottom_left = (self.state_adversary_bottom_left[0] + self.action_coords[action_adversary_bottom_left][0],
                self.state_adversary_bottom_left[1] + self.action_coords[action_adversary_bottom_left][1])
        
        #print()
        #print("-----------------TAKE NEXT STEP------------------------------")
        #print("The Capture Top-Right Flag Status is {}".format(self.flag_captured_top_right))
        #print("The Capture Bottom-Left Flag Status is {}".format(self.flag_captured_bottom_left))
        #print()
        #print("The Current Agent State is {}".format(self.state_agent))
        #print("The Next Agent State is {}".format(state_next_agent))
        #print()
        #print("The Current Top-Right Adversary State is {}".format(self.state_adversary_top_right))
        #print("The Next Top-Right Adversary State is {}".format(state_next_adversary_top_right))
        #print()
        #print("The Current Bottom-Left Adversary State is {}".format(self.state_adversary_bottom_left))
        #print("The Next Bottom-Left Adversary State is {}".format(state_next_adversary_bottom_left))      
        
        
        
        # Collect reward and terminate game status
        
        win = 'N/A'
        
        if state_next_agent == state_next_adversary_top_right or state_next_agent == state_next_adversary_bottom_left:
            reward = self.r_adversary_penalty
            done = True
            win = 0
            #print()
            #print("******** ADVERSARY CAPTURES AGENT ********")
            #print("******** GRIDWORLD GAME IS OVER ********")
        else:    
            reward = self.R[self.state_agent + (action_agent,)]
        
            # Terminate if we reach bottom-right grid corner and the flag has been captured
            done = (state_next_agent[0] == self.Ny - 1) and (state_next_agent[1] == self.Nx - 1) and \
                self.flag_captured_top_right == True and self.flag_captured_bottom_left == True
                
            if done == True:
                win = 1
        
            #print() 
            #print("The One-Step Reward is {}".format(reward))
            #print("The One-Step Reward Matrix is:")
            #print(self.R)
            #print()
            #print("Has the Agent Completed the Game? {}".format(done)) 
        
        
        # Update instance attributes self.state_agent and self.state_adversary
              
        self.state_agent = state_next_agent
        self.state_adversary_top_right = state_next_adversary_top_right
        self.state_adversary_bottom_left = state_next_adversary_bottom_left
        return state_next_agent, state_next_adversary_top_right, \
            state_next_adversary_bottom_left, reward, done, win
    
    
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
        
        # Reward for capturing the flag in the top-right location_by moving down, left, up, or right
        R[ 0 , 7 , self.action_dict["down"]] = self.r_trophy_top_right
        R[ 1 , 8 , self.action_dict["left"]] = self.r_trophy_top_right
        R[ 2 , 7 , self.action_dict["up"]] = self.r_trophy_top_right
        R[ 1 , 6 , self.action_dict["right"]] = self.r_trophy_top_right
        
        # Reward for capturing the flag in the bottom-left location_by moving down, left, up, or right
        R[ 6 , 1 , self.action_dict["down"]] = self.r_trophy_bottom_left
        R[ 7 , 2 , self.action_dict["left"]] = self.r_trophy_bottom_left
        R[ 8 , 1 , self.action_dict["up"]] = self.r_trophy_bottom_left        
        R[ 7 , 0 , self.action_dict["right"]] = self.r_trophy_bottom_left
        return R


    def _update_rewards(self):
        
        
        # Updated reward for capturing the top-right trophy        
        if self.state_agent == (self.Ny-8, self.Nx-2) and self.flag_captured_top_right == False:
            self.R[self.Ny-self.Ny, self.Nx-2, self.action_dict["down"]] = self.r_nongoal
            self.R[self.Ny-8, self.Nx-1, self.action_dict["left"]] = self.r_nongoal
            self.R[self.Ny-7, self.Nx-2, self.action_dict["up"]] = self.r_nongoal
            self.R[self.Ny-8, self.Nx-3, self.action_dict["right"]] = self.r_nongoal 
        
            self.flag_captured_top_right = True

            # Update the exit reward if both trophies are now captured
            if self.flag_captured_bottom_left == True:
                self.R[self.Ny - 2, self.Nx - 1, self.action_dict["down"]] = self.r_exit  # arrive from above
                self.R[self.Ny - 1, self.Nx - 2, self.action_dict["right"]] = self.r_exit  # arrive from the left
                self.exit_open = True


       # Update reward for capturing the bottom-left trophy 
        elif self.state_agent == (self.Ny-2, self.Nx-8) and self.flag_captured_bottom_left == False:
            self.R[self.Ny-3, self.Nx-8, self.action_dict["down"]] = self.r_nongoal
            self.R[self.Ny-2, self.Nx-7, self.action_dict["left"]] = self.r_nongoal
            self.R[self.Ny-1, self.Nx-8, self.action_dict["up"]] = self.r_nongoal
            self.R[self.Ny-2, self.Nx-self.Nx, self.action_dict["right"]] = self.r_nongoal 
        
            self.flag_captured_bottom_left = True

            # Update the exit reward if both trophies are now captured
            if self.flag_captured_top_right == True:
                self.R[self.Ny - 2, self.Nx - 1, self.action_dict["down"]] = self.r_exit  # arrive from above
                self.R[self.Ny - 1, self.Nx - 2, self.action_dict["right"]] = self.r_exit  # arrive from the left
                self.exit_open = True



    def _check_flag_status(self):  

        #print()
        #print("-------------------------------------------------------------")
        #print("Agent moves to state {}".format(self.state_agent))
        #print()
        #print("--------------------TROPHY STATUS IS-------------------------")

        # No need update rewards if both trophies are already captured
        if (self.flag_captured_top_right == True) and (self.flag_captured_bottom_left == True):
            return

        elif self.state_agent == (self.Ny-8, self.Nx-2) and self.flag_captured_top_right == False:
            #print("-------------TOP-RIGHT TROPHY HAS JUST BEEN CAPTURED---------------")
            self._update_rewards()

        elif self.state_agent == (self.Ny-2, self.Nx-8) and self.flag_captured_bottom_left == False:
            #print("-------------BOTTOM-LEFT TROPHY HAS JUST BEEN CAPTURED---------------")
            self._update_rewards()


        '''
        if self.flag_captured_top_right == True and self.flag_captured_bottom_left == False:
            print()
            print("------------TOP-RIGHT TROPHY HAS BEEN CAPTURED---------------")
            print("--------BOTTOM-LEFT TROPHY HAS NOT BEEN CAPTURED---------------")
        elif self.flag_captured_top_right == False and self.flag_captured_bottom_left == True:        
            #print()
            print("---------TOP-RIGHT TROPHY HAS NOT BEEN CAPTURED---------------")
            print("----------BOTTOM-LEFT TROPHY HAS BEEN CAPTURED---------------")
        elif self.flag_captured_top_right == True and self.flag_captured_bottom_left == True:  
            print()
            print("------------BOTH TROPHIES HAVE BEEN CAPTURED---------------")                        
        else:
            print
            print("-----------NEITHER TROPHY HAS BEEN CAPTURED---------------------")
        '''

class Agent:
    
    
    def __init__(self, env):
        # Store state and action dimension 
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        # Agent learning parameters
        self.epsilon = 1.0  # initial exploration probability
        # self.epsilon_decay = 160000000 # Epsilon Decay Control Parameter for 1,500 iterations
        # self.epsilon_decay = 500000000 # Epsilon Decay Control Parameter for 2,000 iterations
        # self.epsilon_decay = 1000000000 # Epsilon Decay Control Parameter for 2,500 iterations
        self.epsilon_decay = 4500000000  # Epsilon Decay Control Parameter for 5,000 iterations
        # self.epsilon_decay = 15000000000  # Epsilon Decay Control Parameter for 50,000 iterations
        # self.epsilon_decay = 120000000000   # Epsilon Decay Control Parameter for 100,000 iterations
        # self.epsilon_decay =  68000000000000000000 # Epsilon Decay Control Parameter for 10,000,000
        self.beta = 0.99  # learning rate
        self.gamma = 0.99  # reward discount factor
        # Initialize Q-Factor Look-Up Table
            # Index 1 -> Trophy status {No trophies taken, Top-right only trophy taken,
                # Bottom-left only trophy taken, both trophies taken}
            # Index 2 ->  Row Position
            # Index 3 ->  Column Position
            # Index 4 ->  Directional Movement {Up, Right, Down, Left}
        self.Q = np.zeros((4,) + self.state_dim + self.action_dim, dtype=float)


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
            state = env.state_agent
            actions_allowed = env.allowed_actions()
            if env.flag_captured_top_right == False and env.flag_captured_bottom_left == False:
                Q_s = self.Q[0, state[0], state[1], actions_allowed]
            elif env.flag_captured_top_right == True and env.flag_captured_bottom_left == False:
                 Q_s = self.Q[1, state[0], state[1], actions_allowed]
            elif env.flag_captured_top_right == False and env.flag_captured_bottom_left == True:
                 Q_s = self.Q[2, state[0], state[1], actions_allowed]    
            elif env.flag_captured_top_right == True and env.flag_captured_bottom_left == True:
                Q_s = self.Q[3, state[0], state[1], actions_allowed]
            actions_greedy = actions_allowed[np.flatnonzero(Q_s == np.max(Q_s))]
            action_agent = np.random.choice(actions_greedy)
            #print("Agent Action Taken is to EXPLOIT --- Agent Moves {}".format(env.movement_dict[action_agent]))
        
        # print()             
        # print("-----------------------Get Action Adversary----------------------------")    
        # print()
        if adversary_movements == 0:
        
            random_adversary_action = random.uniform(0,1)
            #print("Adversaries Take Random Actions")
            #print("Random Number Draw for Adversary Movement is {:.3f}".format(random_adversary_action))
        
            if random_adversary_action <= 0.5:
                #print("Adversaries Move Clockwise")
                action_adversary_top_right = env.clockwise_adversary_top_right[env.state_adversary_top_right]
                action_adversary_bottom_left = env.clockwise_adversary_bottom_left[env.state_adversary_bottom_left]
            
            else:
                #print("Adversaries Move Counterclockwise")
                action_adversary_top_right = env.counterclockwise_adversary_top_right[env.state_adversary_top_right]
                action_adversary_bottom_left = env.counterclockwise_adversary_bottom_left[env.state_adversary_bottom_left]
                
        elif adversary_movements == 1:
        
             #print("Adversaries Move Clockwise")
             action_adversary_top_right = env.clockwise_adversary_top_right[env.state_adversary_top_right]
             action_adversary_bottom_left = env.clockwise_adversary_bottom_left[env.state_adversary_bottom_left]
                        
        elif adversary_movements == 2:
            
            #print("Adversaries Move Counterclockwise")
            action_adversary_top_right = env.counterclockwise_adversary_top_right[env.state_adversary_top_right]
            action_adversary_bottom_left = env.counterclockwise_adversary_bottom_left[env.state_adversary_bottom_left]
            
        return action_agent, action_adversary_top_right, action_adversary_bottom_left


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
        
        #print()
        #print("------------------ Train the Q-Factors ----------------------")
        #print("------------------ Q-Learning Training ----------------------")
        #print()
        #print("Previous Agent State {}".format(state))
        #print("Action Taken was {}".format(env.movement_dict[action]))
        #print("Next Agent State {}".format(state_next))
                      
        if env.flag_captured_top_right == False and env.flag_captured_bottom_left == False:
            Q_level = 0               
        elif env.flag_captured_top_right == True and env.flag_captured_bottom_left == False:
            Q_level = 1
        elif env.flag_captured_top_right == False and env.flag_captured_bottom_left == True:
            Q_level = 2
        elif env.flag_captured_top_right == True and env.flag_captured_bottom_left == True:
            Q_level = 3
        
        sa = (Q_level,) + state + (action,)
        
        #print("Next State Q-Value is {}".format(np.max(self.Q[Q_level,state_next])))
        #print("Previous State Q-Value is {}".format(self.Q[sa]))
        #print()
        #print("Top-Right Flag Capture Status is {}".format(env.flag_captured_top_right))
        #print("Bottom-Left Flag Capture Status is {}".format(env.flag_captured_bottom_left))        
        #print()
        #print("Current Q-Learning Table:")
        #print(self.Q)
        #print()       
        #print("Beta * [reward + discount*[Q-Value Next_State] - [Q-Value Previous_State]]")
        #print(self.beta * (reward + self.gamma*np.max(self.Q[Q_level,state_next]) - self.Q[sa]))
        
        self.Q[sa] += self.beta * (reward + self.gamma*np.max(self.Q[Q_level,state_next]) - self.Q[sa])
        #print()
        #print("Updated previous state Q-Value is {:.3f}".format(self.Q[sa]))
        #print("Updated Q-Learning Table:")
        #print(self.Q)


    def display_greedy_policy(self, env):
        # greedy policy = argmax[a'] Q[s,a']
        greedy_policy = np.zeros((4,) + (self.state_dim[0], self.state_dim[1]), dtype=int)
        for z in range(4):
            for x in range(self.state_dim[0]):
                for y in range(self.state_dim[1]):
                    greedy_policy[z, y, x] = np.argmax(self.Q[z, y, x, :])
        print("\nGreedy policy(flag status, row, column):")
        print(greedy_policy)
        
        buffer = [8, 8, 8, 8, 8, 8, 8, 8, 8]
        pre_trophy_policy = greedy_policy[0,:,:].reshape(env.Ny, env.Nx)
        # pre_trophy_policy = np.vstack((pre_trophy_policy, buffer))        
        top_right_trophy_first = greedy_policy[1,:,:].reshape(env.Ny, env.Nx)        
        bottom_left_trophy_first = greedy_policy[2,:,:].reshape(env.Ny, env.Nx)
        both_trophies_captured = greedy_policy[3,:,:].reshape(env.Ny, env.Nx)
        
        policy = np.vstack((pre_trophy_policy, buffer, top_right_trophy_first,
            buffer, bottom_left_trophy_first, buffer, both_trophies_captured))
        
        np.savetxt("GridWorld_Policy_Results.txt", policy , delimiter=' ; ', 
                    newline="\n" , fmt= ['%d', '%d', '%d', '%d', '%d', '%d' , '%d' , '%d' , '%d'] )
        print()


# Settings
env = Environment(Ny=9, Nx=9)
agent = Agent(env)


# Set Adversary Movements: Either "random", "clockwise", "counterclockwise"
adversary_movements = env.adversary_movement_pattern["counterclockwise"]

# Train agent
print("\nTraining agent...\n")
N_episodes = 5000
N_episode_data = np.zeros((N_episodes,10))
N_epsilon_data = np.zeros(N_episodes)
row_index_episode_data = 0

for episode in range(N_episodes):
    #print()
    #print()
    #print("----------------------------------------------------------------------")    
    #print("START EPISODE {} AND ITERATION 1".format(episode + 1))
    # Generate an episode
    iter_episode, reward_episode = 0, 0
    state_agent = env.reset()               # Starting state
    #print("Starting Agent State is {}".format(env.state_agent))
    #print("Starting Top-Right Adversary State is {}".format(env.state_adversary_top_right))
    #print("Starting Bottom-Left Adversary State is {}".format(env.state_adversary_bottom_left))
    #print("Top-Right Flag Captured Status is {}".format(env.flag_captured_top_right))
    #print("Bottom-Left Flag Captured Status is {}".format(env.flag_captured_bottom_left))
    while True:
        # Get agent action and adversaries actions
        action_agent, action_adversary_top_right, action_adversary_bottom_left \
            = agent.get_action(env, adversary_movements) 

        # Evolve agent state and adversaries state by actions
        state_next_agent, state_next_adversary_top_right, state_next_adversary_bottom_left, reward, done, win \
            =  env.step(action_agent, action_adversary_top_right, action_adversary_bottom_left)
        
        # Train agent
        agent.train((state_agent, action_agent, state_next_agent, reward),env)   
                        
        env._check_flag_status()
        iter_episode += 1

        #print()
        #print("Completed iteration number {}".format(iter_episode))
        #print("--------------------------------------------------------------")
        #print("START ITERATION NUMBER {}".format(iter_episode + 1))
        reward_episode += reward
        if done:
            #print("************************************************************")
            #print("**                                                        **")
            #print("**             GRIDWORLD GAME IS OVER                     **")
            #print("**                                                        **")
            #print("************************************************************")
            break
        state_agent = state_next_agent  # transition to next state


    # Store episode data
    
    N_episode_data[row_index_episode_data,0] = episode + 1
    N_episode_data[row_index_episode_data,1] = N_episodes
    N_episode_data[row_index_episode_data,2] = agent.epsilon
    N_episode_data[row_index_episode_data,3] = iter_episode
    N_episode_data[row_index_episode_data,4] = reward_episode
    N_episode_data[row_index_episode_data,5] = env.flag_captured_top_right
    N_episode_data[row_index_episode_data,6] = env.flag_captured_bottom_left
    N_episode_data[row_index_episode_data,7] = env.R[7,8,2]
    N_episode_data[row_index_episode_data,8] = env.R[8,7,1]
    N_episode_data[row_index_episode_data,9] = win
    row_index_episode_data += 1

    # Update epsilon-decay exploration parameter
    
    # agent.epsilon = max(agent.epsilon * agent.epsilon_decay, 0.01)        
    N_epsilon_data[episode] = agent.epsilon 
    agent.epsilon = agent.epsilon / (1 + (((episode+1)**2) / (agent.epsilon_decay + episode+1)))

    # Print episode data and greedy policy
    if (episode == N_episodes - 1):
        '''
        print()
        print("*********************************************************************")
        print("**                                                                 **")
        print("**                SUMMARY TABLE OF RESULTS                         **")
        print("**                                                                 **")
        print("*********************************************************************") 
        print()
        for i in range(N_episodes):            
            if (i == 0) or (i + 1) % 10 == 0:            
                print("[episode {}/{}] eps = {:.3F} -> iter = {}, rew = {:.1F}, t-right trophy ={}," \
                    "b-left trophy = {}, exit_reward_down = {}, exit_reward_right = {}, win = {}".format(
                    N_episode_data[i][0], N_episode_data[i][1], N_episode_data[i][2], N_episode_data[i][3],
                    N_episode_data[i][4], N_episode_data[i][5], N_episode_data[i][6],
                    N_episode_data[i][7], N_episode_data[i][8], N_episode_data[i][9]))
        '''
        print()
        print("There were {} GridWorld wins over the total {} simulation episodes".format(
            N_episode_data.sum(axis=0)[9], N_episodes))
                
        agent.display_greedy_policy(env)
        
        np.savetxt("GridWorld_Results.txt", N_episode_data, delimiter=' ; ', newline="\n", 
                   fmt= ['%d', '%d', '%.3f', '%d', '%.1f', '%s', '%s', '%d', '%d', '%d'])
        
        with open('Q-Values.txt', 'w') as outfile:
             outfile.write('# Array shape: {0} ; \n'.format(agent.Q.shape))
             trophy_status = ['No Trophy','Top-Right Only','Bottom-Left Only','Both Trophies']
             trophy_indicator = 0
             row_number = 1
             for threeD_data_slice in agent.Q:
                 for twoD_data_slice in threeD_data_slice:
                     outfile.write('# Trophy Status: {}, Row Number: {} ; \n'.format(trophy_status[trophy_indicator],row_number))
                     np.savetxt(outfile, twoD_data_slice, delimiter = ';', fmt='%.2f')
                     row_number +=1
                 trophy_indicator +=1
                 row_number = 1

        
        for (key, val) in sorted(env.action_dict.items(), key=operator.itemgetter(1)):
            # The sorted(env.action_dict.items(), key=operator.itemgetter(1)) sequence converts
            # the env.action_dict = {'up': 0, 'right': 1, 'down': 2, 'left':3}
            # into an ordered list of tuples, that are ordered on the "values" of the action_dict
            # [('up', 0), ('right', 1), ('down', 2), ('left', 3)]
            print(" action['{}'] = {}".format(key, val))
        print()
        
        toc = time.perf_counter()
        
        print(f"GridWorld Training Time Took {toc-tic:0.3f} Seconds")
        
        plt.plot(N_epsilon_data, 'tab:blue')