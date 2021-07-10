# Introduction

This repository contains the codes related to the paper Solving reward-collecting problems with UAVs: a comparison of online optimization and Q-learning. There are three sets of codes for the three methods described in the paper, Deep Q-Learning, Online Optimization, and $\epsilon$-greedy tabular Q-Learning. For each method, we post codes that explore either a 5x5 grid environment with one adversary or a 9x9 grid environment two adversaries. 

# Requirements

* Tensorflow 2.5.0
* Keras 2.4.3
* OpenAI Gym 0.18.3
* Gurobi

# Usage

### Deep Q-Learning

The Deep Q-Learning codes are titled with the structure Deep_Q_Learning_5x5_clockwise_advs, where 5x5 represents the size of the environment, clockwise denotes the type of adversarial movement, and advs representing that there is a single adversary. In the main program of the code, the main variables that can be changed for experimentation purposes are:

* for loop with variable z- number of experimentation repetitions
* maze_size - size of the grid environment
* flags - initial locations of the rewards
* advs - initial locations of the adversaries(ensure the adversaries start in a cell adjacent to the flag)

### Online Optimization

Our Online Optimization codes are named similarly to our Deep Q-Learning codes, with the exception that they are in Jupyter Notebook format. The initial blocks of codes break down the code, while the section under Final Code is what should be used for actual experimentation. The main variables that can be changed for experimentation purposes are:


* rewards - locations of rewards
* maze_size - size of the grid environment
* current_adversary_location - initial locations of the adversaries(ensure the adversaries start in a cell adjacent to the flag)
* for loop with variable e - number of experimentation repetitions.

### $\epsilon$-greedy tabular Q-Learning


# Resources

The basis and inspiration of our Deep Q-Learning code comes from Professor Samy Zafrany's Deep Reinforcement Learning
The Tour De Flags test case, which can be found at: https://www.samyzaf.com/ML/tdf/tdf.html

# Contact

For questions, please contact either Yixuan Liu at ben.liu31492@gmail.com, Professor Chrysafis Vogiatzis at chrys@illinois.edu, Professor Ruriko Yoshida at ryoshida@nps.edu, or Professor Erich Morman at edmorman@nps.edu.
