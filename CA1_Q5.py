# COMP532 Question 5
import numpy as np
import random
import matplotlib.pyplot as plt

"""
Create the reward board
"""
rewards = np.ones((3,12)) * -1
cliff = np.array([[-1, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -1]])
rewards = np.append(rewards, cliff, axis=0)
start = (3,0)
goal = (3, 11)

"""
Functions
"""

def action(state, epsi, qvals):
    """
    Function to choose an action for an agent

    Parameters:
        state: tuple, int - the current state of the agent (x,y), 0 <= x <= 3 and 0 <= y <= 11
        epsi: float - epsilon, the frequency the agent will explore in its next action
        qvals: np.array - the current q-values that the agent knows. Used to select greedy actions

    Return:
        act: int - the action the agent will take. 0:up, 1:right, 2:down, 3:left
    """

    if np.random.random() < epsi: # explore

        act = random.choice(np.arange(4))

    else: # greedy action

        act = np.argmax(qvals[to_index(state)])

    return act

def movement(action, position):
    """
    Function to move the agent to the next state/position

    Parameters:
        action: int - action that the agent is taking. 0:up, 1:right, 2:down, 3:left
        position: tuple, int - the current state of the agent (x,y), 0 <= x <= 3 and 0 <= y <= 11


    Return:
        next_space: tuple, int - the next position of the agent
    """

    if action == 0 and position[0] > 0: # up action
        next_space = (position[0] - 1, position[1])

    elif action == 1 and position[1] < 11: # right action
        next_space = (position[0], position[1] + 1)

    elif action == 2 and position[0] < 3: # down action
        next_space = (position[0] + 1, position[1])

    elif action == 3 and position[1] > 0: # left action
        next_space = (position[0], position[1] -1)
        
    else: # bumping into wall
        next_space = position
    
    return next_space


def to_index(state):
    """
    Function to convert a state/position of an agent to a single int index. From (x,y) -> z where 0 <= z <= 47

    Parameters:
        state: tuple, int - the current state of the agent (x,y), 0 <= x <= 3 and 0 <= y <= 11

    Return:
        state index: int - the index position of the agent
    """
    return state[0] * 12 + state[1]


def Qlearning(rewards, gamma, alpha, iterations, epsilon, runs=10):
    """
    Q-learning function. Agent will learn a path according to the off-policy temporal difference control algorithm.

    Parameters:
        rewards: np.array - array of the rewards board for the cliff walking game. Shape of (4,12)
        gamma: float - value of future reward
        alpha: float - learning rate
        iterations: int - number of episodes in each run
        epsilon: float - rate of exploration 
        runs: int - number of runs to complete, default=10


    Return:
        all_rewards: np.array - all rewards from each episode in each run. Shape of (runs, episodes)
    """

    all_rewards = []
    
    for run in range(runs):

        run_rewards = []
        qvalues = np.zeros((rewards.shape[0]*rewards.shape[1], rewards.shape[0]))

        for episode in range(iterations):

            initial_state = start

            current_state = initial_state

            episode_rewards = 0

            while True:
                # choose action, return 0,1,2, or 3
                next_action = action(current_state, epsilon, qvalues)
                
                # move to next state with selected action, return int of new state
                next_state = movement(next_action, current_state)

                # convert the states from tuple position coordinates (x, y) to single int index based on board positions
                # numbered 0-47
                current_state_idx = to_index(current_state)
                next_state_idx = to_index(next_state)

                # update qvals
                qvalues[current_state_idx][next_action] = qvalues[current_state_idx][next_action] + alpha*(rewards[next_state] + gamma*max(qvalues[next_state_idx]) - qvalues[current_state_idx][next_action])

                # update state
                current_state = next_state

                episode_rewards += rewards[current_state]

                # some logic to break the loop if/when agent reaches cliff/goal state
                if current_state[0] == 3 and current_state[1] >= 1:
                    break
            
            run_rewards.append(episode_rewards)

        all_rewards.append(run_rewards)


    return np.array(all_rewards)
    

def SARSA(rewards, gamma, alpha, iterations, epsilon, runs=10):
    """
    SARSA function. Agent will learn a path according to the on-policy temporal difference control algorithm.

    Parameters:
        rewards: np.array - array of the rewards board for the cliff walking game. Shape of (4,12)
        gamma: float - value of future reward
        alpha: float - learning rate
        iterations: int - number of episodes in each run
        epsilon: float - rate of exploration 
        runs: int - number of runs to complete, default=10


    Return:
        all_rewards: np.array - all rewards from each episode in each run. Shape of (runs, episodes)
    """
    all_rewards = []
    
    for run in range(runs):
        
        run_rewards = []
        qvalues = np.zeros((rewards.shape[0]*rewards.shape[1], rewards.shape[0]))

        for episode in range(iterations):

            initial_state = start

            episode_rewards = 0

            initial_action = action(initial_state, epsilon, qvals=qvalues)

            current_state = initial_state

            while True:

                next_state = movement(initial_action, current_state)

                next_action = action(next_state, epsilon, qvals=qvalues)

                current_state_idx = to_index(current_state)
                next_state_idx = to_index(next_state)


                # update Q values
                qvalues[current_state_idx][initial_action] = qvalues[current_state_idx][initial_action] + alpha*(rewards[next_state] + gamma*qvalues[next_state_idx][next_action] - qvalues[current_state_idx][initial_action])

                current_state = next_state
                initial_action = next_action

                episode_rewards += rewards[current_state]

                if current_state[0] == 3 and current_state[1] >= 1:
                    break
            
            run_rewards.append(episode_rewards)

        all_rewards.append(run_rewards)
        
    return np.array(all_rewards)

def moving_average(vals, n) :
    """
    Function to calculate the moving average of an array.

    Parameters:
        vals: np.array - array of rewards generated from a learning agent's episodes
        n: int - the number of values to include for the running average

    Return:
        moving_av: np.array - array of moving averaged rewards
    """
    cumsum = np.cumsum(vals, dtype=float)
    moving_av = (cumsum[n:] - cumsum[:-n])/n

    return moving_av

def averager(run_reward):
    """
    Function to average the rewards over an agent's runs.

    Parameters:
        run_reward: np.array - array of summed rewards from mulitple runs of shape (runs, episodes)

    Return:
        averaged rewards: np.array - array of averaged rewards over all runs
    """
    
    return sum(run_reward)/len(run_reward)

"""
Train agents
"""

iters = 500
ep_vals = [0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
sars_dict = {}
q_learn_dict = {}

for e_val in ep_vals:
    q_reward = Qlearning(rewards, gamma=1, alpha=0.1, iterations=iters, epsilon=e_val)
    q_av = averager(q_reward)
    sarsa_reward = SARSA(rewards, gamma=1, alpha=0.1, iterations=iters, epsilon=e_val)
    sarsa_av = averager(sarsa_reward)

    q_moving = moving_average(q_av, n=10)
    sarsa_moving = moving_average(sarsa_av, n=10)

    sars_dict[e_val] = sarsa_moving
    q_learn_dict[e_val] = q_moving


"""
Visualize results
"""

for skey in sars_dict:
    plt.figure()
    plt.plot(np.arange(len(sars_dict[skey])), sars_dict[skey], label='SARSA')
    plt.plot(np.arange(len(q_learn_dict[skey])), q_learn_dict[skey], label='Q-Learning')
    plt.ylabel('Sum of rewards during episode')
    plt.xlabel('Episodes')

    plt.legend()

    plt.title(f'Q-learning vs. SARSA Epsilon = {skey}')
    plt.savefig(f'qvsSARSA_ep{skey}.png')
plt.show()