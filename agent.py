import numpy as np
from collections import defaultdict
import random

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = 0.069
        self.gamma = 0.9
        self.max_epsilon=0.01
        self.min_epsilon = 0.0001
    

    def epsilon_greedy_probs(self, env, Q_s, i_episode, eps=None):
        ###########################################################################
        # Obtains the action probabilities corresponding to epsilon-greedy policy #
        ###########################################################################

        epsilon = self.max_epsilon / i_episode
        if eps is not None:
            epsilon = eps
        policy_s = np.ones(self.nA) * epsilon / self.nA
        policy_s[np.argmax(Q_s)] = 1 - epsilon + (epsilon / self.nA)
        return policy_s


    def update_Q(self, Qsa, Qsa_next, reward, alpha, gamma):
        ##############################################################################
        # Updates the action-value function estimate using the most recent time step #
        ##############################################################################
        return Qsa + (alpha * (reward + (gamma * Qsa_next) - Qsa))
    
    def select_action(self, state, env, i_episode):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        ##<-Get epsilon-greedy action probabilities->##
        policy_s = self.epsilon_greedy_probs(env, self.Q[state], i_episode, self.min_epsilon)
        return np.random.choice(np.arange(self.nA), p=policy_s)
        
    def step(self, env, i_episode, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        ##<-Get epsilon-greedy action probabilities->##
        policy_s = self.epsilon_greedy_probs(env, self.Q[state], i_episode, self.min_epsilon)
        #<-Update values using Expected Sarsa->#
        self.Q[state][action] = self.update_Q(self.Q[state][action], np.dot(self.Q[next_state], policy_s), \
                                                  reward, self.alpha, self.gamma)  