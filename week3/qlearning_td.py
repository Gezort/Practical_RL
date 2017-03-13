"""
Q-learning
This file contains the same q-learning agent you implemented in the previous assignment.
The only difference is that it doesn't need any other files with it, so you can use it as a standalone moule.

Here's an example:
>>>from qlearning import QLearningAgent

>>>agent = QLearningAgent(alpha=0.5,epsilon=0.25,discount=0.99,
                       getLegalActions = lambda s: actions_from_that_state)
>>>action = agent.getAction(state)
>>>agent.update(state,action, next_state,reward)
>>>agent.epsilon *= 0.99
"""

import random,math

import numpy as np
from collections import defaultdict

class QLearningTDAgent():
  """
    Q-Learning Agent

    The two main methods are 
    - self.getAction(state) - returns agent's action in that state
    - self.update(state,action,nextState,reward) - returns agent's next action

    Functions you should use
      - self.getLegalActions(state)
        which returns legal actions for a state
      - self.getQValue(state,action)
        which returns Q(state,action)
      - self.setQValue(state,action,value)
        which sets Q(state,action) := value
    
    !!!Important!!!
    NOTE: please avoid using self._qValues directly to make code cleaner
  """
  def __init__(self,alpha,epsilon,discount,lambd,getLegalActions,el_trace_threshold=0.2):
    "We initialize agent and Q-values here."
    self.getLegalActions= getLegalActions
    self._qValues = defaultdict(lambda:defaultdict(lambda:0))
    self.alpha = alpha
    self.epsilon = epsilon
    self.discount = discount
    self.lambd = lambd
    self.el_trace_threshold = el_trace_threshold
    self.e = {}

  def getQValue(self, state, action):
    """
      Returns Q(state,action)
    """
    return self._qValues[state][action]

  def setQValue(self,state,action,value):
    """
      Sets the Qvalue for [state,action] to the given value
    """
    self._qValues[state][action] = value

#---------------------#start of your code#---------------------#

  def getValue(self, state):
    """
      Returns max_action Q(state,action)
      where the max is over legal actions.
    """
    
    possibleActions = self.getLegalActions(state)
    #If there are no legal actions, return 0.0
    if len(possibleActions) == 0:
        return 0.0

    "*** YOUR CODE HERE ***"
    return max([self.getQValue(state, a) for a in possibleActions])
    
  def getPolicy(self, state):
    """
      Compute the best action to take in a state. 
      
    """
    possibleActions = self.getLegalActions(state)

    #If there are no legal actions, return None
    if len(possibleActions) == 0:
        return None
    
    best_action = None

    best_action = possibleActions[np.argmax([self.getQValue(state, a) for a in possibleActions])]
    return best_action

  def getAction(self, state):
    """
      Compute the action to take in the current state, including exploration.  
      
      With probability self.epsilon, we should take a random action.
      otherwise - the best policy action (self.getPolicy).

      HINT: You might want to use util.flipCoin(prob)
      HINT: To pick randomly from a list, use random.choice(list)

    """
    
    # Pick Action
    possibleActions = self.getLegalActions(state)
    action = None
    
    #If there are no legal actions, return None
    if len(possibleActions) == 0:
        return None

    #agent parameters:
    epsilon = self.epsilon

    if np.random.random()<=epsilon:
        return random.choice(possibleActions)
    else:
        action = self.getPolicy(state)
    return action

  def update(self, state, action, nextState, reward):
    """
      You should do your Q-Value update here

      NOTE: You should never call this function,
      it will be called on your behalf


    """
    #agent parameters
    gamma = self.discount
    learning_rate = self.alpha
    lambd = self.lambd
    el_trace = self.e.get((state, action), 0) + 1
    self.e[(state, action)] = el_trace
    keys = list(self.e.keys())

    for s,a in keys:
        el_trace = self.e[(s,a)]
        lr = learning_rate * el_trace
        reference_qvalue = reward + gamma * self.getValue(nextState)
        updated_qvalue = (1-lr) * self.getQValue(state, action) + lr * reference_qvalue
        self.setQValue(state,action,updated_qvalue)
        if el_trace * lambd * gamma > self.el_trace_threshold:
            self.e[(s,a)] = el_trace * lambd * gamma
        else:
            self.e.pop((s,a))
#---------------------#end of your code#---------------------#


