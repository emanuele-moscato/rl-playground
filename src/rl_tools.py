import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import random

class Environment(object):
    """
    Environment object. It stores a state (int from 1 to 10), updates it with
    randomly and returns reward.
    """
    def __init__(self, init_state=None):
        if init_state:
            self.state = init_state
        else:
            self.state = round(np.random.uniform(0.5, 10.5))

    
    def return_state(self):
        return self.state
        
    def update_state(self, action):
        self.state = round(np.random.uniform(0.5, 10.5))

    def return_reward(self, action_choice):
        if action_choice==self.state:
            return 1
        else:
            return 0

class RandomModel(object):
    """
    Random model. Returns an array of 10 elements, each of which contains the
    "value" between 0 and 1 of the corresponding action. The entries are
    normalised as if they were probabilities, even though this is probably not
    needed.
    """
    def __init__(self):
        pass

    def predict(self):
        probs = []
        for _ in range(10):
            probs.append(np.random.uniform(0.0, 1.0))
        probs = np.array(probs)
        probs = probs/probs.sum()
        return probs
        
class NnModel(object):
    pass

class Agent(object):
    """
    Agent object. It has to:
    - Choose an action in action space
    - Contain a model to approximate the Q-value function
    - Compute Q-value funcion taking current state and an action as an input
    - Update the parameters of the model (learn), if any
    """
    def __init__(self, gamma, model, random_only=True):
        self.random_model = RandomModel()
        self.random_only = random_only
        if not random_only:
            self.model = model
        self.memory = []
        self.gamma = gamma

    def compute_q(self, state):
        return self.model.predict(state)

    def choose_action(self, state, eps):
        greedychoice = np.random.uniform(0.0,1.0)
        if greedychoice<eps:
            return np.argmax(self.random_model.predict())+1
        else:
            if self.random_only:
                return np.argmax(self.random_model.predict())+1
            else:
                return np.argmax(self.compute_q(state))+1

    def train(self):
        random.sample(agent.memory, int(len(agent.memory)/10))
        # Continue from here!

class Game(object):
    """
    Game engine.
    """
    def __init__(self, n_attempts=10):
        self.reward = 0
        self.n_actions = 0
        self.n_attempts = n_attempts

    def play_one_step(self, agent, env, eps):
        current_state = env.return_state()
        action = agent.choose_action(current_state, eps)
        reward = env.return_reward(action)
        env.update_state(action)
        next_state = env.state
        
        transition = (
            current_state,
            action,
            reward,
            next_state
        )
        agent.memory.append(transition)
        
        agent.train()

        self.reward += reward
        self.n_actions += 1

        return transition

    def play_one_round(self, agent, env, eps):
        self.reward = 0
        
        for _ in range(self.n_attempts):
            self.play_one_step(agent, env, eps)
        return self.reward