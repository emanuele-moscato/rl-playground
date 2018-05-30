import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.losses import mean_squared_error
import random
from collections import deque

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
    needed. Every agent is equipped with a RandomModel and and epsilon-greedy
    strategy dictates whether to use it or the "learning" model to perform a
    particular action.
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
        self.memory = deque(maxlen=250)# Previously 500
        self.total_memory = []
        self.gamma = gamma
        self.eps = 1.0
        self.model_memory = []
        self.eps_memory = []

    def compute_q(self, state):
        return self.model.predict(state)

    def choose_action(self, state):
        greedychoice = np.random.uniform(0.0,1.0)
        if greedychoice<self.eps:
            self.model_memory.append(0)
            return np.argmax(self.random_model.predict())+1
        else:
            if self.random_only:
                self.model_memory.append(0)
                return np.argmax(self.random_model.predict())+1
            else:
                self.model_memory.append(1)
                return np.argmax(self.compute_q(np.array(state).reshape(1,1)))+1

    def train(self, batch_size=250):# Previously 500
        if not self.random_only:
            if len(self.memory)<batch_size:
                training_batch = np.array(self.memory)
            else:
                training_batch = np.array(
                    random.sample(self.memory, batch_size)
                )
            Y_target = (training_batch[:,2]
                + self.gamma
                * np.amax(self.compute_q(training_batch[:,3]), axis=1))
            Y_pred = np.take(
                self.compute_q(training_batch[:,0]),
                training_batch[:,1]-1
            )
            self.model.fit(Y_target, Y_pred, epochs=1, verbose=0)
        

class Game(object):
    """
    Game engine.
    """
    def __init__(self, n_actions=10):
        self.reward = 0
        self.n_actions = n_actions
        self.action_count = 0
        self.episode_count = 0

    def play_one_action(self, agent, env):
        current_state = env.return_state()
        action = agent.choose_action(current_state)
        reward = env.return_reward(action)
        # env.update_state(action)
        next_state = env.state
        
        transition = (
            current_state,
            action,
            reward,
            next_state
        )
        agent.memory.append(transition)
        agent.total_memory.append(transition)

        self.reward += reward
        self.action_count += 1

        return transition

    def play_one_episode(self, agent, env):
        self.reward = 0
        for _ in range(self.n_actions):
            agent.eps_memory.append(agent.eps)
            self.play_one_action(agent, env)
        self.episode_count += 1
        if self.episode_count>100:
            if agent.eps>0.01:
                agent.eps *= 0.95
        if (self.action_count+1)%125==0:
            agent.train()
        return self.reward
        
def custom_loss(Y_target, Y_pred):
    return mean_squared_error(Y_target, Y_pred)