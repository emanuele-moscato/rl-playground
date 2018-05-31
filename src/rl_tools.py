import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.losses import mean_squared_error
import random
from collections import deque
import plotly.graph_objs as go
from plotly.offline import iplot
from plotly import tools
from tqdm import tqdm_notebook as tqdm
import pickle

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
    def __init__(self, gamma, model, memory_size, 
        batch_size, epochs, random_only=True):
        self.random_model = RandomModel()
        self.random_only = random_only
        if not random_only:
            self.model = model
        self.memory = deque(maxlen=memory_size)# Previously 500
        self.batch_size = batch_size
        self.epochs = epochs
        self.total_memory = []
        self.gamma = gamma
        self.eps = 1.0
        self.model_memory = []
        self.eps_memory = []
        self.training_memory = []

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

    def train(self):
        if not self.random_only:
            if len(self.memory)<self.batch_size:
                training_batch = np.array(self.memory)
            else:
                training_batch = np.array(
                    random.sample(self.memory, self.batch_size)
                )
            Y_target = (training_batch[:,2]
                + self.gamma
                * np.amax(self.compute_q(training_batch[:,3]), axis=1))
            Y_pred = np.take(
                self.compute_q(training_batch[:,0]),
                training_batch[:,1]-1
            )
            self.model.fit(Y_target, Y_pred, epochs=self.epochs, verbose=0)
        

class Game(object):
    """
    Game engine.
    """
    def __init__(self, n_actions=500):
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

    def play_one_episode(self, agent, env, training=True):
        self.reward = 0
        for _ in range(self.n_actions):
            agent.eps_memory.append(agent.eps)
            self.play_one_action(agent, env)
        self.episode_count += 1
        if self.episode_count>100:
            if agent.eps>0.05:# Previously 0.01
                agent.eps *= 0.95
        if training:
            if (self.action_count+1)%1==0:#Previously 125
                agent.train()
                agent.training_memory.append(1)
            else:
                agent.training_memory.append(0)
        return self.reward

def run_game(agent_params, game_params):
    """
    Runs the game once, with the parameters for the NN (nodes, activation, etc.)
    and for the game itself (number of episode, number of actions per episode, 
    etc.) specified in the arguments.
    """
    model = Sequential()
    model.add(Dense(
        agent_params['nodes'],
        input_shape=(1,),
        activation=agent_params['activation']
    ))
    model.add(Dense(10))
    model.compile(
        optimizer=agent_params['optimizer'],
        loss=custom_loss
    )

    env = Environment()
    nn_agent = Agent(
        agent_params['gamma'],
        model,
        agent_params['memory_size'],
        agent_params['batch_size'],
        agent_params['epochs'],
        random_only=False
    )
    game = Game(game_params['n_actions'])

    n_episodes = game_params['n_episodes']
    scores = []

    for _ in tqdm(range(n_episodes)):
        scores.append(game.play_one_episode(nn_agent, env))
    scores = np.array(scores)

    model.save('../models/last_model.h5')
    with open('../scores/'+'last_model'+'_scores.pkl', 'wb') as f:
        pickle.dump(scores, f)

    return scores, nn_agent

def custom_loss(Y_target, Y_pred):
    return mean_squared_error(Y_target, Y_pred)

def plot_scores(scores):
    """
    Plots the collection of scores across episodes.
    """
    trace = go.Scatter(
        x = np.arange(1, len(scores)+1),
        y = scores,
        mode='markers'
    )
    
    layout = go.Layout(
        xaxis = dict(
            title='Episode number'
        ),
        yaxis = dict(
            title='Score'
        ),
        hovermode = 'closest'
    )
    
    data = [trace]
    
    fig = go.Figure(data=data, layout=layout)
    
    iplot(fig)

def describe_scores(scores):
    """
    Computes basic statistics of the scores across episodes.
    """
    print('Average score: '+str(scores.mean()))
    print('Standard deviation: '+str(scores.std()))

def plot_model_switch(model_memory, eps_memory):
    """
    Plots:
    1) Which model (NN or random) is used action by action.
    2) Value of epsilon (greedyness parameter) action by action.
    """
    model_memory_indices = np.arange(0, len(model_memory)-1, len(model_memory)/500).astype(int)

    trace1 = go.Scatter(
        x = model_memory_indices,
        y = np.take(
            model_memory,
            model_memory_indices
        ),
        mode='markers',
        name='Model used'
    )

    trace2 = go.Scatter(
        x = model_memory_indices,
        y = np.take(
            eps_memory,
            model_memory_indices
        ),
        mode='markers',
        name='Epsilon'
    )

    layout = go.Layout(
        xaxis = dict(
            title='Action number'
        ),
        yaxis = dict(
            title='Model used'
        ),
        xaxis2 = dict(
            title='Action number'
        ),
        yaxis2 = dict(
            title='Epsilon'
        )
    )
    fig = tools.make_subplots(rows=1, cols=2)
    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 2)
    fig['layout']['xaxis1'].update(title='Action number')
    fig['layout']['yaxis1'].update(title='Model used')
    fig['layout']['xaxis2'].update(title='Action number')
    fig['layout']['yaxis2'].update(title='Epsilon')
    
    
    iplot(fig)