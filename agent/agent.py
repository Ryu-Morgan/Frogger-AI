import json
import os
import random

from .state import State


class Q_State(State):
    '''Augments the game state with Q-learning information'''
    # start with Q-learning parameters
    alpha = 0.5
    gamma = 0.7
    epsilon = 0.2

    def __init__(self, string):
        super().__init__(string)

        # key stores the state's key string (see notes in _compute_key())
        self.key = self._compute_key()

    def _compute_key(self):
        '''
        Returns a key used to index this state.

        The key should reduce the entire game state to something much smaller
        that can be used for learning. When implementing a Q table as a
        dictionary, this key is used for accessing the Q values for this
        state within the dictionary.
        '''

        # this simple key uses the 3 object characters above the frog
        # and combines them into a key string
        return ''.join([
            self.get(self.frog_x - 1, self.frog_y - 1) or '_',
            self.get(self.frog_x, self.frog_y - 1) or '_',
            self.get(self.frog_x + 1, self.frog_y - 1) or '_',
            self.get(self.frog_x - 1, self.frog_y) or '_', # added left
            self.get(self.frog_x + 1, self.frog_y) or '_', # added right
        ])

    def reward(self):
        '''Returns a reward value for the state.'''

        if self.at_goal:
            return self.score
        elif self.is_done:
            return -10
        else:
            return 0


class Agent:

    def __init__(self, train=None):
        self.train = train
        self.q = {}
        self.name = train or 'q'
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'train', self.name + '.json')
        
        # Load the Q-table once at initialization
        self.load()

    def load(self):
        '''Loads the Q-table from the JSON file'''
        try:
            with open(self.path, 'r') as f:
                self.q = json.load(f)
            print(f"{'Training' if self.train else 'Loaded'} {self.path}")
        except (IOError, json.JSONDecodeError):
            # Only prints if training or fails to load an existing Q-table
            print(f"{'Training' if self.train else 'New'} Q-table initiated at {self.path}")
        return self

    def save(self):
        '''Saves the Q-table to the JSON file'''
        with open(self.path, 'w') as f:
            json.dump(self.q, f, indent=2)  # Using `indent` for easier reading
        return self

    def choose_action(self, state_string):
        '''Returns the action to perform using epsilon-greedy strategy'''
        state = Q_State(state_string)
        key_state = state.key

        # Initialize Q-values if state not in Q-table
        if key_state not in self.q:
            self.q[key_state] = {action: 0 for action in State.ACTIONS}
            self.save()  

        # Epsilon-greedy action selection
        if random.uniform(0, 1) < Q_State.epsilon:
            # Random action and save the Q-value
            action = random.choice(State.ACTIONS)
        else:
            action = max(self.q[key_state], key=self.q[key_state].get)

        self.save()  
        return action