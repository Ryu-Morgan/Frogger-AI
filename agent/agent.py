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

        # this is the state the frog sees 
        return ''.join([
            self.get(self.frog_x - 1, self.frog_y - 1) or '_', # up + left
            self.get(self.frog_x, self.frog_y - 1) or '_', # up
            self.get(self.frog_x + 1, self.frog_y - 1) or '_', # up + right
            self.get(self.frog_x - 1, self.frog_y) or '_', # left
            self.get(self.frog_x + 1, self.frog_y) or '_', # right
        ])

    def reward(self):
        '''Returns a reward value for the state.'''

        if self.at_goal:
            return self.score
        elif self.is_done:
            return -10
        else:
            # This encourages progress instead of standing still
            return (self.max_y - self.frog_y) * 0.1


class Agent:

    def __init__(self, train=None):
        self.train = train
        self.q = {}
        self.name = train or 'q'
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'train', self.name + '.json')
        
        # track the previous state and action after each step
        self.prev_state = None
        self.prev_action = None
        # We need to limit the number of steps that are saved
        self.steps = 0
        self.save_interval = 100
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


    def get_q_values(self, key):
        '''Returns Q-values for a state, initializing to 0 if unseen.'''
        if key not in self.q:
            self.q[key] = {action: 0.0 for action in State.ACTIONS}
        return self.q[key]

    def _update_q(self, prev_state, action, curr_state):
        '''
        Implements the Bellman equation.

        new_q = old_q + alpha * (reward + gamma * best_future - old_q)
        '''
        old_q = self.get_q_values(prev_state.key)
        new_q = self.get_q_values(curr_state.key)

        old_value = old_q[action]
        reward = curr_state.reward()
        best_future = max(new_q.values())

        # Bellman update
        old_q[action] = old_value + Q_State.alpha * (
            reward + Q_State.gamma * best_future - old_value
        )

    def choose_action(self, state_string):
        '''Returns the action to perform'''
        state = Q_State(state_string)

        # q-update we need to use the previous state and action to update the Q-table based on the current state
        if self.prev_action and self.prev_state:
            self._update_q(self.prev_state, self.prev_action, state)

        if random.uniform(0, 1) < Q_State.epsilon:
            action = random.choice(State.ACTIONS)  # explore
        else:
            q_values = self.get_q_values(state.key)
            action = max(q_values, key=q_values.get)  # exploit

        # Remember this state/action for next update
        self.prev_state = state
        self.prev_action = action

        # Save periodically, not every step
        self.steps += 1
        if self.train and self.steps % self.save_interval == 0:
            self.save()

        return action
