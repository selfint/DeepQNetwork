from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from collections import deque, namedtuple
import random


ReplayFrame = namedtuple('ReplayFrame', 'state action reward next_state')


class Agent:

    def __init__(self, observation_space: int, action_space: int, epsilon_max: float, epsilon_min: float, 
        epsilon_decay: float, discount_rate: float, experience_replay_size: int, experience_replay_train: int):
        """generates an agent with a keras neural network that can interact with open ai gym envs
        
        Arguments:
            observation_space {int} -- input dimensions
            action_space {int} -- output dimensions
            epsilon_max {float} -- epsilon start
            epsilon_min {float} -- epsilon end
            epsilon_decay {float} -- epsilon decay rate
            discount_rate {float} -- how much weight does the future have
            experience_replay_size {int} -- how many ReplayFrames to remember
            experience_replay_train {int} -- how many ReplayFrames to select from memory for training
        """
        
        # neural network parameters
        self.observation_space = observation_space
        self.action_space = action_space
        self.network = self.build_model()

        # q learning parameters
        self.epsilon = epsilon_max
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.discount_rate = discount_rate
        self.epsilon_counter = 1

        # experience replay
        self.experience_replay_reel = deque(maxlen=experience_replay_size)
        self.experience_replay_train = experience_replay_train

    def build_model(self) -> Sequential:
        """generate compiled keras neural network
        
        Arguments:
            inputs {int} -- input dimensions
            outputs {int} -- output dimensions
        
        Returns:
            Sequential -- compiled model
        """
        model = Sequential()
        model.add(Dense(24, input_dim=self.observation_space, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        return model


    def choose_action(self, state: np.array) -> int:
        """chooses an action based on an epsilon-greedy policy
        
        Arguments:
            state {np.array} -- current state of environment
        
        Returns:
            int -- action to take
        """

        # decide action based on current epsilon value
        if np.random.sample() < self.epsilon:
            action = np.random.randint(0, self.action_space)
        else:
            action = np.argmax(self.network.predict(state))

    def experience_replay(self):
        """learns from past states and updates the network
        """
        if len(self.experience_replay_reel) < self.experience_replay_train:
            return

        # randomly select frames to train on
        x_train = [state for (state, _, _, _) in random.sample(self.experience_replay_reel, self.experience_replay_train)]

        # set the target values to predict the max reward possible after each reward
        # set negative reward for terminal states
        y_train = [reward + self.discount_rate * np.max(self.network.predict(next_state)) if next_state is not None
                    else -1.0 + self.discount_rate * np.max(self.network.predict(next_state))
                    for (state, action, reward, next_state) in x_train]
        
        # train model
        # self.network.fit(x_train, y_train)
        


    def remember(self, state: np.array, action: int, reward: float, next_state: np.array):
        """adds a tuple contain a state-action and their following reward and next state
        
        Arguments:
            state {np.array} -- state agent was in
            action {int} -- action agent took in state 'state'
            reward {float} -- reward agent recieved for action 'action' in state 'state'
            next_state {np.array} -- state agent was in after taking action 'action' from state 'state'
        """
        self.experience_replay_reel.append(ReplayFrame(state, action, reward, next_state))


if __name__ == "__main__":
    test_agent = Agent(2, 2, 1.0, 0.01, 0.001, 0.99, 2, 2)
    s = np.array([[1, 2]])
    test_agent.remember(s, 1, 1, s)
    test_agent.remember(s, 1, 1, s)
    test_agent.remember(s, 1, 1, s)
    test_agent.experience_replay()


