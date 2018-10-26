"""

https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
"""

import numpy as np
import gym
from keras import Input, Model
from keras.layers import Dense, Flatten, Lambda
from keras import backend as K
from keras.optimizers import RMSprop
from keras.utils import to_categorical

def build_inference_model(num_states, num_actions):
    x_in = Input(shape=(num_states, 1))
    x = Flatten()(x_in)
    x = Dense(16, activation="relu")(x)
    x = Dense(16, activation="relu")(x)
    x_out = Dense(num_actions, activation="linear")(x)
    return Model(inputs=[x_in], outputs=[x_out])

def wrap_training_layers(inference_model, num_actions):
    x_in_masks = Input(shape=(num_actions, 1))
    q_values = inference_model.outputs[0]
    q_value = Lambda(lambda _: K.batch_dot(_, x_in_masks))(q_values)
    return Model(inputs=inference_model.inputs + [x_in_masks], outputs=[q_value])

def huber_loss(y_true, y_pred):
    error = y_true - y_pred
    condition = K.abs(error) < 1.0
    L2 = K.square(error)
    L1 = K.abs(error)
    loss = 0.5 * K.switch(condition, L2, L1)
    return K.mean(loss)

class Memory(object):
    def __init__(self, limit=10000):
        self.experiences = []
        self.limit = limit

    def append(self, experience):
        if len(self.experiences) >= self.limit:
            self.experiences.pop(0)
        self.experiences.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.experiences), batch_size)
        return np.asarray(self.experiences)[indices]

class DQNAgent(object):
    def __init__(self, num_states, num_actions):
        self.memory = Memory()
        self.num_states = num_states
        self.num_actions = num_actions
        self._inference_model = build_inference_model(self.num_states, self.num_actions)

        self._trainable_model = wrap_training_layers(self._inference_model, self.num_actions)
        self._trainable_model.compile(optimizer=RMSprop(lr=1e-4), loss=huber_loss)

        self.batch_size = 32
        self.gamma = 0.99

    def select_action(self, observation):
        epsilon = 0.01 + 0.99 / (1 + len(self.memory.experiences))
        if np.random.random() < epsilon:
            return np.random.randint(self.num_actions)
        else:
            q_values = self._inference_model.predict(np.expand_dims([observation], axis=-1))
            return np.argmax(q_values)

    def learn(self):
        if len(self.memory.experiences) >= self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            states, actions, next_states, rewards = map(np.asarray, zip(*experiences))
            masks = to_categorical(actions, self.num_actions)
            next_q_values = self._inference_model.predict(np.expand_dims(next_states, axis=-1))
            targets = rewards + self.gamma * np.max(next_q_values, axis=1)
            assert targets.shape == rewards.shape
            self._trainable_model.fit([np.expand_dims(states, axis=-1), np.expand_dims(masks, axis=-1)], targets, verbose=0)

def run():
    ENV_NAME = "CartPole-v0"
    MAX_EPISODE = 1000
    MAX_STEP = 200
    env = gym.make(ENV_NAME)
    num_states = env.observation_space.shape[-1]
    num_actions = env.action_space.n
    agent = DQNAgent(num_states, num_actions)
    for episode in range(MAX_EPISODE):
        observation = env.reset()
        score = 0
        for t in range(MAX_STEP):
            action = agent.select_action(observation)
            next_observation, reward, done, info = env.step(action)
            score += 1
            reward = done * (2 * (t >= MAX_STEP - 5) - 1)
            agent.memory.append((observation, action, next_observation, reward))
            agent.learn()
            observation = next_observation
            if done:
                print("episode: %d, frame: %d, score: %d" % (episode, t, score))
                break

if __name__ == "__main__":
    run()

