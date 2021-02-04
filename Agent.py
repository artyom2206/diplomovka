from Paremeters import *
from CNN import *
import pickle as p
import joblib
import os
from random import random, randint, randrange


class Agent:
    def __init__(self, input_shape, action_space, game_name,
                 memory=MAX_EXPERIENCES,
                 epsilon=1.0,
                 min_epsilon=MIN_EPSILON,
                 decay_rate=DECAY_RATE,
                 batch_size=BATCH_SIZE,
                 load_weights=True,
                 test=False,
                 ddqn=False,
                 dueling=False,
                 sarsa=False):

        self.action_set = action_space
        self.input_shape = input_shape
        self.memory_size = memory
        self.epsilon = epsilon
        self.epsilon_min = min_epsilon
        self.decay = decay_rate
        self.batch_size = batch_size
        self.game = game_name
        self.ddqn = ddqn
        self.dueling = dueling
        self.sarsa = sarsa

        if not self.ddqn and not self.dueling and not self.sarsa:
            self.game = self.game + "_DQN"
        elif self.ddqn and not self.dueling and not self.sarsa:
            self.game = self.game + "_DDQN"
        elif not self.ddqn and self.dueling and not self.sarsa:
            self.game = self.game + "_Dueling"
        elif self.ddqn and self.dueling and not self.sarsa:
            self.game = self.game + "_DDQN+Dueling"
        elif self.sarsa:
            self.game = self.game + "_SARSA"

        filepath = str("Weights/" + self.game + "_weights") if load_weights else None

        self.main_network = CNN(self.input_shape,
                                self.action_set,
                                batch_size=self.batch_size,
                                weights=filepath if os.path.exists(filepath) else None,
                                ddqn=ddqn, dueling=dueling)
        if not test and not self.sarsa:
            self.target_network = CNN(self.input_shape, self.action_set, batch_size=self.batch_size,
                                      ddqn=ddqn, dueling=dueling)

            self.target_network.model.set_weights(self.main_network.model.get_weights())
            self.experiences = []

    def action(self, state):
        if random() < self.epsilon:
            return randint(0, self.action_set - 1)
        else:
            return self.main_network.predict(state).argmax()

    def experience_gain(self, current_state, action, reward, next_state, done):
        if self.experiences.__len__() >= self.memory_size:
            self.experiences.pop(0)

        self.experiences.append({'current': current_state,
                                 'action': action,
                                 'reward': reward,
                                 'next_state': next_state,
                                 'done': done})

    def sample_experiences(self):
        batch = []
        for _ in range(self.batch_size):
            batch.append(self.experiences[randrange(0, self.experiences.__len__())])
        return np.asarray(batch)

    def train(self):
        self.main_network.train(self.sample_experiences(), self.target_network)

    def trainSarsa(self, current_state, action, reward, next_state, next_action, done):
        loss = self.main_network.trainSarsa(current_state, action, reward, next_state, next_action, done)
        return loss

    def greedyEpsilon(self):
        if self.epsilon * self.decay > self.epsilon_min:
            self.epsilon *= self.decay
        else:
            self.epsilon = self.epsilon_min

    def update_target_network(self):
        self.target_network.model.set_weights(self.main_network.model.get_weights())
        print("Target Network Updated.")

    def experience_available(self):
        return True if self.experiences.__len__() >= self.batch_size else False

    def save(self):
        self.main_network.save(filepath=str("Weights/" + self.game + "_weights"))

    def save_state(self, ongoing):

        while len(self.experiences) > MAX_EXPERIENCES:
            self.experiences.pop(0)

        print("Saving Experiences...", self.experiences.__len__())
        if not ongoing:
            self.clean()
        gc.disable()
        with open("Experiences/Experiences" + self.game, 'wb') as experience_dump:
            joblib.dump((self.experiences, self.epsilon), experience_dump, compress=6, protocol=p.HIGHEST_PROTOCOL)
        gc.enable()
        experience_dump.close()
        print("Number of Experiences : ", self.experiences.__len__())
        print("Current Episilon Value : ", self.epsilon)
        print("Experiences Saved.")

    def load_state(self, load=True):
        if load and os.path.exists("Experiences/Experiences" + self.game):
            print("Loading Experiences...")
            gc.disable()
            with open("Experiences/Experiences" + self.game, 'rb') as experience_dump:
                self.experiences, self.epsilon = joblib.load(experience_dump)
            gc.enable()
            experience_dump.close()
            gc.collect()
            print(self.experiences.__len__(), " Experiences loaded.\n\n\n")
            print("Current Episilon Value : ", self.epsilon)
            return True
        return False

    def clean(self):
        self.main_network.clean()
        self.target_network.clean()
        del self.target_network
        del self.main_network
        del self.batch_size
        del self.epsilon_min
        del self.input_shape
        del self.action_set
        del self.memory_size
        del self.decay
        gc.collect()
