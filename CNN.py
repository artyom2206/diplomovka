from keras.models import Sequential, Model
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input, Lambda, add, Add
from keras import backend as K
from keras.optimizers import RMSprop, Adam
from keras.callbacks import CSVLogger, History
import gc
import numpy as np
import tensorflow as tf


class CNN:
    def __init__(self, input_dim, action_space,
                 discount_factor=0.99, learning_rate=0.00025, batch_size=32, weights=None,
                 ddqn=False, dueling=False):
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.dueling = dueling
        self.ddqn = ddqn

        if self.dueling:
            input_layer = Input(shape=input_dim)
            conv2dOut = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(input_layer)
            conv2dOut = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(conv2dOut)
            conv2dOut = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(conv2dOut)
            out = Flatten()(conv2dOut)

            advantage = Dense(512, activation='relu')(out)
            advantage = Dense(action_space)(advantage)

            value = Dense(512, activation='relu')(out)
            value = Dense(1)(value)

            advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True),
                               output_shape=(action_space,))(advantage)

            value = Lambda(lambda s: K.expand_dims(s[:, 0], -1),
                           output_shape=(action_space,))(value)

            q_value = add([value, advantage])
            self.model = Model(inputs=input_layer, outputs=q_value)
            self.model.compile(optimizer=RMSprop(lr=learning_rate, rho=0.95, epsilon=0.01),
                               loss="mean_squared_error", metrics=["accuracy"])

            self.model.summary()
            # self.model = model

        else:
            self.model = Sequential()
            self.model.add(Conv2D(32,
                                  8,
                                  strides=(4, 4),
                                  activation='relu',
                                  input_shape=input_dim,
                                  data_format='channels_last'))
            self.model.add(Conv2D(64,
                                  4,
                                  strides=(2, 2),
                                  activation='relu',
                                  data_format='channels_last'))
            self.model.add(Conv2D(64, 3, strides=(1, 1), activation='relu'))

            self.model.add(Flatten())

            self.model.add(Dense(512, activation='relu'))
            # self.model.add(Dense(256, activation='relu'))
            self.model.add(Dense(action_space))

            self.model.compile(optimizer=RMSprop(lr=learning_rate, rho=0.95, epsilon=0.01),
                               loss="mean_squared_error", metrics=["accuracy"])
            self.model.summary()

        if weights is not None:
            print("\n\n\nWeights loaded.\t", weights)
            self.model.load_weights(weights)

    def predict(self, current_state):
        return self.model.predict(np.moveaxis(current_state.astype(np.float64), [1, 2, 3], [3, 1, 2]), batch_size=1)

    def trainSarsa(self, state, action, reward, next_state, next_action, done):

        next_state_pred = self.predict(next_state).ravel()
        target = list(self.predict(state)[0])

        if done:
            target[action] = reward
        else:
            target[action] = (reward + self.discount_factor * next_state_pred[next_action])

        target = np.reshape(target, [1, 6])

        loss = self.model.fit(np.moveaxis(np.asarray(state.astype(np.float64)), [1, 2, 3], [3, 1, 2]),
                              np.asarray(target), epochs=1, verbose=0)
        # print(loss.history["loss"])

        return loss.history["loss"]

    def train(self, batch, target_network):
        x = []
        y = []

        for experience in batch:
            x.append(experience['current'].astype(np.float64))
            # state = experience['current'].astype(np.float64)

            next_state = experience['next_state'].astype(np.float64)
            next_state_pred = target_network.predict(next_state).ravel()
            next_q_value = np.max(next_state_pred)

            # print(next_q_value)

            target = list(self.predict(experience['current'])[0])
            if experience['done']:
                target[experience['action']] = experience['reward']
            else:
                if self.ddqn:
                    model_val = self.predict(next_state)  # self.model.predict(next_state).ravel()
                    model_q_value = np.argmax(model_val)
                    target[experience['action']] = experience['reward'] + self.discount_factor * next_state_pred[
                        model_q_value]
                else:
                    target[experience['action']] = experience['reward'] + self.discount_factor * next_q_value
            y.append(target)
            # target = np.reshape(target, [1, 6])

            # print(np.asarray(y).shape)
            # print(np.moveaxis(np.asarray(x).squeeze(axis=1), [1, 2, 3], [3, 1, 2]).shape)
            loss = self.model.fit(np.moveaxis(np.asarray(x).squeeze(axis=1), [1, 2, 3], [3, 1, 2]),
                                  np.asarray(y),
                                  batch_size=self.batch_size,
                                  epochs=1,
                                  verbose=0)
            # loss = self.model.fit(np.moveaxis(np.asarray(state), [1, 2, 3], [3, 1, 2]),
            #                       np.asarray(target),
            #                       batch_size=self.batch_size,
            #                       epochs=1,
            #                       verbose=0)
            # print(loss.history["loss"])

    def save(self, filepath):
        # print("Saving Weights.")
        self.model.save_weights(filepath)

    def clean(self):
        del self.model
        del self.batch_size
        del self.discount_factor
        gc.collect()
