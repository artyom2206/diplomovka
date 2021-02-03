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
        # self.logger = CSVLogger("History.txt", append=True)
        self.discount_factor = discount_factor
        self.batch_size = batch_size

        self.dueling = dueling
        self.ddqn = ddqn
        if self.dueling:  # dueling architecture

            # initial layers are the same
            input = Input(shape=input_dim)
            cnnout = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(input)
            cnnout = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(cnnout)
            cnnout = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(cnnout)
            out = Flatten()(cnnout)

            advantage = Dense(512, activation='relu')(out)
            advantage = Dense(action_space)(advantage)

            value = Dense(512, activation='relu')(out)
            value = Dense(1)(value)

            # before aggregating, we subtract average advantage to acc elerate training
            advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True),
                               output_shape=(action_space,))(advantage)

            value = Lambda(lambda s: K.expand_dims(s[:, 0], -1),
                           output_shape=(action_space,))(value)

            # sums advantage and value to estimate q-value
            q_value = add([value, advantage])
            self.model = Model(inputs=input, outputs=q_value)
            self.model.compile(optimizer=RMSprop(lr=learning_rate, rho=0.95, epsilon=0.01),
                               loss="mean_squared_error", metrics=["accuracy"])

            # backbone = tf.keras.Sequential([
            #     Input(shape=(84, 84, 4)),
            #     Dense(32, activation='relu'),
            #     Dense(16, activation='relu')
            # ])
            # # state_input = Input((input_dim,))
            # state_input = Input(shape=(84, 84, 4))
            # backbone_1 = Dense(32, activation='relu')(state_input)
            # backbone_2 = Dense(16, activation='relu')(backbone_1)

            # input = Input(shape=input_dim)
            # cnnout = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(input)
            # cnnout = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(cnnout)
            # cnnout = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(cnnout)
            # value_output = Dense(1)(cnnout)
            # advantage_output = Dense(action_space)(cnnout)
            #
            # # value_output = Dense(1)(backbone_2)
            # # advantage_output = Dense(action_space)(backbone_2)
            # output = Add()([value_output, advantage_output])
            # # model = tf.keras.Model(state_input, output)
            # self.model = Model(inputs=input, outputs=output)
            # # model.compile(loss='mse', optimizer=Adam(learning_rate))
            # # model.compile(loss='mse', optimizer=Adam(lr=learning_rate, epsilon=0.01))
            # self.model.compile(optimizer=RMSprop(lr=learning_rate, rho=0.95, epsilon=0.01),
            #                    loss="mse", metrics=["accuracy"])

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
            print("\n\n\nWeights Found!!!\t", weights)
            self.model.load_weights(weights)

    def predict(self, current_state):
        return self.model.predict(np.moveaxis(current_state.astype(np.float64), [1, 2, 3], [3, 1, 2]), batch_size=1)

    # vyberie batch(32) nahodnych exp z celkovych 20k, naplni sa x,y a spusti sa model.fit
    def train(self, batch, target_network):
        x = []  # stav prostredia
        y = []  # reward za akciu čo spravil v danom stave ^^

        for experience in batch:
            x.append(experience['current'].astype(np.float64))

            next_state = experience['next_state'].astype(np.float64)
            next_state_pred = target_network.predict(next_state).ravel()
            next_q_value = np.max(next_state_pred)

            # print(next_q_value)

            target = list(self.predict(experience['current'])[0])
            if experience['done']:
                # za akciu čo spravil dostane odmenu ak skončí epizoda
                target[experience['action']] = experience['reward']
            else:
                # pre danu akciu rozhodnutia priradi hodnotu experience['reward'] + self.discount_factor * next_q_value
                if self.ddqn:
                    # pozicka najvysieho q v modeli do target modelu
                    model_val = self.predict(next_state)  # self.model.predict(next_state).ravel()
                    model_q_value = np.argmax(model_val)
                    target[experience['action']] = experience['reward'] + self.discount_factor * next_state_pred[
                        model_q_value]
                else:
                    # max q hodnota
                    target[experience['action']] = experience['reward'] + self.discount_factor * next_q_value
            y.append(target)

            # print(np.asarray(y))
            # print(np.moveaxis(np.asarray(x).squeeze(axis=1), [1, 2, 3], [3, 1, 2]).shape)
            loss = self.model.fit(np.moveaxis(np.asarray(x).squeeze(axis=1), [1, 2, 3], [3, 1, 2]),
                                  np.asarray(y),
                                  batch_size=self.batch_size,
                                  epochs=1,
                                  verbose=0)  # ,
            # print(loss.history["loss"])

    def save(self, filepath):
        # print("Saving Weights!!")
        self.model.save_weights(filepath)

    def clean(self):
        del self.model
        del self.batch_size
        del self.discount_factor
        gc.collect()
