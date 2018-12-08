from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Conv2D, BatchNormalization, Activation, Flatten, Add, TimeDistributed, LSTM, MaxPooling2D, Reshape, ConvLSTM2D
from keras import regularizers, utils, optimizers
from keras.preprocessing import sequence
from keras import backend as K
from reconBoard import ReconBoard
from chess import Move, SQUARE_NAMES
from keras.layers.merge import concatenate
import numpy as np
import tensorflow as tf
from tensorflow import set_random_seed
from keras.utils import plot_model

K.clear_session()

class ChessModel:
    def __init__(self, hyperparamters, load_from_file=False, training=True):
        self.hp = hyperparamters
        self.training = training

        if load_from_file:
            self.load_all()
        else:
            self.belief_state = self.build_belief_state_network()
            self.sense_policy = self.build_sensing_policy_network()
            self.move_policy = self.build_move_policy_network()

        # Needed for multi threading
        self.belief_state._make_predict_function()
        self.sense_policy._make_predict_function()
        self.move_policy._make_predict_function()
        self.session = K.get_session()
        self.graph = tf.get_default_graph()


    def build_belief_state_network(self, training=True):
        flatten = Sequential()
        for i in range(self.hp["num_conv"]):
            flatten.add(Conv2D(self.hp["conv_filters"],
                               (self.hp["conv_kernel"], self.hp["conv_kernel"]),
                               input_shape=(8, 8, 13)))
        flatten.add(Flatten())

        lstm = Sequential()
        if not training:
            # batch_input_shape required to be stateful
            lstm.add(TimeDistributed(flatten, input_shape=(None, 8, 8, 13),
                                     batch_input_shape=(1, 1, 8, 8, 13)))
        else:
            lstm.add(TimeDistributed(flatten, input_shape=(None, 8, 8, 13)))

        # An lstm is stateful if it can take in entries one by one instead of being given
        # the entire sequence up front.
        for i in range(self.hp["num_lstm"]):
            stateful = not training
            lstm.add(LSTM(self.hp["lstm_size"], return_sequences=True, stateful=stateful))

        model = Sequential()
        for i in range(self.hp["num_dense"]):
            model.add(Dense(self.hp["dense_size"]))
        model.add(Dense(64 * 13))
        model.add(Reshape((64, 13)))
        model.add(Activation('softmax'))
        model.add(Reshape((8, 8, 13)))

        lstm.add(TimeDistributed(model, input_shape=(None, 8 * 8 * 13)))

        sgd = optimizers.SGD(lr=self.hp["lr"], momentum=self.hp["momentum"])
        lstm.compile(loss='kld', optimizer=sgd)
        return lstm

    # Makes a prediction of the next board state given an observation
    def get_belief_state(self, observation):
        with self.session.as_default():
            with self.graph.as_default():
                batch = np.asarray([np.asarray([observation])])
                return self.belief_state.predict(batch)[0]

    # Trains the model on inputs of board state + observation to next boardstate
    def train_belief_state(self, _input, _output, epochs=1):
        max_len = max(x.shape[0] for x in _input)
        _input = sequence.pad_sequences(_input, maxlen=max_len)
        _output = sequence.pad_sequences(_output, maxlen=max_len)
        return self.belief_state.fit(_input, _output, batch_size=self.hp["batch_size"],
                                     epochs=epochs, verbose=0, validation_split=0)

    # 8 x 8 board with 14 channels
    # (6 my pieces + 6 their pieces + empty squares) + 1 for sensing
    # Outputs a 1x64 policy for where to sense
    def build_sensing_policy_network(self):
        input_shape = (14, 8, 8)

        # Convolutional part
        cnn = Sequential()
        cnn.add(Conv2D(14, (2,2), activation='relu',
                       data_format="channels_first",
                       padding='same', input_shape=input_shape))
        cnn.add(MaxPooling2D(pool_size=(2, 2)))
        cnn.add(Conv2D(14, (2,2), data_format="channels_first",
                       activation='relu', padding='same'))
        cnn.add(MaxPooling2D(pool_size=(2, 2)))
        cnn.add(Flatten())
        cnn.add(Dense(64, activation='softmax'))
        cnn.compile(loss='mean_squared_error', optimizer='adam')
        return cnn

    # 8 x 8 board with 14 channels
    # (6 my pieces + 6 their pieces + empty squares) + 1 for sensing
    # Outputs a 73 x 8 x 8 policy of moves
    def build_move_policy_network(self):
        input_shape = (14, 8, 8)

        # Convolutional part
        cnn = Sequential()
        cnn.add(Conv2D(14, (2,2), activation='relu',
                       data_format="channels_first",
                       padding='same', input_shape=input_shape))
        cnn.add(MaxPooling2D(pool_size=(2, 2)))
        cnn.add(Conv2D(14, (2,2), data_format="channels_first",
                       activation='relu', padding='same'))
        cnn.add(MaxPooling2D(pool_size=(2, 2)))
        cnn.add(Flatten())
        cnn.add(Dense(8 * 8 * (7 * 8 + 8 + 9), activation='softmax'))
        cnn.compile(loss='mean_squared_error', optimizer='adam')
        return cnn

    def train_sensing_policy(self, states, actions, rewards):
        action_onehot = utils.to_categorical(actions,
                                             num_classes=self.sense_policy.output_shape[1])
        self._train_sensing([states, action_onehot, rewards])

    def train_move_policy(self, states, actions, rewards):
        action_onehot = utils.to_categorical(actions,
                                             num_classes=self.move_policy.output_shape[1])
        self._train_move([states, action_onehot, rewards])

    def update_belief_state(self, belief_input):
        batch = np.array([belief_input])
        return self.belief_state.predict(batch)[0]

    def get_sensing_policy(self, sense_input):
        batch = np.array([sense_input])
        return self.sense_policy.predict(batch)[0]

    def get_move_policy(self, move_input):
        batch = np.array([move_input])
        return self.move_policy.predict(batch)[0]

    def save_all(self):
        self.move_policy.save('move_policy.h5')
        self.sense_policy.save('sense_policy.h5')
        self.belief_state.save('belief_state.h5')

    def load_all(self):
        self.move_policy = load_model('move_policy.h5')
        self.sense_policy = load_model('sense_policy.h5')
        self.belief_state = load_model('belief_state.h5')
        # If we are not training, we want to be stateful
        # Make a stateful model and transfer the weights
        if not self.training:
            stateful = self.build_belief_state_network(training=False)
            stateful.set_weights(self.belief_state.get_weights())
            self.belief_state = stateful

    def save_belief(self):
        self.belief_state.save('belief_state.h5')


def kl_divergence(true_state, belief_state):
    total = 0
    for k in range(len(belief_state)):
        for i in range(len(belief_state[k])):
            for j in range(len(belief_state[k][i])):
                true_val = (0.0001 if true_state[k][i][j] == 0
                            else true_state[k][i][j])
                belief_val = (0.0001 if belief_state[k][i][j] == 0
                              else belief_state[k][i][j])
                total += (belief_val * np.log(belief_val / true_val))
    return total
