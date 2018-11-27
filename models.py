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
set_random_seed(12321)

K.clear_session()

class ChessModel:
    def __init__(self, load_from_file=False, training=True):
        self.training = training

        if load_from_file:
            self.load_all()
        else:
            self.belief_state = self.build_belief_state_network()
            self.sense_policy = self.build_sensing_policy_network()
            self.move_policy = self.build_move_policy_network()

        # Build custom loss function
        self._train_sensing = self.__build_train_fn(self.sense_policy)
        self._train_move = self.__build_train_fn(self.move_policy)

        # Needed for multi threading
        self.belief_state._make_predict_function()
        self.sense_policy._make_predict_function()
        self.move_policy._make_predict_function()
        self.session = K.get_session()
        self.graph = tf.get_default_graph()

        plot_model(self.belief_state, to_file='model.png')
        # print(self.belief_state.summary())


    # 8x8 observation with 13 channels
    def build_belief_state_network(self, training=True):
        input_shape = (8, 8, 13)

        # Convolutional part for each timestep
        # cnn = Sequential()
        # cnn.add(Conv2D(384, (2,2),
        #               activation='relu',
        #               data_format="channels_last",
        #               padding='same',
        #               kernel_regularizer=regularizers.l2(0.01),
        #               input_shape=input_shape))
        # cnn.add(Flatten())

        flatten = Sequential()
        flatten.add(Conv2D(52, (3, 3), input_shape=(8, 8, 13)))
        flatten.add(Flatten())
        # flatten.add(Flatten(input_shape=(8, 8, 13)))

        lstm = Sequential()
        if not training:
            lstm.add(TimeDistributed(flatten, input_shape=(None, 8, 8, 13),
                                     batch_input_shape=(1, 1, 8, 8, 13)))
        else:
            lstm.add(TimeDistributed(flatten, input_shape=(None, 8, 8, 13)))

        for i in range(2):
            stateful = not training
            lstm.add(LSTM(200, return_sequences=True, stateful=stateful))

        model = Sequential()
        for i in range(1):
            model.add(Dense(1000))
        model.add(Dense(64 * 13))
        model.add(Reshape((64, 13)))
        model.add(Activation('softmax'))
        model.add(Reshape((8, 8, 13)))

        lstm.add(TimeDistributed(model, input_shape=(None, 8 * 8 * 13)))

        sgd = optimizers.SGD(lr=0.1, momentum=0.1)
        lstm.compile(loss='kld', optimizer=sgd)
        return lstm

    def get_belief_state(self, observation):
        with self.session.as_default():
            with self.graph.as_default():
                batch = np.asarray([np.asarray([observation])])
                return self.belief_state.predict(batch)[0]


    def train_belief_state(self, _input, _output, epochs=1):
        max_len = max(x.shape[0] for x in _input)
        _input = sequence.pad_sequences(_input, maxlen=max_len)
        _output = sequence.pad_sequences(_output, maxlen=max_len)
        return self.belief_state.fit(_input, _output, batch_size=32, epochs=epochs,
                                     verbose=0, validation_split=0)

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
        self.belief_state = load_model('belief_state_1.h5')
        # If we are not training, we want to be stateful
        if not self.training:
            stateful = self.build_belief_state_network(training=False)
            stateful.set_weights(self.belief_state.get_weights())
            self.belief_state = stateful


    def save_belief(self):
        self.belief_state.save('belief_state.h5')

    def __build_train_fn(self, model):
        """Create a train function
        It replaces `model.fit(X, y)` because we use the output of model and use it for training.
        For example, we need action placeholder
        called `action_one_hot` that stores, which action we took at state `s`.
        Hence, we can update the same action.
        This function will create
        `self.train_fn([state, action_one_hot, discount_reward])`
        which would train the model.
        """
        action_prob_placeholder = model.output
        action_onehot_placeholder = K.placeholder(shape=(None, model.output_shape[1]),
                                                  name="action_onehot")
        discount_reward_placeholder = K.placeholder(shape=(None,),
                                                    name="discount_reward")

        action_prob = K.sum(action_prob_placeholder * action_onehot_placeholder, axis=1)
        log_action_prob = K.log(action_prob)

        loss = - log_action_prob * discount_reward_placeholder
        loss = K.mean(loss)

        adam = optimizers.Adam()

        updates = adam.get_updates(params=model.trainable_weights,
                                   loss=loss)

        return K.function(inputs=[model.input,
                                action_onehot_placeholder,
                                discount_reward_placeholder],
                                outputs=[],
                                updates=updates)


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
