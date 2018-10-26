from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Conv2D, BatchNormalization, Activation, Flatten, Add, TimeDistributed, LSTM, MaxPooling2D, Reshape
from keras import regularizers, utils, optimizers
from keras import backend as K
from reconBoard import ReconBoard
from chess import Move, SQUARE_NAMES
from keras.layers.merge import concatenate
import numpy as np
import tensorflow as tf
from tensorflow import set_random_seed
from keras.utils import plot_model
set_random_seed(2)

K.clear_session()

class ChessModel:
    def __init__(self, load_from_file=False):
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

        # plot_model(self.belief_state, to_file='model.png')
        # print(self.belief_state.summary())


    # 8x8 board with 26 channels (13 channels from old belief state, 13 from observation)
    # 8x8x13 belief state
    def build_belief_state_network(self):
        input_shape = (8, 8, 26)

        # Convolutional part
        main_input = Input(shape=input_shape, name='belief_input')
        cnn = (Conv2D(10, (2,2),
                          activation='relu',
                          data_format="channels_last",
                          padding='same',
                          kernel_regularizer=regularizers.l2(0.01)))(main_input)
        for i in range(5):
            for i in range(3):
                cnn = (Conv2D(10, (2,2),
                              activation='relu',
                              data_format="channels_last",
                              padding='same',
                              kernel_regularizer=regularizers.l2(0.01)))(cnn)
            cnn = MaxPooling2D(pool_size=(2, 2),
                               data_format="channels_last",
                               padding='same')(cnn)
        cnn = Flatten()(cnn)

        # Input for previous sense and ply num
        scalar_input = Input(shape=(2,), name='scalar_input')
        merge = concatenate([cnn, scalar_input])

        for i in range(3):
            merge = Dense(1000)(merge)

        output = Dense(8 * 8 * 13)(merge)
        output = Reshape((64, 13))(output)
        output = Activation('softmax')(output)
        output = Reshape((8, 8, 13))(output)

        model = Model(inputs=[main_input, scalar_input], outputs=[output])
        sgd = optimizers.SGD(lr=0.0005)
        model.compile(loss='kld', optimizer=sgd)
        return model


    def update_belief_state_multi(self, belief_input, scalar_input):
        with self.session.as_default():
            with self.graph.as_default():
                batch = [[belief_input], [scalar_input]]
                return self.belief_state.predict(batch)[0]


    def train_belief_state(self, belief_input, scalar_input, _output):
        _input = [belief_input, scalar_input]
        return self.belief_state.fit(_input, _output, verbose=0, validation_split=0.1)

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
        self.belief_state = load_model('belief_state_lr0005.h5')

    def save_belief(self):
        self.belief_state.save('belief_state_lr001.h5')

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
