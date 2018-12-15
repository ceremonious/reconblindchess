from multiprocessing import Queue, Value
from reconBoard import ReconBoard
from chess import Move, SQUARE_NAMES, BaseBoard
from models import ChessModel
from keras.preprocessing import sequence
from keras import backend as K
import time
import threading
import numpy as np
import json
import h5py
import random
np.set_printoptions(threshold=np.nan, linewidth=2000, suppress=True)


# This method plays games against itself and outputs the sequence of observations/board states to
# the queue. This method can be run in parallel in multiple threads
def self_play(should_play, queue):
    num_games = 0
    while True:

        while should_play.value:
            # Sequence of moves to be played. If there are multiple moves at an index, one is
            # chosen at random
            moves = [["e2e4", "a2a4", "h2h4"], ["e7e5"]]
            sensing = [[28], [28, 24, 31]]

            board = ReconBoard()

            ply_num = 0
            belief_input_training = [[board.get_current_state(True)],
                                     [board.get_current_state(True)]]
            belief_output_training = [[board.get_current_state(True)],
                                      [board.get_current_state(True)]]

            while not board.is_game_over() and should_play.value and ply_num < len(moves):
                color_name = "White" if board.turn else "Black"
                true_state = board.get_current_state(board.turn)
                previous_sense = board.get_previous_sense()
                scalar_input = np.array([previous_sense, ply_num])

                # observation is everything observed over the previous 2 ply
                observation = board.get_pre_turn_observation()
                belief_input_training[board.turn].append(observation)
                belief_output_training[board.turn].append(true_state)

                # Choose where to sense based on policy or exploration
                # square = np.random.randint(64)
                square = random.choice(sensing[ply_num])

                # print("{} observing at square {}".format(color_name, SQUARE_NAMES[square]))

                # Get observation from sensing
                observation = board.sense(square)

                belief_input_training[board.turn].append(observation)
                belief_output_training[board.turn].append(true_state)

                legal_moves = board.get_pseudo_legal_moves()
                # move = np.random.choice(legal_moves)
                move = random.choice(moves[ply_num])
                move = Move.from_uci(move)

                # print("{} making move {}".format(color_name, str(move)))

                board.push(move)
                ply_num += 1

            # Convert array to numpy array
            for i in range(2):
                belief_input_training[i] = np.asarray(belief_input_training[i])
                belief_output_training[i] = np.asarray(belief_output_training[i])

            if should_play.value:
                queue.put((belief_input_training, belief_output_training))

            num_games += 1

        # Wait here while the main thread is training the model
        while not should_play.value:
            pass


def train_model():
    hp = {'num_conv': 3, 'conv_filters': 70, 'conv_kernel': 3, 'num_lstm': 1, 'lstm_size': 250,
          'num_dense': 8, 'dense_size': 1500, 'lr': 0.1, 'momentum': 0.3, 'batch_size': 128}
    queue = Queue()
    should_play = Value('i', 1)

    num_threads = 1
    train_iteration = 20
    model = ChessModel(hp, 1)

    for i in range(num_threads):
        thread = threading.Thread(name=str(i),
                                  target=self_play,
                                  args=(should_play, queue))
        thread.setDaemon(True)
        thread.start()

    num_trained = 0

    # These are lists of lists. Each internal list is a game
    observations = []
    true_states = []

    print("Start: " + str(time.time()))
    while True:
        next_set = queue.get()
        for i in range(1):
            observations.append(next_set[0][i])
            true_states.append(next_set[1][i])

        num_trained += 1
        if num_trained % 20 == 0:
            print(num_trained)

        if num_trained % train_iteration == 0:
            should_play.value = 0
            while not queue.empty():
                next_set = queue.get()
                for i in range(1):
                    observations.append(next_set[0][i])
                    true_states.append(next_set[1][i])
                num_trained += 1

            _input = np.asarray(observations)
            _output = np.asarray(true_states)

            # _input and _output are lists of list. Each list is a game. A game is a sequence of
            # observations (input) with a corresponding sequence of true states (output)
            result = model.train_belief_state(_input, _output, 100)
            loss = result.history['loss'][-1]
            print(loss)

            print("Time: " + str(time.time()))
            print("Saving")
            model.save_belief()
            print("Saved")

            observations, true_states = [], []
            should_play.value = 1


# Tries different combinations of hyperparameters and sees which one performs the best in a fixed
# amount of training time
def hyper_opt():
    hp_space = {
        "num_conv": (1, 2, 3),
        "conv_filters": (10, 30, 50, 70, 90),
        "conv_kernel": (1, 2, 3),
        "num_lstm": (1, 2, 4, 6),
        "lstm_size": (100, 150, 200, 250, 300),
        "num_dense": (1, 2, 4, 8, 10),
        "dense_size": (500, 1000, 1500),
        "lr": (0.01, 0.1, 0.001),
        "momentum": (0, 0.1, 0.2, 0.3),
        "batch_size": (32, 64, 128)
    }
    queue = Queue()
    should_play = Value('i', 1)

    num_threads = 1

    for i in range(num_threads):
        thread = threading.Thread(name=str(i),
                                  target=self_play,
                                  args=(should_play, queue))
        thread.setDaemon(True)
        thread.start()

    num_trained = 0

    # These are lists of lists. Each internal list is a game
    observations = []
    true_states = []

    while True:
        next_set = queue.get()
        for i in range(1):
            observations.append(next_set[0][i])
            true_states.append(next_set[1][i])

        num_trained += 1

        if num_trained == 2:
            _input = np.asarray(observations)
            _output = np.asarray(true_states)
            choices = [0] * len(hp_space.items())
            num_hp = 0
            for key, val in hp_space.items():
                losses = []
                for i in range(len(val)):
                    choices[num_hp] = i
                    hp = generate_hp(hp_space, choices)
                    model = ChessModel(hp)
                    result = model.train_belief_state(_input, _output, 5000)
                    losses.append(result.history["loss"][-1])

                min_loss = min(losses)
                choices[num_hp] = losses.index(min_loss)
                print(losses)
                print(generate_hp(hp_space, choices))
                num_hp += 1


def generate_hp(hp_space, choices):
    hp = {}
    i = 0
    for key, val in hp_space.items():
        hp[key] = val[choices[i]]
        i += 1
    return hp


if __name__ == '__main__':
    train_model()
