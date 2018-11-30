from multiprocessing import Queue, Value, Process
from reconBoard import ReconBoard
from chess import Move, SQUARE_NAMES, BaseBoard
from models import ChessModel
from keras.losses import kullback_leibler_divergence
from keras.preprocessing import sequence
from keras import backend as K
import time
import threading
import numpy as np
import json
import h5py
import random
np.set_printoptions(threshold=np.nan, linewidth=2000, suppress=True)


def self_play(should_play, queue, move_queue):
    num_games = 0
    while True:
        print("playing")
        while should_play.value:
            moves = [["e2e4", "a2a4", "h2h4"], ["e7e5"]]
            sensing = [[28], [28, 24, 31]]

            board = ReconBoard()

            ply_num = 0
            belief_input_training = [[board.get_current_state(True)],
                                     [board.get_current_state(True)]]
            belief_output_training = [[board.get_current_state(True)],
                                      [board.get_current_state(True)]]
            # board_states = []
            while not board.is_game_over() and should_play.value and ply_num < len(moves):
                color_name = "White" if board.turn else "Black"
                true_state = board.get_current_state(board.turn)
                previous_sense = board.get_previous_sense()
                scalar_input = np.array([previous_sense, ply_num])

                # Update our belief state based on the results of the last two ply
                # Any observations we made about empty squares / captures
                # And where our pieces are now
                observation = board.get_pre_turn_observation()
                belief_input_training[board.turn].append(observation)
                belief_output_training[board.turn].append(true_state)
                # board_states.append(board.board_fen())

                # Choose where to sense based on policy or exploration
                # square = np.random.randint(64)
                square = random.choice(sensing[ply_num])

                # print("{} observing at square {}".format(color_name, SQUARE_NAMES[square]))

                # Update belief state based on where we sense
                observation = board.sense(square)

                # Store training data for belief state
                belief_input_training[board.turn].append(observation)
                belief_output_training[board.turn].append(true_state)
                # board_states.append(board.board_fen())

                legal_moves = board.get_pseudo_legal_moves()
                # move = np.random.choice(legal_moves)
                move = random.choice(moves[ply_num])
                move = Move.from_uci(move)

                # print("{} making move {}".format(color_name, str(move)))

                board.push(move)
                ply_num += 1

            for i in range(2):
                belief_input_training[i] = np.asarray(belief_input_training[i])
                belief_output_training[i] = np.asarray(belief_output_training[i])

            if should_play.value:
                queue.put((belief_input_training, belief_output_training))

            num_games += 1

        # Wait here while the main thread is training the model
        while not should_play.value:
            pass

        new_move = move_queue.get()
        if new_move:
            moves[0].append(new_move[0])
            sensing[1].append(new_move[1])


def train_model():
    hp = {'num_conv': 3, 'conv_filters': 70, 'conv_kernel': 3, 'num_lstm': 1, 'lstm_size': 250,
          'num_dense': 8, 'dense_size': 1500, 'lr': 0.1, 'momentum': 0.3, 'batch_size': 128}
    queue = Queue()
    move_queue = Queue()
    should_play = Value('i', 1)

    num_processes = 1
    train_iteration = 20
    save_iteration = 20
    board = ReconBoard()
    model = ChessModel(hp, True)

    for i in range(num_processes):
        thread = threading.Thread(name=str(i),
                                  target=self_play,
                                  args=(should_play, queue, move_queue))
        thread.setDaemon(True)
        thread.start()

    num_trained = 0

    # These are lists of lists. Each internal list is a game
    observations = []
    true_states = []
    loss_over_time = []

    # Moves to add
    moves_to_add = [("b2b4", 25), ("c2c4", 26), ("d2d4", 27), ("f2f4", 29),
                    ("g2g4", 30), ("g1f3", 21), ("b1a3", 16)]
    move_index = 0
    num_epochs = 2

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

            loss = 10
            while loss > 0.03:
                result = model.train_belief_state(_input, _output, num_epochs)
                loss = result.history['loss'][-1]
                print(loss)

            print("Time: " + str(time.time()))
            print("Saving")
            model.save_belief()
            print("Saved")

            observations, true_states = [], []
            move_queue.put(moves_to_add[move_index])
            move_index += 1
            num_epochs = int(num_epochs * 1.5)
            should_play.value = 1


def evaluate_model(load_from_file):
    queue = Queue()
    should_play = Value('i', 1)

    num_processes = 3
    model = ChessModel(load_from_file)
    model = ChessModel(load_from_file)

    for i in range(num_processes):
        thread = threading.Thread(name=str(i),
                                  target=self_play,
                                  args=(should_play, queue, model))
        thread.setDaemon(True)
        thread.start()

    board = ReconBoard()
    observations = []
    true_states = []
    num_trained = 0

    while num_trained < 20:
        next_set = queue.get()
        for i in range(2):
            observations.append(next_set[0][i])
            true_states.append(next_set[1][i])

        num_trained += 1
        print(num_trained)

    should_play.value = 0

    _input = np.asarray(observations)
    _output = np.asarray(true_states)
    max_len = max(x.shape[0] for x in _input)
    _input = sequence.pad_sequences(_input, maxlen=max_len)
    _output = sequence.pad_sequences(_output, maxlen=max_len)

    baseline = np.asarray([board.get_current_state(True)] * max_len)

    print(model.belief_state.evaluate(_input, _output))
    _eval = K.eval(kullback_leibler_divergence(K.variable(_output), K.variable(baseline)))
    print(np.average(_eval, None))


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

    num_processes = 1

    for i in range(num_processes):
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
    # hyper_opt()
    # evaluate_model(True)
