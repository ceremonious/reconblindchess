from multiprocessing import Queue, Value, Process
from reconBoard import ReconBoard
from chess import Move, SQUARE_NAMES, BaseBoard
from models import ChessModel
from keras.losses import kullback_leibler_divergence
from keras import backend as K
import time
import threading
import numpy as np
import json
import h5py
np.random.seed(2133)
np.set_printoptions(threshold=np.nan, linewidth=2000, suppress=True)


def self_play(should_play, queue, model):
    num_games = 0
    while True:
        print("playing")
        while should_play.value:
            board = ReconBoard()

            ply_num = 0
            belief_input_training = []
            scalar_input_training = []
            belief_output_training = []
            # board_states = []
            colors = []
            while not board.is_game_over() and should_play.value:
                color_name = "White" if board.turn else "Black"
                true_state = board.get_current_state(board.turn)
                previous_sense = board.get_previous_sense()
                scalar_input = np.array([previous_sense, ply_num])

                # Update our belief state based on the results of the last two ply
                if ply_num > 0:
                    observation = board.observation[board.turn]
                    belief_input = np.concatenate((board.belief_state[board.turn], observation),
                                                  axis=2)
                    belief_input_training.append(belief_input)
                    scalar_input_training.append(scalar_input)
                    belief_output_training.append(true_state)
                    # board_states.append(board.board_fen())
                    colors.append(board.turn)

                    board.belief_state[board.turn] = model.update_belief_state_multi(belief_input,
                                                                                     scalar_input)

                # Choose where to sense based on policy or exploration
                square = np.random.randint(64)

                # print("{} observing at square {}".format(color_name, SQUARE_NAMES[square]))

                # Update belief state based on where we sense
                observation = board.sense(square)
                belief_input = np.concatenate((board.belief_state[board.turn], observation),
                                              axis=2)
                board.belief_state[board.turn] = model.update_belief_state_multi(belief_input,
                                                                                 scalar_input)

                # Store training data for belief state
                belief_input_training.append(belief_input)
                scalar_input_training.append(scalar_input)
                belief_output_training.append(true_state)
                # board_states.append(board.board_fen())
                colors.append(board.turn)

                legal_moves = board.get_pseudo_legal_moves()
                move = np.random.choice(legal_moves)

                # print("{} making move {}".format(color_name, str(move)))

                board.push(move)
                ply_num += 1

            if should_play.value:
                queue.put((np.asarray(belief_input_training),
                           np.asarray(scalar_input_training),
                           np.asarray(belief_output_training),
                           colors))

        # Wait here while the main thread is training the model
        while not should_play.value:
            pass


def train_model():
    queue = Queue()
    should_play = Value('i', 1)

    num_processes = 4
    train_iteration = 10
    save_iteration = 2000
    model = ChessModel(False)

    for i in range(num_processes):
        thread = threading.Thread(name=str(i),
                                  target=self_play,
                                  args=(should_play, queue, model))
        thread.setDaemon(True)
        thread.start()

    num_trained = 0
    belief_input, scalar_input, _output, loss_over_time = [], [], [], []
    while True:
        next_set = queue.get()
        belief_input.append(next_set[0])
        scalar_input.append(next_set[1])
        _output.append(next_set[2])

        num_trained += 1
        if num_trained % 25 == 0:
            print(num_trained)

        if num_trained % train_iteration == 0:
            should_play.value = 0
            while not queue.empty():
                next_set = queue.get()
                belief_input.append(next_set[0])
                scalar_input.append(next_set[1])
                _output.append(next_set[2])
                num_trained += 1

            belief_input = np.concatenate(tuple(belief_input))
            scalar_input = np.concatenate(tuple(scalar_input))
            _output = np.concatenate(tuple(_output))
            print(np.round(belief_input[312], 2))
            print(np.round(_output[312]), 2)
            exit()

            result = model.train_belief_state(belief_input, scalar_input, _output)
            loss_over_time.append(result.history['val_loss'][0])
            print(result.history['val_loss'][0])

            if num_trained % save_iteration == 0:
                print("Saving old")
                print(loss_over_time)
                model.save_belief()

            belief_input, scalar_input, _output = [], [], []
            should_play.value = 1


def evaluate_model(load_from_file):
    queue = Queue()
    should_play = Value('i', 1)

    num_processes = 3
    model = ChessModel(load_from_file)

    for i in range(num_processes):
        thread = threading.Thread(name=str(i),
                                  target=self_play,
                                  args=(should_play, queue, model))
        thread.setDaemon(True)
        thread.start()

    board = ReconBoard()
    belief_input_set = []
    scalar_input_set = []
    belief_output_set = []
    baseline = []
    baseline_color = [board.get_current_state(False), board.get_current_state(True)]
    num_trained = 0

    while num_trained < 20:
        next_set = queue.get()
        belief_input_set.append(next_set[0])
        scalar_input_set.append(next_set[1])
        belief_output_set.append(next_set[2])
        # If it's white turn, the baseline is starting pos
        for color in next_set[3]:
            baseline.append(baseline_color[color])
        num_trained += 1
        print(num_trained)

    should_play.value = 0

    belief_input = np.concatenate(tuple(belief_input_set))
    scalar_input = np.concatenate(tuple(scalar_input_set))
    belief_output = np.concatenate(tuple(belief_output_set))
    baseline = np.asarray(baseline)
    print(model.belief_state.evaluate([belief_input, scalar_input], belief_output))
    _eval = K.eval(kullback_leibler_divergence(K.variable(belief_output), K.variable(baseline)))
    print(np.average(_eval, None))

if __name__ == '__main__':
    train_model()
    # evaluate_model(True)
