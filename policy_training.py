from multiprocessing import Process, Queue, Value, Lock
from chess import ReconBoard, Move, SQUARE_NAMES
import time
import numpy as np
import json
import h5py
np.random.seed(2)

DECAY_GAMMA = 0.97
SENSE_LAMBDA = 4


def self_play(should_play, queue):
    from models import ChessModel, kl_divergence
    model = ChessModel()
    num_games = 0
    while True:
        print("playing")
        while should_play.value:
            board = ReconBoard()

            ply_num = 0
            belief_input_training = []
            belief_output_training = []
            sense_policy_input = [[], []]
            sense_policy_action = [[], []]
            sense_policy_reward = [[], []]
            move_policy_input = [[], []]
            move_policy_action = [[], []]
            move_policy_reward = [[], []]
            illegal_move_input = []
            illegal_move_action = []

            while not board.is_game_over() and should_play.value:
                #print(should_play.value)
                color_name = "White" if board.turn else "Black"
                # Gets the previous square we sensed at as a one-hot 8x8 array
                previous_sense = board.sense_to_array(board.get_previous_sense())
                # Gets a 7x8x8 array of the current state of the opponent's board (+ empty squares)
                # This is the optimal output for our belief state network
                true_opponent_state = board.get_opponent_state(board.turn)

                # Update our belief state based on the results of the last two ply
                if ply_num > 0:
                    observation = board.observation[board.turn]
                    belief_input = np.vstack((board.belief_state[board.turn], observation, previous_sense))
                    belief_input_training.append(belief_input)
                    belief_output_training.append(true_opponent_state)

                    board.belief_state[board.turn] = model.update_belief_state(belief_input)

                # Choose where to sense
                sense_input = np.vstack((board.belief_state[board.turn], previous_sense))
                sensing_policy = model.get_sensing_policy(sense_input)
                square = np.random.choice(np.arange(sensing_policy.shape[0]), p=sensing_policy)

                # print("{} observing at square {}".format(color_name, SQUARE_NAMES[square]))

                # Update belief state based on where we sense
                observation = board.sense(square)
                belief_input = np.vstack((board.belief_state[board.turn], observation, previous_sense))
                old_opp_state = board.belief_state[board.turn][6:]
                board.belief_state[board.turn] = model.update_belief_state(belief_input)
                new_opp_state = board.belief_state[board.turn][6:]

                pre_sense_divergence = kl_divergence(true_opponent_state, old_opp_state)
                post_sense_divergence = kl_divergence(true_opponent_state, new_opp_state)
                delta_divergence = post_sense_divergence - pre_sense_divergence

                # Store training data for belief state
                belief_input_training.append(belief_input)
                belief_output_training.append(true_opponent_state)

                # Store training data for sense policy
                sense_policy_input[board.turn].append(sense_input)
                sense_policy_action[board.turn].append(square)
                sense_policy_reward[board.turn].append(-1 * SENSE_LAMBDA * delta_divergence)

                legal_moves = board.get_pseudo_legal_moves()
                move_input = np.vstack((board.belief_state[board.turn], previous_sense))
                move_policy = model.get_move_policy(move_input)
                move = None
                while move is None:
                    move_num = np.random.choice(np.arange(move_policy.shape[0]), p=move_policy)
                    move = board.get_move_from_model(move_num)
                    if move not in legal_moves:
                        illegal_move_input.append(move_input)
                        illegal_move_action.append(move_num)
                        move = None

                # print("{} making move {}".format(color_name, str(move)))
                turn = board.turn
                reward = board.push(move)

                move_policy_input[turn].append(move_input)
                move_policy_action[turn].append(move_num)
                move_policy_reward[turn].append(reward)

                ply_num += 1

            result = board.result()

            # Calculate rewards for white
            i = len(move_policy_reward[True]) - 1
            move_policy_reward[True][i] = result
            i -= 1
            while i >= 0:
                move_policy_reward[True][i] += DECAY_GAMMA * move_policy_reward[True][i + 1]

            # Calculate rewards for black
            i = len(move_policy_reward[False]) - 1
            move_policy_reward[False][i] = -1 * result
            i -= 1
            while i >= 0:
                move_policy_reward[False][i] += DECAY_GAMMA * move_policy_reward[False][i + 1]

            illegal_move_reward = [-1000] * len(illegal_move_input)

            sense_input = np.asarray(sense_policy_input[0] + sense_policy_input[1])
            sense_action = np.asarray(sense_policy_action[0] + sense_policy_action[1])
            sense_reward = np.asarray(sense_policy_reward[0] + sense_policy_reward[1])
            move_input = np.asarray(move_policy_input[0] + move_policy_input[1] + illegal_move_input)
            move_action = np.asarray(move_policy_action[0] + move_policy_action[1] + illegal_move_action)
            move_reward = np.asarray(move_policy_reward[0] + move_policy_reward[1] + illegal_move_reward)

            if should_play.value:
                queue.put((np.asarray(belief_input_training),
                           np.asarray(belief_output_training),
                           sense_input, sense_action, sense_reward,
                           move_input, move_action, move_reward))

        # Wait here while the main thread is training the model
        while not should_play.value:
            pass

        # Load the new model and continue playing
        model = ChessModel(True)


def save_training_data(data):
    labels = ["belief_input", "belief_output", "sense_input", "sense_action", "sense_reward",
              "move_input", "move_action", "move_reward"]

    with h5py.File('training_data.h5', 'a') as hf:
        for i in range(len(data)):
            num_adding = data[i].shape[0]
            hf[labels[i]].resize(hf[labels[i]].shape[0] + num_adding, axis=0)
            hf[labels[i]][-num_adding:] = data[i]


if __name__ == '__main__':
    queue = Queue()
    should_play = Value('i', 1)

    with h5py.File('training_data.h5', 'w') as hf:
        hf.create_dataset("belief_input", shape=(0, 21, 8, 8), maxshape=(None, 21, 8, 8))
        hf.create_dataset("belief_output", shape=(0, 7, 8, 8), maxshape=(None, 7, 8, 8))
        hf.create_dataset("sense_input", shape=(0, 14, 8, 8), maxshape=(None, 14, 8, 8))
        hf.create_dataset("sense_action", shape=(0, 1), maxshape=(None, 1))
        hf.create_dataset("sense_reward", shape=(0, 1), maxshape=(None, 1))
        hf.create_dataset("move_input", shape=(0, 14, 8, 8), maxshape=(None, 14, 8, 8))
        hf.create_dataset("move_action", shape=(0, 1), maxshape=(None, 1))
        hf.create_dataset("move_reward", shape=(0, 1), maxshape=(None, 1))

    num_processes = 6
    save_iteration = 20000

    processes = [None] * num_processes
    for i in range(num_processes):
        processes[i] = Process(target=self_play, args=(should_play, queue))
        processes[i].start()

    # Import keras in each process seperately to make it work
    from models import ChessModel
    num_trained = 0
    saved = False
    while True:
        # Whenever a game is finished playing, we save it to the training data file
        next_set = queue.get()
        print(next_set.shape)
        exit()
        save_training_data(next_set)
        num_trained += 1
        if num_trained % 25 == 0:
            print(num_trained)

        # Every x iterations, we train and save the model
        if num_trained % save_iteration == 0:

            # Tell the workers to stop playing and empty the queue
            should_play.value = 0
            while not queue.empty():
                save_training_data(queue.get())
                num_trained += 1

            if saved:
                model = ChessModel(True)
            else:
                model = ChessModel()

            with h5py.File('training_data.h5', 'r') as hf:
                # Choose 50000 random samples and train on them
                num_data = hf['belief_input'].shape[0]
                num_select = min(num_data, 50000)
                indices = list(np.random.choice(num_data, size=num_select, replace=False))
                indices.sort()
                model.train_belief_state(hf['belief_input'][indices], hf['belief_output'][indices])

            # Save the model and resume play
            model.save_belief()
            saved = True
            should_play.value = 1
