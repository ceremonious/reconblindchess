from multiprocessing import Process, Queue, Value, Lock
from chess import ReconBoard, Move, SQUARE_NAMES
from models import kl_divergence
import random
import time
import numpy as np

# Hyperparameters
DECAY_GAMMA = 0.97
EPSILON = 0.7
SENSE_LAMBDA = 2

def self_play(model, should_play, file_lock, queue):
    num_games = 0
    """if file_lock:
        file_lock.acquire()
        model = ChessModel(load_from_file=True)
        print("loaded")
        file_lock.release()
    else:
        model = ChessModel(False)"""

    while should_play.value:
        board = ReconBoard()

        ply_num = 0
        sense_training = []
        belief_training = []
        move_training = []
        while not board.is_game_over() and should_play.value:
            color_name = "White" if board.turn else "Black"
            previous_sense = board.sense_to_array(board.get_previous_sense())
            true_opponent_state = board.get_opponent_state(board.turn)

            if ply_num > 0:
                observation = board.observation[board.turn]
                belief_input = np.vstack((board.belief_state[board.turn], observation,
                                          previous_sense))
                belief_training.append({
                    "belief_input": belief_input,
                    "true_state": true_opponent_state
                })

                # update belief state with observation from our/opp move
                board.belief_state[board.turn] = model.update_belief_state(belief_input)

            # Get sensing policy
            sense_input = np.vstack((board.belief_state[board.turn], previous_sense))
            sensing_policy = model.get_sensing_policy(sense_input)

            # Choose where to sense based on policy
            if np.random.rand(1) < EPSILON:
                square = np.random.randint(64)
            else:
                square = np.argmax(sensing_policy)

            # print("{} observing at square {}".format(color_name, SQUARE_NAMES[square]))

            # Update belief state based on where we sense
            observation = board.sense(square)
            belief_input = np.vstack((board.belief_state[board.turn], observation, previous_sense))
            old_opp_state = board.belief_state[board.turn][6:]
            board.belief_state[board.turn] = model.update_belief_state(belief_input)
            new_opp_state = board.belief_state[board.turn][6:]


            # Store training data for sensing
            pre_sense_divergence = kl_divergence(true_opponent_state, old_opp_state)
            post_sense_divergence = kl_divergence(true_opponent_state, new_opp_state)
            sense_training.append({
                "sense_input": sense_input,
                "policy": sensing_policy,
                "square": square,
                "change_in_divergence": post_sense_divergence - pre_sense_divergence,
                "ply_num": ply_num,
                "color": board.turn
            })

            # Store training data for belief state
            belief_training.append({
                "belief_input": belief_input,
                "true_state": true_opponent_state
            })

            # Get move policy based on belief state
            move_input = np.vstack((board.belief_state[board.turn], previous_sense))
            move_policy = model.get_move_policy(move_input)

            legal_moves = board.get_pseudo_legal_moves()
            if np.random.rand(1) < EPSILON:
                move = random.choice(legal_moves)
            else:
                # Sort in descending order
                moves = np.argsort(-move_policy)
                # Iterate through best moves until we find a legal move
                for move in moves:
                    move = board.get_move_from_model(move)
                    if move in legal_moves:
                        break

            #print("{} making move {}".format(color_name, str(move)))

            move_training.append({
                "move_input": move_input,
                "policy": move_policy,
                "move": board.get_model_num_from_move(move),
                "ply_num": ply_num,
                "color": board.turn
            })

            board.push(move)
            ply_num += 1

        # print(board)
        if should_play.value:
            result = board.result()
            if result == 0.5:
                result = 0
            queue.put((result, ply_num, sense_training[:], belief_training[:], move_training[:]))
            num_games += 1

    queue.close()

def train_models(model, result, total_ply, sense_training, belief_training, move_training):
    _input = []
    _output = []
    for obvs in belief_training:
        _input.append(obvs['belief_input'])
        _output.append(obvs['true_state'])
    model.train_belief_state(np.array(_input), np.array(_output))

    _input = []
    _output = []
    for sense in sense_training:
        base_reward = result if sense['color'] else -1 * result
        decay = DECAY_GAMMA * (total_ply - sense['ply_num'])
        reward = decay * base_reward - SENSE_LAMBDA * sense['change_in_divergence']
        policy = sense['policy']
        policy[sense['square']] = reward
        _input.append(sense['sense_input'])
        _output.append(policy)
    model.train_sensing_policy(np.array(_input), np.array(_output))

    _input = []
    _output = []
    for move in move_training:
        base_reward = result if move['color'] else -1 * result
        decay = DECAY_GAMMA * (total_ply - move['ply_num'])
        reward = decay * base_reward
        policy = move['policy']
        policy[move['move']] = reward
        _input.append(move['move_input'])
        _output.append(policy)
    model.train_move_policy(np.array(_input), np.array(_output))


if __name__ == '__main__':
    from models import ChessModel
    np.random.seed(12)
    queue = Queue()
    models = [ChessModel(False), ChessModel(False), ChessModel(False)]
    should_play = Value('i', 1)
    file_lock = Lock()
    num_processes = 3
    processes = [None] * num_processes
    for i in range(num_processes):
        processes[i] = Process(target=self_play, args=(models[i], should_play, None, queue,))
        processes[i].start()


    model = ChessModel()
    num_trained = 0
    while True:
        next_set = queue.get()
        train_models(model, *next_set)
        num_trained += 1
        print(num_trained)

        if num_trained % 5 == 0:
            should_play.value = 0
            for process in processes:
                process.join()

            # export HDF5_USE_FILE_LOCKING="FALSE" to be able to save
            model.save_all()
            models = [ChessModel(True), ChessModel(True), ChessModel(True)]
            should_play.value = 1
            for i in range(num_processes):
                processes[i] = Process(target=self_play, args=(models[i], should_play,
                                                               file_lock, queue,))
                processes[i].start()
                print("started")

