from models import ChessModel
from chess import SQUARE_NAMES, Move
from reconBoard import ReconBoard
import numpy as np
np.random.seed(5)
np.set_printoptions(threshold=np.nan, linewidth=1000, suppress=True)

def test_belief_state(load_from_file=True):
    models = [ChessModel(load_from_file), ChessModel(load_from_file)]
    board = ReconBoard()
    ply_num = 0

    moves = ["e2e4", "e7e5", "f1b5", "c7c6", "g1f3", "d7d6", "e1g1"]
    sensing = [50, 50, 50, 50, 50, 50, 50, 5]
    while not board.is_game_over():

        color_name = "White" if board.turn else "Black"
        true_state = board.get_current_state(board.turn)
        previous_sense = board.get_previous_sense()

        observation = np.add(board.observation[board.turn],
                             board.my_pieces_observation(board.turn))

        starting_board_state = models[board.turn].get_belief_state(observation)
        # if ply_num == 7:
        #     print(board)
        #     old = board_state

        # Choose where to sense based on policy or exploration
        square = np.random.randint(64) # sensing[ply_num]

        print("{} observing at square {}".format(color_name, SQUARE_NAMES[square]))

        # Update belief state based on where we sense
        observation = board.sense(square)

        next_board_state = models[board.turn].get_belief_state(observation)

        print(np.round(next_board_state - starting_board_state), 3)

        legal_moves = board.get_pseudo_legal_moves()
        move = np.random.choice(legal_moves) # Move.from_uci(moves[ply_num])

        print("{} making move {}".format(color_name, str(move)))

        board.push(move)
        ply_num += 1


def simple_test():
    model = ChessModel(True, training=False)
    board = ReconBoard()
    # observation = board.my_pieces_observation(board.turn)
    observation = board.get_current_state(True)
    model.get_belief_state(observation)

    board.sense(3)
    board.push(Move.from_uci("e2e4"))

    observation = board.my_pieces_observation(board.turn)
    pre_sense = np.round(model.get_belief_state(observation), 3)
    print(pre_sense)
    observation = board.sense(28)

    post_sense = np.round(model.get_belief_state(observation), 2)
    print(post_sense)


def seq_test():
    model = ChessModel(True, training=True)
    board = ReconBoard()

    observations = []
    # observation = board.my_pieces_observation(board.turn)
    observations.append(board.get_current_state(True))

    board.sense(3)
    board.push(Move.from_uci("a2a4"))

    observations.append(board.my_pieces_observation(board.turn))
    observations.append(board.sense(28))

    batch = np.asarray([np.asarray(observations)])
    print(model.belief_state.predict(batch)[0][-1])


def play():
    model = ChessModel(True)
    board = ReconBoard()

    ply_num = 0
    while not board.is_game_over():
        color_name = "White" if board.turn else "Black"
        previous_sense = board.sense_to_array(board.get_previous_sense())
        true_opponent_state = board.get_opponent_state(board.turn)

        if ply_num > 0:
          observation = board.observation[board.turn]
          belief_input = np.vstack((board.belief_state[board.turn], observation,
                                    previous_sense))
          # update belief state with observation from our/opp move
          board.belief_state[board.turn] = model.update_belief_state(belief_input)

        # Get sensing policy
        sense_input = np.vstack((board.belief_state[board.turn], previous_sense))
        sensing_policy = model.get_sensing_policy(sense_input)

        # Choose where to sense based on policy
        square = np.argmax(sensing_policy)
        print(sensing_policy)

        print("{} observing at square {}".format(color_name, SQUARE_NAMES[square]))

        # Update belief state based on where we sense
        observation = board.sense(square)
        belief_input = np.vstack((board.belief_state[board.turn], observation, previous_sense))
        old_opp_state = board.belief_state[board.turn][6:]
        board.belief_state[board.turn] = model.update_belief_state(belief_input)
        new_opp_state = board.belief_state[board.turn][6:]

        # Get move policy based on belief state
        move_input = np.vstack((board.belief_state[board.turn], previous_sense))
        move_policy = model.get_move_policy(move_input)

        legal_moves = board.get_pseudo_legal_moves()
        moves = np.argsort(-move_policy)
        # Iterate through best moves until we find a legal move
        for move in moves:
            move = board.get_move_from_model(move)
            if move in legal_moves:
                break

        print("{} making move {}".format(color_name, str(move)))

        board.push(move)

if __name__ == '__main__':
    seq_test()
    # model = ChessModel(load_from_file=True, training=False)
    #test_belief_state()
    # model = ChessModel(True)
    # print(board.get_current_state(False))
    """board = ReconBoard()
    observation = board.sense(2)
    _input = np.array([[observation]])
    print(_input.shape)
    print(model.belief_state.predict(_input))
    # test_belief_state(True)
    # simple_test()"""
    """model = ChessModel(True)
    board = ReconBoard()
    board.push(Move.from_uci("e2e4"))
    board.push(Move.from_uci("e7e5"))
    board.push(Move.from_uci("f1b5"))
    board.push(Move.from_uci("c7c6"))

    board.push(Move.from_uci("g1f3"))
    board.push(Move.from_uci("d7d6"))
    board.push(Move.from_uci("e1g1"))
    print(board.belief_state[False])
    board.sense(15)
    print(board.belief_state[False])
    #board.push(Move.from_uci("c6b5"))

    print(board)
    exit()
    #print(board.belief_state[True])
    #print(board.observation[True])
    c = np.concatenate((board.belief_state[True], board.observation[True]), axis=2)
    print(c)
    model = ChessModel()"""
