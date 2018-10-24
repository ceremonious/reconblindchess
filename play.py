from models import ChessModel
from chess import SQUARE_NAMES, Move
from reconBoard import ReconBoard
import numpy as np
np.random.seed(4)
np.set_printoptions(threshold=np.nan, linewidth=200, suppress=True)

def test_belief_state(load_from_file=True):
    model = ChessModel(load_from_file)
    board = ReconBoard()
    ply_num = 0

    moves = ["e2e4", "e7e5", "f1b5", "c7c6", "g1f3", "d7d6", "e1g1"]
    sensing = [50, 50, 50, 50, 50, 50, 50, 5]
    while not board.is_game_over():
        color_name = "White" if board.turn else "Black"
        true_state = board.get_current_state(board.turn)
        previous_sense = board.get_previous_sense()
        scalar_input = np.array([previous_sense, ply_num])

        # Update our belief state based on the results of the last two ply
        if ply_num > 0:
            observation = board.observation[board.turn]
            belief_input = np.concatenate((board.belief_state[board.turn], observation),
                                          axis=2)

            board.belief_state[board.turn] = model.update_belief_state_multi(belief_input,
                                                                             scalar_input)

        # Choose where to sense based on policy or exploration
        # square = np.random.randint(64)
        square = sensing[ply_num]

        print("{} observing at square {}".format(color_name, SQUARE_NAMES[square]))

        # Update belief state based on where we sense
        if ply_num == 7:
            print(board)
            old = np.round(board.belief_state[False], 2)
        observation = board.sense(square)
        # print(observation)
        belief_input = np.concatenate((board.belief_state[board.turn], observation),
                                      axis=2)
        board.belief_state[board.turn] = model.update_belief_state_multi(belief_input,
                                                                         scalar_input)

        if ply_num == 7:
            print(board)
            print(np.round(board.belief_state[False] - old, 2))
        # if ply_num == 7:
        #     print(np.round(board.belief_state[False], 2) - old)

        legal_moves = board.get_pseudo_legal_moves()
        # move = np.random.choice(legal_moves)
        move = Move.from_uci(moves[ply_num])

        print("{} making move {}".format(color_name, str(move)))

        board.push(move)

        ply_num += 1

def simple_test():
    model = ChessModel(True)
    board = ReconBoard()
    print(board.belief_state[True])
    observation = board.sense(50)
    belief_input = np.concatenate((board.belief_state[board.turn], observation),
                                   axis=2)
    scalar_input = np.array([0, 0])
    print(model.update_belief_state_multi(belief_input, scalar_input))


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
    simple_test()
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
