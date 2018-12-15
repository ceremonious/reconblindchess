from models import ChessModel
from chess import SQUARE_NAMES, Move
from reconBoard import ReconBoard
import numpy as np
np.random.seed(5)
np.set_printoptions(threshold=np.nan, linewidth=1000, suppress=True)


# Tests whether the model can respond to a single move made by white.
# White makes a move. Black observes on a square. The resulting belief state is printed
def simple_test(move, sense_square):
    hp = {'num_conv': 3, 'conv_filters': 70, 'conv_kernel': 3, 'num_lstm': 1, 'lstm_size': 250,
          'num_dense': 8, 'dense_size': 1500, 'lr': 0.1, 'momentum': 0.3, 'batch_size': 128}
    model = ChessModel(hp, True, training=False)
    board = ReconBoard()

    # Start of game
    observation = board.get_current_state(board.turn)
    print(np.round(model.get_belief_state(observation), 2))

    board.push(Move.from_uci(move))

    # Turn 1
    observation = board.get_pre_turn_observation()
    print(np.round(model.get_belief_state(observation), 2))
    observation = board.sense(sense_square)
    print(np.round(model.get_belief_state(observation), 2))

# Does the same thing as simple_test but the entire sequence of observations is passed in at once
# Should result in the same output
def seq_test(move, sense_square):
    model = ChessModel(True, training=True)
    board = ReconBoard()

    observations = []
    observations.append(board.get_current_state(board.turn))

    board.push(Move.from_uci(move))
    observations.append(board.get_pre_turn_observation())
    observations.append(board.sense(sense_square))

    batch = np.asarray([np.asarray(observations)])
    print(np.round(model.belief_state.predict(batch)[0][-1]), 2)


if __name__ == '__main__':
    simple_test("b2b4", 25)
