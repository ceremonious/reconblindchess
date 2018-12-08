from models import ChessModel
from chess import SQUARE_NAMES, Move
from reconBoard import ReconBoard
import numpy as np
np.random.seed(5)
np.set_printoptions(threshold=np.nan, linewidth=1000, suppress=True)


def simple_test():
    hp = {'num_conv': 3, 'conv_filters': 70, 'conv_kernel': 3, 'num_lstm': 1, 'lstm_size': 250,
          'num_dense': 8, 'dense_size': 1500, 'lr': 0.1, 'momentum': 0.3, 'batch_size': 128}
    model = ChessModel(hp, True, training=False)
    board = ReconBoard()

    # Start of game
    observation = board.get_current_state(board.turn)
    print(np.round(model.get_belief_state(observation), 2))

    board.push(Move.from_uci("b2b4"))

    # Turn 1
    observation = board.get_pre_turn_observation()
    print(np.round(model.get_belief_state(observation), 2))
    observation = board.sense(25)
    print(np.round(model.get_belief_state(observation), 2))


def seq_test():
    model = ChessModel(True, training=True)
    board = ReconBoard()

    observations = []
    observations.append(board.get_current_state(board.turn))

    board.push(Move.from_uci("a2a4"))
    observations.append(board.get_pre_turn_observation())
    observations.append(board.sense(28))

    batch = np.asarray([np.asarray(observations)])
    print(np.round(model.belief_state.predict(batch)[0][-1]), 2)


if __name__ == '__main__':
    simple_test()
