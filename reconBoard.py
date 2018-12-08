from chess import Board, BaseBoard, Move, BB_ALL, scan_reversed, BB_SQUARES, BB_PAWN_ATTACKS, WHITE, BLACK, square_rank, square_file, square_distance, BB_RANK_1, BB_RANK_3, BB_RANK_4, BB_RANK_5, BB_RANK_6, BB_RANK_8, A1, A8, D1, D8, F1, F8, H1, H8, BB_FILE_B, BB_FILE_C, BB_FILE_D, BB_FILE_F, BB_FILE_G, KNIGHT, BISHOP, ROOK, QUEEN, STARTING_FEN
import numpy as np
from keras.utils import to_categorical

# Converts an integer to a numpy bit array of length 64
def int_to_bit_array(num):
    _str = np.binary_repr(num, width=64)
    arr = np.asarray(np.fromstring(_str, dtype='u1', count=64) - ord('0'))
    return np.reshape(arr, (1, 8, 8))

# Bitwise not
def bit_not(n, numbits=64):
    return (1 << numbits) - 1 - n


class ReconBoard(Board):

    def __init__(self, fen=STARTING_FEN):
        super().__init__(fen=fen)
        # Visible state emulates what JHUAPL website shows on its board while playing
        self.visible_state = [BaseBoard(), BaseBoard()]
        self.observation = [np.zeros((8, 8, 13), dtype='float32'),
                            np.zeros((8, 8, 13), dtype='float32')]
        self.sense_history = [[], []]

    def remove_opp_pieces(self, color):
        self.removed_pieces = {
            'occupied': self.occupied_co[color],
            'pawns': self.pawns,
            'knights': self.knights,
            'bishops': self.bishops,
            'rooks': self.rooks,
            'queens': self.queens,
            'kings': self.kings,
            'promoted': self.promoted
        }
        mask = self.occupied_co[color]

        self.pawns ^= mask
        self.knights ^= mask
        self.bishops ^= mask
        self.rooks ^= mask
        self.queens ^= mask
        self.kings ^= mask

        self.occupied ^= mask
        self.occupied_co[color] = 0

        self.promoted &= ~mask

    def restore_opp_pieces(self, color):
        if not self.removed_pieces:
            return

        mask = self.removed_pieces['occupied']

        self.pawns = self.removed_pieces['pawns']
        self.knights = self.removed_pieces['knights']
        self.bishops = self.removed_pieces['bishops']
        self.rooks = self.removed_pieces['rooks']
        self.queens = self.removed_pieces['queens']
        self.kings = self.removed_pieces['kings']

        self.occupied ^= mask
        self.occupied_co[color] = self.removed_pieces['occupied']

        self.promoted = self.removed_pieces['promoted']

        self.removed_pieces = None

    # Remove all of the opponent's pieces before generating legal moves
    def get_pseudo_legal_moves(self):
        self.remove_opp_pieces(not self.turn)
        moves = list(self.generate_moves())
        self.restore_opp_pieces(not self.turn)
        return moves


    # This function is taken from the python-chess library
    # Only modification is making all pawn captures pseudo-legal
    def generate_moves(self, from_mask=BB_ALL, to_mask=BB_ALL):
        our_pieces = self.occupied_co[self.turn]

        # Generate piece moves.
        non_pawns = our_pieces & ~self.pawns & from_mask
        for from_square in scan_reversed(non_pawns):
            moves = self.attacks_mask(from_square) & ~our_pieces & to_mask
            for to_square in scan_reversed(moves):
                yield Move(from_square, to_square)

        # Generate castling moves.
        if from_mask & self.kings:
            yield from self.generate_castling_moves(from_mask, to_mask)

        # The remaining moves are all pawn moves.
        pawns = self.pawns & self.occupied_co[self.turn] & from_mask
        if not pawns:
            return

        # Generate pawn captures.
        capturers = pawns
        for from_square in scan_reversed(capturers):
            # All pawn captures are now pseudo-legal
            targets = (BB_PAWN_ATTACKS[self.turn][from_square] & to_mask)

            for to_square in scan_reversed(targets):
                if square_rank(to_square) in [0, 7]:
                    yield Move(from_square, to_square, QUEEN)
                    yield Move(from_square, to_square, ROOK)
                    yield Move(from_square, to_square, BISHOP)
                    yield Move(from_square, to_square, KNIGHT)
                else:
                    yield Move(from_square, to_square)

        # Prepare pawn advance generation.
        if self.turn == WHITE:
            single_moves = pawns << 8 & ~self.occupied
            double_moves = single_moves << 8 & ~self.occupied & (BB_RANK_3 | BB_RANK_4)
        else:
            single_moves = pawns >> 8 & ~self.occupied
            double_moves = single_moves >> 8 & ~self.occupied & (BB_RANK_6 | BB_RANK_5)

        single_moves &= to_mask
        double_moves &= to_mask

        # Generate single pawn moves.
        for to_square in scan_reversed(single_moves):
            from_square = to_square + (8 if self.turn == BLACK else -8)

            if square_rank(to_square) in [0, 7]:
                yield Move(from_square, to_square, QUEEN)
                yield Move(from_square, to_square, ROOK)
                yield Move(from_square, to_square, BISHOP)
                yield Move(from_square, to_square, KNIGHT)
            else:
                yield Move(from_square, to_square)

        # Generate double pawn moves.
        for to_square in scan_reversed(double_moves):
            from_square = to_square + (16 if self.turn == BLACK else -16)
            yield Move(from_square, to_square)

    # Override parent push method.
    # Fixes a given valid pseudo-legal move
    # Replace illegal moves with null move
    # Replace moves that go through a piece with the corresponding capture
    # Returns reward for move
    def push(self, move):
        backrank = BB_RANK_1 if self.turn == WHITE else BB_RANK_8
        is_legal = True
        dest_sq = move.to_square
        clear_squares = 0
        # Observations that are made as a result of this move are encoded by this var
        observation = np.zeros((8, 8, 13), dtype='float32')

        # TODO: Add observation from failed castling
        # TODO: Add observation from sliding pawn move

        # Castling is legal if the squares between the king and rook are empty
        if self.is_kingside_castling(move):
            cols = BB_FILE_F | BB_FILE_G
            squares = cols & backrank
            is_legal = (squares & self.occupied) == 0
        elif self.is_queenside_castling(move):
            cols = BB_FILE_B | BB_FILE_C | BB_FILE_D
            squares = cols & backrank
            is_legal = (squares & self.occupied) == 0
        elif BB_SQUARES[move.from_square] & self.pawns:
            # Pawn moves that are straight need to go to empty squares
            if move.from_square % 8 == move.to_square % 8:
                is_legal = (BB_SQUARES[move.to_square] & self.occupied) == 0
            # Pawn moves that are diagonal need to be captures (accounts for ep)
            else:
                is_legal = self.is_capture(move)
        elif (BB_SQUARES[move.from_square] &
              (self.bishops | self.rooks | self.queens)):
            # Returns the new destination and a mask for all squares that were revealed to be empty
            dest_sq, clear_squares = self.adjust_sliding_move(move.from_square, move.to_square)


        true_move = Move(move.from_square, dest_sq, promotion=move.promotion)
        if not is_legal:
            true_move = Move.null()
            # The square the pawn is moving to is empty
            if BB_SQUARES[move.from_square] & self.pawns:
                observation[square_rank(move.to_square)][square_file(move.to_square)][12] = 1

        capture = None

        if true_move != Move.null():

            # Updates visible board. Moves pieces from from_square to to_square
            # Special case for promotion and castling needed
            visible = self.visible_state[self.turn]
            if true_move.promotion is None:
                visible.set_piece_at(true_move.to_square, visible.piece_at(true_move.from_square))
            else:
                visible._set_piece_at(true_move.to_square, true_move.promotion, self.turn, True)
            visible.remove_piece_at(true_move.from_square)
            if self.is_castling(move):
                if self.is_kingside_castling(move):
                    rook_from = H1 if self.turn else H8
                    rook_to = F1 if self.turn else F8
                else:
                    rook_from = A1 if self.turn else A8
                    rook_to = D1 if self.turn else D8
                visible.set_piece_at(rook_to, visible.piece_at(rook_from))
                visible.remove_piece_at(rook_from)

            capture = self.is_capture(true_move)
            # Update our visible board to be empty on any squares we moved through
            if clear_squares:
                observation = self.get_current_state(self.turn, clear_squares)

            from_rank = square_rank(true_move.from_square)
            from_file = square_file(true_move.from_square)
            to_rank = square_rank(true_move.to_square)
            to_file = square_file(true_move.to_square)
            if capture:
                # If you capture something, update your opponent's visibility
                self.visible_state[not self.turn].remove_piece_at(true_move.to_square)

                for i in range(6, 12):
                    # We observe a -1 for all their pieces on that square
                    observation[to_rank][to_file][i] = -1
                    # Our opponent observes a 1 for all our pieces on that square
                    self.observation[not self.turn][to_rank][to_file][i] = 1

        self.observation[self.turn] = observation

        super().push(true_move)

        # Return reward for move
        if true_move == Move.null():
            return -5
        elif capture:
            return 5
        else:
            return 0

    # It's possible to make a move that would slide past an opponent piece
    # This stops the sliding at the first capture
    def adjust_sliding_move(self, from_square, to_square):

        # Either moving along rank, file, left-right diagonal, right-left diagonal
        delta = None
        if abs(from_square - to_square) % 7 == 0:
            delta = 7
        elif abs(from_square - to_square) % 8 == 0:
            delta = 8
        elif abs(from_square - to_square) % 9 == 0:
            delta = 9
        elif from_square // 8 == to_square // 8:
            delta = 1

        # Invalid sliding attack
        if delta is None:
            return None

        # If moving backwards
        if to_square < from_square:
            delta *= -1

        sq = from_square
        dest_square = from_square
        clear_squares = 0

        # Keep sliding until we hit our target or a sqaure that's occupied
        while True:
            sq += delta
            if not (0 <= sq < 64) or square_distance(sq, sq - delta) > 2:
                break

            dest_square = sq

            if (self.occupied & BB_SQUARES[sq]) or dest_square == to_square:
                break

            # Keep track of all the empty squares we passed through bc that contains info
            clear_squares |= BB_SQUARES[sq]

        return dest_square, clear_squares

    # Senses on the given square (0-63) and returns an observation on those squares
    # Any squares on the edge are moved inwards to make a 3x3 sqaure
    def sense(self, square):
        if square_file(square) == 0:
            square += 1
        elif square_file(square) == 7:
            square -= 1

        if square_rank(square) == 0:
            square += 8
        elif square_rank(square) == 7:
            square -= 8

        mask = 0
        for i in [-9, -8, -7, -1, 0, 1, 7, 8, 9]:
            mask |= BB_SQUARES[square + i]

        self.sense_history[self.turn].append(square)

        return self.sense_mask(mask)

    def get_previous_sense(self):
        return self.sense_history[self.turn][-1] if len(self.sense_history[self.turn]) > 0 else 0


    # Updates the visible board over the given mask and returns an observation over the mask
    def sense_mask(self, mask):
        def update_visible(visible, true, sense_mask):
            visible &= ~sense_mask
            return (visible | (sense_mask & true))

        board = self.visible_state[self.turn]

        board.pawns = update_visible(board.pawns, self.pawns, mask)
        board.knights = update_visible(board.knights, self.knights, mask)
        board.bishops = update_visible(board.bishops, self.bishops, mask)
        board.rooks = update_visible(board.rooks, self.rooks, mask)
        board.queens = update_visible(board.queens, self.queens, mask)
        board.kings = update_visible(board.kings, self.kings, mask)
        board.occupied = update_visible(board.occupied, self.occupied, mask)
        board.occupied_co[not self.turn] = update_visible(board.occupied_co[not self.turn],
                                                          self.occupied_co[not self.turn], mask)

        return self.get_current_state(self.turn, mask)

    # Returns the observation made during the previous move overlayed on top of
    # an observation of all your pieces
    def get_pre_turn_observation(self):
        return np.add(self.observation[self.turn], self.my_pieces_observation(self.turn))

    # Returns an observation of your own pieces
    def my_pieces_observation(self, color):
        return self.get_current_state(color, mask=self.occupied_co[color])

    # Returns an observation of the current board
    def get_current_state(self, color, mask=BB_ALL):
        ranks = range(8) if color else range(7, -1, -1)
        squares = []
        for i in ranks:
            row = []
            for j in range(8):
                square = i * 8 + j
                piece = self.piece_at(square)
                num = 12
                if piece is not None:
                    if piece.color == color:
                        num = piece.piece_type - 1
                    else:
                        num = piece.piece_type - 1 + 6
                if BB_SQUARES[square] & mask:
                    row.append(to_categorical(num, num_classes=13))
                else:
                    row.append(np.zeros((13), dtype='float32'))
            squares.append(row)
        return np.asarray(squares)

    # Decodes a number to a move
    def get_move_from_model(self, move):
        starting_square = move // (7 * 8 + 8 + 9)
        _type = move % (7 * 8 + 8 + 9)
        promotion = None

        # Sliding move
        if _type < 56:
            direction = _type // 7
            distance = (_type % 7) + 1
            directions = [-1, 1, -7, 7, -8, 8, -9, 9]
            direction = directions[direction]
            ending_square = starting_square + distance * direction
            # Direction should be 1 if and only if the squares are on the same rank
            if ((square_rank(starting_square) == square_rank(ending_square))
                != (abs(direction) == 1)):
                return None
            # Direction should be 8 if and only if the squares are on the same file
            if ((square_file(starting_square) == square_file(ending_square))
                != abs(direction) == 8):
                return None

            # If promoting, promote as queen
            if ((self.pawns & self.occupied_co[self.turn] & BB_SQUARES[starting_square]) and
                ((self.turn and starting_square // 8 == 6) or
                 (not self.turn and starting_square // 8 == 1))):
                promotion = 5
        # Knight move
        elif _type >= 56 and _type < 64:
            knight_moves = [17, 15, 10, 6, -17, -15, -10, -6]
            ending_square = starting_square + knight_moves[_type - 56]
            if (square_distance(starting_square, ending_square) != 2):
                return None
        # Underpromotion
        else:
            _type = _type - 64
            direction = _type // 3
            promotion = (_type % 3) + 2
            directions = [7, 8, 9]
            direction = directions[direction]
            if not self.turn:
                direction *= -1
            ending_square = starting_square + direction

        if ending_square < 0 or ending_square > 63:
            return None
        else:
            return Move(starting_square, ending_square, promotion=promotion)

    # Encodes a move as a number
    def get_model_num_from_move(self, move):
        directions = [-1, 1, -7, 7, -8, 8, -9, 9]
        knight_moves = [17, 15, 10, 6, -17, -15, -10, -6]
        diff = move.to_square - move.from_square

        # Underpromotion
        if move.promotion != 5 and move.promotion != None:
            directions = [7, 8, 9]
            for d in directions:
                if abs(diff) == d:
                    direction = d
            _type = 3 * directions.index(direction) + move.promotion - 2 + 64
        # Knight move
        elif diff in knight_moves and square_distance(move.to_square, move.from_square) == 2:
            _type = 56 + knight_moves.index(diff)
        else:
            direction = 0
            for d in reversed(directions):
                if np.sign(diff) == np.sign(d) and diff % d == 0:
                    direction = d
                    break
            if square_rank(move.to_square) == square_rank(move.from_square):
                direction = np.sign(diff) * 1
            elif square_file(move.to_square) == square_file(move.from_square):
                direction = np.sign(diff) * 8

            distance = abs(diff / direction)
            _type = 7 * directions.index(direction) + (distance - 1)

        move_num = move.from_square * (7 * 8 + 8 + 9) + _type
        return int(move_num)


    def is_game_over(self):
        return self.result() is not None

    # Redefine a win as when the opponent king is missing
    def result(self):
        if (self.kings & self.occupied_co[WHITE]) == 0:
            return -1
        elif (self.kings & self.occupied_co[BLACK]) == 0:
            return 1
        elif self.can_claim_fifty_moves():
            return 0

        return None
