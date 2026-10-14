from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np

from game import TwoPlayerGameState
from tournament import StudentHeuristic
from reversi import get_valid_moves


BOARD_WEIGHTS_8X8: Dict[Tuple[int, int], int] = {
    (1, 1): 100, (1, 2): -20, (1, 3):  10, (1, 4):   5, (1, 5):   5, (1, 6):  10, (1, 7): -20, (1, 8): 100,
    (2, 1): -20, (2, 2): -40, (2, 3):  -2, (2, 4):  -2, (2, 5):  -2, (2, 6):  -2, (2, 7): -40, (2, 8): -20,
    (3, 1):  10, (3, 2):  -2, (3, 3):  -1, (3, 4):  -1, (3, 5):  -1, (3, 6):  -1, (3, 7):  -2, (3, 8):  10,
    (4, 1):   5, (4, 2):  -2, (4, 3):  -1, (4, 4):   0, (4, 5):   0, (4, 6):  -1, (4, 7):  -2, (4, 8):   5,
    (5, 1):   5, (5, 2):  -2, (5, 3):  -1, (5, 4):   0, (5, 5):   0, (5, 6):  -1, (5, 7):  -2, (5, 8):   5,
    (6, 1):  10, (6, 2):  -2, (6, 3):  -1, (6, 4):  -1, (6, 5):  -1, (6, 6):  -1, (6, 7):  -2, (6, 8):  10,
    (7, 1): -20, (7, 2): -40, (7, 3):  -2, (7, 4):  -2, (7, 5):  -2, (7, 6):  -2, (7, 7): -40, (7, 8): -20,
    (8, 1): 100, (8, 2): -20, (8, 3):  10, (8, 4):   5, (8, 5):   5, (8, 6):  10, (8, 7): -20, (8, 8): 100,
}

# esquinas y sus celdas adyacentes peligrosas
CORNERS_AND_NEIGHBORS = [
    ((1, 1), [(1, 2), (2, 1), (2, 2)]),
    ((1, 8), [(1, 7), (2, 8), (2, 7)]),
    ((8, 1), [(7, 1), (8, 2), (7, 2)]),
    ((8, 8), [(8, 7), (7, 8), (7, 7)]),
]


class Solution1(StudentHeuristic):

    def get_name(self) -> str:
        return "posicional_esquinas"

    def evaluation_function(self, state: TwoPlayerGameState) -> float:
        if state.end_of_game:
            scores = state.scores
            if scores is not None:
                diff = (scores[0] - scores[1]) if state.is_player_max(state.player1) else (scores[1] - scores[0])
                if diff > 0:
                    return 100000.0 + float(diff)
                elif diff < 0:
                    return -100000.0 + float(diff)
            return 0.0

        board = state.board
        if not isinstance(board, dict):
            return 0.0

        if state.is_player_max(state.player1):
            my_label = state.player1.label
            op_label = state.player2.label
        else:
            my_label = state.player2.label
            op_label = state.player1.label

        score = 0.0
        my_coins = 0
        op_coins = 0

        # ajustar pesos segun quien controla cada esquina
        dynamic_weights = dict(BOARD_WEIGHTS_8X8)
        for corner, neighbors in CORNERS_AND_NEIGHBORS:
            owner = board.get(corner)
            if owner == my_label:
                for n in neighbors:
                    dynamic_weights[n] = 10
            elif owner == op_label:
                for n in neighbors:
                    dynamic_weights[n] = -10

        for pos, piece in board.items():
            if piece == my_label:
                my_coins += 1
                score += dynamic_weights.get(pos, 0)
            elif piece == op_label:
                op_coins += 1
                score -= dynamic_weights.get(pos, 0)

        total_coins = my_coins + op_coins
        if total_coins > 0:
            score += 10.0 * (my_coins - op_coins) / total_coins

        return float(score)


class Solution2(StudentHeuristic):

    def get_name(self) -> str:
        return "estrategica_movilidad"

    def evaluation_function(self, state: TwoPlayerGameState) -> float:
        if state.end_of_game:
            scores = state.scores
            if scores is not None:
                diff = (scores[0] - scores[1]) if state.is_player_max(state.player1) else (scores[1] - scores[0])
                if diff > 0:
                    return 100000.0 + float(diff)
                elif diff < 0:
                    return -100000.0 + float(diff)
            return 0.0

        board = state.board
        if not isinstance(board, dict):
            return 0.0

        if state.is_player_max(state.player1):
            my_label = state.player1.label
            op_label = state.player2.label
        else:
            my_label = state.player2.label
            op_label = state.player1.label

        # esquinas
        corners = [(1, 1), (1, 8), (8, 1), (8, 8)]
        my_corners = sum(1 for c in corners if board.get(c) == my_label)
        op_corners = sum(1 for c in corners if board.get(c) == op_label)

        corner_score = 0.0
        if (my_corners + op_corners) > 0:
            corner_score = 100.0 * (my_corners - op_corners) / (my_corners + op_corners)

        # movilidad
        h = getattr(state.game, 'height', 8)
        w = getattr(state.game, 'width', 8)
        blocked = getattr(state.game, 'blocked_cell_label', 'O')
        ib = getattr(state.game, 'ignore_block_cells_in_captures', False)

        my_moves = len(get_valid_moves(board, h, w, my_label, op_label, blocked, ib))
        op_moves = len(get_valid_moves(board, h, w, op_label, my_label, blocked, ib))
        

        mob = 0.0
        if (my_moves + op_moves) > 0:
            mob = 100.0 * (my_moves - op_moves) / (my_moves + op_moves)

        # puntuacion posicional
        pos_score = 0.0
        for pos, piece in board.items():
            w8 = BOARD_WEIGHTS_8X8.get(pos, 0)
            if piece == my_label:
                pos_score += w8
            elif piece == op_label:
                pos_score -= w8

        return float(40.0 * corner_score + 35.0 * mob + 25.0 * pos_score)
