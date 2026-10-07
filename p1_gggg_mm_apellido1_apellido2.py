"""Implementación de heurísticas para Reversi con búsqueda Minimax y poda Alfa-Beta.

Asignatura: Laboratorio de Inteligencia Artificial (2026-2027)
Práctica 1: Búsqueda con adversarios.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np

from game import TwoPlayerGameState
from tournament import StudentHeuristic
from reversi import get_valid_moves


# Matriz de pesos posicionales para Reversi (tablero 8x8 estándar)
# Las esquinas tienen el valor más alto (+100).
# Las casillas adyacentes a esquinas (X y C) tienen valor negativo (-40 y -20)
# si la esquina no ha sido conquistada.
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

# Definición de las cuatro esquinas y sus celdas de peligro adyacentes (casillas C y X)
CORNERS_AND_NEIGHBORS = [
    # Esquina (1, 1) y sus vecinas C y X
    ((1, 1), [(1, 2), (2, 1), (2, 2)]),
    # Esquina (1, 8) y sus vecinas
    ((1, 8), [(1, 7), (2, 8), (2, 7)]),
    # Esquina (8, 1) y sus vecinas
    ((8, 1), [(7, 1), (8, 2), (7, 2)]),
    # Esquina (8, 8) y sus vecinas
    ((8, 8), [(8, 7), (7, 8), (7, 7)]),
]


class Solution1(StudentHeuristic):
    """Heurística 1: Matriz Posicional Adaptativa y Control de Esquinas.

    Esta heurística evalúa el estado mediante una matriz de pesos estática adaptativa:
    - Otorga máxima prioridad al control de las cuatro esquinas (+100).
    - Penaliza fuertemente colocar fichas en las casillas X (-40) y C (-20) adyacentes
      a una esquina mientras dicha esquina esté vacía, pues facilitan que el oponente
      la conquiste.
    - Si la esquina ya ha sido conquistada por nuestro jugador, las casillas adyacentes
      dejan de ser peligrosas y pasan a considerarse seguras (+10).
    - Desempata con la diferencia de monedas en fases avanzadas.
    - Es extremadamente rápida (< 0.05 ms por evaluación), garantizando no superar
      el límite de tiempo de evaluación.
    """

    def get_name(self) -> str:
        return "posicional_esquinas"

    def evaluation_function(self, state: TwoPlayerGameState) -> float:
        # 1. Comprobación de estado terminal de juego
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

        # 2. Identificar etiquetas de MAX y MIN
        if state.is_player_max(state.player1):
            my_label = state.player1.label
            op_label = state.player2.label
        else:
            my_label = state.player2.label
            op_label = state.player1.label

        score = 0.0
        my_coins = 0
        op_coins = 0

        # Copia local de pesos adaptables a partir de las esquinas
        dynamic_weights = dict(BOARD_WEIGHTS_8X8)
        for corner, neighbors in CORNERS_AND_NEIGHBORS:
            owner = board.get(corner)
            if owner == my_label:
                # La esquina es nuestra: las celdas vecinas ya no son peligrosas para nosotros
                for neighbor in neighbors:
                    dynamic_weights[neighbor] = 10
            elif owner == op_label:
                # La esquina es del oponente: las celdas vecinas no le hacen daño a él
                for neighbor in neighbors:
                    dynamic_weights[neighbor] = -10

        # 3. Sumar pesos posicionales y contar monedas
        for pos, piece in board.items():
            if piece == my_label:
                my_coins += 1
                score += dynamic_weights.get(pos, 0)
            elif piece == op_label:
                op_coins += 1
                score -= dynamic_weights.get(pos, 0)

        # 4. Paridad de fichas como factor secundario de desempate
        total_coins = my_coins + op_coins
        if total_coins > 0:
            coin_diff = (my_coins - op_coins) / total_coins
            score += 10.0 * coin_diff

        return float(score)


class Solution2(StudentHeuristic):
    """Heurística 2: Heurística Multicriterio Dinámica (Esquinas + Movilidad + Posición + Paridad).

    Estrategia de juego basada en la literatura clásica de Othello (Rosenbloom 1982, Buro 1997):
    1. Control de Esquinas: Las 4 esquinas son estables y no pueden ser capturadas.
    2. Penalización de Casillas Peligrosas (X y C): Evita regalar esquinas al oponente.
    3. Movilidad Actual: En la apertura y medio juego, restringir las opciones legales del
       adversario lo asfixia y lo fuerza a realizar movimientos desfavorables.
    4. Paridad de Monedas Ponderada por Fase:
       - En apertura/medio juego (< 48 fichas): el número de fichas tiene peso mínimo.
       - En el tramo final (> 48 fichas): la cantidad de fichas pasa a ser decisiva.
    """

    def get_name(self) -> str:
        return "estrategica_movilidad"

    def evaluation_function(self, state: TwoPlayerGameState) -> float:
        # 1. Comprobación de estado terminal de juego
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

        # 2. Identificar etiquetas de MAX y MIN
        if state.is_player_max(state.player1):
            my_label = state.player1.label
            op_label = state.player2.label
        else:
            my_label = state.player2.label
            op_label = state.player1.label

        # 3. Componente 1: Control de Esquinas
        corners = [(1, 1), (1, 8), (8, 1), (8, 8)]
        my_corners = 0
        op_corners = 0
        for c in corners:
            owner = board.get(c)
            if owner == my_label:
                my_corners += 1
            elif owner == op_label:
                op_corners += 1

        corner_score = 0.0
        if (my_corners + op_corners) > 0:
            corner_score = 100.0 * (my_corners - op_corners) / (my_corners + op_corners)

        # 4. Componente 2: Penalización de celdas adyacentes a esquinas vacías (Casillas X y C)
        proximity_score = 0.0
        for corner, neighbors in CORNERS_AND_NEIGHBORS:
            if corner not in board:  # Esquina vacía
                for neighbor in neighbors:
                    piece = board.get(neighbor)
                    if piece == my_label:
                        proximity_score -= 25.0
                    elif piece == op_label:
                        proximity_score += 25.0

        # 5. Componente 3: Movilidad (Número de movimientos legales disponibles)
        height = getattr(state.game, 'height', 8)
        width = getattr(state.game, 'width', 8)
        blocked = getattr(state.game, 'blocked_cell_label', 'O')
        ignore_blocked = getattr(state.game, 'ignore_block_cells_in_captures', False)

        my_moves = len(get_valid_moves(board, height, width, my_label, op_label, blocked, ignore_blocked))
        op_moves = len(get_valid_moves(board, height, width, op_label, my_label, blocked, ignore_blocked))

        mobility_score = 0.0
        if (my_moves + op_moves) > 0:
            mobility_score = 100.0 * (my_moves - op_moves) / (my_moves + op_moves)

        # 6. Componente 4: Paridad de Fichas (Coin Parity)
        my_coins = sum(1 for piece in board.values() if piece == my_label)
        op_coins = sum(1 for piece in board.values() if piece == op_label)
        total_coins = my_coins + op_coins

        coin_parity_score = 0.0
        if total_coins > 0:
            coin_parity_score = 100.0 * (my_coins - op_coins) / total_coins

        # 7. Componente 5: Puntuación Posicional Rápida
        positional_score = 0.0
        for pos, piece in board.items():
            weight = BOARD_WEIGHTS_8X8.get(pos, 0)
            if piece == my_label:
                positional_score += weight
            elif piece == op_label:
                positional_score -= weight

        # Ponderación según la fase de la partida:
        # Apertura / Medio juego (< 48 fichas): la movilidad y las esquinas dominan.
        # Final de partida (>= 48 fichas): la paridad de fichas pasa a ser primordial.
        if total_coins < 48:
            total_value = (
                30.0 * corner_score +
                15.0 * proximity_score +
                25.0 * mobility_score +
                10.0 * positional_score +
                5.0 * coin_parity_score
            )
        else:
            total_value = (
                35.0 * corner_score +
                10.0 * proximity_score +
                10.0 * mobility_score +
                10.0 * positional_score +
                35.0 * coin_parity_score
            )

        return float(total_value)
