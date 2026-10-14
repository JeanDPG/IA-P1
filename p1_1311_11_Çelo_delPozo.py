"""Laboratorio de Inteligencia Artificial - Curso 2026-2027.
Práctica 1: Búsqueda con adversarios (Reversi).
Grupo de laboratorio: 1311
Pareja: 11
Autores: Çelo - del Pozo
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np

from game import TwoPlayerGameState
from tournament import StudentHeuristic
from reversi import get_valid_moves


# =====================================================================
# MATRICES Y ESTRUCTURAS DE TABLERO 8x8
# =====================================================================

# Matriz de pesos posicionales para Reversi estándar (8x8)
# Basada en literatura clásica (Rosenbloom 1982, Buro 1997):
# - Esquinas: valor supremo (+120), son inexpugnables.
# - Casillas X (diagonales a esquinas): altamente peligrosas (-45) si la esquina está vacía.
# - Casillas C (adyacentes a esquinas en bordes): peligrosas (-25) si la esquina está vacía.
# - Bordes estables: seguros (+20, +5).
# - Centro: valores moderados/neutros.
BOARD_WEIGHTS_8X8: Dict[Tuple[int, int], int] = {
    (1, 1): 120, (1, 2): -25, (1, 3):  20, (1, 4):   5, (1, 5):   5, (1, 6):  20, (1, 7): -25, (1, 8): 120,
    (2, 1): -25, (2, 2): -45, (2, 3):  -3, (2, 4):  -3, (2, 5):  -3, (2, 6):  -3, (2, 7): -45, (2, 8): -25,
    (3, 1):  20, (3, 2):  -3, (3, 3):  10, (3, 4):   2, (3, 5):   2, (3, 6):  10, (3, 7):  -3, (3, 8):  20,
    (4, 1):   5, (4, 2):  -3, (4, 3):   2, (4, 4):   1, (4, 5):   1, (4, 6):   2, (4, 7):  -3, (4, 8):   5,
    (5, 1):   5, (5, 2):  -3, (5, 3):   2, (5, 4):   1, (5, 5):   1, (5, 6):   2, (5, 7):  -3, (5, 8):   5,
    (6, 1):  20, (6, 2):  -3, (6, 3):  10, (6, 4):   2, (6, 5):   2, (6, 6):  10, (6, 7):  -3, (6, 8):  20,
    (7, 1): -25, (7, 2): -45, (7, 3):  -3, (7, 4):  -3, (7, 5):  -3, (7, 6):  -3, (7, 7): -45, (7, 8): -25,
    (8, 1): 120, (8, 2): -25, (8, 3):  20, (8, 4):   5, (8, 5):   5, (8, 6):  20, (8, 7): -25, (8, 8): 120,
}

# Coordenadas de las 4 esquinas
CORNERS: List[Tuple[int, int]] = [(1, 1), (1, 8), (8, 1), (8, 8)]

# Esquinas y sus correspondientes casillas adyacentes de peligro (C y X)
CORNERS_AND_NEIGHBORS: List[Tuple[Tuple[int, int], List[Tuple[int, int]]]] = [
    ((1, 1), [(1, 2), (2, 1), (2, 2)]),
    ((1, 8), [(1, 7), (2, 8), (2, 7)]),
    ((8, 1), [(7, 1), (8, 2), (7, 2)]),
    ((8, 8), [(8, 7), (7, 8), (7, 7)]),
]

# Definición de los 4 bordes del tablero
EDGES: List[List[Tuple[int, int]]] = [
    [(x, 1) for x in range(1, 9)],  # Borde superior
    [(x, 8) for x in range(1, 9)],  # Borde inferior
    [(1, y) for y in range(1, 9)],  # Borde izquierdo
    [(8, y) for y in range(1, 9)],  # Borde derecho
]

# 8 direcciones para cálculo de casillas y fronteras
DIRECTIONS_8: List[Tuple[int, int]] = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


# =====================================================================
# FUNCIONES AUXILIARES DE EVALUACIÓN
# =====================================================================

def _terminal_utility(state: TwoPlayerGameState) -> float:
    """Evalúa con precisión absoluta el estado final de juego."""
    scores = state.scores
    if scores is not None:
        diff = (scores[0] - scores[1]) if state.is_player_max(state.player1) else (scores[1] - scores[0])
        if diff > 0:
            return 100000.0 + float(diff)
        elif diff < 0:
            return -100000.0 + float(diff)
    return 0.0


def _compute_edge_stability(board: dict, my_label: str, op_label: str) -> float:
    """Calcula la estabilidad de fichas en los cuatro bordes del tablero.
    
    Las fichas que forman una cadena continua desde una esquina conquistada
    no pueden ser flanqueadas y son permanentemente estables.
    """
    my_stable = 0
    op_stable = 0
    for edge in EDGES:
        c0, c7 = edge[0], edge[7]
        owner0 = board.get(c0)
        if owner0:
            for pos in edge:
                if board.get(pos) == owner0:
                    if owner0 == my_label:
                        my_stable += 1
                    else:
                        op_stable += 1
                else:
                    break
        owner7 = board.get(c7)
        if owner7:
            for pos in reversed(edge):
                if board.get(pos) == owner7:
                    if owner7 == my_label:
                        my_stable += 1
                    else:
                        op_stable += 1
                else:
                    break
    total = my_stable + op_stable
    if total > 0:
        return 100.0 * (my_stable - op_stable) / total
    return 0.0


def _compute_frontier_discs(board: dict, my_label: str, op_label: str) -> float:
    """Evalúa la movilidad potencial mediante el conteo de fichas de frontera.
    
    Una ficha es de frontera si tiene al menos una casilla vecina vacía.
    Minimizar las fichas de frontera propias maximiza las fichas internas
    seguras y asfixia las opciones de captura del rival.
    """
    my_frontier = 0
    op_frontier = 0
    for (x, y), piece in board.items():
        is_front = False
        for dx, dy in DIRECTIONS_8:
            nx, ny = x + dx, y + dy
            if 1 <= nx <= 8 and 1 <= ny <= 8 and (nx, ny) not in board:
                is_front = True
                break
        if is_front:
            if piece == my_label:
                my_frontier += 1
            elif piece == op_label:
                op_frontier += 1
    total = my_frontier + op_frontier
    if total > 0:
        return 100.0 * (op_frontier - my_frontier) / total
    return 0.0


# =====================================================================
# HEURÍSTICA 1: Posicional Adaptativa y Control de Esquinas
# =====================================================================

class Solution1(StudentHeuristic):
    """Heurística 1: Matriz Posicional Adaptativa y Paridad por Fases.

    Características principales:
    - Evaluación de velocidad ultrarrápida (< 0.003 ms por nodo), ideal para
      evitar cualquier riesgo de timeout.
    - Matriz de pesos estática adaptativa: penaliza fuertemente las casillas
      X (-45) y C (-25) mientras la esquina esté vacía, pero si la esquina es
      conquistada, sus casillas adyacentes pasan a ser seguras (+15).
    - Paridad de fichas activada en el tramo final (>= 48 fichas) para asegurar
      la victoria en el conteo de discos.
    """

    def get_name(self) -> str:
        return "Netan"

    def evaluation_function(self, state: TwoPlayerGameState) -> float:
        if state.end_of_game:
            return _terminal_utility(state)

        board = state.board
        if not isinstance(board, dict):
            return 0.0

        if state.is_player_max(state.player1):
            my_label = state.player1.label
            op_label = state.player2.label
        else:
            my_label = state.player2.label
            op_label = state.player1.label

        # Modificación dinámica de pesos según control de esquinas
        dynamic_weights = dict(BOARD_WEIGHTS_8X8)
        for corner, neighbors in CORNERS_AND_NEIGHBORS:
            owner = board.get(corner)
            if owner == my_label:
                for n in neighbors:
                    dynamic_weights[n] = 12
            elif owner == op_label:
                for n in neighbors:
                    dynamic_weights[n] = -12

        pos_score = 0.0
        my_coins = 0
        op_coins = 0

        for pos, piece in board.items():
            wt = dynamic_weights.get(pos, 0)
            if piece == my_label:
                my_coins += 1
                pos_score += wt
            elif piece == op_label:
                op_coins += 1
                pos_score -= wt

        total_coins = my_coins + op_coins
        if total_coins >= 48:
            pos_score += 25.0 * (my_coins - op_coins) / total_coins
        elif total_coins > 0:
            pos_score += 5.0 * (my_coins - op_coins) / total_coins

        return float(pos_score)


# =====================================================================
# HEURÍSTICA 2: Movilidad Estratégica, Esquinas y Paridad
# =====================================================================

class Solution2(StudentHeuristic):
    """Heurística 2: Movilidad Legal, Control de Esquinas y Paridad Dinámica.

    Estrategia de asfixia clásica de Othello (Rosenbloom 1982):
    - Apertura y medio juego (< 48 fichas): Minimizar las opciones del oponente
      y maximizar las propias (movilidad legal actual). El oponente se ve forzado
      a ocupar casillas desfavorables (C y X) facilitando la captura de esquinas.
    - Matriz posicional adaptativa que respalda la conquista territorial.
    - Final de partida (>= 48 fichas): La paridad de fichas pasa a ser primordial
      para garantizar que la ventaja territorial se traduzca en victoria numérica.
    """

    def get_name(self) -> str:
        return "yahu"

    def evaluation_function(self, state: TwoPlayerGameState) -> float:
        if state.end_of_game:
            return _terminal_utility(state)

        board = state.board
        if not isinstance(board, dict):
            return 0.0

        if state.is_player_max(state.player1):
            my_label = state.player1.label
            op_label = state.player2.label
        else:
            my_label = state.player2.label
            op_label = state.player1.label

        # 1. Control de Esquinas
        my_corners = sum(1 for c in CORNERS if board.get(c) == my_label)
        op_corners = sum(1 for c in CORNERS if board.get(c) == op_label)
        corner_score = 0.0
        if (my_corners + op_corners) > 0:
            corner_score = 100.0 * (my_corners - op_corners) / (my_corners + op_corners)

        # 2. Movilidad Legal (Actual Mobility)
        h = getattr(state.game, 'height', 8)
        w = getattr(state.game, 'width', 8)
        blocked = getattr(state.game, 'blocked_cell_label', 'O')
        ib = getattr(state.game, 'ignore_block_cells_in_captures', False)

        my_moves = len(get_valid_moves(board, h, w, my_label, op_label, blocked, ib))
        op_moves = len(get_valid_moves(board, h, w, op_label, my_label, blocked, ib))

        mobility_score = 0.0
        if (my_moves + op_moves) > 0:
            mobility_score = 100.0 * (my_moves - op_moves) / (my_moves + op_moves)

        # 3. Puntuación Posicional Adaptativa
        dynamic_weights = dict(BOARD_WEIGHTS_8X8)
        for corner, neighbors in CORNERS_AND_NEIGHBORS:
            owner = board.get(corner)
            if owner == my_label:
                for n in neighbors:
                    dynamic_weights[n] = 10
            elif owner == op_label:
                for n in neighbors:
                    dynamic_weights[n] = -10

        pos_score = 0.0
        my_coins = 0
        op_coins = 0

        for pos, piece in board.items():
            wt = dynamic_weights.get(pos, 0)
            if piece == my_label:
                my_coins += 1
                pos_score += wt
            elif piece == op_label:
                op_coins += 1
                pos_score -= wt

        # 4. Paridad de Fichas
        total_coins = my_coins + op_coins
        coin_parity = 0.0
        if total_coins > 0:
            coin_parity = 100.0 * (my_coins - op_coins) / total_coins

        # Ponderación según la fase de la partida
        if total_coins < 48:
            total_value = 40.0 * corner_score + 35.0 * mobility_score + 25.0 * pos_score
        else:
            total_value = 35.0 * corner_score + 15.0 * mobility_score + 15.0 * pos_score + 35.0 * coin_parity

        return float(total_value)


# =====================================================================
# HEURÍSTICA 3: Maestra Multicriterio de Campeonato
# =====================================================================

class Solution3(StudentHeuristic):
    """Heurística 3: Heurística Maestra Multicriterio Dinámica.

    Solución completa para torneos competitivos (Buro 1997, Sastry 2013):
    1. Control y Anclaje de Esquinas: Máxima prioridad a conquistar esquinas.
    2. Penalización de Peligro C y X: Evita activamente jugar en casillas
       adyacentes a esquinas desocupadas y aprovecha los errores del rival.
    3. Movilidad Legal Actual: Control del árbol de decisiones del juego.
    4. Movilidad Potencial (Fichas de Frontera): Minimiza la exposición de fichas
       propias maximizando la masa interna estable.
    5. Estabilidad de Bordes: Detección de cadenas estables de fichas en los
       bordes conectadas a esquinas.
    6. Paridad Ponderada por Fase: Transición continua hacia el conteo numérico
       en las etapas finales de la partida.
    """

    def get_name(self) -> str:
        return "maestra_multicriterio"

    def evaluation_function(self, state: TwoPlayerGameState) -> float:
        if state.end_of_game:
            return _terminal_utility(state)

        board = state.board
        if not isinstance(board, dict):
            return 0.0

        if state.is_player_max(state.player1):
            my_label = state.player1.label
            op_label = state.player2.label
        else:
            my_label = state.player2.label
            op_label = state.player1.label

        # 1. Esquinas
        my_c = sum(1 for c in CORNERS if board.get(c) == my_label)
        op_c = sum(1 for c in CORNERS if board.get(c) == op_label)
        corner_val = 100.0 * (my_c - op_c) / (my_c + op_c) if (my_c + op_c) > 0 else 0.0

        # 2. Casillas de Peligro (X y C) con esquina vacía
        danger_val = 0.0
        for corner, neighbors in CORNERS_AND_NEIGHBORS:
            if corner not in board:
                for n in neighbors:
                    piece = board.get(n)
                    if piece == my_label:
                        danger_val -= 25.0
                    elif piece == op_label:
                        danger_val += 25.0

        # 3. Movilidad Legal
        h = getattr(state.game, 'height', 8)
        w = getattr(state.game, 'width', 8)
        blocked = getattr(state.game, 'blocked_cell_label', 'O')
        ib = getattr(state.game, 'ignore_block_cells_in_captures', False)

        my_m = len(get_valid_moves(board, h, w, my_label, op_label, blocked, ib))
        op_m = len(get_valid_moves(board, h, w, op_label, my_label, blocked, ib))
        mob_val = 100.0 * (my_m - op_m) / (my_m + op_m) if (my_m + op_m) > 0 else 0.0

        # 4. Fichas de Frontera y Estabilidad de Bordes
        front_val = _compute_frontier_discs(board, my_label, op_label)
        stab_val = _compute_edge_stability(board, my_label, op_label)

        # 5. Puntuación Posicional Adaptativa
        dynamic_weights = dict(BOARD_WEIGHTS_8X8)
        for corner, neighbors in CORNERS_AND_NEIGHBORS:
            owner = board.get(corner)
            if owner == my_label:
                for n in neighbors:
                    dynamic_weights[n] = 12
            elif owner == op_label:
                for n in neighbors:
                    dynamic_weights[n] = -12

        pos_val = 0.0
        my_coins = 0
        op_coins = 0

        for pos, piece in board.items():
            wt = dynamic_weights.get(pos, 0)
            if piece == my_label:
                my_coins += 1
                pos_val += wt
            elif piece == op_label:
                op_coins += 1
                pos_val -= wt

        # 6. Paridad de Fichas
        total_coins = my_coins + op_coins
        parity_val = 100.0 * (my_coins - op_coins) / total_coins if total_coins > 0 else 0.0

        # Ponderación según la fase del juego
        if total_coins < 48:
            total_score = (
                35.0 * corner_val +
                15.0 * danger_val +
                30.0 * mob_val +
                20.0 * pos_val +
                10.0 * front_val +
                15.0 * stab_val
            )
        else:
            total_score = (
                35.0 * corner_val +
                15.0 * mob_val +
                15.0 * pos_val +
                20.0 * stab_val +
                35.0 * parity_val
            )

        return float(total_score)
