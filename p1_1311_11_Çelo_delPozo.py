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

# Matriz de pesos posicionales clásica para Reversi 8x8.
# Basada en literatura experta (Rosenbloom 1982, Buro 1997, Sannidhanam 2015):
#   - Esquinas (+120): absolutamente inexpugnables una vez conquistadas.
#   - Casillas X (-45): diagonalmente adyacentes a esquinas, peligrosísimas si la esquina está vacía.
#   - Casillas C (-25): lateralmente adyacentes a esquinas, peligrosas si la esquina está vacía.
#   - Bordes internos (+20, +5): seguros y de difícil captura.
#   - Sub-esquinas (+10): casillas (3,3) que fortalecen el control del centro.
#   - Centro (+1): ligera preferencia por la posición central en apertura.
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

# Coordenadas de las 4 esquinas del tablero 8x8
CORNERS: List[Tuple[int, int]] = [(1, 1), (1, 8), (8, 1), (8, 8)]

# Esquinas y sus casillas adyacentes de peligro (C y X)
CORNERS_AND_NEIGHBORS: List[Tuple[Tuple[int, int], List[Tuple[int, int]]]] = [
    ((1, 1), [(1, 2), (2, 1), (2, 2)]),
    ((1, 8), [(1, 7), (2, 8), (2, 7)]),
    ((8, 1), [(7, 1), (8, 2), (7, 2)]),
    ((8, 8), [(8, 7), (7, 8), (7, 7)]),
]

# 4 bordes del tablero 8x8
EDGES: List[List[Tuple[int, int]]] = [
    [(x, 1) for x in range(1, 9)],  # Borde superior
    [(x, 8) for x in range(1, 9)],  # Borde inferior
    [(1, y) for y in range(1, 9)],  # Borde izquierdo
    [(8, y) for y in range(1, 9)],  # Borde derecho
]

# 8 direcciones para cálculo de casillas frontera
DIRECTIONS_8: List[Tuple[int, int]] = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


# =====================================================================
# FUNCIONES AUXILIARES DE EVALUACIÓN
# =====================================================================

def _terminal_utility(state: TwoPlayerGameState) -> float:
    """Valor exacto para estados terminales: victoria/derrota con margen numérico."""
    scores = state.scores
    if scores is not None:
        diff = (scores[0] - scores[1]) if state.is_player_max(state.player1) else (scores[1] - scores[0])
        if diff > 0:
            return 100000.0 + float(diff)
        elif diff < 0:
            return -100000.0 + float(diff)
    return 0.0


def _get_labels(state: TwoPlayerGameState):
    """Extrae (mi_etiqueta, etiqueta_rival) correctamente para MAX."""
    if state.is_player_max(state.player1):
        return state.player1.label, state.player2.label
    return state.player2.label, state.player1.label


def _get_board_params(state: TwoPlayerGameState):
    """Extrae parámetros del tablero de forma robusta (incluye soporte de obstáculos)."""
    h = getattr(state.game, 'height', 8)
    w = getattr(state.game, 'width', 8)
    blocked = getattr(state.game, 'blocked_cell_label', 'O')
    ib = getattr(state.game, 'ignore_block_cells_in_captures', False)
    return h, w, blocked, ib


def _compute_edge_stability_robust(board: dict, my_label: str, op_label: str,
                                   h: int, w: int, blocked: str) -> float:
    """Calcula la estabilidad en los bordes ROBUSTA a tableros con obstáculos.

    Una ficha es estable en borde si forma una cadena continua desde una esquina
    conquistada. Los obstáculos ('O') interrumpen la cadena (no cuentan como fichas
    estables propias) pero también protegen la cadena al bloquear capturas desde ese lado.
    La versión robusta recalcula los bordes según las dimensiones reales del tablero.
    """
    real_edges = [
        [(x, 1) for x in range(1, w + 1)],  # superior
        [(x, h) for x in range(1, w + 1)],  # inferior
        [(1, y) for y in range(1, h + 1)],  # izquierdo
        [(w, y) for y in range(1, h + 1)],  # derecho
    ]

    my_stable = 0
    op_stable = 0

    for edge in real_edges:
        n = len(edge)
        if n == 0:
            continue

        # Desde la primera esquina del borde
        c0 = edge[0]
        owner0 = board.get(c0)
        if owner0 and owner0 != blocked:
            for pos in edge:
                cell = board.get(pos)
                if cell == owner0:
                    if owner0 == my_label:
                        my_stable += 1
                    else:
                        op_stable += 1
                else:
                    break

        # Desde la última esquina del borde
        c_last = edge[-1]
        owner_last = board.get(c_last)
        if owner_last and owner_last != blocked:
            for pos in reversed(edge):
                cell = board.get(pos)
                if cell == owner_last:
                    if owner_last == my_label:
                        my_stable += 1
                    else:
                        op_stable += 1
                else:
                    break

    total = my_stable + op_stable
    return 100.0 * (my_stable - op_stable) / total if total > 0 else 0.0


def _compute_frontier_robust(board: dict, my_label: str, op_label: str,
                              h: int, w: int) -> float:
    """Fichas de frontera: expuestas a al menos una casilla vacía adyacente.

    Penaliza NUESTRAS fichas de frontera y premia las del rival. Minimizar fichas
    propias en frontera protege el interior y reduce opciones de flanqueo.
    Versión robusta: usa las dimensiones reales del tablero (h, w) en lugar de 8.
    """
    my_frontier = 0
    op_frontier = 0
    for (x, y), piece in board.items():
        if piece != my_label and piece != op_label:
            continue
        is_front = False
        for dx, dy in DIRECTIONS_8:
            nx, ny = x + dx, y + dy
            if 1 <= nx <= w and 1 <= ny <= h and (nx, ny) not in board:
                is_front = True
                break
        if is_front:
            if piece == my_label:
                my_frontier += 1
            else:
                op_frontier += 1
    total = my_frontier + op_frontier
    return 100.0 * (op_frontier - my_frontier) / total if total > 0 else 0.0


def _compute_potential_mobility(board: dict, my_label: str, op_label: str,
                                h: int, w: int, blocked: str) -> float:
    """Movilidad potencial: casillas vacías adyacentes a fichas rivales.

    Indica cuánto espacio de flanqueo tendré en movimientos futuros.
    A mayor número de casillas vacías junto al rival, mayor es mi movilidad futura.
    Esta métrica complementa la movilidad legal actual, más costosa de calcular.
    """
    my_potential = set()
    op_potential = set()

    for (x, y), piece in board.items():
        if piece != my_label and piece != op_label:
            continue
        for dx, dy in DIRECTIONS_8:
            nx, ny = x + dx, y + dy
            if 1 <= nx <= w and 1 <= ny <= h:
                cell = board.get((nx, ny))
                if cell is None:  # vacía
                    if piece == op_label:  # vacía junto a rival → útil para mí
                        my_potential.add((nx, ny))
                    else:
                        op_potential.add((nx, ny))

    total = len(my_potential) + len(op_potential)
    return 100.0 * (len(my_potential) - len(op_potential)) / total if total > 0 else 0.0


def _compute_corner_closeness(board: dict, my_label: str, op_label: str,
                               blocked: str) -> float:
    """Penaliza las casillas C y X ocupadas cuando la esquina adyacente está vacía.

    Diferencia respecto a la penalización de pesos estáticos: evalúa únicamente
    las casillas 'peligrosas' de forma explícita e independiente, con mayor peso.
    """
    danger = 0.0
    for corner, neighbors in CORNERS_AND_NEIGHBORS:
        cell_corner = board.get(corner)
        if cell_corner is None or cell_corner == blocked:
            # Esquina vacía o bloqueada: las casillas adyacentes son peligrosas
            for n in neighbors:
                piece = board.get(n)
                if piece == my_label:
                    danger -= 30.0
                elif piece == op_label:
                    danger += 30.0
    return danger


# =====================================================================
# HEURÍSTICA 1: Posicional Adaptativa Ultra-Rápida
# =====================================================================

class Solution1(StudentHeuristic):
    """Heurística 1: Posicional Adaptativa + Esquinas + Movilidad Potencial.

    Diseñada para máxima velocidad (< 0.004 ms/nodo). Combina:
    - Matriz de pesos adaptativa (desbloquea casillas X/C al conquistar esquinas).
    - Penalización explícita de casillas de peligro C y X.
    - Movilidad potencial (casillas vacías junto al rival) sin coste de get_valid_moves.
    - Paridad de fichas activada sólo en el tramo final (>= 48 fichas).

    Referencias: Sannidhanam & Muthukaruppan (2015); Rosenbloom (1982).
    """

    def get_name(self) -> str:
        return "Netan"

    def evaluation_function(self, state: TwoPlayerGameState) -> float:
        if state.end_of_game:
            return _terminal_utility(state)

        board = state.board
        if not isinstance(board, dict):
            return 0.0

        my_label, op_label = _get_labels(state)
        h, w, blocked, _ = _get_board_params(state)

        # ---- Pesos adaptativos ----
        dynamic_weights = dict(BOARD_WEIGHTS_8X8)
        for corner, neighbors in CORNERS_AND_NEIGHBORS:
            owner = board.get(corner)
            if owner == my_label:
                for n in neighbors:
                    dynamic_weights[n] = 15
            elif owner == op_label:
                for n in neighbors:
                    dynamic_weights[n] = -15

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

        # ---- Casillas de peligro C y X ----
        danger = _compute_corner_closeness(board, my_label, op_label, blocked)

        # ---- Movilidad potencial (rápida, sin get_valid_moves) ----
        pot_mob = _compute_potential_mobility(board, my_label, op_label, h, w, blocked)

        # ---- Paridad de fichas ----
        total_coins = my_coins + op_coins
        parity = 0.0
        if total_coins > 0:
            parity = 100.0 * (my_coins - op_coins) / total_coins

        if total_coins < 48:
            return float(pos_score + 0.5 * danger + 0.3 * pot_mob)
        else:
            return float(pos_score + 0.5 * danger + 0.2 * pot_mob + 2.5 * parity)


# =====================================================================
# HEURÍSTICA 2: Movilidad Estratégica con Obstáculos
# =====================================================================

class Solution2(StudentHeuristic):
    """Heurística 2: Movilidad Legal + Esquinas + Posicional Adaptativo.

    Estrategia de asfixia clásica (Rosenbloom 1982) mejorada con:
    - Penalización explícita de casillas C y X peligrosas (robusta a obstáculos).
    - Movilidad legal real con soporte de obstáculos y 'ignore_block_cells'.
    - Movilidad potencial como complemento rápido de la movilidad legal.
    - 3 fases de juego: apertura, medio juego y endgame con pesos distintos.
    """

    def get_name(self) -> str:
        return "yahu"

    def evaluation_function(self, state: TwoPlayerGameState) -> float:
        if state.end_of_game:
            return _terminal_utility(state)

        board = state.board
        if not isinstance(board, dict):
            return 0.0

        my_label, op_label = _get_labels(state)
        h, w, blocked, ib = _get_board_params(state)

        # ---- Esquinas ----
        my_c = sum(1 for c in CORNERS if board.get(c) == my_label)
        op_c = sum(1 for c in CORNERS if board.get(c) == op_label)
        corner_score = 100.0 * (my_c - op_c) / (my_c + op_c) if (my_c + op_c) > 0 else 0.0

        # ---- Penalización C y X ----
        danger = _compute_corner_closeness(board, my_label, op_label, blocked)

        # ---- Movilidad legal ----
        my_moves = len(get_valid_moves(board, h, w, my_label, op_label, blocked, ib))
        op_moves = len(get_valid_moves(board, h, w, op_label, my_label, blocked, ib))
        mobility_score = 100.0 * (my_moves - op_moves) / (my_moves + op_moves) if (my_moves + op_moves) > 0 else 0.0

        # ---- Movilidad potencial ----
        pot_mob = _compute_potential_mobility(board, my_label, op_label, h, w, blocked)

        # ---- Posicional adaptativo ----
        dyn = dict(BOARD_WEIGHTS_8X8)
        for corner, neighbors in CORNERS_AND_NEIGHBORS:
            owner = board.get(corner)
            if owner == my_label:
                for n in neighbors: dyn[n] = 12
            elif owner == op_label:
                for n in neighbors: dyn[n] = -12

        pos_score = 0.0
        my_coins = 0
        op_coins = 0
        for pos, piece in board.items():
            wt = dyn.get(pos, 0)
            if piece == my_label:
                my_coins += 1
                pos_score += wt
            elif piece == op_label:
                op_coins += 1
                pos_score -= wt

        # ---- Paridad ----
        total_coins = my_coins + op_coins
        parity = 100.0 * (my_coins - op_coins) / total_coins if total_coins > 0 else 0.0

        # ---- Ponderación por fase ----
        if total_coins < 20:
            # Apertura: movilidad y posición dominan; evitar peligro
            return float(
                40.0 * corner_score +
                20.0 * danger / 30.0 +
                35.0 * mobility_score +
                15.0 * pot_mob +
                10.0 * pos_score
            )
        elif total_coins < 48:
            # Medio juego: movilidad + estabilidad posicional
            return float(
                40.0 * corner_score +
                15.0 * danger / 30.0 +
                30.0 * mobility_score +
                10.0 * pot_mob +
                20.0 * pos_score +
                5.0 * parity
            )
        else:
            # Endgame: paridad de fichas es decisiva
            return float(
                35.0 * corner_score +
                10.0 * mobility_score +
                15.0 * pos_score +
                40.0 * parity
            )


# =====================================================================
# HEURÍSTICA 3: Maestra Multicriterio Avanzada
# =====================================================================

class Solution3(StudentHeuristic):
    """Heurística 3: Maestra Multicriterio Avanzada con Soporte de Obstáculos.

    Versión mejorada de la estrategia ganadora del torneo anterior. Integra:
    1. Esquinas: Máxima prioridad (inexpugnables).
    2. Penalización C y X: Penalización explícita, doblada respecto a versiones anteriores.
    3. Movilidad Legal Real: Soporte de obstáculos y 'ignore_block_cells'.
    4. Movilidad Potencial: Complemento rápido, robusto con dimensiones reales.
    5. Fichas de Frontera: Minimiza exposición propia, maximiza la del rival.
    6. Estabilidad de Bordes: Versión robusta a obstáculos y dimensiones variables.
    7. Posicional Adaptativo: Desbloqueo dinámico de casillas adyacentes a esquinas.
    8. Paridad por 3 Fases: Transición suave apertura → medio juego → endgame.

    Referencias:
      Buro, M. (1997). Logistello's learning algorithm.
      Rosenbloom, P. (1982). A world-championship-level Othello program.
      Sannidhanam, V., & Muthukaruppan, A. (2015). Analysis of heuristics in Othello.
    """

    def get_name(self) -> str:
        return "maestra_multicriterio"

    def evaluation_function(self, state: TwoPlayerGameState) -> float:
        if state.end_of_game:
            return _terminal_utility(state)

        board = state.board
        if not isinstance(board, dict):
            return 0.0

        my_label, op_label = _get_labels(state)
        h, w, blocked, ib = _get_board_params(state)

        # ---- 1. Esquinas ----
        my_c = sum(1 for c in CORNERS if board.get(c) == my_label)
        op_c = sum(1 for c in CORNERS if board.get(c) == op_label)
        corner_val = 100.0 * (my_c - op_c) / (my_c + op_c) if (my_c + op_c) > 0 else 0.0

        # ---- 2. Penalización C y X (robusta a obstáculos) ----
        danger_val = _compute_corner_closeness(board, my_label, op_label, blocked)

        # ---- 3. Movilidad Legal ----
        my_m = len(get_valid_moves(board, h, w, my_label, op_label, blocked, ib))
        op_m = len(get_valid_moves(board, h, w, op_label, my_label, blocked, ib))
        mob_val = 100.0 * (my_m - op_m) / (my_m + op_m) if (my_m + op_m) > 0 else 0.0

        # ---- 4. Movilidad Potencial ----
        pot_val = _compute_potential_mobility(board, my_label, op_label, h, w, blocked)

        # ---- 5. Fichas de Frontera (robusta con h, w reales) ----
        front_val = _compute_frontier_robust(board, my_label, op_label, h, w)

        # ---- 6. Estabilidad de Bordes (robusta con h, w, blocked) ----
        stab_val = _compute_edge_stability_robust(board, my_label, op_label, h, w, blocked)

        # ---- 7. Posicional Adaptativo ----
        dyn = dict(BOARD_WEIGHTS_8X8)
        for corner, neighbors in CORNERS_AND_NEIGHBORS:
            owner = board.get(corner)
            if owner == my_label:
                for n in neighbors: dyn[n] = 15
            elif owner == op_label:
                for n in neighbors: dyn[n] = -15

        pos_val = 0.0
        my_coins = 0
        op_coins = 0
        for pos, piece in board.items():
            wt = dyn.get(pos, 0)
            if piece == my_label:
                my_coins += 1
                pos_val += wt
            elif piece == op_label:
                op_coins += 1
                pos_val -= wt

        # ---- 8. Paridad ----
        total_coins = my_coins + op_coins
        parity_val = 100.0 * (my_coins - op_coins) / total_coins if total_coins > 0 else 0.0

        # ---- Ponderación por 3 fases ----
        if total_coins < 20:
            # Apertura: movilidad + posición; evitar peligro; paridad irrelevante
            return float(
                35.0 * corner_val +
                1.0  * danger_val +
                35.0 * mob_val +
                15.0 * pot_val +
                10.0 * front_val +
                15.0 * stab_val +
                20.0 * (pos_val / 10.0)
            )
        elif total_coins < 52:
            # Medio juego: balance entre movilidad, estabilidad y posición
            return float(
                35.0 * corner_val +
                1.0  * danger_val +
                25.0 * mob_val +
                10.0 * pot_val +
                10.0 * front_val +
                20.0 * stab_val +
                15.0 * (pos_val / 10.0) +
                10.0 * parity_val
            )
        else:
            # Endgame: paridad decisiva + estabilidad
            return float(
                30.0 * corner_val +
                10.0 * mob_val +
                25.0 * stab_val +
                15.0 * (pos_val / 10.0) +
                40.0 * parity_val
            )
