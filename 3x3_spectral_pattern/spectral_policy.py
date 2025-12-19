#!/usr/bin/env python3
"""
- Pure spectral (DCT) features with optional polynomial expansion.
- Deterministic ridge-regression training from minimax supervision.
- Deterministic move selection with tie-breaking by cell index.
- Simple evaluation vs random opponents, perfect opponent on 3x3, and self-play.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

TREE_FILE = 'optimal_decision_tree.json'
DEFAULT_3X3_POLY_ORDER = 4
DEFAULT_3X3_WEIGHT_FILE = Path('best_poly4_weights.json')

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


@dataclass(frozen=True)
class TicTacToeNxN:
    size: int

    def index(self, r: int, c: int) -> int:
        return r * self.size + c

    def get_valid_moves(self, board: Sequence[int]) -> List[int]:
        return [i for i, v in enumerate(board) if v == 0]

    def check_winner(self, board: Sequence[int]) -> int:
        n = self.size
        # rows
        for r in range(n):
            s = sum(board[self.index(r, c)] for c in range(n))
            if s == n:
                return 1
            if s == -n:
                return -1
        # columns
        for c in range(n):
            s = sum(board[self.index(r, c)] for r in range(n))
            if s == n:
                return 1
            if s == -n:
                return -1
        # diagonals
        diag = sum(board[self.index(i, i)] for i in range(n))
        if diag == n:
            return 1
        if diag == -n:
            return -1
        anti = sum(board[self.index(i, n - 1 - i)] for i in range(n))
        if anti == n:
            return 1
        if anti == -n:
            return -1
        return 0

    def is_terminal(self, board: Sequence[int]) -> Tuple[bool, int]:
        winner = self.check_winner(board)
        if winner != 0:
            return True, winner
        if 0 not in board:
            return True, 0
        return False, 0


# ---------------------------------------------------------------------------
# DCT spectral features
# ---------------------------------------------------------------------------

_DCT_CACHE: Dict[int, np.ndarray] = {}


def dct_matrix(n: int) -> np.ndarray:
    if n in _DCT_CACHE:
        return _DCT_CACHE[n]
    alpha = [math.sqrt(1 / n)] + [math.sqrt(2 / n)] * (n - 1)
    C = np.zeros((n, n), dtype=float)
    for u in range(n):
        for x in range(n):
            C[u, x] = alpha[u] * math.cos(math.pi * (2 * x + 1) * u / (2 * n))
    _DCT_CACHE[n] = C
    return C


def board_channels(board: Sequence[int], n: int) -> Tuple[np.ndarray, np.ndarray]:
    X = np.zeros((n, n), dtype=float)
    O = np.zeros((n, n), dtype=float)
    for idx, val in enumerate(board):
        r, c = divmod(idx, n)
        if val == 1:
            X[r, c] = 1.0
        elif val == -1:
            O[r, c] = 1.0
    return X, O


def spectral_features_after_move(state: Sequence[int], player: int, move: int, n: int) -> np.ndarray:
    board = list(state)
    board[move] = player
    X, O = board_channels(board, n)
    C = dct_matrix(n)
    DX = (C @ X @ C.T).reshape(-1)
    DO = (C @ O @ C.T).reshape(-1)
    return np.concatenate([DX, DO])


# ---------------------------------------------------------------------------
# Polynomial expansion
# ---------------------------------------------------------------------------


class PolynomialFeatureMap:
    def __init__(self, base_dim: int, order: int):
        self.base_dim = base_dim
        self.order = order
        self.monomials: List[Tuple[int, ...]] = []
        for deg in range(1, order + 1):
            self.monomials.extend(combinations_with_replacement(range(base_dim), deg))
        self.output_dim = len(self.monomials)

    def transform_vector(self, x: np.ndarray) -> np.ndarray:
        feats = np.empty(self.output_dim, dtype=float)
        for idx, combo in enumerate(self.monomials):
            prod = 1.0
            for j in combo:
                prod *= x[j]
            feats[idx] = prod
        return feats

    def transform_matrix(self, X: np.ndarray) -> np.ndarray:
        rows = [self.transform_vector(row) for row in X]
        return np.vstack(rows)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def parse_player(token: str) -> int:
    return 1 if token.upper() in ('X', '1') else -1


def downsample_by_state(entries: List[Tuple[np.ndarray, float, Tuple[Tuple[int, ...], int, int]]],
                        fraction: float, seed: int) -> List[Tuple[np.ndarray, float, Tuple[Tuple[int, ...], int, int]]]:
    if fraction >= 0.9999:
        return entries
    groups: Dict[Tuple[Tuple[int, ...], int], List[int]] = {}
    for idx, (_, _, meta) in enumerate(entries):
        state, _, player = meta
        groups.setdefault((state, player), []).append(idx)
    keys = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)
    target = max(1, int(len(keys) * fraction))
    keep = set(keys[:target])
    kept_rows: List[Tuple[np.ndarray, float, Tuple[Tuple[int, ...], int, int]]] = []
    for key in keep:
        for idx in groups[key]:
            kept_rows.append(entries[idx])
    return kept_rows


def read_weight_file(path: Path) -> Tuple[np.ndarray, float]:
    data = json.loads(path.read_text(encoding='utf-8'))
    weights = np.array(data['weights'], dtype=float)
    lam = float(data.get('lambda', 0.0))
    return weights, lam


def write_weight_file(path: Path, weights: np.ndarray, lam: float, size: int, poly_order: int) -> None:
    payload = {
        'size': size,
        'poly_order': poly_order,
        'lambda': lam,
        'weights': weights.tolist(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def load_exhaustive_states() -> Dict[str, Dict]:
    payload = json.loads(Path(TREE_FILE).read_text(encoding='utf-8'))
    return payload.get('states', payload)


def opponent_immediate_wins(board: List[int], opponent: int, game: TicTacToeNxN) -> List[int]:
    """Return list of empty cells where opponent would win if they played there."""
    wins: List[int] = []
    for i, v in enumerate(board):
        if v != 0:
            continue
        board[i] = opponent
        if game.check_winner(board) == opponent:
            wins.append(i)
        board[i] = 0
    return wins


def build_exhaustive_dataset(data_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray, List[Tuple[Tuple[int, ...], int, int]]]:
    """Build dataset with binary labels (1.0 for optimal, 0.0 for non-optimal) and block oversampling.
    
    Uses the tree's pre-computed move_details scores for determining optimal moves,
    matching the original spectral_policy_3x3.py behavior.
    """
    states = load_exhaustive_states()
    entries: List[Tuple[np.ndarray, float, Tuple[Tuple[int, ...], int, int]]] = []
    game3 = TicTacToeNxN(3)
    for state_str, info in states.items():
        state = tuple(json.loads(state_str))
        if 0 not in state:
            continue
        player = parse_player(info.get('player', 'X'))
        valid = [i for i, v in enumerate(state) if v == 0]
        
        # Determine optimal moves from tree's move_details (matching original)
        best_moves: set = set()
        details = info.get('move_details', [])
        if details:
            max_score = max(d.get('score', -1e9) for d in details)
            best_moves = {int(d['move']) for d in details if abs(d.get('score', -1e9) - max_score) <= 1e-6}
        if not best_moves and 'move' in info:
            best_moves = {int(info['move'])}
        
        # Get opponent's immediate win cells (for block oversampling)
        board_list = list(state)
        opp_wins = set(opponent_immediate_wins(board_list, -player, game3))
        
        for move in valid:
            phi = spectral_features_after_move(state, player, move, 3)
            label = 1.0 if move in best_moves else 0.0
            entries.append((phi, label, (state, move, player)))
            # Oversample block-critical moves
            if move in best_moves and move in opp_wins:
                entries.append((phi, label, (state, move, player)))
    if data_fraction < 1.0:
        entries = downsample_by_state(entries, data_fraction, seed)
    X = np.vstack([e[0] for e in entries])
    y = np.array([e[1] for e in entries], dtype=float)
    meta = [e[2] for e in entries]
    return X, y, meta


def generate_random_state(game: TicTacToeNxN, rng: random.Random, max_empties: int) -> Optional[Tuple[Tuple[int, ...], int]]:
    total = game.size * game.size
    empties = rng.randint(2, max_empties)
    board = [0] * total
    order = list(range(total))
    rng.shuffle(order)
    player = 1
    moves_to_play = total - empties
    for idx in range(moves_to_play):
        mv = order[idx]
        board[mv] = player
        player = -player
        if game.check_winner(board) != 0:
            return None
    term, _ = game.is_terminal(board)
    if term:
        return None
    return tuple(board), player


def minimax(game: TicTacToeNxN, state: Tuple[int, ...], player: int,
            cache: Dict[Tuple[Tuple[int, ...], int], int]) -> int:
    key = (state, player)
    if key in cache:
        return cache[key]
    term, winner = game.is_terminal(state)
    if term:
        cache[key] = winner
        return winner
    board = list(state)
    best = -2 if player == 1 else 2
    for move in game.get_valid_moves(state):
        board[move] = player
        nxt = tuple(board)
        score = minimax(game, nxt, -player, cache)
        board[move] = 0
        if player == 1:
            best = max(best, score)
            if best == 1:
                break
        else:
            best = min(best, score)
            if best == -1:
                break
    cache[key] = best
    return best


def optimal_moves(game: TicTacToeNxN, state: Tuple[int, ...], player: int,
                  cache: Dict[Tuple[Tuple[int, ...], int], int]) -> List[int]:
    valid = game.get_valid_moves(state)
    if not valid:
        return []
    board = list(state)
    scores: List[Tuple[int, int]] = []
    target: Optional[int] = None
    for mv in valid:
        board[mv] = player
        nxt = tuple(board)
        score = minimax(game, nxt, -player, cache)
        board[mv] = 0
        scores.append((score, mv))
        if target is None:
            target = score
        else:
            target = max(target, score) if player == 1 else min(target, score)
    if target is None:
        return []
    return [mv for score, mv in scores if score == target]


def build_sampled_dataset(size: int, target_states: int, max_empties: int, seed: int) -> Tuple[np.ndarray, np.ndarray, List[Tuple[Tuple[int, ...], int, int]]]:
    game = TicTacToeNxN(size)
    rng = random.Random(seed)
    cache: Dict[Tuple[Tuple[int, ...], int], int] = {}
    entries: List[Tuple[np.ndarray, float, Tuple[Tuple[int, ...], int, int]]] = []
    states_used = 0
    attempts = 0
    max_attempts = target_states * 50
    while states_used < target_states and attempts < max_attempts:
        attempts += 1
        sample = generate_random_state(game, rng, max_empties)
        if sample is None:
            continue
        state, player = sample
        best = optimal_moves(game, state, player, cache)
        if not best:
            continue
        best_set = set(best)
        valid = game.get_valid_moves(state)
        for move in valid:
            phi = spectral_features_after_move(state, player, move, size)
            label = 1.0 if move in best_set else 0.0
            entries.append((phi, label, (state, move, player)))
        states_used += 1
    if states_used == 0:
        raise RuntimeError('Failed to sample any non-terminal states for training.')
    X = np.vstack([e[0] for e in entries])
    y = np.array([e[1] for e in entries], dtype=float)
    meta = [e[2] for e in entries]
    return X, y, meta


# ---------------------------------------------------------------------------
# Ridge regression
# ---------------------------------------------------------------------------


def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    d = X.shape[1]
    A = X.T @ X + lam * np.eye(d)
    b = X.T @ y
    return np.linalg.solve(A, b)


# ---------------------------------------------------------------------------
# Policy + agents
# ---------------------------------------------------------------------------


class SpectralPolicy:
    def __init__(self, size: int, weights: np.ndarray, poly_map: Optional[PolynomialFeatureMap], safety: bool = True):
        self.size = size
        self.weights = weights
        self.poly_map = poly_map
        self.safety = safety
        self.game = TicTacToeNxN(size)

    def score_move(self, board: Sequence[int], player: int, move: int) -> float:
        phi = spectral_features_after_move(board, player, move, self.size)
        if self.poly_map is not None:
            phi = self.poly_map.transform_vector(phi)
        if phi.shape[0] != self.weights.shape[0]:
            raise ValueError('Feature dimension mismatch between weights and polynomial map.')
        return float(self.weights @ phi)

    def choose_move(self, board: Sequence[int], player: int) -> int:
        moves = [i for i, v in enumerate(board) if v == 0]
        # Safety override: check for immediate win or block
        if self.safety:
            board_list = list(board)
            # Check for immediate win
            for mv in moves:
                board_list[mv] = player
                if self.game.check_winner(board_list) == player:
                    board_list[mv] = 0
                    return mv
                board_list[mv] = 0
            # Check for immediate block
            opp = -player
            for mv in moves:
                board_list[mv] = opp
                if self.game.check_winner(board_list) == opp:
                    board_list[mv] = 0
                    return mv
                board_list[mv] = 0
        # Spectral score with deterministic tie-break
        best_move = moves[0]
        best_score = -1e9
        for mv in moves:
            score = self.score_move(board, player, mv)
            if score > best_score or (score == best_score and mv < best_move):
                best_score = score
                best_move = mv
        return best_move


class RandomAgent:
    """Uses global random module (like original spectral_policy_3x3.py)."""
    def __init__(self, seed: Optional[int] = None):
        pass  # Uses global random, seed set via set_seed()

    def choose_move(self, board: Sequence[int], player: int) -> int:
        moves = [i for i, v in enumerate(board) if v == 0]
        return random.choice(moves)


class Perfect3x3Agent:
    def __init__(self):
        self.cache: Dict[Tuple[Tuple[int, ...], int], int] = {}
        self.game = TicTacToeNxN(3)

    def minimax(self, state: Tuple[int, ...], player: int) -> int:
        key = (state, player)
        if key in self.cache:
            return self.cache[key]
        term, winner = self.game.is_terminal(state)
        if term:
            self.cache[key] = winner
            return winner
        board = list(state)
        best = -2 if player == 1 else 2
        for mv in self.game.get_valid_moves(state):
            board[mv] = player
            nxt = tuple(board)
            score = self.minimax(nxt, -player)
            board[mv] = 0
            if player == 1:
                best = max(best, score)
                if best == 1:
                    break
            else:
                best = min(best, score)
                if best == -1:
                    break
        self.cache[key] = best
        return best

    def choose_move(self, board: Sequence[int], player: int) -> int:
        valid = [i for i, v in enumerate(board) if v == 0]
        best_move = valid[0]
        best_score = -2 if player == 1 else 2
        for mv in valid:
            tmp = list(board)
            tmp[mv] = player
            score = self.minimax(tuple(tmp), -player)
            if player == 1:
                if score > best_score or (score == best_score and mv < best_move):
                    best_score = score
                    best_move = mv
            else:
                if score < best_score or (score == best_score and mv < best_move):
                    best_score = score
                    best_move = mv
        return best_move


# ---------------------------------------------------------------------------
# Gameplay evaluation
# ---------------------------------------------------------------------------


def play_game(size: int, agent_x, agent_o) -> int:
    game = TicTacToeNxN(size)
    board = [0] * (size * size)
    current = 1
    while True:
        agent = agent_x if current == 1 else agent_o
        move = agent.choose_move(board, current)
        board[move] = current
        term, winner = game.is_terminal(board)
        if term:
            return winner
        current *= -1


def run_matches(size: int, agent, opponent, games: int, swap_first: bool = False) -> Tuple[int, int, int]:
    wins = draws = losses = 0
    for g in range(games):
        if swap_first and (g % 2 == 1):
            result = play_game(size, opponent, agent)
            # result is from X perspective; convert so policy is opponent's success
            if result == 1:
                losses += 1
            elif result == -1:
                wins += 1
            else:
                draws += 1
        else:
            result = play_game(size, agent, opponent)
            if result == 1:
                wins += 1
            elif result == -1:
                losses += 1
            else:
                draws += 1
    return wins, draws, losses


def run_self_play(size: int, agent, games: int) -> Tuple[int, int, int]:
    wins = draws = losses = 0
    for _ in range(games):
        result = play_game(size, agent, agent)
        if result == 1:
            wins += 1
        elif result == -1:
            losses += 1
        else:
            draws += 1
    return wins, draws, losses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description='Minimal spectral policy for tic-tac-toe (3x3 or NxN).')
    parser.add_argument('--size', type=int, default=3, help='Board size (3 for exhaustive dataset, >3 uses sampled states).')
    parser.add_argument('--poly-order', type=int, default=4, help='Polynomial expansion order (>=1).')
    parser.add_argument('--lam', type=float, default=1e-4, help='Ridge regularization lambda.')
    parser.add_argument('--data-fraction', type=float, default=1.0, help='Fraction of 3x3 states to keep (unused for NxN).')
    parser.add_argument('--sample-states', type=int, default=500, help='Number of sampled states for NxN training.')
    parser.add_argument('--max-empties', type=int, default=6, help='Maximum empty cells when sampling NxN states.')
    parser.add_argument('--random-games', type=int, default=50, help='Evaluation games vs random opponent.')
    parser.add_argument('--perfect-games', type=int, default=50, help='Evaluation games vs perfect opponent (3x3 only).')
    parser.add_argument('--self-play', type=int, default=50, help='Number of self-play games.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed.')
    parser.add_argument('--save-weights', type=str, default='', help='Optional path to save learned weights.')
    parser.add_argument('--load-weights', type=str, default='', help='Optional path to load pre-trained weights.')
    parser.add_argument('--max-weight-print', type=int, default=10, help='How many weights to print (use -1 for all).')
    parser.add_argument('--no-safety', action='store_true', help='Disable safety overrides (no immediate win/block).')
    args = parser.parse_args()

    if args.poly_order < 1:
        parser.error('--poly-order must be >= 1')

    set_seed(args.seed)

    if args.size == 3:
        print('Building dataset from exhaustive minimax tree...')
        X, y, meta = build_exhaustive_dataset(args.data_fraction, args.seed or 0)
    else:
        print(f'Sampling {args.sample_states} states for size {args.size}...')
        X, y, meta = build_sampled_dataset(args.size, args.sample_states, args.max_empties, args.seed or 0)
    base_dim = X.shape[1]
    print(f'Samples: {len(y)} | Base Dim: {base_dim}')

    poly_map = None
    if args.poly_order > 1:
        poly_map = PolynomialFeatureMap(base_dim, args.poly_order)
        X = poly_map.transform_matrix(X)
        print(f'Effective Dim after poly order {args.poly_order}: {X.shape[1]}')
    else:
        print('Polynomial expansion disabled (order=1).')

    feature_dim = X.shape[1]
    weights: Optional[np.ndarray] = None
    chosen_lambda = args.lam
    default_cache_path: Optional[Path] = None
    if args.size == 3 and args.poly_order == DEFAULT_3X3_POLY_ORDER:
        default_cache_path = DEFAULT_3X3_WEIGHT_FILE

    def _validate_loaded(source: str) -> bool:
        nonlocal weights
        if weights is None:
            return False
        if len(weights) != feature_dim:
            print(f'Ignored weights from {source} because dim={len(weights)} but expected {feature_dim}.')
            weights = None
            return False
        return True

    if args.load_weights:
        custom_path = Path(args.load_weights)
        weights, chosen_lambda = read_weight_file(custom_path)
        if _validate_loaded(str(custom_path)):
            print(f'Loaded weights from {custom_path} (lambda={chosen_lambda}).')
    elif default_cache_path and default_cache_path.exists():
        weights, chosen_lambda = read_weight_file(default_cache_path)
        if _validate_loaded(str(default_cache_path)):
            print(f'Loaded cached 3x3 poly{DEFAULT_3X3_POLY_ORDER} weights from {default_cache_path} (lambda={chosen_lambda}).')
    elif default_cache_path:
        print(f'Cached 3x3 poly{DEFAULT_3X3_POLY_ORDER} weights not found at {default_cache_path}; training new weights...')

    if weights is None:
        print(f'Training ridge regression (lambda={chosen_lambda})...')
        weights = fit_ridge(X, y, chosen_lambda)
        if default_cache_path:
            write_weight_file(default_cache_path, weights, chosen_lambda, args.size, args.poly_order)
            print(f'Saved default 3x3 poly{DEFAULT_3X3_POLY_ORDER} weights to {default_cache_path}.')

    total_weights = len(weights)
    to_print = total_weights if args.max_weight_print < 0 else min(args.max_weight_print, total_weights)
    print('Weights preview:')
    for i in range(to_print):
        print(f'  w[{i}]: {weights[i]:+.4f}')
    if to_print < total_weights:
        print(f'  ... {total_weights - to_print} more weights')

    if args.save_weights:
        out_path = Path(args.save_weights)
        write_weight_file(out_path, weights, chosen_lambda, args.size, args.poly_order)
        print(f'Saved weights to {out_path}')

    policy = SpectralPolicy(args.size, weights, poly_map, safety=not args.no_safety)
    random_agent = RandomAgent(seed=args.seed)

    if args.random_games > 0:
        w, d, l = run_matches(args.size, policy, random_agent, args.random_games, swap_first=False)
        print(f'Vs Random ({args.random_games}): W:{w} D:{d} L:{l}')

    if args.size == 3 and args.perfect_games > 0:
        perfect = Perfect3x3Agent()
        w, d, l = run_matches(3, policy, perfect, args.perfect_games, swap_first=True)
        print(f'Vs Perfect ({args.perfect_games}): W:{w} D:{d} L:{l}')

    if args.self_play > 0:
        w, d, l = run_self_play(args.size, policy, args.self_play)
        print(f'Self-play ({args.self_play}): W:{w} D:{d} L:{l}')


if __name__ == '__main__':
    main()
