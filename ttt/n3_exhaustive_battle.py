from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

if __package__ in (None, "", __name__):
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from ttt.common import StateN3, load_n3_exhaustive  # type: ignore
else:
    from .common import StateN3, load_n3_exhaustive


Board = List[int]


WIN_LINES: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)


def legal_moves(board: Board) -> List[int]:
    return [i for i, v in enumerate(board) if v == 0]


def format_board(board: Board) -> str:
    """
    Format board as a 3x3 grid string with borders.
    +1 -> 'X', -1 -> 'O', 0 -> '.'
    Returns a multi-line string with borders.
    """
    symbols = {+1: 'X', -1: 'O', 0: '.'}
    rows = []
    rows.append("+---+---+---+")
    for r in range(3):
        row_chars = [symbols[board[r * 3 + c]] for c in range(3)]
        rows.append(f"| {' | '.join(row_chars)} |")
        rows.append("+---+---+---+")
    return '\n'.join(rows)


def evaluate_terminal(board: Board) -> Optional[int]:
    """
    Return +1 if X wins, -1 if O wins, 0 if draw, None otherwise.
    """
    for a, b, c in WIN_LINES:
        s = board[a] + board[b] + board[c]
        if s == 3:
            return +1
        if s == -3:
            return -1
    if all(v != 0 for v in board):
        return 0
    return None


class Agent:
    def select_move(self, board: Board, turn: int) -> int:
        raise NotImplementedError


class ExhaustiveAgent(Agent):
    def __init__(self, states: Iterable[StateN3], rng: random.Random, epsilon: float = 1e-12, lambda_draw: float = 0.5):
        self._rng = rng
        self._epsilon = epsilon
        self._lambda_draw = lambda_draw
        self._state_map: Dict[Tuple[Tuple[int, ...], int], StateN3] = {}
        for st in states:
            key = (tuple(st.board), st.turn)
            self._state_map[key] = st

    def select_move(self, board: Board, turn: int) -> int:
        key = (tuple(board), turn)
        state = self._state_map.get(key)
        legal = legal_moves(board)
        if state is None or not state.per_move:
            return self._rng.choice(legal)

        best_score = None
        best_moves: List[int] = []
        for move_str, stats in state.per_move.items():
            move = int(move_str)
            if move not in legal:
                continue
            wins = stats.get("wins", 0)
            draws = stats.get("draws", 0)
            losses = stats.get("losses", 0)
            total = wins + draws + losses
            if total <= 0:
                continue
            # Use same peff formula as data generation: (wins + lambda_draw*draws) / total
            score = (wins + self._lambda_draw * draws) / total
            if (best_score is None) or (score > best_score + self._epsilon):
                best_score = score
                best_moves = [move]
            elif abs(score - (best_score or 0.0)) <= self._epsilon:
                best_moves.append(move)

        if not best_moves:
            return self._rng.choice(legal)
        return self._rng.choice(best_moves)


class RandomAgent(Agent):
    def __init__(self, rng: random.Random):
        self._rng = rng

    def select_move(self, board: Board, turn: int) -> int:
        return self._rng.choice(legal_moves(board))


class SmartAgent(Agent):
    """
    A "smart" agent that:
    1. Wins if possible
    2. Blocks opponent from winning
    3. Prefers center, then corners, then edges
    """
    def __init__(self, rng: random.Random):
        self._rng = rng

    def _can_win(self, board: Board, move: int, player: int) -> bool:
        """Check if playing at move would make player win."""
        board[move] = player
        result = evaluate_terminal(board)
        board[move] = 0
        return result == player

    def select_move(self, board: Board, turn: int) -> int:
        legal = legal_moves(board)
        if not legal:
            return 0

        # 1. Win if possible
        for move in legal:
            if self._can_win(board, move, turn):
                return move

        # 2. Block opponent from winning
        for move in legal:
            if self._can_win(board, move, -turn):
                return move

        # 3. Heuristic: prefer center (4), then corners (0,2,6,8), then edges (1,3,5,7)
        center = 4
        corners = [0, 2, 6, 8]
        edges = [1, 3, 5, 7]

        if center in legal:
            return center

        for pos in corners:
            if pos in legal:
                return pos

        for pos in edges:
            if pos in legal:
                return pos

        # Fallback: random (shouldn't happen)
        return self._rng.choice(legal)


class MinimaxAgent(Agent):
    def __init__(self):
        self._cache: Dict[Tuple[Tuple[int, ...], int], int] = {}

    def _solve(self, board: Board, turn: int) -> int:
        key = (tuple(board), turn)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        terminal = evaluate_terminal(board)
        if terminal is not None:
            if terminal == 0:
                self._cache[key] = 0
                return 0
            val = terminal * turn
            self._cache[key] = val
            return val

        best = -2
        for move in legal_moves(board):
            board[move] = turn
            score = -self._solve(board, -turn)
            board[move] = 0
            if score > best:
                best = score
            if best == 1:
                break
        self._cache[key] = best
        return best

    def select_move(self, board: Board, turn: int) -> int:
        best_value = -2
        best_moves: List[int] = []
        for move in legal_moves(board):
            board[move] = turn
            score = -self._solve(board, -turn)
            board[move] = 0
            if score > best_value:
                best_value = score
                best_moves = [move]
            elif score == best_value:
                best_moves.append(move)
        # Deterministic but stable: prefer smallest index on ties
        return min(best_moves)


@dataclass
class SeriesResult:
    wins: int = 0
    losses: int = 0
    draws: int = 0

    def record(self, outcome: int, pov: int) -> None:
        """
        outcome: +1 for X win, -1 for O win, 0 for draw
        pov: +1 if tracking agent played as X, -1 if played as O
        """
        if outcome == 0:
            self.draws += 1
        elif outcome == pov:
            self.wins += 1
        else:
            self.losses += 1

    def as_dict(self) -> Dict[str, int]:
        return {"wins": self.wins, "draws": self.draws, "losses": self.losses}
    
    def total(self) -> int:
        return self.wins + self.draws + self.losses
    
    def win_rate(self) -> float:
        total = self.total()
        return (self.wins / total) if total > 0 else 0.0
    
    def __str__(self) -> str:
        total = self.total()
        wr = self.win_rate()
        return f"W:{self.wins} D:{self.draws} L:{self.losses} (Win Rate: {wr:.1%}, Total: {total})"


def play_game(agent_x: Agent, agent_o: Agent, record_history: bool = False) -> tuple:
    """
    Returns (outcome, history) where:
    - outcome: +1 (X wins), -1 (O wins), 0 (draw)
    - history: list of (board_state, move, turn) if record_history=True, else None
    """
    board = [0] * 9
    turn = +1
    agents = {+1: agent_x, -1: agent_o}
    history = [] if record_history else None
    
    while True:
        move = agents[turn].select_move(board, turn)
        if record_history:
            history.append((board[:], move, turn))
        board[move] = turn
        outcome = evaluate_terminal(board)
        if outcome is not None:
            if record_history:
                # Add final board state
                history.append((board[:], None, None))
            return outcome, history
        turn = -turn


def run_series(main_agent: Agent, opponent: Agent, games: int, rng: random.Random, record_losses: bool = False) -> tuple:
    """
    Returns (result, loss_games) where:
    - result: SeriesResult
    - loss_games: list of game histories for losses (if record_losses=True)
    """
    result = SeriesResult()
    loss_games = []
    
    for i in range(games):
        if i % 2 == 0:
            outcome, history = play_game(main_agent, opponent, record_history=record_losses)
            result.record(outcome, pov=+1)
            if record_losses and outcome == -1:  # main agent lost (O won)
                formatted_history = []
                if history:
                    for board, move, turn in history:
                        entry = {
                            "board": list(board),
                            "board_str": format_board(board)
                        }
                        if move is not None:
                            entry["move"] = move
                            entry["turn"] = turn
                            entry["turn_symbol"] = "X" if turn == 1 else "O"
                        else:
                            entry["final_state"] = True
                        formatted_history.append(entry)
                loss_games.append({
                    "game_id": i,
                    "main_agent_played_as": "X",
                    "outcome": "loss",
                    "history": formatted_history
                })
        else:
            outcome, history = play_game(opponent, main_agent, record_history=record_losses)
            # when opponent starts, main_agent plays as O -> pov -1
            result.record(outcome, pov=-1)
            if record_losses and outcome == +1:  # main agent lost (X won)
                formatted_history = []
                if history:
                    for board, move, turn in history:
                        entry = {
                            "board": list(board),
                            "board_str": format_board(board)
                        }
                        if move is not None:
                            entry["move"] = move
                            entry["turn"] = turn
                            entry["turn_symbol"] = "X" if turn == 1 else "O"
                        else:
                            entry["final_state"] = True
                        formatted_history.append(entry)
                loss_games.append({
                    "game_id": i,
                    "main_agent_played_as": "O",
                    "outcome": "loss",
                    "history": formatted_history
                })
    
    return result, loss_games


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate 3x3 exhaustive policy against random and minimax opponents.")
    parser.add_argument("--states", default="data/n3_exhaustive.json", help="Path to exhaustive state data.")
    parser.add_argument("--games_random", type=int, default=1000, help="Number of games vs random opponent.")
    parser.add_argument("--games_smart", type=int, default=0, help="Number of games vs smart opponent (wins/blocks/heuristic).")
    parser.add_argument("--games_minimax", type=int, default=50, help="Number of games vs minimax opponent.")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for tie-breaking and random opponent.")
    parser.add_argument("--lambda_draw", type=float, default=None, help="Single lambda_draw value to test (overrides sweep).")
    parser.add_argument("--sweep_lambda", action="store_true", help="Test lambda_draw from 0.1 to 1.3 (10%% to 130%%).")
    parser.add_argument("--out_json", type=str, default=None, help="Output JSON file to save results.")
    parser.add_argument("--record_losses", action="store_true", help="Record detailed history of lost games.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    states = load_n3_exhaustive(args.states)
    random_agent = RandomAgent(rng=rng)
    smart_agent = SmartAgent(rng=rng)
    minimax_agent = MinimaxAgent()

    # Determine lambda_draw values to test
    if args.sweep_lambda:
        # Test from 10% to 130% (0.1 to 1.3) in steps of 0.1
        lambda_values = [round(0.1 + i * 0.1, 1) for i in range(13)]  # 0.1, 0.2, ..., 1.3
    elif args.lambda_draw is not None:
        lambda_values = [args.lambda_draw]
    else:
        lambda_values = [0.5]  # default

    print(f"Testing {len(lambda_values)} lambda_draw value(s)")
    print("=" * 80)

    results = []
    for lambda_draw in lambda_values:
        print(f"\nlambda_draw = {lambda_draw:.1f} ({lambda_draw*100:.0f}%)")
        print("-" * 80)
        
        exhaustive_agent = ExhaustiveAgent(states, rng=rng, lambda_draw=lambda_draw)
        
        result_entry = {
            "lambda_draw": lambda_draw,
            "lambda_draw_percent": round(lambda_draw * 100, 0)
        }
        
        if args.games_random > 0:
            res_random, loss_games_random = run_series(exhaustive_agent, random_agent, args.games_random, rng, record_losses=args.record_losses)
            print(f"  Against Random: {res_random}")
            result_entry["vs_random"] = res_random.as_dict()
            result_entry["vs_random"]["win_rate"] = round(res_random.win_rate(), 6)
            result_entry["vs_random"]["total"] = res_random.total()
            if args.record_losses and loss_games_random:
                result_entry["vs_random"]["loss_games"] = loss_games_random

        if args.games_smart > 0:
            res_smart, loss_games_smart = run_series(exhaustive_agent, smart_agent, args.games_smart, rng, record_losses=args.record_losses)
            print(f"  Against Smart: {res_smart}")
            result_entry["vs_smart"] = res_smart.as_dict()
            result_entry["vs_smart"]["win_rate"] = round(res_smart.win_rate(), 6)
            result_entry["vs_smart"]["total"] = res_smart.total()
            if args.record_losses and loss_games_smart:
                result_entry["vs_smart"]["loss_games"] = loss_games_smart

        if args.games_minimax > 0:
            res_minimax, loss_games_minimax = run_series(exhaustive_agent, minimax_agent, args.games_minimax, rng, record_losses=args.record_losses)
            print(f"  Against Minimax: {res_minimax}")
            result_entry["vs_minimax"] = res_minimax.as_dict()
            result_entry["vs_minimax"]["win_rate"] = round(res_minimax.win_rate(), 6)
            result_entry["vs_minimax"]["total"] = res_minimax.total()
            if args.record_losses and loss_games_minimax:
                result_entry["vs_minimax"]["loss_games"] = loss_games_minimax
        
        results.append(result_entry)

    # Save results to file if specified
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)
        with open(args.out_json, 'w') as f:
            json.dump({
                "config": {
                    "games_random": args.games_random,
                    "games_smart": args.games_smart,
                    "games_minimax": args.games_minimax,
                    "seed": args.seed,
                    "states_file": args.states
                },
                "results": results
            }, f, indent=2)
        print(f"\nResults saved to: {args.out_json}")


if __name__ == "__main__":
    main()

