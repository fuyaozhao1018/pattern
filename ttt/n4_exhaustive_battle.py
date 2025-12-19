# ttt/n4_exhaustive_battle.py
"""Battle evaluation for 4x4 connect-4 exhaustive policy against baselines.

Usage:
  python -m ttt.n4_exhaustive_battle \
    --probs data/n4_dual/n4_exhaustive_probs.json \
    --sweep_lambda \
    --games_random 1000 \
    --games_minimax 50 \
    --out_json out/runs/n4_battle_results.json

"""
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# 4x4 Connect-4 (K=4) constants
N = 4
K = 4

Board = List[str]  # 16 chars: 'X', 'O', or ' '


def new_board() -> Board:
    return [' '] * (N * N)


def legal_moves(board: Board) -> List[int]:
    return [i for i, c in enumerate(board) if c == ' ']


def switch_turn(turn: str) -> str:
    return 'O' if turn == 'X' else 'X'


def format_board(board: Board) -> str:
    """Format board as 4x4 grid with borders."""
    rows = []
    rows.append("+---+---+---+---+")
    for r in range(N):
        row_chars = [board[r * N + c] for c in range(N)]
        rows.append(f"| {' | '.join(row_chars)} |")
        rows.append("+---+---+---+---+")
    return '\n'.join(rows)


def line_winner(seq: List[str]) -> Optional[str]:
    """Check if any player has K consecutive in seq."""
    for player in ('X', 'O'):
        run = 0
        for c in seq:
            run = run + 1 if c == player else 0
            if run >= K:
                return player
    return None


def check_winner(board: Board) -> Optional[str]:
    """Return 'X', 'O', or None."""
    # rows
    for r in range(N):
        w = line_winner([board[r * N + c] for c in range(N)])
        if w: return w
    # cols
    for c in range(N):
        w = line_winner([board[r * N + c] for r in range(N)])
        if w: return w
    # diag ↘
    for sr in range(N):
        seq = []
        r, c = sr, 0
        while r < N and c < N:
            seq.append(board[r * N + c])
            r += 1
            c += 1
        if len(seq) >= K:
            w = line_winner(seq)
            if w: return w
    for sc in range(1, N):
        seq = []
        r, c = 0, sc
        while r < N and c < N:
            seq.append(board[r * N + c])
            r += 1
            c += 1
        if len(seq) >= K:
            w = line_winner(seq)
            if w: return w
    # anti-diag ↙
    for sr in range(N):
        seq = []
        r, c = sr, N - 1
        while r < N and c >= 0:
            seq.append(board[r * N + c])
            r += 1
            c -= 1
        if len(seq) >= K:
            w = line_winner(seq)
            if w: return w
    for sc in range(N - 2, -1, -1):
        seq = []
        r, c = 0, sc
        while r < N and c >= 0:
            seq.append(board[r * N + c])
            r += 1
            c -= 1
        if len(seq) >= K:
            w = line_winner(seq)
            if w: return w
    return None


def is_full(board: Board) -> bool:
    return all(c != ' ' for c in board)


def evaluate_terminal(board: Board) -> Optional[str]:
    """Return 'X', 'O', 'D' (draw), or None (not terminal)."""
    w = check_winner(board)
    if w is not None:
        return w
    if is_full(board):
        return 'D'
    return None


class Agent:
    def select_move(self, board: Board, turn: str) -> int:
        raise NotImplementedError


class ExhaustiveAgent(Agent):
    """Agent using exhaustive probability map with configurable lambda_draw."""
    
    def __init__(self, probs_file: str, rng: random.Random, epsilon: float = 1e-12, lambda_draw: float = 0.5):
        self._rng = rng
        self._epsilon = epsilon
        self._lambda_draw = lambda_draw
        self._state_map: Dict[Tuple[Tuple[str, ...], str], Dict] = {}
        
        # Load probability map - only keep per_move data to save memory
        print(f"  Loading probability map (lambda_draw={lambda_draw})...")
        with open(probs_file, 'r') as f:
            states = json.load(f)
        
        for st in states:
            key = (tuple(st['board']), st['turn'])
            # Only store per_move to reduce memory footprint
            self._state_map[key] = {'per_move': st.get('per_move', {})}

    def select_move(self, board: Board, turn: str) -> int:
        key = (tuple(board), turn)
        state = self._state_map.get(key)
        legal = legal_moves(board)
        
        if state is None or not state.get('per_move'):
            return self._rng.choice(legal)

        best_score = None
        best_moves: List[int] = []
        
        for move_str, stats in state['per_move'].items():
            move = int(move_str)
            if move not in legal:
                continue
            
            # Compute mixed effectiveness: peff = win_prob + lambda_draw * draw_prob
            win_prob = stats.get('win_prob', 0.0)
            draw_prob = stats.get('draw_prob', 0.0)
            score = win_prob + self._lambda_draw * draw_prob
            
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

    def select_move(self, board: Board, turn: str) -> int:
        return self._rng.choice(legal_moves(board))


class SmartAgent(Agent):
    """Agent that: 1) Wins if possible, 2) Blocks opponent, 3) Prefers center/corners."""
    
    def __init__(self, rng: random.Random):
        self._rng = rng

    def _can_win(self, board: Board, move: int, player: str) -> bool:
        """Check if playing at move would win for player."""
        board[move] = player
        result = check_winner(board)
        board[move] = ' '
        return result == player

    def select_move(self, board: Board, turn: str) -> int:
        legal = legal_moves(board)
        if not legal:
            return 0

        opponent = switch_turn(turn)

        # 1. Win if possible
        for move in legal:
            if self._can_win(board, move, turn):
                return move

        # 2. Block opponent from winning
        for move in legal:
            if self._can_win(board, move, opponent):
                return move

        # 3. Heuristic: prefer center positions, then corners, then edges
        # For 4x4: centers are 5,6,9,10; corners are 0,3,12,15
        centers = [5, 6, 9, 10]
        corners = [0, 3, 12, 15]
        
        for pos in centers:
            if pos in legal:
                return pos
        
        for pos in corners:
            if pos in legal:
                return pos

        # Fallback: random
        return self._rng.choice(legal)


class MinimaxAgent(Agent):
    """Minimax agent with alpha-beta pruning and cache size limit."""
    
    def __init__(self, max_cache_size: int = 50000):
        self._cache: Dict[Tuple[Tuple[str, ...], str], int] = {}
        self._max_cache_size = max_cache_size

    def _solve(self, board: Board, turn: str, alpha: int = -2, beta: int = 2) -> int:
        """Return score from turn's perspective: +1 win, 0 draw, -1 loss."""
        key = (tuple(board), turn)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        terminal = evaluate_terminal(board)
        if terminal is not None:
            if terminal == 'D':
                self._cache[key] = 0
                return 0
            # terminal is 'X' or 'O'
            val = 1 if terminal == turn else -1
            self._cache[key] = val
            return val

        best = -2
        for move in legal_moves(board):
            board[move] = turn
            score = -self._solve(board, switch_turn(turn), -beta, -alpha)
            board[move] = ' '
            if score > best:
                best = score
            if best >= beta:
                break
            if best > alpha:
                alpha = best

        # Limit cache size to prevent OOM
        if len(self._cache) < self._max_cache_size:
            self._cache[key] = best
        return best

    def select_move(self, board: Board, turn: str) -> int:
        best_value = -2
        best_moves: List[int] = []
        for move in legal_moves(board):
            board[move] = turn
            score = -self._solve(board, switch_turn(turn))
            board[move] = ' '
            if score > best_value:
                best_value = score
                best_moves = [move]
            elif score == best_value:
                best_moves.append(move)
        # Deterministic: prefer smallest index on ties
        return min(best_moves)


@dataclass
class SeriesResult:
    wins: int = 0
    losses: int = 0
    draws: int = 0

    def record(self, outcome: str, pov: str) -> None:
        """
        outcome: 'X', 'O', or 'D'
        pov: 'X' or 'O' (tracking agent's side)
        """
        if outcome == 'D':
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


def play_game(agent_x: Agent, agent_o: Agent, record_history: bool = False) -> Tuple:
    """
    Returns (outcome, history) where:
    - outcome: 'X', 'O', or 'D'
    - history: list of (board_state, move, turn) if record_history=True, else None
    """
    board = new_board()
    turn = 'X'
    agents = {'X': agent_x, 'O': agent_o}
    history = [] if record_history else None
    
    while True:
        move = agents[turn].select_move(board, turn)
        if record_history:
            history.append((board[:], move, turn))
        board[move] = turn
        outcome = evaluate_terminal(board)
        if outcome is not None:
            if record_history:
                history.append((board[:], None, None))
            return outcome, history
        turn = switch_turn(turn)


def run_series(main_agent: Agent, opponent: Agent, games: int, rng: random.Random, 
               record_losses: bool = False) -> Tuple:
    """
    Returns (result, loss_games) where:
    - result: SeriesResult
    - loss_games: list of game histories for losses (if record_losses=True)
    """
    result = SeriesResult()
    loss_games = []
    
    for i in range(games):
        if i % 2 == 0:
            # main agent plays as X
            outcome, history = play_game(main_agent, opponent, record_history=record_losses)
            result.record(outcome, pov='X')
            if record_losses and outcome == 'O':  # main agent lost
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
            # main agent plays as O
            outcome, history = play_game(opponent, main_agent, record_history=record_losses)
            result.record(outcome, pov='O')
            if record_losses and outcome == 'X':  # main agent lost
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
    parser = argparse.ArgumentParser(description="Evaluate 4x4 exhaustive policy against baselines.")
    parser.add_argument("--probs", required=True, help="Path to exhaustive probability map (n4_exhaustive_probs.json).")
    parser.add_argument("--games_random", type=int, default=1000, help="Number of games vs random opponent.")
    parser.add_argument("--games_smart", type=int, default=0, help="Number of games vs smart opponent.")
    parser.add_argument("--games_minimax", type=int, default=50, help="Number of games vs minimax opponent.")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed.")
    parser.add_argument("--lambda_draw", type=float, default=None, help="Single lambda_draw value to test.")
    parser.add_argument("--sweep_lambda", action="store_true", help="Test lambda_draw from 0.0 to 1.0 (0%% to 100%%).")
    parser.add_argument("--out_json", type=str, default=None, help="Output JSON file to save results.")
    parser.add_argument("--record_losses", action="store_true", help="Record detailed history of lost games.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    random_agent = RandomAgent(rng=rng)
    smart_agent = SmartAgent(rng=rng)
    minimax_agent = MinimaxAgent()

    # Determine lambda_draw values to test
    if args.sweep_lambda:
        # Test from 0% to 100% (0.0 to 1.0) in steps of 0.1
        lambda_values = [round(i * 0.1, 1) for i in range(11)]  # 0.0, 0.1, ..., 1.0
    elif args.lambda_draw is not None:
        lambda_values = [args.lambda_draw]
    else:
        lambda_values = [0.5]  # default

    print(f"Loading probability map from: {args.probs}")
    print(f"Testing {len(lambda_values)} lambda_draw value(s)")
    print("=" * 80)

    results = []
    for lambda_draw in lambda_values:
        print(f"\nlambda_draw = {lambda_draw:.1f} ({lambda_draw*100:.0f}%)")
        print("-" * 80)
        
        exhaustive_agent = ExhaustiveAgent(args.probs, rng=rng, lambda_draw=lambda_draw)
        
        result_entry = {
            "lambda_draw": lambda_draw,
            "lambda_draw_percent": round(lambda_draw * 100, 0)
        }
        
        if args.games_random > 0:
            res_random, loss_games_random = run_series(exhaustive_agent, random_agent, args.games_random, rng, 
                                                       record_losses=args.record_losses)
            print(f"  Against Random: {res_random}")
            result_entry["vs_random"] = res_random.as_dict()
            result_entry["vs_random"]["win_rate"] = round(res_random.win_rate(), 6)
            result_entry["vs_random"]["total"] = res_random.total()
            if args.record_losses and loss_games_random:
                result_entry["vs_random"]["loss_games"] = loss_games_random

        if args.games_smart > 0:
            res_smart, loss_games_smart = run_series(exhaustive_agent, smart_agent, args.games_smart, rng, 
                                                     record_losses=args.record_losses)
            print(f"  Against Smart: {res_smart}")
            result_entry["vs_smart"] = res_smart.as_dict()
            result_entry["vs_smart"]["win_rate"] = round(res_smart.win_rate(), 6)
            result_entry["vs_smart"]["total"] = res_smart.total()
            if args.record_losses and loss_games_smart:
                result_entry["vs_smart"]["loss_games"] = loss_games_smart

        if args.games_minimax > 0:
            res_minimax, loss_games_minimax = run_series(exhaustive_agent, minimax_agent, args.games_minimax, rng, 
                                                         record_losses=args.record_losses)
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
                    "probs_file": args.probs,
                    "games_random": args.games_random,
                    "games_smart": args.games_smart,
                    "games_minimax": args.games_minimax,
                    "seed": args.seed
                },
                "results": results
            }, f, indent=2)
        print(f"\nResults saved to: {args.out_json}")
    
    print("\n" + "=" * 80)
    print("Summary:")
    print("-" * 80)
    for entry in results:
        lam = entry['lambda_draw']
        print(f"\nλ={lam:.1f} ({lam*100:.0f}%):")
        if 'vs_random' in entry:
            wr = entry['vs_random']['win_rate']
            print(f"  vs Random: {wr:.1%} win rate")
        if 'vs_smart' in entry:
            wr = entry['vs_smart']['win_rate']
            print(f"  vs Smart: {wr:.1%} win rate")
        if 'vs_minimax' in entry:
            wr = entry['vs_minimax']['win_rate']
            print(f"  vs Minimax: {wr:.1%} win rate")


if __name__ == "__main__":
    main()
