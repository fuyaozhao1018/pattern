# ttt/n9_exhaustive_battle.py
"""Battle evaluation for 9x9 connect-4 exhaustive policy against baselines.

Usage:
  python -m ttt.n9_exhaustive_battle \
    --probs data/n9_dual/n9_exhaustive_probs.json \
    --lambda_draw 0.3 \
    --games_random 500 \
    --out_json out/runs/n9_battle_results.json

"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# 9x9 Connect-4 (K=4) constants
N = 9
K = 4

Board = List[str]  # 81 chars: 'X', 'O', or ' '


def new_board() -> Board:
    return [' '] * (N * N)


def legal_moves(board: Board) -> List[int]:
    return [i for i, c in enumerate(board) if c == ' ']


def switch_turn(turn: str) -> str:
    return 'O' if turn == 'X' else 'X'


def format_board(board: Board) -> str:
    """Format board as 9x9 grid with borders."""
    rows = []
    rows.append("+---" * N + "+")
    for r in range(N):
        row_chars = [board[r * N + c] for c in range(N)]
        rows.append(f"| {' | '.join(row_chars)} |")
        rows.append("+---" * N + "+")
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
    # diagonals ↘
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
    # anti-diagonals ↙
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
        
        # Load probability map - only keep per_move to save memory
        print(f"  Loading probability map (lambda_draw={lambda_draw})...")
        with open(probs_file, 'r') as f:
            states = json.load(f)
        
        for st in states:
            key = (tuple(st['board']), st['turn'])
            self._state_map[key] = {'per_move': st.get('per_move', {})}
        
        print(f"  Loaded {len(self._state_map):,} states")

    def select_move(self, board: Board, turn: str) -> int:
        key = (tuple(board), turn)
        state = self._state_map.get(key)
        legal = legal_moves(board)
        
        if state is None or not state.get('per_move'):
            # Fallback: random from legal moves
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
    """Agent that: 1) Wins if possible, 2) Blocks opponent, 3) Prefers center."""
    
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

        # 3. Heuristic: prefer center
        center = N * N // 2
        if center in legal:
            return center

        # 4. Random
        return self._rng.choice(legal)


@dataclass
class SeriesResult:
    wins: int = 0
    losses: int = 0
    draws: int = 0

    def record(self, outcome: str, pov: str) -> None:
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
    board = new_board()
    turn = 'X'
    agents = {'X': agent_x, 'O': agent_o}
    history = [] if record_history else None
    
    move_count = 0
    max_moves = N * N
    
    while move_count < max_moves:
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
        move_count += 1
    
    # Max moves reached - force draw
    if record_history:
        history.append((board[:], None, None))
    return 'D', history


def run_series(main_agent: Agent, opponent: Agent, games: int, rng: random.Random, 
               record_losses: bool = False) -> Tuple:
    result = SeriesResult()
    loss_games = []
    
    for i in range(games):
        if i % 10 == 0:
            print(f"  Game {i}/{games}...", end='\r')
        
        if i % 2 == 0:
            outcome, history = play_game(main_agent, opponent, record_history=record_losses)
            result.record(outcome, pov='X')
            if record_losses and outcome == 'O':
                loss_games.append({
                    "game_id": i,
                    "main_agent_played_as": "X",
                    "outcome": "loss",
                    "history": [{"board": list(b), "move": m, "turn": t} 
                               for b, m, t in history]
                })
        else:
            outcome, history = play_game(opponent, main_agent, record_history=record_losses)
            result.record(outcome, pov='O')
            if record_losses and outcome == 'X':
                loss_games.append({
                    "game_id": i,
                    "main_agent_played_as": "O",
                    "outcome": "loss",
                    "history": [{"board": list(b), "move": m, "turn": t} 
                               for b, m, t in history]
                })
    
    print()  # Clear progress line
    return result, loss_games


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate 9x9 exhaustive policy against baselines.")
    parser.add_argument("--probs", required=True, help="Path to exhaustive probability map.")
    parser.add_argument("--games_random", type=int, default=100, help="Number of games vs random.")
    parser.add_argument("--games_smart", type=int, default=0, help="Number of games vs smart.")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed.")
    parser.add_argument("--lambda_draw", type=float, default=0.3, help="Draw weight.")
    parser.add_argument("--out_json", type=str, default=None, help="Output JSON file.")
    parser.add_argument("--record_losses", action="store_true", help="Record loss games.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    random_agent = RandomAgent(rng=rng)
    smart_agent = SmartAgent(rng=rng)

    print(f"Loading probability map from: {args.probs}")
    exhaustive_agent = ExhaustiveAgent(args.probs, rng=rng, lambda_draw=args.lambda_draw)
    
    print()
    print(f"λ = {args.lambda_draw} ({args.lambda_draw*100:.0f}%)")
    print("=" * 80)
    
    result_entry = {
        "lambda_draw": args.lambda_draw,
        "lambda_draw_percent": round(args.lambda_draw * 100, 0)
    }
    
    if args.games_random > 0:
        print(f"Playing {args.games_random} games vs Random...")
        res_random, loss_games_random = run_series(exhaustive_agent, random_agent, 
                                                   args.games_random, rng, 
                                                   record_losses=args.record_losses)
        print(f"  Result: {res_random}")
        result_entry["vs_random"] = res_random.as_dict()
        result_entry["vs_random"]["win_rate"] = round(res_random.win_rate(), 6)
        result_entry["vs_random"]["total"] = res_random.total()
        if args.record_losses and loss_games_random:
            result_entry["vs_random"]["loss_games"] = loss_games_random

    if args.games_smart > 0:
        print(f"Playing {args.games_smart} games vs Smart...")
        res_smart, loss_games_smart = run_series(exhaustive_agent, smart_agent, 
                                                 args.games_smart, rng, 
                                                 record_losses=args.record_losses)
        print(f"  Result: {res_smart}")
        result_entry["vs_smart"] = res_smart.as_dict()
        result_entry["vs_smart"]["win_rate"] = round(res_smart.win_rate(), 6)
        result_entry["vs_smart"]["total"] = res_smart.total()
        if args.record_losses and loss_games_smart:
            result_entry["vs_smart"]["loss_games"] = loss_games_smart

    if args.out_json:
        output = {
            "config": {
                "probs_file": args.probs,
                "games_random": args.games_random,
                "games_smart": args.games_smart,
                "seed": args.seed
            },
            "results": [result_entry]
        }
        with open(args.out_json, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {args.out_json}")


if __name__ == "__main__":
    main()
