# ttt/n9_convolution_battle.py
"""
Battle evaluation for 9×9 convolution agent (using 4×4 exhaustive data).

Usage:
  python -m ttt.n9_convolution_battle \
    --probs data/n4_dual/n4_exhaustive_probs.json \
    --lambda_draw 0.3 \
    --games_random 100 \
    --games_smart 50 \
    --out_json out/runs/n9_conv_lambda0.3.json \
    --record_losses \
    --seed 2025
"""
from __future__ import annotations
import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from ttt.n9_convolution_agent import (
    ConvolutionAgent, RandomAgent9x9, SmartAgent9x9,
    play_game_9x9, N9
)


def format_board_9x9(board: List[str]) -> str:
    """Format 9×9 board with borders."""
    rows = []
    rows.append("+---" * 9 + "+")
    for r in range(N9):
        row_chars = [board[r * N9 + c] for c in range(N9)]
        rows.append("| " + " | ".join(row_chars) + " |")
        rows.append("+---" * 9 + "+")
    return '\n'.join(rows)


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


def run_series(main_agent, opponent, games: int, rng: random.Random, 
               record_losses: bool = False) -> Tuple:
    """Run a series of games."""
    result = SeriesResult()
    loss_games = []
    
    for i in range(games):
        if i % 2 == 0:
            # main agent plays as X
            outcome, history = play_game_9x9(main_agent, opponent, record_history=record_losses)
            result.record(outcome, pov='X')
            if record_losses and outcome == 'O':  # main agent lost
                formatted_history = []
                if history:
                    for board, move, turn in history:
                        entry = {
                            "board": list(board),
                            "board_str": format_board_9x9(board)
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
            outcome, history = play_game_9x9(opponent, main_agent, record_history=record_losses)
            result.record(outcome, pov='O')
            if record_losses and outcome == 'X':  # main agent lost
                formatted_history = []
                if history:
                    for board, move, turn in history:
                        entry = {
                            "board": list(board),
                            "board_str": format_board_9x9(board)
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
    parser = argparse.ArgumentParser(description="Evaluate 9×9 convolution policy against baselines.")
    parser.add_argument("--probs", required=True, help="Path to 4×4 exhaustive probability map.")
    parser.add_argument("--games_random", type=int, default=100, help="Number of games vs random opponent.")
    parser.add_argument("--games_smart", type=int, default=0, help="Number of games vs smart opponent.")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed.")
    parser.add_argument("--lambda_draw", type=float, default=0.3, help="Lambda draw value to test.")
    parser.add_argument("--out_json", type=str, default=None, help="Output JSON file to save results.")
    parser.add_argument("--record_losses", action="store_true", help="Record detailed history of lost games.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    
    print(f"Loading convolution agent with λ={args.lambda_draw}...")
    conv_agent = ConvolutionAgent(args.probs, lambda_draw=args.lambda_draw, rng=rng)
    
    random_agent = RandomAgent9x9(rng=rng)
    smart_agent = SmartAgent9x9(rng=rng)

    print(f"\nTesting λ={args.lambda_draw} ({args.lambda_draw*100:.0f}% draw weight)")
    print("=" * 80)
    
    result_entry = {
        "lambda_draw": args.lambda_draw,
        "lambda_draw_percent": round(args.lambda_draw * 100, 0)
    }
    
    all_loss_games = []
    
    if args.games_random > 0:
        print(f"\nPlaying {args.games_random} games vs Random...")
        res_random, loss_games_random = run_series(conv_agent, random_agent, args.games_random, rng, 
                                                   record_losses=args.record_losses)
        print(f"  Against Random: {res_random}")
        result_entry["vs_random"] = res_random.as_dict()
        result_entry["vs_random"]["win_rate"] = round(res_random.win_rate(), 6)
        result_entry["vs_random"]["total"] = res_random.total()
        if args.record_losses and loss_games_random:
            for game in loss_games_random:
                game["opponent"] = "random"
            all_loss_games.extend(loss_games_random)

    if args.games_smart > 0:
        print(f"\nPlaying {args.games_smart} games vs Smart...")
        res_smart, loss_games_smart = run_series(conv_agent, smart_agent, args.games_smart, rng, 
                                                 record_losses=args.record_losses)
        print(f"  Against Smart: {res_smart}")
        result_entry["vs_smart"] = res_smart.as_dict()
        result_entry["vs_smart"]["win_rate"] = round(res_smart.win_rate(), 6)
        result_entry["vs_smart"]["total"] = res_smart.total()
        if args.record_losses and loss_games_smart:
            for game in loss_games_smart:
                game["opponent"] = "smart"
            all_loss_games.extend(loss_games_smart)

    # Save results to file if specified
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)
        with open(args.out_json, 'w') as f:
            json.dump({
                "config": {
                    "probs_file": args.probs,
                    "games_random": args.games_random,
                    "games_smart": args.games_smart,
                    "seed": args.seed,
                    "board_size": "9x9",
                    "method": "convolution_4x4"
                },
                "results": [result_entry]
            }, f, indent=2)
        print(f"\nResults saved to: {args.out_json}")
        
        # Save losses to separate file
        if args.record_losses and all_loss_games:
            losses_file = args.out_json.replace('.json', '_losses.json')
            with open(losses_file, 'w') as f:
                json.dump({
                    "lambda_draw": args.lambda_draw,
                    "total_losses": len(all_loss_games),
                    "loss_games": all_loss_games
                }, f, indent=2)
            print(f"Loss games saved to: {losses_file}")
    
    print("\n" + "=" * 80)
    print("Summary:")
    print("-" * 80)
    print(f"λ={args.lambda_draw:.1f} ({args.lambda_draw*100:.0f}%):")
    if 'vs_random' in result_entry:
        wr = result_entry['vs_random']['win_rate']
        print(f"  vs Random: {wr:.1%} win rate")
    if 'vs_smart' in result_entry:
        wr = result_entry['vs_smart']['win_rate']
        print(f"  vs Smart: {wr:.1%} win rate")
    if args.record_losses:
        print(f"  Total losses recorded: {len(all_loss_games)}")


if __name__ == "__main__":
    main()
