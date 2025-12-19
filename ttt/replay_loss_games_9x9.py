# ttt/replay_loss_games_9x9.py
"""
Replay and visualize 9×9 loss games from battle test results.

Usage:
  python -m ttt.replay_loss_games_9x9 out/runs/n9_conv_lambda0.3.json --opponent random
  python -m ttt.replay_loss_games_9x9 out/runs/n9_conv_lambda0.3.json --game_id 5 --delay 0.5
  python -m ttt.replay_loss_games_9x9 out/runs/n9_conv_lambda0.3.json --limit 3 --style simple
"""
from __future__ import annotations
import argparse
import json
import os
import time
from typing import List, Optional


def position_name_9x9(idx: int) -> str:
    """Convert index to A1-I9 format."""
    row = idx // 9
    col = idx % 9
    row_name = str(row + 1)
    col_name = chr(ord('A') + col)
    return f"{col_name}{row_name}"


def format_board_fancy_9x9(board: List[str], last_move: Optional[int] = None) -> str:
    """Format 9×9 board with Unicode box-drawing and color highlights."""
    N = 9
    
    # ANSI color codes
    X_COLOR = "\033[94m"  # Blue
    O_COLOR = "\033[91m"  # Red
    LAST_MOVE_COLOR = "\033[93m"  # Yellow
    RESET = "\033[0m"
    
    def colorize(char: str, pos: int) -> str:
        if pos == last_move:
            return f"{LAST_MOVE_COLOR}{char}{RESET}"
        elif char == 'X':
            return f"{X_COLOR}{char}{RESET}"
        elif char == 'O':
            return f"{O_COLOR}{char}{RESET}"
        else:
            return char
    
    rows = []
    # Column labels
    col_labels = "   " + "   ".join(chr(ord('A') + c) for c in range(N))
    rows.append(col_labels)
    
    # Top border
    rows.append("  ┏" + "━━━┳" * (N - 1) + "━━━┓")
    
    for r in range(N):
        # Row content
        row_chars = []
        for c in range(N):
            pos = r * N + c
            char = board[pos]
            row_chars.append(f" {colorize(char, pos)} ")
        row_str = f"{r + 1} ┃" + "┃".join(row_chars) + "┃"
        rows.append(row_str)
        
        # Middle border
        if r < N - 1:
            rows.append("  ┣" + "━━━╋" * (N - 1) + "━━━┫")
    
    # Bottom border
    rows.append("  ┗" + "━━━┻" * (N - 1) + "━━━┛")
    
    return '\n'.join(rows)


def format_board_simple_9x9(board: List[str]) -> str:
    """Simple ASCII board without fancy unicode."""
    N = 9
    rows = []
    rows.append("+---" * N + "+")
    for r in range(N):
        row_chars = [board[r * N + c] for c in range(N)]
        rows.append("| " + " | ".join(row_chars) + " |")
        rows.append("+---" * N + "+")
    return '\n'.join(rows)


def replay_game(game_entry: dict, delay: float = 1.0, style: str = "fancy") -> None:
    """Replay a single game step by step with animation."""
    game_id = game_entry.get("game_id", "?")
    # Support multiple field names for different battle script versions
    played_as = game_entry.get("strategic_agent_played_as") or game_entry.get("main_agent_played_as", "?")
    history = game_entry.get("history", [])
    
    print("\n" + "=" * 80)
    print(f"GAME #{game_id} - Convolution Agent played as {played_as} - LOSS")
    print("=" * 80)
    
    if not history:
        print("No history recorded for this game.")
        return
    
    for step_idx, entry in enumerate(history):
        board = entry.get("board", [])
        move = entry.get("move", None)
        turn = entry.get("turn", "?")
        is_final = entry.get("final_state", False)
        
        # Clear screen (optional, works in most terminals)
        # print("\033[2J\033[H", end="")
        
        print(f"\n--- Step {step_idx + 1} ---")
        if move is not None:
            move_name = position_name_9x9(move)
            print(f"Player {turn} plays at {move_name} (index {move})")
        else:
            print("Final board state")
        
        if style == "fancy":
            print(format_board_fancy_9x9(board, last_move=move))
        else:
            print(format_board_simple_9x9(board))
        
        if not is_final:
            time.sleep(delay)
    
    print("\n" + "=" * 80)
    print(f"END OF GAME #{game_id}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Replay 9×9 loss games with animation")
    parser.add_argument("json_file", help="Path to battle results JSON")
    parser.add_argument("--game_id", type=int, default=None, help="Replay specific game ID")
    parser.add_argument("--opponent", type=str, default=None, 
                       help="Filter by opponent type (random, smart)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of games to replay")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between moves in seconds")
    parser.add_argument("--style", type=str, choices=["fancy", "simple"], default="fancy",
                       help="Board display style")
    args = parser.parse_args()
    
    if not os.path.exists(args.json_file):
        print(f"Error: File not found: {args.json_file}")
        return
    
    with open(args.json_file, 'r') as f:
        data = json.load(f)
    
    # Extract loss games (support both formats)
    all_loss_games = []
    
    # Format 1: Separate losses file with "loss_games" key
    if "loss_games" in data:
        for game in data["loss_games"]:
            # Use "opponent" field from losses file
            game["opponent_type"] = game.get("opponent", "unknown")
            all_loss_games.append(game)
    
    # Format 2: Battle results file with nested loss_games
    elif "results" in data:
        results = data.get("results", [])
        for result in results:
            # From vs_random
            if "vs_random" in result and "loss_games" in result["vs_random"]:
                for game in result["vs_random"]["loss_games"]:
                    game["opponent_type"] = "random"
                    all_loss_games.append(game)
            
            # From vs_smart
            if "vs_smart" in result and "loss_games" in result["vs_smart"]:
                for game in result["vs_smart"]["loss_games"]:
                    game["opponent_type"] = "smart"
                    all_loss_games.append(game)
    
    if not all_loss_games:
        print(f"No loss games found in {args.json_file}")
        return
    
    # Filter by game_id
    if args.game_id is not None:
        all_loss_games = [g for g in all_loss_games if g.get("game_id") == args.game_id]
        if not all_loss_games:
            print(f"Game ID {args.game_id} not found")
            return
    
    # Filter by opponent
    if args.opponent:
        all_loss_games = [g for g in all_loss_games if g.get("opponent_type") == args.opponent.lower()]
        if not all_loss_games:
            print(f"No loss games found against {args.opponent}")
            return
    
    # Limit number of games
    if args.limit:
        all_loss_games = all_loss_games[:args.limit]
    
    print(f"\nFound {len(all_loss_games)} loss game(s) to replay")
    
    # Replay each game
    for game in all_loss_games:
        replay_game(game, delay=args.delay, style=args.style)
        
        # Ask to continue if multiple games
        if len(all_loss_games) > 1 and game != all_loss_games[-1]:
            resp = input("Press Enter to continue, or 'q' to quit: ")
            if resp.lower() == 'q':
                break


if __name__ == "__main__":
    main()
