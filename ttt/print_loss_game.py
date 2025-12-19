#!/usr/bin/env python3
"""
Print a specific loss game from the battle results JSON file.
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, Any, List


def print_game(game: Dict[str, Any], game_index: int = 0) -> None:
    """Print a single loss game in a readable format."""
    print("=" * 80)
    print(f"Loss Game #{game_index + 1}")
    print("=" * 80)
    print(f"Game ID: {game['game_id']}")
    print(f"Main Agent Played As: {game['main_agent_played_as']}")
    print(f"Outcome: {game['outcome']}")
    print()
    
    history = game.get('history', [])
    if not history:
        print("No history recorded.")
        return
    
    print("Game History:")
    print("-" * 80)
    
    for step_idx, step in enumerate(history):
        if step.get('final_state'):
            print(f"\nStep {step_idx + 1}: FINAL STATE")
        else:
            move = step.get('move')
            turn_symbol = step.get('turn_symbol', '?')
            print(f"\nStep {step_idx + 1}: {turn_symbol} plays at position {move}")
        
        board_str = step.get('board_str', '')
        if board_str:
            print(board_str)
        else:
            # Fallback: format from board array
            board = step.get('board', [])
            if board:
                symbols = {+1: 'X', -1: 'O', 0: '.'}
                print("+---+---+---+")
                for r in range(3):
                    row_chars = [symbols[board[r * 3 + c]] for c in range(3)]
                    print(f"| {' | '.join(row_chars)} |")
                    print("+---+---+---+")
        print()


def main():
    parser = argparse.ArgumentParser(description="Print loss games from battle results.")
    parser.add_argument("--json_file", required=True, help="Path to battle results JSON file.")
    parser.add_argument("--game_index", type=int, default=0, help="Index of loss game to print (0-based, default: 0 = first game).")
    parser.add_argument("--vs", choices=['random', 'minimax'], default='random', help="Which opponent's losses to show.")
    args = parser.parse_args()
    
    with open(args.json_file) as f:
        data = json.load(f)
    
    # Get the first result (assuming single lambda_draw test)
    if not data.get('results'):
        print("No results found in file.")
        return
    
    result = data['results'][0]
    
    # Get loss games
    vs_key = f"vs_{args.vs}"
    if vs_key not in result:
        print(f"No {args.vs} results found.")
        return
    
    vs_data = result[vs_key]
    loss_games = vs_data.get('loss_games', [])
    
    if not loss_games:
        print(f"No loss games found for {args.vs} opponent.")
        return
    
    if args.game_index >= len(loss_games):
        print(f"Game index {args.game_index} out of range. Total loss games: {len(loss_games)}")
        return
    
    game = loss_games[args.game_index]
    print_game(game, args.game_index)
    
    print(f"\nTotal loss games: {len(loss_games)}")
    print(f"Use --game_index to view other games (0 to {len(loss_games) - 1})")


if __name__ == '__main__':
    main()

