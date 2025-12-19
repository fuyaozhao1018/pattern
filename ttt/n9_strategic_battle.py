#!/usr/bin/env python3
"""
9×9 Connect-4 Battle Test for Strategic Convolution Agent
Features:
- Opening area restriction (center region only for first move)
- Shape-aware heuristics (connectivity bonuses)
- Must-win/must-block tactical rules
- Convolution evaluation using 4×4 exhaustive data

Tests Strategic agent vs Random opponent over 500 games.
Saves loss games to separate JSON file for analysis.
"""

import json
import random
import argparse
from pathlib import Path
from typing import List
from ttt.n9_strategic_convolution_agent import StrategicConvolutionAgent
from ttt.n9_convolution_agent import RandomAgent9x9, SmartAgent9x9, check_winner_9x9


def format_board_9x9(board: List[str]) -> str:
    """Format 9×9 board as string with fancy borders."""
    lines = []
    lines.append("  +" + "---+" * 9)
    
    for row in range(9):
        row_str = f"{row} |"
        for col in range(9):
            idx = row * 9 + col
            cell = board[idx] if board[idx] != ' ' else '.'
            row_str += f" {cell} |"
        lines.append(row_str)
        lines.append("  +" + "---+" * 9)
    
    lines.append("    " + "   ".join("ABCDEFGHI"))
    return "\n".join(lines)


def play_game(agent1, agent2, agent1_symbol='X', agent2_symbol='O', record_history=False):
    """
    Play one game between two agents.
    
    Returns:
        winner: 'X', 'O', or 'draw'
        history: list of board states (if record_history=True)
    """
    board = [' '] * 81  # 9×9 = 81
    history = []
    
    for turn_idx in range(81):
        current_player = agent1_symbol if turn_idx % 2 == 0 else agent2_symbol
        current_agent = agent1 if turn_idx % 2 == 0 else agent2
        
        if record_history:
            history.append({
                'step': turn_idx,
                'board': board.copy(),
                'board_str': format_board_9x9(board),
                'turn': current_player,
                'turn_symbol': current_player
            })
        
        # Agent selects move
        move = current_agent.select_move(board, current_player)
        
        if move is None:
            # No valid moves (should not happen before board is full)
            return 'draw', history
        
        # Apply move
        board[move] = current_player
        
        # Check winner
        winner = check_winner_9x9(board)
        if winner:
            if record_history:
                history.append({
                    'step': turn_idx + 1,
                    'board': board.copy(),
                    'board_str': format_board_9x9(board),
                    'turn': current_player,
                    'turn_symbol': current_player,
                    'move': move,
                    'winner': winner
                })
            return winner, history
    
    # Board full, draw
    if record_history:
        history.append({
            'step': 81,
            'board': board.copy(),
            'board_str': format_board_9x9(board),
            'turn': 'draw',
            'outcome': 'draw'
        })
    
    return 'draw', history


def run_battle_series(n4_probs_path, lambda_draw, num_games, seed=None, record_losses=False):
    """
    Run battle test series: Strategic Agent vs Random
    
    Args:
        n4_probs_path: Path to 4×4 exhaustive probabilities JSON
        lambda_draw: Draw weight for convolution scoring
        num_games: Number of games to play
        seed: Random seed for reproducibility
        record_losses: Whether to record full game history for losses
    
    Returns:
        results: dict with win/draw/loss stats and optional loss histories
    """
    if seed is not None:
        random.seed(seed)
    
    print(f"\n{'='*80}")
    print(f"Strategic Convolution Agent vs Random Agent")
    print(f"{'='*80}")
    print(f"λ (draw weight): {lambda_draw}")
    print(f"Games: {num_games}")
    print(f"Seed: {seed}")
    print(f"Record losses: {record_losses}")
    print(f"{'='*80}\n")
    
    # Initialize agents
    strategic_agent = StrategicConvolutionAgent(n4_probs_path, lambda_draw)
    random_agent = RandomAgent9x9()
    
    # Stats
    wins = 0
    draws = 0
    losses = 0
    loss_games = []
    
    # Play games (alternate who plays X)
    for game_idx in range(num_games):
        strategic_plays_x = (game_idx % 2 == 0)
        
        if strategic_plays_x:
            agent1 = strategic_agent
            agent2 = random_agent
            agent1_symbol = 'X'
            agent2_symbol = 'O'
        else:
            agent1 = random_agent
            agent2 = strategic_agent
            agent1_symbol = 'X'
            agent2_symbol = 'O'
        
        # Play game
        winner, history = play_game(
            agent1, agent2,
            agent1_symbol, agent2_symbol,
            record_history=record_losses
        )
        
        # Update stats
        if winner == 'draw':
            draws += 1
        elif (winner == 'X' and strategic_plays_x) or (winner == 'O' and not strategic_plays_x):
            wins += 1
        else:
            losses += 1
            if record_losses:
                loss_games.append({
                    'game_id': game_idx,
                    'lambda_draw': lambda_draw,
                    'opponent': 'random',
                    'strategic_agent_played_as': 'X' if strategic_plays_x else 'O',
                    'outcome': 'loss',
                    'winner': winner,
                    'history': history
                })
        
        # Progress
        if (game_idx + 1) % 50 == 0:
            current_wr = wins / (game_idx + 1)
            print(f"Game {game_idx+1}/{num_games}: W:{wins} D:{draws} L:{losses} (WR: {current_wr:.1%})")
    
    # Final results
    total = wins + draws + losses
    win_rate = wins / total if total > 0 else 0.0
    
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Wins:   {wins}/{total} ({wins/total:.1%})")
    print(f"Draws:  {draws}/{total} ({draws/total:.1%})")
    print(f"Losses: {losses}/{total} ({losses/total:.1%})")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"{'='*80}\n")
    
    results = {
        'lambda_draw': lambda_draw,
        'seed': seed,
        'vs_random': {
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'total': total,
            'win_rate': win_rate
        }
    }
    
    if record_losses and loss_games:
        print(f"Recorded {len(loss_games)} loss games")
    
    return results, loss_games


def main():
    parser = argparse.ArgumentParser(description='9×9 Strategic Convolution Agent Battle Test')
    parser.add_argument('--probs', required=True, help='Path to n4_exhaustive_probs.json')
    parser.add_argument('--lambda_draw', type=float, default=0.3, help='Draw weight (default: 0.3)')
    parser.add_argument('--games', type=int, default=500, help='Number of games (default: 500)')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed (default: 2025)')
    parser.add_argument('--out_json', required=True, help='Output JSON path for results')
    parser.add_argument('--record_losses', action='store_true', help='Record full history for losses')
    
    args = parser.parse_args()
    
    # Validate inputs
    probs_path = Path(args.probs)
    if not probs_path.exists():
        print(f"ERROR: Probs file not found: {probs_path}")
        return 1
    
    # Run battle
    results, loss_games = run_battle_series(
        n4_probs_path=str(probs_path),
        lambda_draw=args.lambda_draw,
        num_games=args.games,
        seed=args.seed,
        record_losses=args.record_losses
    )
    
    # Save main results
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'config': {
            'agent': 'StrategicConvolutionAgent',
            'n4_probs': str(probs_path),
            'seed': args.seed,
            'num_games': args.games
        },
        'results': [results]
    }
    
    with open(out_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to: {out_path}")
    
    # Save losses to separate file
    if args.record_losses and loss_games:
        loss_path = out_path.parent / f"{out_path.stem}_losses.json"
        
        loss_data = {
            'lambda_draw': args.lambda_draw,
            'total_losses': len(loss_games),
            'loss_games': loss_games
        }
        
        with open(loss_path, 'w') as f:
            json.dump(loss_data, f, indent=2)
        
        print(f"Loss games saved to: {loss_path}")
        print(f"Total losses: {len(loss_games)}")
    
    return 0


if __name__ == '__main__':
    exit(main())
