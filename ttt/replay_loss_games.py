#!/usr/bin/env python3
"""
Replay loss games with animated board display for creating demo videos.

Usage:
  python -m ttt.replay_loss_games --losses out/runs/n4_lambda0.3_losses.json --game_id 0
  python -m ttt.replay_loss_games --losses out/runs/n4_lambda0.3_losses.json --opponent smart --limit 3
  python -m ttt.replay_loss_games --losses out/runs/n4_lambda0.3_losses.json --all

"""
import argparse
import json
import time
import sys


def clear_screen():
    """Clear terminal screen."""
    print('\033[2J\033[H', end='')


def format_board_fancy(board, last_move=None):
    """
    Format 4x4 board with fancy Unicode box-drawing characters.
    Highlight the last move if provided.
    """
    symbols = {'X': 'X', 'O': 'O', ' ': '·'}
    lines = []
    
    lines.append('┏━━━┳━━━┳━━━┳━━━┓')
    for r in range(4):
        row_chars = []
        for c in range(4):
            pos = r * 4 + c
            cell = board[pos]
            sym = symbols[cell]
            
            # Highlight last move
            if last_move is not None and pos == last_move:
                if cell == 'X':
                    sym = '\033[1;91mX\033[0m'  # Bold red X
                elif cell == 'O':
                    sym = '\033[1;94mO\033[0m'  # Bold blue O
            
            row_chars.append(f' {sym} ')
        
        lines.append('┃' + '┃'.join(row_chars) + '┃')
        
        if r < 3:
            lines.append('┣━━━╋━━━╋━━━╋━━━┫')
        else:
            lines.append('┗━━━┻━━━┻━━━┻━━━┛')
    
    return '\n'.join(lines)


def format_board_simple(board, last_move=None):
    """Format 4x4 board with simple ASCII characters."""
    symbols = {'X': 'X', 'O': 'O', ' ': '.'}
    lines = []
    
    lines.append('+---+---+---+---+')
    for r in range(4):
        row_chars = []
        for c in range(4):
            pos = r * 4 + c
            cell = board[pos]
            sym = symbols[cell]
            
            # Highlight last move with *
            if last_move is not None and pos == last_move:
                sym = f'*{sym}*'[1]  # Just use the symbol for now
            
            row_chars.append(sym)
        
        lines.append('| ' + ' | '.join(row_chars) + ' |')
        lines.append('+---+---+---+---+')
    
    return '\n'.join(lines)


def position_name(pos):
    """Convert position index to human-readable name (e.g., 0 -> A1)."""
    row = pos // 4
    col = pos % 4
    return f"{chr(65 + col)}{row + 1}"


def replay_game(game, delay=1.0, style='fancy', show_info=True):
    """
    Replay a single game with animated display.
    
    Args:
        game: Game dict with history
        delay: Delay in seconds between moves
        style: 'fancy' or 'simple' board style
        show_info: Whether to show game info header
    """
    if style == 'fancy':
        format_func = format_board_fancy
    else:
        format_func = format_board_simple
    
    history = game['history']
    
    # Header
    if show_info:
        clear_screen()
        print('=' * 60)
        print(f"Game ID: {game['game_id']}")
        print(f"Lambda: {game['lambda_draw']}")
        print(f"Opponent: {game['opponent'].upper()}")
        print(f"Exhaustive Agent played as: {game['main_agent_played_as']}")
        print(f"Outcome: {game['outcome'].upper()}")
        print('=' * 60)
        print()
        time.sleep(delay * 1.5)
    
    move_num = 0
    for i, step in enumerate(history):
        board = step['board']
        move = step.get('move')
        turn = step.get('turn')
        is_final = step.get('final_state', False)
        
        clear_screen()
        
        # Header
        if show_info:
            print('=' * 60)
            print(f"Game {game['game_id']} | λ=0.3 vs {game['opponent'].upper()} | "
                  f"Exhaustive={game['main_agent_played_as']} | Move {move_num}")
            print('=' * 60)
            print()
        
        # Show move info
        if move is not None:
            move_num += 1
            agent_type = "Exhaustive" if turn == game['main_agent_played_as'] else game['opponent'].capitalize()
            print(f"Move {move_num}: {turn} ({agent_type}) plays at {position_name(move)} (index {move})")
            print()
        elif is_final:
            print(f"FINAL STATE - {game['outcome'].upper()}")
            print()
        
        # Display board
        last_move = move if move is not None else None
        print(format_func(board, last_move=last_move))
        print()
        
        # Position guide
        if show_info and not is_final:
            print("Position Guide:")
            print("  A1  A2  A3  A4")
            print("  B1  B2  B3  B4")
            print("  C1  C2  C3  C4")
            print("  D1  D2  D3  D4")
        
        # Wait before next move
        if i < len(history) - 1:
            time.sleep(delay)
        else:
            # Final state - wait longer
            time.sleep(delay * 2)


def main():
    parser = argparse.ArgumentParser(description='Replay 4x4 loss games with animated display.')
    parser.add_argument('--losses', required=True, help='Path to losses JSON file')
    parser.add_argument('--game_id', type=int, default=None, help='Specific game ID to replay')
    parser.add_argument('--opponent', type=str, default=None, 
                       choices=['random', 'smart', 'minimax'],
                       help='Filter by opponent type')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of games to replay')
    parser.add_argument('--all', action='store_true', help='Replay all games')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between moves in seconds')
    parser.add_argument('--style', type=str, default='fancy', 
                       choices=['fancy', 'simple'],
                       help='Board display style')
    parser.add_argument('--no-info', action='store_true', help='Hide game info header')
    args = parser.parse_args()
    
    # Load loss games
    with open(args.losses, 'r') as f:
        data = json.load(f)
    
    games = data['loss_games']
    
    # Filter by opponent
    if args.opponent:
        games = [g for g in games if g['opponent'] == args.opponent]
    
    # Filter by game_id
    if args.game_id is not None:
        games = [g for g in games if g['game_id'] == args.game_id]
    
    if not games:
        print(f"No games found matching criteria.")
        return
    
    # Limit number of games
    if args.limit and not args.all:
        games = games[:args.limit]
    
    print(f"Found {len(games)} game(s) to replay.")
    print(f"Lambda: {data['lambda_draw']}")
    print()
    
    if len(games) > 1 and not args.all:
        response = input(f"Replay {len(games)} games? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("Cancelled.")
            return
    
    # Replay games
    show_info = not args.no_info
    for i, game in enumerate(games):
        if i > 0:
            print()
            print('=' * 60)
            print(f"Next game ({i+1}/{len(games)})...")
            print('=' * 60)
            time.sleep(2)
        
        replay_game(game, delay=args.delay, style=args.style, show_info=show_info)
        
        if i < len(games) - 1:
            print()
            input("Press Enter to continue to next game...")


if __name__ == '__main__':
    main()
