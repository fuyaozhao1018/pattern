#!/usr/bin/env python3
"""
Generate FULL exhaustive 4×4 Connect-4 data (43,046,721 states)

This includes ALL possible board configurations, including:
- Unbalanced states (15 X's, 1 O)
- Impossible states (both players won)
- All 3^16 = 43M combinations

This takes ~4-6 hours to compute and produces ~12-15GB of data.
"""
from __future__ import annotations
import argparse
import json
import os
from typing import List, Tuple, Dict
from collections import defaultdict
from itertools import product

N = 4
K = 4


def new_board(): 
    return [' '] * (N * N)


def legal_moves(board): 
    return [i for i, c in enumerate(board) if c == ' ']


def switch(turn): 
    return 'O' if turn == 'X' else 'X'


def check_winner(board):
    """Return 'X', 'O', or None"""
    def line_winner(seq):
        for p in ('X', 'O'):
            run = 0
            for c in seq:
                run = run + 1 if c == p else 0
                if run >= K:
                    return p
        return None

    # rows
    for r in range(N):
        w = line_winner([board[r * N + c] for c in range(N)])
        if w:
            return w
    
    # cols
    for c in range(N):
        w = line_winner([board[r * N + c] for r in range(N)])
        if w:
            return w
    
    # diagonals
    for start_r in range(N):
        seq = []
        r, c = start_r, 0
        while r < N and c < N:
            seq.append(board[r * N + c])
            r += 1
            c += 1
        if len(seq) >= K:
            w = line_winner(seq)
            if w:
                return w
    
    for start_c in range(1, N):
        seq = []
        r, c = 0, start_c
        while r < N and c < N:
            seq.append(board[r * N + c])
            r += 1
            c += 1
        if len(seq) >= K:
            w = line_winner(seq)
            if w:
                return w
    
    # anti-diagonals
    for start_r in range(N):
        seq = []
        r, c = start_r, N - 1
        while r < N and c >= 0:
            seq.append(board[r * N + c])
            r += 1
            c -= 1
        if len(seq) >= K:
            w = line_winner(seq)
            if w:
                return w
    
    for start_c in range(N - 2, -1, -1):
        seq = []
        r, c = 0, start_c
        while r < N and c >= 0:
            seq.append(board[r * N + c])
            r += 1
            c -= 1
        if len(seq) >= K:
            w = line_winner(seq)
            if w:
                return w
    
    return None


def is_full(board):
    return ' ' not in board


def minimax(board: List[str], turn: str, memo: Dict) -> Tuple[float, float, float]:
    """
    Return (p_win, p_draw, p_loss) from perspective of 'turn' player.
    Using full minimax with memoization.
    """
    key = (tuple(board), turn)
    if key in memo:
        return memo[key]
    
    winner = check_winner(board)
    if winner == turn:
        result = (1.0, 0.0, 0.0)  # Win
        memo[key] = result
        return result
    elif winner == switch(turn):
        result = (0.0, 0.0, 1.0)  # Loss
        memo[key] = result
        return result
    elif is_full(board):
        result = (0.0, 1.0, 0.0)  # Draw
        memo[key] = result
        return result
    
    # Recursive case
    legal = legal_moves(board)
    if not legal:
        result = (0.0, 1.0, 0.0)
        memo[key] = result
        return result
    
    best_win = -1.0
    best_draw = -1.0
    best_loss = 2.0
    
    for move in legal:
        board[move] = turn
        opp_win, opp_draw, opp_loss = minimax(board, switch(turn), memo)
        board[move] = ' '
        
        # From our perspective: opponent's loss is our win
        my_win = opp_loss
        my_draw = opp_draw
        my_loss = opp_win
        
        # Choose move that maximizes win, then draw, then minimizes loss
        if (my_win > best_win or 
            (my_win == best_win and my_draw > best_draw) or
            (my_win == best_win and my_draw == best_draw and my_loss < best_loss)):
            best_win = my_win
            best_draw = my_draw
            best_loss = my_loss
    
    result = (best_win, best_draw, best_loss)
    memo[key] = result
    return result


def generate_all_states(lambda_draw: float = 0.0):
    """
    Generate ALL 3^16 = 43,046,721 possible 4×4 board states.
    For each state, compute optimal move probabilities for both X and O.
    """
    states = []
    memo = {}
    
    # Statistics tracking
    stats = {
        'X': {'wins': 0, 'draws': 0, 'losses': 0, 'total': 0},
        'O': {'wins': 0, 'draws': 0, 'losses': 0, 'total': 0}
    }
    
    total_states = 3 ** 16
    print(f"Generating {total_states:,} states (3^16)...")
    print("This will take 4-6 hours. Progress updates every 1M states.")
    
    state_id = 0
    
    # Generate all possible boards using itertools.product
    for board_tuple in product([' ', 'X', 'O'], repeat=16):
        board = list(board_tuple)
        state_id += 1
        
        if state_id % 1_000_000 == 0:
            print(f"  Processed {state_id:,} / {total_states:,} ({state_id/total_states*100:.1f}%)")
        
        # Skip if board is already terminal (someone won or full)
        legal = legal_moves(board)
        if not legal:
            continue  # No legal moves, skip this board layout
        
        # For each possible turn (X and O) - evaluate from both perspectives
        for turn in ['X', 'O']:
            per_move = {}
            
            for move in legal:
                # Make move
                board[move] = turn
                
                # Compute probabilities using minimax (perfect play from here)
                p_win, p_draw, p_loss = minimax(board, turn, memo)
                
                # Restore board
                board[move] = ' '
                
                per_move[str(move)] = {
                    'p_win': p_win,
                    'p_draw': p_draw,
                    'p_loss': p_loss
                }
            
            # Compute best move and probability map
            best_move_idx = None
            best_score = (-1, -1, 2)  # (p_win, p_draw, p_loss)
            
            # Calculate move scores with lambda_draw
            move_scores = {}
            for move_str, probs in per_move.items():
                score = probs['p_win'] + lambda_draw * probs['p_draw']
                move_scores[move_str] = score
                
                # Track best move (maximize win, then draw, minimize loss)
                move_tuple = (probs['p_win'], probs['p_draw'], probs['p_loss'])
                if (move_tuple[0] > best_score[0] or
                    (move_tuple[0] == best_score[0] and move_tuple[1] > best_score[1]) or
                    (move_tuple[0] == best_score[0] and move_tuple[1] == best_score[1] and move_tuple[2] < best_score[2])):
                    best_score = move_tuple
                    best_move_idx = move_str
            
            # Compute probability map (softmax-like distribution)
            total_score = sum(move_scores.values())
            probability_map = {}
            if total_score > 0:
                for move_str, score in move_scores.items():
                    probability_map[move_str] = score / total_score
            else:
                # All moves are equally bad, uniform distribution
                uniform_prob = 1.0 / len(per_move)
                for move_str in per_move.keys():
                    probability_map[move_str] = uniform_prob
            
            # Update statistics (use best move outcome)
            if per_move:
                stats[turn]['total'] += 1
                if best_score[0] == 1.0:
                    stats[turn]['wins'] += 1
                elif best_score[2] == 1.0:
                    stats[turn]['losses'] += 1
                elif best_score[1] == 1.0:
                    stats[turn]['draws'] += 1
            
            # Save state - ONE entry per (board_layout, turn) pair
            # This ignores turn order rules - just "what if X/O moves from this layout?"
            state_data = {
                'board': board.copy(),
                'turn': turn,
                'per_move': per_move,
                'best_move': best_move_idx,
                'probability_map': probability_map
            }
            states.append(state_data)
    
    print(f"\nGenerated {len(states)} total states (both turns)")
    print(f"Memo cache size: {len(memo):,}")
    
    # Print statistics
    print("\n" + "="*60)
    print("Statistics by Player:")
    print("="*60)
    for player in ['X', 'O']:
        total = stats[player]['total']
        if total > 0:
            win_rate = stats[player]['wins'] / total * 100
            draw_rate = stats[player]['draws'] / total * 100
            loss_rate = stats[player]['losses'] / total * 100
            print(f"\n{player} to move:")
            print(f"  Total states: {total:,}")
            print(f"  Wins:  {stats[player]['wins']:,} ({win_rate:.2f}%)")
            print(f"  Draws: {stats[player]['draws']:,} ({draw_rate:.2f}%)")
            print(f"  Losses: {stats[player]['losses']:,} ({loss_rate:.2f}%)")
    
    return states, stats


def main():
    parser = argparse.ArgumentParser(description='Generate FULL 4×4 exhaustive data')
    parser.add_argument('--lambda_draw', type=float, default=0.0, help='Draw weight (default: 0)')
    parser.add_argument('--out_dir', default='data/n4_full', help='Output directory')
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("="*80)
    print("Generating FULL 4×4 Connect-4 Exhaustive Data")
    print("="*80)
    print(f"Lambda draw: {args.lambda_draw}")
    print(f"Output: {args.out_dir}")
    print()
    
    states, stats = generate_all_states(args.lambda_draw)
    
    # Save states to JSON
    states_file = os.path.join(args.out_dir, 'n4_exhaustive_probs_full.json')
    print(f"\nSaving states to {states_file}...")
    
    with open(states_file, 'w') as f:
        json.dump(states, f)
    
    # Save statistics to separate file
    stats_file = os.path.join(args.out_dir, 'n4_full_statistics.json')
    print(f"Saving statistics to {stats_file}...")
    
    stats_data = {
        'lambda_draw': args.lambda_draw,
        'total_states': len(states),
        'generated_date': '2025-11-25',
        'board_size': f'{N}x{N}',
        'connect_k': K,
        'statistics': stats
    }
    
    with open(stats_file, 'w') as f:
        json.dump(stats_data, f, indent=2)
    
    # Print file statistics
    states_size_mb = os.path.getsize(states_file) / (1024 * 1024)
    stats_size_kb = os.path.getsize(stats_file) / 1024
    
    print(f"\n{'='*60}")
    print(f"Done!")
    print(f"{'='*60}")
    print(f"States file: {states_file}")
    print(f"  Size: {states_size_mb:.1f} MB")
    print(f"  Count: {len(states):,}")
    print(f"\nStatistics file: {stats_file}")
    print(f"  Size: {stats_size_kb:.1f} KB")
    print(f"\nLambda draw: {args.lambda_draw}")


if __name__ == '__main__':
    main()
