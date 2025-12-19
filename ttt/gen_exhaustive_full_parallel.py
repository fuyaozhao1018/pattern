#!/usr/bin/env python3
"""
Generate FULL exhaustive 4×4 Connect-4 data (43,046,721 states) with MULTIPROCESSING

This includes ALL possible board configurations with parallel processing for speed.
Expected speedup: 4-8x on multi-core machines.
"""
from __future__ import annotations
import argparse
import json
import os
from typing import List, Tuple, Dict
from collections import defaultdict
from itertools import product
from multiprocessing import Pool, cpu_count, Manager
import time

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


def process_board_batch(args):
    """Process a batch of boards (for parallel processing)"""
    board_tuples, lambda_draw, batch_id = args
    
    states = []
    memo = {}
    local_stats = {
        'X': {'wins': 0, 'draws': 0, 'losses': 0, 'total': 0},
        'O': {'wins': 0, 'draws': 0, 'losses': 0, 'total': 0}
    }
    
    for board_tuple in board_tuples:
        board = list(board_tuple)
        
        # Skip if board is already terminal
        legal = legal_moves(board)
        if not legal:
            continue
        
        # For each possible turn (X and O)
        for turn in ['X', 'O']:
            per_move = {}
            
            for move in legal:
                # Make move
                board[move] = turn
                
                # Compute probabilities using minimax
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
            best_score = (-1, -1, 2)
            
            # Calculate move scores with lambda_draw
            move_scores = {}
            for move_str, probs in per_move.items():
                score = probs['p_win'] + lambda_draw * probs['p_draw']
                move_scores[move_str] = score
                
                # Track best move
                move_tuple = (probs['p_win'], probs['p_draw'], probs['p_loss'])
                if (move_tuple[0] > best_score[0] or
                    (move_tuple[0] == best_score[0] and move_tuple[1] > best_score[1]) or
                    (move_tuple[0] == best_score[0] and move_tuple[1] == best_score[1] and move_tuple[2] < best_score[2])):
                    best_score = move_tuple
                    best_move_idx = move_str
            
            # Compute probability map
            total_score = sum(move_scores.values())
            probability_map = {}
            if total_score > 0:
                for move_str, score in move_scores.items():
                    probability_map[move_str] = score / total_score
            else:
                uniform_prob = 1.0 / len(per_move)
                for move_str in per_move.keys():
                    probability_map[move_str] = uniform_prob
            
            # Update statistics
            if per_move:
                local_stats[turn]['total'] += 1
                if best_score[0] == 1.0:
                    local_stats[turn]['wins'] += 1
                elif best_score[2] == 1.0:
                    local_stats[turn]['losses'] += 1
                elif best_score[1] == 1.0:
                    local_stats[turn]['draws'] += 1
            
            # Save state
            state_data = {
                'board': board.copy(),
                'turn': turn,
                'per_move': per_move,
                'best_move': best_move_idx,
                'probability_map': probability_map
            }
            states.append(state_data)
    
    return states, local_stats, batch_id


def generate_all_states_parallel(lambda_draw: float = 0.0, num_workers: int = None):
    """
    Generate ALL 3^16 = 43,046,721 possible 4×4 board states using multiprocessing.
    Uses streaming to avoid loading all combinations into memory.
    """
    if num_workers is None:
        num_workers = cpu_count()
    
    print(f"Using {num_workers} parallel workers")
    
    total_states = 3 ** 16
    print(f"Generating {total_states:,} board layouts (3^16)...")
    
    # Don't generate all boards at once - stream them in batches
    # This saves memory (avoids 5-6 GB list allocation)
    batch_size = 1_000_000  # Process 1M boards per batch
    num_batches = (total_states + batch_size - 1) // batch_size
    
    print(f"Will process {num_batches} batches of ~{batch_size:,} boards each")
    print(f"Starting parallel processing...")
    print()
    
    # Generate batches on-the-fly using a generator
    def batch_generator():
        """Generate batches of board tuples without loading all into memory"""
        batch = []
        batch_id = 0
        for board_tuple in product([' ', 'X', 'O'], repeat=16):
            batch.append(board_tuple)
            if len(batch) >= batch_size:
                yield (batch, lambda_draw, batch_id)
                batch = []
                batch_id += 1
        # Don't forget the last partial batch
        if batch:
            yield (batch, lambda_draw, batch_id)
    
    # Process in parallel
    start_time = time.time()
    all_states = []
    combined_stats = {
        'X': {'wins': 0, 'draws': 0, 'losses': 0, 'total': 0},
        'O': {'wins': 0, 'draws': 0, 'losses': 0, 'total': 0}
    }
    
    with Pool(num_workers) as pool:
        completed = 0
        for states, local_stats, batch_id in pool.imap_unordered(process_board_batch, batch_generator()):
            all_states.extend(states)
            
            # Merge statistics
            for player in ['X', 'O']:
                for key in ['wins', 'draws', 'losses', 'total']:
                    combined_stats[player][key] += local_stats[player][key]
            
            completed += 1
            if completed % 10 == 0 or completed == num_batches:
                elapsed = time.time() - start_time
                progress = completed / num_batches * 100
                states_processed = len(all_states)
                rate = states_processed / elapsed if elapsed > 0 else 0
                eta = (num_batches - completed) * elapsed / completed if completed > 0 else 0
                
                print(f"  Progress: {completed}/{num_batches} batches ({progress:.1f}%) | "
                      f"States: {states_processed:,} | "
                      f"Rate: {rate:.0f} states/sec | "
                      f"ETA: {eta/60:.1f} min")
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Parallel processing complete!")
    print(f"Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
    print(f"Total states generated: {len(all_states):,}")
    print(f"Average rate: {len(all_states)/elapsed:.0f} states/sec")
    
    # Print statistics
    print(f"\n{'='*60}")
    print("Statistics by Player:")
    print("="*60)
    for player in ['X', 'O']:
        total = combined_stats[player]['total']
        if total > 0:
            win_rate = combined_stats[player]['wins'] / total * 100
            draw_rate = combined_stats[player]['draws'] / total * 100
            loss_rate = combined_stats[player]['losses'] / total * 100
            print(f"\n{player} to move:")
            print(f"  Total states: {total:,}")
            print(f"  Wins:  {combined_stats[player]['wins']:,} ({win_rate:.2f}%)")
            print(f"  Draws: {combined_stats[player]['draws']:,} ({draw_rate:.2f}%)")
            print(f"  Losses: {combined_stats[player]['losses']:,} ({loss_rate:.2f}%)")
    
    return all_states, combined_stats


def main():
    parser = argparse.ArgumentParser(description='Generate FULL 4×4 exhaustive data (parallel)')
    parser.add_argument('--lambda_draw', type=float, default=0.0, help='Draw weight (default: 0)')
    parser.add_argument('--out_dir', default='data/n4_full', help='Output directory')
    parser.add_argument('--workers', type=int, default=None, help='Number of workers (default: CPU count)')
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("="*80)
    print("Generating FULL 4×4 Connect-4 Exhaustive Data (PARALLEL)")
    print("="*80)
    print(f"Lambda draw: {args.lambda_draw}")
    print(f"Output: {args.out_dir}")
    print(f"Workers: {args.workers or cpu_count()}")
    print()
    
    states, stats = generate_all_states_parallel(args.lambda_draw, args.workers)
    
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
        'workers': args.workers or cpu_count(),
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
