# ttt/gen_exhaustive_n4_dual.py
"""Generate exhaustive 4x4 (connect-4) game states with separate win-rate and draw-rate effectiveness.

Outputs:
  - n4_exhaustive_states.json : list of states with per-move raw (wins, draws, losses)
  - n4_exhaustive_best_win.json : best moves by pure win-rate (wins/total)
  - n4_exhaustive_best_draw.json : best moves by pure draw-rate (draws/total)
  - n4_exhaustive_best_mix_lambdaX.json : best moves by mixed peff=(wins + lambda*draws)/total for given lambda
You can request multiple lambda values.

Usage example:
  python -m ttt.gen_exhaustive_n4_dual \
    --out_dir out/runs/n4_dual \
    --lambdas 0.25 0.50 0.75

"""
from __future__ import annotations
import argparse, json, os
from typing import Dict, List, Tuple

# Basic 4x4 connect-4 (K=4) definitions
N = 4
K = 4
PLAYERS = ('X','O')

def new_board():
    return [' '] * (N*N)

def legal_moves(board):
    return [i for i,c in enumerate(board) if c == ' ']

def switch(turn: str) -> str:
    return 'O' if turn == 'X' else 'X'

def line_winner(seq: List[str]) -> str | None:
    for p in PLAYERS:
        run = 0
        for c in seq:
            run = run + 1 if c == p else 0
            if run >= K:
                return p
    return None

def check_winner(board: List[str]) -> str | None:
    # rows
    for r in range(N):
        w = line_winner([board[r*N + c] for c in range(N)])
        if w: return w
    # cols
    for c in range(N):
        w = line_winner([board[r*N + c] for r in range(N)])
        if w: return w
    # diag ↘ chains (starting positions)
    for sr in range(N):
        seq=[]; r=sr; c=0
        while r<N and c<N:
            seq.append(board[r*N+c]); r+=1; c+=1
        if len(seq)>=K:
            w=line_winner(seq)
            if w: return w
    for sc in range(1,N):
        seq=[]; r=0; c=sc
        while r<N and c<N:
            seq.append(board[r*N+c]); r+=1; c+=1
        if len(seq)>=K:
            w=line_winner(seq)
            if w: return w
    # anti diag ↙ chains
    for sr in range(N):
        seq=[]; r=sr; c=N-1
        while r<N and c>=0:
            seq.append(board[r*N+c]); r+=1; c-=1
        if len(seq)>=K:
            w=line_winner(seq)
            if w: return w
    for sc in range(N-2,-1,-1):
        seq=[]; r=0; c=sc
        while r<N and c>=0:
            seq.append(board[r*N+c]); r+=1; c-=1
        if len(seq)>=K:
            w=line_winner(seq)
            if w: return w
    return None

def is_full(board: List[str]) -> bool:
    return all(c != ' ' for c in board)

# Memo key: (tuple(board), turn, root_player)
MemoType = Dict[Tuple[Tuple[str,...], str, str], Tuple[int,int,int]]

def count_all(board: List[str], turn: str, root: str, memo: MemoType) -> Tuple[int,int,int]:
    key = (tuple(board), turn, root)
    if key in memo:
        return memo[key]
    w = check_winner(board)
    if w is not None:
        if w == root:
            memo[key] = (1,0,0)
        else:
            memo[key] = (0,0,1)
        return memo[key]
    if is_full(board):
        memo[key] = (0,1,0)
        return memo[key]
    wins=draws=losses=0
    for m in legal_moves(board):
        board[m] = turn
        w2,d2,l2 = count_all(board, switch(turn), root, memo)
        board[m] = ' '
        wins += w2; draws += d2; losses += l2
    memo[key] = (wins, draws, losses)
    return memo[key]

# State id generator

def gen_state_id():
    i=1
    while True:
        yield f"n4_state_{i:07d}"; i+=1


def build_exhaustive_streaming(out_file: str, batch_size: int = 100000):
    """
    Generate all 3^16 boards with streaming write and periodic memo cleanup.
    
    Args:
        out_file: Output JSON file path
        batch_size: Clear memo every N boards (default: 100K)
    """
    from itertools import product
    import time
    
    total_boards = 3 ** 16
    print(f"Generating {total_boards:,} board states (3^16)...", flush=True)
    print(f"Memo cleared every {batch_size:,} boards", flush=True)
    print(f"Writing to: {out_file}", flush=True)
    print(flush=True)
    
    memo: MemoType = {}
    
    # Open file for streaming write
    with open(out_file, 'w') as f:
        f.write('[\n')
        
        board_count = 0
        state_count = 0
        start_time = time.time()
        
        # Enumerate ALL 3^16 boards (not just reachable via DFS)
        for board_tuple in product([' ', 'X', 'O'], repeat=16):
            board = list(board_tuple)
            board_count += 1
            
            # Progress reporting (but memo cleared after EVERY board)
            if board_count % batch_size == 0:
                elapsed = time.time() - start_time
                rate = board_count / elapsed if elapsed > 0 else 0
                eta_seconds = (total_boards - board_count) / rate if rate > 0 else 0
                progress_pct = board_count / total_boards * 100
                
                print(f"  Processed: {board_count:,} / {total_boards:,} ({progress_pct:.1f}%)", flush=True)
                print(f"  States: {state_count:,}", flush=True)
                print(f"  Rate: {rate:.0f} boards/sec", flush=True)
                print(f"  ETA: {eta_seconds/3600:.1f} hours", flush=True)
                print(f"  Memo size: {len(memo):,}", flush=True)
                print(flush=True)
            
            # Get legal moves
            legal = legal_moves(board)
            
            # Process for both X and O
            for turn in ['X', 'O']:
                per_move = {}
                
                if legal:
                    # Calculate win/draw/loss for each move
                    for m in legal:
                        board[m] = turn
                        wdl = count_all(board, switch(turn), turn, memo)
                        board[m] = ' '
                        
                        wins, draws, losses = wdl
                        tot = wins + draws + losses
                        win_prob = wins / tot if tot > 0 else 0.0
                        draw_prob = draws / tot if tot > 0 else 0.0
                        loss_prob = losses / tot if tot > 0 else 0.0
                        
                        per_move[str(m)] = {
                            "wins": wins,
                            "draws": draws,
                            "losses": losses,
                            "total": tot,
                            "win_prob": round(win_prob, 6),
                            "draw_prob": round(draw_prob, 6),
                            "loss_prob": round(loss_prob, 6)
                        }
                
                # Write state record
                board_str = ''.join(board)
                record = {
                    'board': board_str,
                    'turn': turn,
                    'legal': legal,
                    'per_move': per_move
                }
                
                if state_count > 0:
                    f.write(',\n')
                json.dump(record, f)
                state_count += 1
                
                # Flush every 1000 states
                if state_count % 1000 == 0:
                    f.flush()
            
            # CRITICAL: Clear memo after each board to prevent memory explosion
            # (count_all recursion generates too many intermediate states)
            memo.clear()
        
        f.write('\n]')
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}", flush=True)
    print(f"Generation complete!", flush=True)
    print(f"Total time: {elapsed/3600:.2f} hours", flush=True)
    print(f"Total boards: {board_count:,}", flush=True)
    print(f"Total states: {state_count:,}", flush=True)
    print(f"Final memo size: {len(memo):,}", flush=True)
    print(flush=True)
    
    return state_count


def build_exhaustive():
    """OLD DFS-based function - kept for compatibility but not used"""
    init = new_board(); turn='X'
    memo: MemoType = {}
    seen = set()
    sid_gen = gen_state_id()

    states: List[Dict] = []

    def dfs(board: List[str], turn: str):
        key=(tuple(board),turn)
        if key in seen: return
        seen.add(key)

        w = check_winner(board)
        full = is_full(board)
        terminal = (w is not None) or full

        legal = legal_moves(board)
        per_move = {}

        # enumerate moves and compute probabilities
        for m in legal:
            board[m] = turn
            wdl = count_all(board, switch(turn), turn, memo)
            board[m] = ' '
            wins, draws, losses = wdl
            tot = wins + draws + losses
            win_prob = wins / tot if tot>0 else 0.0
            draw_prob = draws / tot if tot>0 else 0.0
            loss_prob = losses / tot if tot>0 else 0.0
            per_move[str(m)] = {
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "total": tot,
                "win_prob": round(win_prob, 6),
                "draw_prob": round(draw_prob, 6),
                "loss_prob": round(loss_prob, 6)
            }

        # state record
        st_id = next(sid_gen)
        states.append({
            'id': st_id,
            'turn': turn,
            'board': board.copy(),
            'legal': legal.copy(),
            'terminal': terminal,
            'per_move': per_move
        })

        if not terminal:
            for m in legal:
                board[m] = turn
                dfs(board, switch(turn))
                board[m] = ' '

    dfs(init, turn)

    return states


def main():
    ap = argparse.ArgumentParser(description='Exhaustive 4x4 connect-4 generator - ALL 3^16 states with streaming')
    ap.add_argument('--out_dir', required=True, help='Output directory')
    ap.add_argument('--batch_size', type=int, default=100000, help='Clear memo every N boards (default: 100K)')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print('='*80)
    print('Generating FULL 4×4 exhaustive data (ALL 3^16 states)')
    print('='*80)
    print(f'Output dir: {args.out_dir}')
    print(f'Batch size: {args.batch_size:,}')
    print()

    # Use streaming version
    path_states = os.path.join(args.out_dir, 'n4_exhaustive_probs.json')
    total_states = build_exhaustive_streaming(path_states, args.batch_size)

    print(f'\nWrote: {path_states}')
    print(f'Total states: {total_states:,}')
    print('Each state includes per-move: wins, draws, losses, total, win_prob, draw_prob, loss_prob')

if __name__ == '__main__':
    main()
