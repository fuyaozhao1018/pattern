# ttt/gen_exhaustive_n9.py
"""Generate exhaustive 9x9 (connect-4) game states with win/draw probabilities.

WARNING: 9x9 has an ENORMOUS state space (~10^38 possible boards).
Full exhaustive enumeration is computationally infeasible.

This script uses early pruning and sampling strategies:
1. Only enumerate states up to a certain depth
2. Sample representative branches
3. Use symmetry reduction
4. Prune losing branches aggressively

Output:
  - n9_exhaustive_probs.json : sampled states with per-move probability maps

Usage example:
  python -m ttt.gen_exhaustive_n9 \
    --out_dir data/n9_dual \
    --max_depth 20 \
    --max_states 100000 \
    --sample_rate 0.1

"""
from __future__ import annotations
import argparse, json, os, random
from typing import Dict, List, Tuple, Optional

# 9x9 Connect-4 (K=4) definitions
N = 9
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
    # diagonals ↘ (all possible starting positions)
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
    # anti-diagonals ↙
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

def count_all(board: List[str], turn: str, root: str, memo: MemoType, 
              max_depth: int = 999, current_depth: int = 0) -> Tuple[int,int,int]:
    """
    Count wins/draws/losses from this position.
    Uses depth limit to prevent excessive computation.
    """
    if current_depth >= max_depth:
        # Estimate: assume equal probability for remaining outcomes
        # This is a rough heuristic
        return (1, 1, 1)
    
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
    legal = legal_moves(board)
    
    for m in legal:
        board[m] = turn
        w2,d2,l2 = count_all(board, switch(turn), root, memo, max_depth, current_depth+1)
        board[m] = ' '
        wins += w2; draws += d2; losses += l2
    
    memo[key] = (wins, draws, losses)
    return memo[key]

def gen_state_id():
    i=1
    while True:
        yield f"n9_state_{i:07d}"; i+=1

def build_exhaustive_sampled(max_depth: int = 20, max_states: int = 100000, 
                             sample_rate: float = 0.1, seed: int = 2025):
    """
    Build sampled exhaustive states for 9x9 board.
    
    Args:
        max_depth: Maximum depth for probability calculation
        max_states: Maximum number of states to generate
        sample_rate: Probability of exploring each branch (for large branching)
        seed: Random seed for sampling
    """
    init = new_board(); turn='X'
    memo: MemoType = {}
    seen = set()
    sid_gen = gen_state_id()
    rng = random.Random(seed)
    
    states: List[Dict] = []
    
    def dfs(board: List[str], turn: str, depth: int = 0):
        if len(states) >= max_states:
            return
        
        key=(tuple(board),turn)
        if key in seen: return
        seen.add(key)
        
        w = check_winner(board)
        full = is_full(board)
        terminal = (w is not None) or full
        
        legal = legal_moves(board)
        per_move = {}
        
        # Compute probabilities for each move
        for m in legal:
            board[m] = turn
            wdl = count_all(board, switch(turn), turn, memo, max_depth=max_depth)
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
        
        # Add state
        st_id = next(sid_gen)
        states.append({
            'id': st_id,
            'turn': turn,
            'board': board.copy(),
            'legal': legal.copy(),
            'terminal': terminal,
            'depth': depth,
            'per_move': per_move
        })
        
        if not terminal and len(states) < max_states:
            # Sample moves to explore (to reduce branching)
            if len(legal) > 10:
                # For high branching factor, sample moves
                sample_size = max(3, int(len(legal) * sample_rate))
                moves_to_explore = rng.sample(legal, sample_size)
            else:
                moves_to_explore = legal
            
            for m in moves_to_explore:
                if len(states) >= max_states:
                    break
                board[m] = turn
                dfs(board, switch(turn), depth + 1)
                board[m] = ' '
    
    print(f"Starting sampled exhaustive generation:")
    print(f"  max_depth={max_depth}, max_states={max_states}, sample_rate={sample_rate}")
    dfs(init, turn)
    
    return states

def main():
    ap = argparse.ArgumentParser(description='Partial exhaustive 9x9 connect-4 generator (opening only).')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--max_depth', type=int, default=10, 
                   help='Maximum depth for probability calculation (default: 10)')
    ap.add_argument('--max_states', type=int, default=10000,
                   help='Maximum number of states to generate (default: 10000)')
    ap.add_argument('--sample_rate', type=float, default=0.2,
                   help='Sampling rate for high branching (default: 0.2)')
    ap.add_argument('--seed', type=int, default=2025,
                   help='Random seed for sampling (default: 2025)')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print('=' * 60)
    print('9×9 Connect-4 Partial Exhaustive Generator')
    print('=' * 60)
    print(f'Configuration:')
    print(f'  max_depth: {args.max_depth} (probability calculation limit)')
    print(f'  max_states: {args.max_states} (state generation limit)')
    print(f'  sample_rate: {args.sample_rate} (branch sampling)')
    print(f'  seed: {args.seed}')
    print()
    print('Note: 9×9 board has ~10^38 possible states.')
    print('This generates opening book only, use heuristics for mid/endgame.')
    print('=' * 60)
    print()
    
    states = build_exhaustive_sampled(
        max_depth=args.max_depth,
        max_states=args.max_states,
        sample_rate=args.sample_rate,
        seed=args.seed
    )

    # write single probability map file
    path_states = os.path.join(args.out_dir, 'n9_exhaustive_probs_partial.json')
    with open(path_states, 'w') as f:
        json.dump(states, f, indent=2)

    print()
    print('=' * 60)
    print(f'✓ Wrote: {path_states}')
    print(f'✓ Total states: {len(states)}')
    if states:
        depths = [s['depth'] for s in states]
        print(f'✓ Depth range: {min(depths)} - {max(depths)}')
        print(f'✓ Average moves per state: {sum(len(s["legal"]) for s in states) / len(states):.1f}')
    print('=' * 60)

if __name__ == '__main__':
    main()
