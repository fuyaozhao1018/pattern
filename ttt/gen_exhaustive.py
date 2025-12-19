# ttt/gen_exhaustive.py
from __future__ import annotations
import argparse, json, os
from collections import defaultdict

# ----- Simple generic tic-tac-toe engine (no symmetry reduction) -----

def new_board(N): return [' '] * (N*N)

def legal_moves(board): return [i for i,c in enumerate(board) if c == ' ']

def switch(turn): return 'O' if turn == 'X' else 'X'

def check_winner(board, N, K):
    """Return 'X' / 'O' / None; if board is full and no winner, caller handles draw."""
    def line_winner(seq):
        if ' ' in seq: pass
        for p in ('X', 'O'):
            run = 0
            for c in seq:
                run = run + 1 if c == p else 0
                if run >= K: 
                    return p
        return None

    # rows
    for r in range(N):
        w = line_winner([board[r*N + c] for c in range(N)])
        if w: return w
    # cols
    for c in range(N):
        w = line_winner([board[r*N + c] for r in range(N)])
        if w: return w
    # main diagonals
    for start_r in range(N):
        seq=[]
        r,c=start_r,0
        while r<N and c<N:
            seq.append(board[r*N+c]); r+=1; c+=1
        if len(seq)>=K:
            w=line_winner(seq)
            if w: return w
    for start_c in range(1,N):
        seq=[]
        r,c=0,start_c
        while r<N and c<N:
            seq.append(board[r*N+c]); r+=1; c+=1
        if len(seq)>=K:
            w=line_winner(seq)
            if w: return w
    # anti-diagonals
    for start_r in range(N):
        seq=[]
        r,c=start_r,N-1
        while r<N and c>=0:
            seq.append(board[r*N+c]); r+=1; c-=1
        if len(seq)>=K:
            w=line_winner(seq)
            if w: return w
    for start_c in range(N-2,-1,-1):
        seq=[]
        r,c=0,start_c
        while r<N and c>=0:
            seq.append(board[r*N+c]); r+=1; c-=1
        if len(seq)>=K:
            w=line_winner(seq)
            if w: return w
    return None

def is_full(board): return all(c != ' ' for c in board)

# Count all completions from a given (board, turn) for root_player
def count_all_completions(board, turn, root_player, N, K, memo):
    key = (tuple(board), turn, root_player)
    if key in memo: return memo[key]

    w = check_winner(board, N, K)
    if w is not None:
        if w == root_player: 
            memo[key] = (1,0,0); return memo[key]
        else:
            memo[key] = (0,0,1); return memo[key]
    if is_full(board):
        memo[key] = (0,1,0); return memo[key]

    wins=draws=losses=0
    for m in legal_moves(board):
        board[m] = turn
        w2,d2,l2 = count_all_completions(board, switch(turn), root_player, N, K, memo)
        board[m] = ' '
        wins += w2; draws += d2; losses += l2
    memo[key] = (wins, draws, losses)
    return memo[key]

def state_id_gen(N):
    i=1
    while True:
        yield f"n{N}_state_{i:06d}"
        i+=1

# DFS enumerate all reachable states; compute per-move counts
def build_dataset(N, K, lambda_draw, include_terminal=True):
    init = new_board(N); turn = 'X'
    memo_counts = {}  # (board, turn, root) -> (w,d,l)
    seen = set()      # (board, turn)
    sid = state_id_gen(N)

    states_out = []
    best_map = {}

    def dfs(board, turn):
        key = (tuple(board), turn)
        if key in seen: return
        seen.add(key)

        w = check_winner(board, N, K)
        full = is_full(board)
        terminal = (w is not None) or full
        if terminal and not include_terminal:
            return

        legal = legal_moves(board)
        per_move = {}
        best_p = None

        # For current player, compute (win,draw,loss) for each move (root = current player)
        for m in legal:
            board[m] = turn
            wdl = count_all_completions(board, switch(turn), turn, N, K, memo_counts)
            board[m] = ' '
            wins, draws, losses = wdl
            per_move[str(m)] = {"wins": wins, "draws": draws, "losses": losses}
            tot = wins + draws + losses
            peff = (wins + lambda_draw * draws) / tot if tot > 0 else 0.0
            if best_p is None or peff > best_p:
                best_p = peff

        # store state
        st = {
            "id": next(sid),
            "turn": turn,
            "board": board.copy(),
            "legal": legal.copy(),
            "per_move": per_move
        }
        states_out.append(st)

        # build best-move set
        best_moves = []
        for m in legal:
            rec = per_move[str(m)]
            tot = rec["wins"] + rec["draws"] + rec["losses"]
            peff = (rec["wins"] + lambda_draw * rec["draws"]) / tot if tot > 0 else 0.0
            if abs(peff - best_p) <= 1e-12:
                best_moves.append(m)
        best_map[st["id"]] = best_moves

        # continue DFS
        if not terminal:
            for m in legal:
                board[m] = turn
                dfs(board, switch(turn))
                board[m] = ' '

    dfs(init, turn)
    return states_out, best_map

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, choices=[3,4], required=True, help='board size N×N')
    ap.add_argument('--K', type=int, default=None, help='in-a-row needed to win (default: N)')
    ap.add_argument('--lambda_draw', type=float, default=0.5, help='Peff = (win + λ*draw)/total')
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()

    N = args.N
    K = args.K if args.K is not None else N
    os.makedirs(args.out_dir, exist_ok=True)

    states, best_map = build_dataset(N, K, args.lambda_draw, include_terminal=True)

    # write files
    path_states = os.path.join(args.out_dir, f"n{N}_exhaustive_states.json")
    path_best   = os.path.join(args.out_dir, f"n{N}_exhaustive_best.json")
    with open(path_states,'w') as f: json.dump(states, f)
    with open(path_best,'w') as f: json.dump(best_map, f)

    print(f"Wrote {len(states)} states ->")
    print(" ", path_states)
    print(" ", path_best)

if __name__ == '__main__':
    main()
