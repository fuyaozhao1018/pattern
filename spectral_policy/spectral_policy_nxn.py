#!/usr/bin/env python3
"""
Deterministic NxN spectral policy with DCT features, ridge regression, and optional symmetry-averaged features.

- Features: DCT-II(NxN) of X and O occupancy grids, concatenated (2*N*N dims)
- Labels:
  * default: Monte Carlo rollouts vs Random for sampled states (works for any N)
  * optional (N=3 only): exhaustive optimal labels from optimal_decision_tree.json (like spectral_policy_3x3)
- Policy: safety (immediate win/block), else argmax linear score w^T phi
- Eval: W/D/L vs Random and optional small MCTS

CLI:
  python spectral_policy_nxn.py --n 4 --samples 600 --rollouts 8 --games-random 100 --games-mcts 20 --mcts-iters 200 --sym-avg
"""
import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

TREE_FILE = 'optimal_decision_tree.json'

# ---------- Game core ----------
class TicTacToeNxN:
    def __init__(self, n:int=4):
        self.n = n
        self.board = [0]*(n*n)  # 0 empty, 1 X, -1 O
        self.current_player = 1
    def clone(self):
        g = TicTacToeNxN(self.n)
        g.board = self.board.copy()
        g.current_player = self.current_player
        return g
    def get_valid_moves(self) -> List[int]:
        return [i for i,v in enumerate(self.board) if v==0]
    def make_move(self, m:int):
        self.board[m] = self.current_player
        self.current_player *= -1
    def check_winner(self) -> int:
        n=self.n; b=self.board
        # rows
        for r in range(n):
            s = sum(b[r*n+c] for c in range(n))
            if s==n: return 1
            if s==-n: return -1
        # cols
        for c in range(n):
            s = sum(b[r*n+c] for r in range(n))
            if s==n: return 1
            if s==-n: return -1
        # diags
        s = sum(b[i*n+i] for i in range(n))
        if s==n: return 1
        if s==-n: return -1
        s = sum(b[i*n+(n-1-i)] for i in range(n))
        if s==n: return 1
        if s==-n: return -1
        return 0
    def is_terminal(self) -> bool:
        return self.check_winner()!=0 or 0 not in self.board

# ---------- DCT-II ----------
def dct_basis_1d(N:int) -> np.ndarray:
    alpha = [math.sqrt(1/N)] + [math.sqrt(2/N)]*(N-1)
    C = np.zeros((N,N), dtype=float)
    for u in range(N):
        for x in range(N):
            C[u,x] = alpha[u]*math.cos(math.pi*(2*x+1)*u/(2*N))
    return C

def dct2(A: np.ndarray, C: np.ndarray) -> np.ndarray:
    return C @ A @ C.T

# ---------- Feature extraction with optional symmetry averaging ----------
def board_to_channels(board: List[int], n:int) -> Tuple[np.ndarray,np.ndarray]:
    X = np.zeros((n,n), dtype=float)
    O = np.zeros((n,n), dtype=float)
    for idx,val in enumerate(board):
        r,c = divmod(idx,n)
        if val==1: X[r,c]=1.0
        elif val==-1: O[r,c]=1.0
    return X,O

def board_symmetries(board: List[int], n:int) -> List[List[int]]:
    """Return 8 D4 symmetries: 4 rotations + horizontal flip and its rotations."""
    X, O = board_to_channels(board, n)
    syms: List[List[int]] = []
    def grid_to_board(X_sym: np.ndarray, O_sym: np.ndarray) -> List[int]:
        b=[0]*(n*n)
        for i in range(n*n):
            r,c = divmod(i,n)
            if X_sym[r,c] > 0.5: b[i]=1
            elif O_sym[r,c] > 0.5: b[i]=-1
        return b
    # identity + 3 rotations
    Xr, Or = X.copy(), O.copy()
    syms.append(grid_to_board(Xr, Or))
    for _ in range(3):
        Xr = np.rot90(Xr); Or = np.rot90(Or)
        syms.append(grid_to_board(Xr, Or))
    # horizontal flip + its 3 rotations
    Xf, Of = np.fliplr(X), np.fliplr(O)
    syms.append(grid_to_board(Xf, Of))
    for _ in range(3):
        Xf = np.rot90(Xf); Of = np.rot90(Of)
        syms.append(grid_to_board(Xf, Of))
    return syms

def features_from_board(board: List[int], n:int, C: np.ndarray, sym_avg: bool=False) -> np.ndarray:
    X,O = board_to_channels(board, n)
    DX = dct2(X, C); DO = dct2(O, C)
    if sym_avg:
        DX_acc = np.zeros_like(DX); DO_acc = np.zeros_like(DO)
        for b_sym in board_symmetries(board, n):
            Xs, Os = board_to_channels(b_sym, n)
            DX_acc += dct2(Xs, C)
            DO_acc += dct2(Os, C)
        DX = DX_acc / 8.0; DO = DO_acc / 8.0
    return np.concatenate([DX.flatten(), DO.flatten()])

# ---------- Labels ----------
def random_playout_from_state(g: TicTacToeNxN) -> int:
    gg = g.clone()
    while not gg.is_terminal():
        m = random.choice(gg.get_valid_moves())
        gg.make_move(m)
    return gg.check_winner()

def estimate_move_value(state_board: List[int], player: int, move: int, n:int, rollouts:int) -> float:
    g=TicTacToeNxN(n)
    g.board = state_board.copy()
    g.current_player = player
    g.make_move(move)
    total = 0.0
    for _ in range(rollouts):
        winner = random_playout_from_state(g)
        if winner==player: total += 1.0
        elif winner==0: total += 0.0
        else: total += -1.0
    return total/rollouts

# 3x3 optimal dataset (optional)
def parse_player_to_int(p_val) -> int:
    if isinstance(p_val, str):
        return 1 if p_val.upper()=='X' else -1
    try:
        return int(p_val)
    except Exception:
        return 1

def build_dataset_optimal_3x3(C: np.ndarray, sym_avg: bool=False) -> Tuple[np.ndarray,np.ndarray,List[Tuple[Tuple[int,...],int,int]]]:
    states = json.loads(Path(TREE_FILE).read_text(encoding='utf-8'))['states']
    X_rows: List[np.ndarray] = []
    y: List[float] = []
    meta: List[Tuple[Tuple[int,...],int,int]] = []
    for state_str,info in states.items():
        try:
            s = tuple(json.loads(state_str))
        except Exception:
            continue
        if 0 not in s:
            continue
        move_opt = info.get('move')
        player = parse_player_to_int(info.get('player','X'))
        valid = [i for i,v in enumerate(s) if v==0]
        for m in valid:
            b = list(s)
            b[m]=player
            phi = features_from_board(b, 3, C, sym_avg=sym_avg)
            X_rows.append(phi)
            y.append(1.0 if m==move_opt else 0.0)
            meta.append((s,m,player))
    X = np.vstack(X_rows); y = np.array(y, dtype=float)
    return X,y,meta

# Monte Carlo dataset for any N

def sample_states_for_dataset(n:int, samples:int, rollouts:int, C: np.ndarray, sym_avg: bool=False, seed: Optional[int]=None):
    if seed is not None:
        random.seed(seed)
    X_rows: List[np.ndarray] = []; y: List[float] = []
    for _ in range(samples):
        g=TicTacToeNxN(n)
        k = random.randint(0, n*n-1)
        for _ in range(k):
            if g.is_terminal(): break
            m = random.choice(g.get_valid_moves())
            g.make_move(m)
        if g.is_terminal():
            continue
        moves = g.get_valid_moves(); p = g.current_player
        for m in moves:
            val = estimate_move_value(g.board, p, m, n, rollouts)
            b = g.board.copy(); b[m]=p
            phi = features_from_board(b, n, C, sym_avg=sym_avg)
            X_rows.append(phi)
            y.append(val)
    if not X_rows:
        return None, None
    return np.vstack(X_rows), np.array(y, dtype=float)

# ---------- Ridge regression ----------
def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float=1e-3) -> np.ndarray:
    d=X.shape[1]
    A = X.T @ X + lam*np.eye(d)
    b = X.T @ y
    return np.linalg.solve(A,b)

# ---------- Policy ----------
class SpectralPolicyNxN:
    def __init__(self, w: np.ndarray, n:int, C: np.ndarray, sym_avg: bool=False, safety: bool=True):
        self.w=w; self.n=n; self.C=C; self.sym_avg=sym_avg; self.safety=safety
    def score_move(self, board: List[int], player:int, move:int) -> float:
        b=board.copy(); b[move]=player
        phi = features_from_board(b, self.n, self.C, sym_avg=self.sym_avg)
        return float(self.w @ phi)
    def choose_move(self, g: TicTacToeNxN) -> int:
        moves = g.get_valid_moves(); p=g.current_player
        # immediate win/block (optional)
        if self.safety:
            for m in moves:
                g.board[m]=p
                if g.check_winner()==p:
                    g.board[m]=0; return m
                g.board[m]=0
            for m in moves:
                g.board[m]=-p
                if g.check_winner()==-p:
                    g.board[m]=0; return m
                g.board[m]=0
        # spectral
        best=moves[0]; best_s=-1e9
        for m in moves:
            s=self.score_move(g.board, p, m)
            if s>best_s: best_s=s; best=m
        return best

class RandomPlayerNxN:
    def choose_move(self, g: TicTacToeNxN) -> int:
        return random.choice(g.get_valid_moves())

# ---------- Small MCTS opponent ----------
class MCTSNxN:
    def __init__(self, iterations: int = 200, c: float = 1.414):
        self.iterations = iterations; self.c=c
        self.N: Dict[Tuple[Tuple[int,...],int,int], int] = {}
        self.W: Dict[Tuple[Tuple[int,...],int,int], float] = {}
        self.children: Dict[Tuple[Tuple[int,...],int,int], List[Tuple[int,Tuple[Tuple[int,...],int,int]]]] = {}
    def key(self, g: TicTacToeNxN):
        return (tuple(g.board), g.current_player, g.n)
    def expand(self, g: TicTacToeNxN):
        k = self.key(g)
        if k in self.children: return
        kids=[]
        for m in g.get_valid_moves():
            gg = g.clone(); gg.make_move(m)
            kids.append((m, self.key(gg)))
        self.children[k]=kids
        self.N.setdefault(k,0); self.W.setdefault(k,0.0)
        for _,ck in kids:
            self.N.setdefault(ck,0); self.W.setdefault(ck,0.0)
    def uct_select(self, k):
        total = self.N.get(k,1) + 1e-9
        best=None; best_sc=-1e9
        for m, ck in self.children.get(k, []):
            n=self.N.get(ck,0); w=self.W.get(ck,0.0)
            q=(w/n) if n>0 else 0.0
            u=self.c*math.sqrt(math.log(total)/(n+1e-9))
            sc=q+u
            if sc>best_sc: best_sc=sc; best=(m,ck)
        return best
    def rollout(self, g: TicTacToeNxN) -> int:
        gg=g.clone()
        while not gg.is_terminal():
            m=random.choice(gg.get_valid_moves())
            gg.make_move(m)
        return gg.check_winner()
    def choose_move(self, g: TicTacToeNxN) -> int:
        root=g.clone(); root_k=self.key(root)
        self.expand(root)
        for _ in range(self.iterations):
            path=[]
            node=root.clone(); k=self.key(node)
            path.append((k, node.current_player))
            while True:
                if node.is_terminal(): break
                self.expand(node)
                if any(self.N.get(ck,0)==0 for _,ck in self.children[k]):
                    unvisited=[(m,ck) for m,ck in self.children[k] if self.N.get(ck,0)==0]
                    m,ck=random.choice(unvisited)
                    node.make_move(m); k=ck
                    path.append((k, node.current_player))
                    break
                sel=self.uct_select(k)
                if sel is None: break
                m,ck=sel
                node.make_move(m); k=ck
                path.append((k, node.current_player))
            winner=self.rollout(node)
            for state_k, player_to_move in path:
                self.N[state_k]=self.N.get(state_k,0)+1
                if winner==0: val=0.0
                elif winner==player_to_move: val=1.0
                else: val=-1.0
                self.W[state_k]=self.W.get(state_k,0.0)+val
        # pick child with highest N
        best_m=None; best_n=-1
        for m,ck in self.children.get(root_k, []):
            n=self.N.get(ck,0)
            if n>best_n: best_n=n; best_m=m
        return best_m if best_m is not None else random.choice(g.get_valid_moves())

# ---------- Utilities ----------
def fmt_board(board: List[int], n:int) -> str:
    sym={1:'X', -1:'O', 0:'.'}
    rows=[' '.join(sym[board[r*n+c]] for c in range(n)) for r in range(n)]
    return '\n'.join(rows)

def play_game(p1, p2, n:int) -> int:
    g=TicTacToeNxN(n)
    players=[p1,p2]
    while not g.is_terminal():
        m=players[0].choose_move(g)
        g.make_move(m)
        players=[players[1], players[0]]
    return g.check_winner()

# ---------- Main ----------
def main():
    ap=argparse.ArgumentParser(description='NxN spectral policy via DCT features + ridge regression (MC labels or 3x3 optimal)')
    ap.add_argument('--n', type=int, default=4, help='Board size N (win condition = N in a line)')
    ap.add_argument('--labels', type=str, default='mc', choices=['mc','optimal'], help='Label source: Monte Carlo (any N) or optimal (N=3 only)')
    ap.add_argument('--samples', type=int, default=600, help='Number of sampled states for MC dataset')
    ap.add_argument('--rollouts', type=int, default=8, help='MC rollouts per candidate move')
    ap.add_argument('--lam', type=float, default=1e-3, help='Ridge lambda')
    ap.add_argument('--sym-avg', action='store_true', help='Use symmetry-averaged DCT features')
    ap.add_argument('--games-random', type=int, default=100, help='Games vs Random for eval')
    ap.add_argument('--games-mcts', type=int, default=0, help='Games vs MCTS for eval (0 to skip)')
    ap.add_argument('--mcts-iters', type=int, default=200, help='MCTS iterations per move')
    ap.add_argument('--seed', type=int, default=0, help='Global seed')
    ap.add_argument('--no-examples', action='store_true', help='Disable visualization example')
    ap.add_argument('--no-safety', action='store_true', help='Disable safety overrides (no immediate win/block).')
    ap.add_argument('--self-play', type=int, default=0, help='Number of self-play games (policy vs itself).')
    args=ap.parse_args()

    random.seed(args.seed)
    n=args.n
    C = dct_basis_1d(n)
    # Build dataset
    if args.labels=='optimal':
        if n!=3:
            print('Optimal labels only available for N=3; falling back to MC labels.')
            args.labels='mc'
        if args.labels=='optimal':
            if not Path(TREE_FILE).exists():
                print(f'{TREE_FILE} not found; falling back to MC labels.')
                args.labels='mc'
    if args.labels=='optimal':
        print(f'Building optimal dataset for 3x3 (mode={"sym-avg" if args.sym_avg else "standard"})...')
        X,y,meta = build_dataset_optimal_3x3(C, sym_avg=args.sym_avg)
        print('Dataset:', X.shape)
    else:
        print(f'Building MC dataset for {n}x{n}: samples={args.samples}, rollouts={args.rollouts}, mode={"sym-avg" if args.sym_avg else "standard"} ...')
        X,y = sample_states_for_dataset(n, args.samples, args.rollouts, C, sym_avg=args.sym_avg, seed=args.seed)
        if X is None:
            print('No samples gathered (all sampled states were terminal). Try increasing --samples.')
            return
        print('Dataset:', X.shape, '| mean target:', float(y.mean()))

    # Fit
    w = fit_ridge(X,y,lam=args.lam)
    print('\nLearned weights (first 10 shown):', ', '.join(f'{w[i]:+.4f}' for i in range(min(10,len(w)))))

    # Policy
    policy = SpectralPolicyNxN(w, n, C, sym_avg=args.sym_avg, safety=(not args.no_safety))

    # Eval vs Random
    wins=draws=loss=0
    for i in range(args.games_random):
        if i%2==0:
            r=play_game(policy, RandomPlayerNxN(), n)
        else:
            r=play_game(RandomPlayerNxN(), policy, n); r=-r
        if r==1: wins+=1
        elif r==0: draws+=1
        else: loss+=1
    print(f"\nVs Random ({args.games_random}): W:{wins} D:{draws} L:{loss}")
    # Self-play
    if args.self_play > 0:
        wins=draws=loss=0
        for i in range(args.self_play):
            r=play_game(policy, policy, n)
            if r==1: wins+=1
            elif r==0: draws+=1
            else: loss+=1
        print(f"Self-play ({args.self_play}): W:{wins} D:{draws} L:{loss}")

    # Eval vs MCTS
    if args.games_mcts>0:
        mcts = MCTSNxN(iterations=args.mcts_iters)
        wins=draws=loss=0
        for i in range(args.games_mcts):
            if i%2==0:
                r=play_game(policy, mcts, n)
            else:
                r=play_game(mcts, policy, n); r=-r
            if r==1: wins+=1
            elif r==0: draws+=1
            else: loss+=1
        print(f"Vs MCTS(iters={args.mcts_iters}) ({args.games_mcts}): W:{wins} D:{draws} L:{loss}")

    # Optional one example
    if not args.no_examples:
        print('\nOne example game vs Random:')
        print(fmt_board(TicTacToeNxN(n).board, n))
        # Just play silently; visualization grid omitted for brevity
        r=play_game(policy, RandomPlayerNxN(), n)
        print('Example result:', 'X wins' if r==1 else ('O wins' if r==-1 else 'Draw'))

if __name__=='__main__':
    main()
