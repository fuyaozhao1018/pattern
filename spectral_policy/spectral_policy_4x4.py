#!/usr/bin/env python3
"""
Deterministic 4x4 spectral policy using the same pattern family (DCT features) as 3x3.

We don't assume an optimal 4x4 tree. Instead, we estimate move values via
Monte Carlo rollouts vs Random to build a regression dataset, then fit ridge
regression s(phi)=w^T phi on DCT(X), DCT(O) features (32 dims).

- Board: 4x4, win condition is a full row/col/main diag/anti diag of same player.
- Features: DCT-II(4x4) coefficients of X and O occupancy grids concatenated.
- Labels: For sampled states and candidate moves, average rollout outcome
  (+1 win, 0 draw, -1 loss) as target.
- Policy: immediate win/block, else argmax s(phi).
- Eval: W/D/L vs Random over N games.
- Visualization: optional one game with 4x4 score grid and top-K feature contributions.

This transfers the pattern idea (frequency-domain basis) cleanly to 4x4.
"""
import math
import random
import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np

# ---------- 4x4 TicTacToe core ----------
class TicTacToeNxN:
    def __init__(self, n=4):
        self.n = n
        self.board = [0]*(n*n)
        self.current_player = 1
    def clone(self):
        g=TicTacToeNxN(self.n)
        g.board=self.board.copy()
        g.current_player=self.current_player
        return g
    def get_valid_moves(self):
        return [i for i,v in enumerate(self.board) if v==0]
    def make_move(self, m):
        self.board[m]=self.current_player
        self.current_player*=-1
    def check_winner(self):
        n=self.n; b=self.board
        # rows
        for r in range(n):
            s=sum(b[r*n+c] for c in range(n))
            if s==n: return 1
            if s==-n: return -1
        # cols
        for c in range(n):
            s=sum(b[r*n+c] for r in range(n))
            if s==n: return 1
            if s==-n: return -1
        # diags
        s=sum(b[i*n+i] for i in range(n))
        if s==n: return 1
        if s==-n: return -1
        s=sum(b[i*n+(n-1-i)] for i in range(n))
        if s==n: return 1
        if s==-n: return -1
        return 0
    def is_terminal(self):
        return self.check_winner()!=0 or 0 not in self.board

# ---------- DCT-II NxN ----------
def dct_basis_1d(N:int) -> np.ndarray:
    alpha = [math.sqrt(1/N)] + [math.sqrt(2/N)]*(N-1)
    C = np.zeros((N,N), dtype=float)
    for u in range(N):
        for x in range(N):
            C[u,x] = alpha[u]*math.cos(math.pi*(2*x+1)*u/(2*N))
    return C

def dct2(A: np.ndarray, C: np.ndarray) -> np.ndarray:
    return C @ A @ C.T

# Features: DCT(X) + DCT(O)

def board_to_channels(board: List[int], n:int) -> Tuple[np.ndarray,np.ndarray]:
    X=np.zeros((n,n), dtype=float)
    O=np.zeros((n,n), dtype=float)
    for idx,val in enumerate(board):
        r,c=divmod(idx,n)
        if val==1: X[r,c]=1.0
        elif val==-1: O[r,c]=1.0
    return X,O

def board_symmetries_nxn(board: List[int], n: int) -> List[List[int]]:
    """Generate all 8 symmetries (4 rotations + 4 flips) of an NxN board."""
    X, O = board_to_channels(board, n)
    syms = []
    
    def grid_to_board(X_sym, O_sym):
        b = [0] * (n*n)
        for i in range(n*n):
            r, c = divmod(i, n)
            if X_sym[r, c] > 0.5:
                b[i] = 1
            elif O_sym[r, c] > 0.5:
                b[i] = -1
        return b
    
    # Original
    syms.append(board)
    
    # 3 rotations (90, 180, 270 degrees)
    X_rot, O_rot = X.copy(), O.copy()
    for _ in range(3):
        X_rot = np.rot90(X_rot)
        O_rot = np.rot90(O_rot)
        syms.append(grid_to_board(X_rot, O_rot))
    
    # Flip horizontal, then apply 3 rotations
    X_flip, O_flip = np.fliplr(X), np.fliplr(O)
    syms.append(grid_to_board(X_flip, O_flip))
    for _ in range(3):
        X_flip = np.rot90(X_flip)
        O_flip = np.rot90(O_flip)
        syms.append(grid_to_board(X_flip, O_flip))
    
    return syms

def features_from_board(board: List[int], n:int, C: np.ndarray, sym_avg: bool = False) -> np.ndarray:
    X,O = board_to_channels(board, n)
    DX = dct2(X, C)
    DO = dct2(O, C)
    if sym_avg:
        # Average DCT coefficients across symmetries (rotations + flips)
        DX_avg = np.zeros_like(DX)
        DO_avg = np.zeros_like(DO)
        for board_sym in board_symmetries_nxn(board, n):
            X_sym, O_sym = board_to_channels(board_sym, n)
            DX_sym = dct2(X_sym, C)
            DO_sym = dct2(O_sym, C)
            DX_avg += DX_sym
            DO_avg += DO_sym
        DX = DX_avg / 8.0
        DO = DO_avg / 8.0
    return np.concatenate([DX.flatten(), DO.flatten()])

# ---------- Monte Carlo rollout labels ----------

def random_playout_from_state(g: TicTacToeNxN) -> int:
    # returns outcome from perspective of player who just moved to reach g.current state? We'll compute at root.
    # We'll return winner (1,0,-1) from X perspective when game ends; we will transform sign at call site for player perspective.
    gg=g.clone()
    while not gg.is_terminal():
        m=random.choice(gg.get_valid_moves())
        gg.make_move(m)
    return gg.check_winner()


def estimate_move_value(state_board: List[int], player: int, move: int, n:int, rollouts:int) -> float:
    # Place move, then average rollout outcome from current player's perspective.
    g=TicTacToeNxN(n)
    g.board=state_board.copy()
    g.current_player=player
    g.make_move(move)
    # Now it's opponent's turn; we want expected result for the player who just played.
    total=0.0
    for _ in range(rollouts):
        winner = random_playout_from_state(g)
        # If winner equals player, +1; if draw 0; if opponent -1
        if winner==player: total += 1.0
        elif winner==0: total += 0.0
        else: total += -1.0
    return total/rollouts


def sample_states_for_dataset(n:int, samples:int, rollouts:int, seed:int=None, sym_avg: bool = False):
    if seed is not None:
        random.seed(seed)
    C = dct_basis_1d(n)
    X_rows=[]; y=[]
    # Sample via random partial games to get diverse states
    for s in range(samples):
        g=TicTacToeNxN(n)
        # play k random moves (k uniform 0..n*n-1) to get a state; stop if terminal
        k = random.randint(0, n*n-1)
        for i in range(k):
            if g.is_terminal(): break
            m = random.choice(g.get_valid_moves())
            g.make_move(m)
        if g.is_terminal():
            continue
        moves = g.get_valid_moves()
        p = g.current_player
        for m in moves:
            # label
            val = estimate_move_value(g.board, p, m, n, rollouts)
            # features from resulting board
            b = g.board.copy(); b[m]=p
            phi = features_from_board(b, n, C, sym_avg=sym_avg)
            X_rows.append(phi)
            y.append(val)
    if not X_rows:
        return None, None, C
    X=np.vstack(X_rows)
    y=np.array(y, dtype=float)
    return X,y,C

# ---------- Ridge regression ----------

def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float=1e-3) -> np.ndarray:
    d=X.shape[1]
    A = X.T @ X + lam*np.eye(d)
    b = X.T @ y
    w = np.linalg.solve(A,b)
    return w

# ---------- Policy ----------
class SpectralPolicy4x4:
    def __init__(self, w: np.ndarray, n:int, C: np.ndarray, sym_avg: bool = False, safety: bool = True):
        self.w=w; self.n=n; self.C=C; self.sym_avg=sym_avg; self.safety=safety
    def score_move(self, board: List[int], player:int, move:int) -> float:
        b=board.copy(); b[move]=player
        phi = features_from_board(b, self.n, self.C, sym_avg=self.sym_avg)
        return float(self.w @ phi)
    def choose_move(self, g: TicTacToeNxN):
        moves=g.get_valid_moves(); p=g.current_player
        # immediate win/block (optional)
        if self.safety:
            for m in moves:
                g.board[m]=p
                if g.check_winner()==p:
                    g.board[m]=0
                    return m
                g.board[m]=0
            for m in moves:
                g.board[m]=-p
                if g.check_winner()==-p:
                    g.board[m]=0
                    return m
                g.board[m]=0
        best=moves[0]; best_s=-1e9
        for m in moves:
            s=self.score_move(g.board, p, m)
            if s>best_s:
                best_s=s; best=m
        return best

class RandomPlayer4x4:
    def choose_move(self, g):
        return random.choice(g.get_valid_moves())

# ---------- Small MCTS opponent ----------
class MCTSPlayer4x4:
    def __init__(self, iterations: int = 200, c: float = 1.414):
        self.iterations = iterations
        self.c = c
        # stats: N(state), W(state), children[state] -> list of (move, next_state)
        self.N = {}
        self.W = {}
        self.children = {}

    def key(self, g: TicTacToeNxN):
        return (tuple(g.board), g.current_player, g.n)

    def expand(self, g: TicTacToeNxN):
        k = self.key(g)
        if k in self.children:
            return
        moves = g.get_valid_moves()
        kids = []
        for m in moves:
            gg = g.clone()
            gg.make_move(m)
            kids.append((m, (tuple(gg.board), gg.current_player, gg.n)))
        self.children[k] = kids
        self.N.setdefault(k, 0)
        self.W.setdefault(k, 0.0)
        for _, child_k in kids:
            self.N.setdefault(child_k, 0)
            self.W.setdefault(child_k, 0.0)

    def uct_select(self, k):
        # pick child that maximizes UCT
        total = self.N.get(k, 1) + 1e-9
        best = None
        best_score = -1e9
        for m, ck in self.children.get(k, []):
            n = self.N.get(ck, 0)
            w = self.W.get(ck, 0.0)
            q = (w / n) if n > 0 else 0.0
            u = self.c * math.sqrt(math.log(total) / (n + 1e-9))
            sc = q + u
            if sc > best_score:
                best_score = sc
                best = (m, ck)
        return best

    def rollout(self, g: TicTacToeNxN) -> int:
        gg = g.clone()
        while not gg.is_terminal():
            m = random.choice(gg.get_valid_moves())
            gg.make_move(m)
        return gg.check_winner()

    def choose_move(self, g: TicTacToeNxN):
        root = g.clone()
        root_k = self.key(root)
        self.expand(root)

        for _ in range(self.iterations):
            # selection
            path = []
            node = root.clone()
            k = self.key(node)
            path.append((k, node.current_player))
            while True:
                if node.is_terminal():
                    break
                self.expand(node)
                if any(self.N.get(ck, 0) == 0 for _, ck in self.children[k]):
                    # expand an unvisited child
                    unvisited = [(m, ck) for m, ck in self.children[k] if self.N.get(ck, 0) == 0]
                    m, ck = random.choice(unvisited)
                    node.make_move(m)
                    k = ck
                    path.append((k, node.current_player))
                    break
                else:
                    sel = self.uct_select(k)
                    if sel is None:
                        break
                    m, ck = sel
                    node.make_move(m)
                    k = ck
                    path.append((k, node.current_player))

            # rollout
            winner = self.rollout(node)
            # backprop: +1 if winner == player_to_move_at_node, -1 if opposite, 0 draw
            for state_k, player_to_move in path:
                self.N[state_k] = self.N.get(state_k, 0) + 1
                if winner == 0:
                    val = 0.0
                elif winner == player_to_move:
                    val = 1.0
                else:
                    val = -1.0
                self.W[state_k] = self.W.get(state_k, 0.0) + val

        # pick root child with highest N
        best_m = None
        best_n = -1
        for m, ck in self.children.get(root_k, []):
            n = self.N.get(ck, 0)
            if n > best_n:
                best_n = n
                best_m = m
        if best_m is None:
            # fallback
            moves = g.get_valid_moves()
            return random.choice(moves)
        return best_m

# ---------- Visualization ----------

def fmt_board_4x4(board: List[int], n:int) -> str:
    sym={1:'X', -1:'O', 0:'.'}
    rows=[]
    for r in range(n):
        rows.append(' '.join(sym[board[r*n+c]] for c in range(n)))
    return '\n'.join(rows)

def score_grid(policy: SpectralPolicy4x4, board: List[int], player:int) -> np.ndarray:
    n=policy.n
    grid=np.full((n,n), np.nan)
    for i in range(n*n):
        if board[i]==0:
            grid[i//n, i%n] = policy.score_move(board, player, i)
    return grid

def visualize_one_game(policy: SpectralPolicy4x4, opponent, first='policy', topk=8, seed=0):
    random.seed(seed)
    n=policy.n
    g=TicTacToeNxN(n)
    players=[policy, opponent] if first=='policy' else [opponent, policy]
    turn=0
    print('\n=== Visualizing one 4x4 game ===')
    while not g.is_terminal():
        cur=players[0]; p=g.current_player
        you=(cur is policy)
        print(f"\nTurn {turn} | Player: {'Policy' if you else type(cur).__name__} ({'X' if p==1 else 'O'})")
        print(fmt_board_4x4(g.board,n))
        if you:
            S=score_grid(policy, g.board, p)
            print('Spectral scores:')
            for r in range(n):
                print(' '.join(f"{S[r,c]:7.3f}" if not np.isnan(S[r,c]) else '  ----  ' for c in range(n)))
        m = cur.choose_move(g)
        g.make_move(m)
        players=[players[1], players[0]]
        turn+=1
    print('\nFinal board:')
    print(fmt_board_4x4(g.board,n))
    w=g.check_winner()
    print('Result:', 'X wins' if w==1 else ('O wins' if w==-1 else 'Draw'))

# ---------- Play utility ----------

def play_game(p1, p2, n=4):
    g=TicTacToeNxN(n)
    players=[p1,p2]
    while not g.is_terminal():
        m=players[0].choose_move(g)
        g.make_move(m)
        players=[players[1], players[0]]
    return g.check_winner()

# ---------- Main ----------

def main():
    ap=argparse.ArgumentParser(description='4x4 spectral policy via DCT features + regression with MC labels')
    ap.add_argument('--samples', type=int, default=600, help='Number of sampled states for dataset')
    ap.add_argument('--rollouts', type=int, default=8, help='MC rollouts per candidate move')
    ap.add_argument('--lam', type=float, default=1e-3, help='Ridge lambda')
    ap.add_argument('--sym-avg', action='store_true', help='Use symmetry-averaged DCT features instead of raw features.')
    ap.add_argument('--games-random', type=int, default=100, help='Games vs Random for eval')
    ap.add_argument('--no-safety', action='store_true', help='Disable safety overrides (no immediate win/block).')
    ap.add_argument('--games-mcts', type=int, default=0, help='Games vs MCTS for eval (0 to skip)')
    ap.add_argument('--mcts-iters', type=int, default=200, help='MCTS iterations per move')
    ap.add_argument('--seed', type=int, default=0, help='Global seed')
    ap.add_argument('--no-examples', action='store_true', help='Disable visualization example')
    ap.add_argument('--self-play', type=int, default=0, help='Number of self-play games (policy vs itself).')
    args=ap.parse_args()

    random.seed(args.seed)
    n=4
    print(f'Building dataset for {n}x{n} via MC rollouts: samples={args.samples}, rollouts={args.rollouts}, mode={"sym-avg" if args.sym_avg else "standard"} ...')
    X,y,C = sample_states_for_dataset(n, args.samples, args.rollouts, seed=args.seed, sym_avg=args.sym_avg)
    if X is None:
        print('No samples gathered (all sampled states were terminal). Try increasing --samples.')
        return
    print('Dataset:', X.shape, '| mean target:', float(y.mean()))

    w = fit_ridge(X,y,lam=args.lam)
    print('\nLearned weights (first 10 shown):', ', '.join(f'{w[i]:+.4f}' for i in range(min(10,len(w)))))

    policy = SpectralPolicy4x4(w, n, C, sym_avg=args.sym_avg, safety=(not args.no_safety))
    # Eval vs Random
    wins=draws=loss=0
    for i in range(args.games_random):
        # alternate first
        if i%2==0:
            r=play_game(policy, RandomPlayer4x4(), n)
        else:
            r=play_game(RandomPlayer4x4(), policy, n)
            r=-r
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

    if args.games_mcts > 0:
        # Eval vs MCTS
        mcts = MCTSPlayer4x4(iterations=args.mcts_iters)
        wins=draws=loss=0
        for i in range(args.games_mcts):
            # alternate first
            if i%2==0:
                r=play_game(policy, mcts, n)
            else:
                r=play_game(mcts, policy, n)
                r=-r
            if r==1: wins+=1
            elif r==0: draws+=1
            else: loss+=1
        print(f"Vs MCTS(iters={args.mcts_iters}) ({args.games_mcts}): W:{wins} D:{draws} L:{loss}")

    if not args.no_examples:
        visualize_one_game(policy, RandomPlayer4x4(), first='policy', topk=8, seed=args.seed)

if __name__=='__main__':
    main()
