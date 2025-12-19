#!/usr/bin/env python3
"""
Deterministic 3x3 policy from exhaustive data using interpretable spectral features.

Idea:
- For each non-terminal state, for each valid move, compute a feature vector
  from the resulting board using 3x3 DCT-II coefficients for X and O channels.
- Train a single linear scoring function s(phi) = w^T phi via ridge regression
  to give higher scores to the optimal move from the exhaustive tree.
- At inference, for a given board, score each valid move and select argmax.
- No neural nets, no heuristics; deterministic closed-form solution.

Outputs:
- prints weights per feature (DCT[u,v] for X and O), training accuracy,
  and W/D/L vs Random and vs Perfect (minimax) over 50 games.

Run:
    C:/Users/ASUS/anaconda3/python.exe spectral_policy_3x3.py
"""
import json
import math
import numpy as np
from typing import List, Tuple, Dict, Iterable, Set
import random
import argparse
from pathlib import Path

TREE_FILE = 'optimal_decision_tree.json'

# ---------- TicTacToe 3x3 core ----------
class TicTacToe3x3:
    def __init__(self):
        self.board = [0]*9  # 0 empty, 1 X, -1 O
        self.current_player = 1
    def get_valid_moves(self):
        return [i for i in range(9) if self.board[i]==0]
    def check_winner(self):
        b = self.board
        lines = [
            (0,1,2),(3,4,5),(6,7,8),
            (0,3,6),(1,4,7),(2,5,8),
            (0,4,8),(2,4,6)
        ]
        for a,b1,c in lines:
            if self.board[a]==self.board[b1]==self.board[c]!=0:
                return self.board[a]
        return 0
    def is_terminal(self):
        return self.check_winner()!=0 or 0 not in self.board
    def make_move(self, m):
        self.board[m]=self.current_player
        self.current_player*=-1

# ---------- DCT-II (3x3) ----------
# Orthonormal 1D DCT-II basis for N=3
N=3
alpha = [math.sqrt(1/N)] + [math.sqrt(2/N)]*(N-1)
C1 = np.zeros((N,N), dtype=float)
for u in range(N):
    for x in range(N):
        C1[u,x] = alpha[u]*math.cos(math.pi*(2*x+1)*u/(2*N))
# 2D DCT via separability: C * A * C^T

def dct2_3x3(A: np.ndarray) -> np.ndarray:
    return C1 @ A @ C1.T

# Feature extractor: given board (after a candidate move), build 18-dim features
# channels: X grid and O grid (1 where occupies, 0 else)

def board_to_channels(board: List[int]) -> Tuple[np.ndarray,np.ndarray]:
    X = np.zeros((3,3), dtype=float)
    O = np.zeros((3,3), dtype=float)
    for idx,val in enumerate(board):
        r,c = divmod(idx,3)
        if val==1: X[r,c]=1.0
        elif val==-1: O[r,c]=1.0
    return X,O

def board_symmetries_3x3(board: List[int]) -> List[List[int]]:
    """Generate all 8 symmetries (4 rotations + 4 flips) of a 3x3 board."""
    X, O = board_to_channels(board)
    syms = []
    grids = [X, O]
    
    def grid_to_board(X_sym, O_sym):
        b = [0] * 9
        for i in range(9):
            r, c = divmod(i, 3)
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

def features_from_board(board: List[int], sym_avg: bool = False) -> np.ndarray:
    X,O = board_to_channels(board)
    DX = dct2_3x3(X)
    DO = dct2_3x3(O)
    if sym_avg:
        # Average DCT coefficients across symmetries (rotations + flips)
        DX_avg = np.zeros_like(DX)
        DO_avg = np.zeros_like(DO)
        for board_sym in board_symmetries_3x3(board):
            X_sym, O_sym = board_to_channels(board_sym)
            DX_sym = dct2_3x3(X_sym)
            DO_sym = dct2_3x3(O_sym)
            DX_avg += DX_sym
            DO_avg += DO_sym
        DX = DX_avg / 8.0
        DO = DO_avg / 8.0
    # vectorize in fixed order (u,v) row-major
    return np.concatenate([DX.flatten(), DO.flatten()])  # 18 dims

# ---------- Dataset from exhaustive tree ----------

def load_tree():
    with open(TREE_FILE,'r',encoding='utf-8') as f:
        data = json.load(f)
    return data['states']

def parse_player_to_int(p_val) -> int:
    if isinstance(p_val, str):
        return 1 if p_val.upper()=='X' else -1
    try:
        return int(p_val)
    except Exception:
        return 1

def build_optimal_moves_map() -> Dict[Tuple[Tuple[int,...],int], Set[int]]:
    states = load_tree()
    opt_map: Dict[Tuple[Tuple[int,...],int], Set[int]] = {}
    for state_str, info in states.items():
        try:
            s = tuple(json.loads(state_str))
        except Exception:
            continue
        if 0 not in s:
            continue
        p = parse_player_to_int(info.get('player','X'))
        details = info.get('move_details', [])
        if not details:
            best = info.get('move', None)
            if best is not None:
                opt_map[(s,p)] = {int(best)}
            continue
        max_score = max(d.get('score', -1e9) for d in details)
        tol = 1e-6
        best_moves = {int(d['move']) for d in details if abs(d.get('score', -1e9) - max_score) <= tol}
        if not best_moves and 'move' in info:
            best_moves = {int(info['move'])}
        opt_map[(s,p)] = best_moves
    return opt_map


def build_dataset() -> Tuple[np.ndarray,np.ndarray,List[Tuple[Tuple[int,...],int,int]],bool]:
    states = load_tree()
    X_rows=[]
    y=[]
    meta=[]  # (state_tuple, move, player)
    sym_avg_mode = False  # Flag to track if sym_avg is enabled globally
    for state_str,info in states.items():
        try:
            s = tuple(json.loads(state_str))
        except Exception:
            continue
        if 0 not in s:  # terminal
            continue
        move_opt = info.get('move')
        p_str = info.get('player', 'X')
        player = parse_player_to_int(p_str)
        # Build candidate rows for all valid moves
        valid = [i for i,v in enumerate(s) if v==0]
        for m in valid:
            b = list(s)
            b[m] = player
            phi = features_from_board(b, sym_avg=sym_avg_mode)
            X_rows.append(phi)
            y.append(1.0 if m==move_opt else 0.0)
            meta.append((s,m,player))
    X = np.vstack(X_rows)
    y = np.array(y, dtype=float)
    return X,y,meta,sym_avg_mode

def build_dataset_with_sym_avg(sym_avg: bool = False) -> Tuple[np.ndarray,np.ndarray,List[Tuple[Tuple[int,...],int,int]]]:
    states = load_tree()
    X_rows=[]
    y=[]
    meta=[]  # (state_tuple, move, player)
    for state_str,info in states.items():
        try:
            s = tuple(json.loads(state_str))
        except Exception:
            continue
        if 0 not in s:  # terminal
            continue
        move_opt = info.get('move')
        p_str = info.get('player', 'X')
        player = parse_player_to_int(p_str)
        # Build candidate rows for all valid moves
        valid = [i for i,v in enumerate(s) if v==0]
        for m in valid:
            b = list(s)
            b[m] = player
            phi = features_from_board(b, sym_avg=sym_avg)
            X_rows.append(phi)
            y.append(1.0 if m==move_opt else 0.0)
            meta.append((s,m,player))
    X = np.vstack(X_rows)
    y = np.array(y, dtype=float)
    return X,y,meta

# ---------- Closed-form ridge regression ----------

def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float=1e-3) -> np.ndarray:
    d = X.shape[1]
    A = X.T @ X + lam*np.eye(d)
    b = X.T @ y
    w = np.linalg.solve(A,b)
    return w

# ---------- Grouped accuracy (state-level) ----------
def grouped_accuracy(w: np.ndarray, meta: List[Tuple[Tuple[int,...],int,int]],
                     X: np.ndarray, idxs: Iterable[int]) -> float:
    by_state: Dict[Tuple[Tuple[int,...],int], List[Tuple[float,int,float]]] = {}
    for i in idxs:
        s,m,p = meta[i]
        yy = 1.0 if True else 0.0  # placeholder; labels not needed here
        score = float(w @ X[i])
        by_state.setdefault((s,p), []).append((score, m, 1.0 if yy>0.5 else 0.0))
    correct=0; total=0
    for key,lst in by_state.items():
        # infer labels from meta/X pairing: positive label corresponds to the single optimal move per state
        # we can infer because in our dataset construction y==1 for the optimal move; we need the index mapping.
        # Easiest is to pick the best score and check if it corresponds to the move that was labeled 1 in meta.
        # But labels are not provided here; we'll rebuild y_map inline during CV using a separate map.
        # This function will be only used when we construct with provided y_map.
        pass
    return 0.0

def build_state_index(meta: List[Tuple[Tuple[int,...],int,int]], y: np.ndarray) -> Dict[Tuple[Tuple[int,...],int], Dict[str,object]]:
    groups: Dict[Tuple[Tuple[int,...],int], Dict[str,object]] = {}
    for i, ((s,m,p), yy) in enumerate(zip(meta, y)):
        g = groups.setdefault((s,p), {'idxs': [], 'pos_move': None})
        g['idxs'].append(i)
        if yy>0.5:
            g['pos_move'] = m
    return groups

def accuracy_from_groups(w: np.ndarray, X: np.ndarray, groups: Dict[Tuple[Tuple[int,...],int], Dict[str,object]],
                         keys: Iterable[Tuple[Tuple[int,...],int]], meta: List[Tuple[Tuple[int,...],int,int]]) -> float:
    correct=0; total=0
    for key in keys:
        g = groups[key]
        idxs = g['idxs']
        pos_move = g['pos_move']
        scores = [(float(w @ X[i]), meta[i][1]) for i in idxs]
        best_m = max(scores, key=lambda t: t[0])[1]
        correct += 1 if best_m == pos_move else 0
        total += 1
    return correct/total if total>0 else 0.0

def kfold_cv_lambda(X: np.ndarray, y: np.ndarray, meta: List[Tuple[Tuple[int,...],int,int]],
                    lam_list: List[float], k: int=5, seed: int=0) -> Tuple[float, Dict[float,float]]:
    rng = random.Random(seed)
    groups = build_state_index(meta, y)
    keys = list(groups.keys())
    rng.shuffle(keys)
    folds = [keys[i::k] for i in range(k)]
    lam_scores: Dict[float,float] = {lam:0.0 for lam in lam_list}
    for lam in lam_list:
        acc_sum = 0.0
        for fi in range(k):
            val_keys = set(folds[fi])
            train_keys = [kk for kk in keys if kk not in val_keys]
            # collect train indices
            train_idxs = [i for kk in train_keys for i in groups[kk]['idxs']]
            val_idxs_keys = list(val_keys)
            # fit
            w = fit_ridge(X[train_idxs], y[train_idxs], lam=lam)
            acc = accuracy_from_groups(w, X, groups, val_idxs_keys, meta)
            acc_sum += acc
        lam_scores[lam] = acc_sum / k
    best_lam = max(lam_scores.items(), key=lambda kv: (kv[1], -math.log10(kv[0]+1e-20)))[0]
    return best_lam, lam_scores

# ---------- Policy ----------
class SpectralPolicy3x3:
    def __init__(self, w: np.ndarray, sym_avg: bool = False, safety: bool = True):
        self.w = w
        self.sym_avg = sym_avg
        self.safety = safety
    def score_move(self, board: List[int], player: int, move: int) -> float:
        b = board.copy()
        b[move]=player
        phi = features_from_board(b, sym_avg=self.sym_avg)
        return float(self.w @ phi)
    def choose_move(self, game: TicTacToe3x3):
        # Safety: immediate win/block (optional)
        moves = game.get_valid_moves()
        p = game.current_player
        if self.safety:
            for m in moves:
                game.board[m]=p
                if game.check_winner()==p:
                    game.board[m]=0
                    return m
                game.board[m]=0
            for m in moves:
                game.board[m]=-p
                if game.check_winner()==-p:
                    game.board[m]=0
                    return m
                game.board[m]=0
        # Spectral score
        best=moves[0]
        best_s=-1e9
        for m in moves:
            s = self.score_move(game.board, p, m)
            if s>best_s:
                best_s=s
                best=m
        return best

# ---------- Visualization helpers ----------
def fmt_board(board: List[int]) -> str:
    sym = {1:'X', -1:'O', 0:'.'}
    rows=[]
    for r in range(3):
        rows.append(' '.join(sym[board[3*r+c]] for c in range(3)))
    return '\n'.join(rows)

def score_grid(policy: SpectralPolicy3x3, board: List[int], player: int) -> np.ndarray:
    grid = np.full((3,3), np.nan)
    for i in range(9):
        if board[i]==0:
            grid[i//3, i%3] = policy.score_move(board, player, i)
    return grid

def immediate_win_moves(board: List[int], player: int) -> List[int]:
    g=TicTacToe3x3(); g.board=board.copy()
    wins=[]
    for m in [i for i,v in enumerate(board) if v==0]:
        g.board[m]=player
        if g.check_winner()==player:
            wins.append(m)
        g.board[m]=0
    return wins

def choose_move_and_explain(policy: SpectralPolicy3x3, game: TicTacToe3x3, topk:int=5):
    moves = game.get_valid_moves()
    p = game.current_player
    # safety (optional)
    if policy.safety:
        wins = immediate_win_moves(game.board, p)
        if wins:
            return wins[0], {'mode':'win', 'scores': None, 'explain': []}
        blocks = immediate_win_moves(game.board, -p)
        if blocks:
            return blocks[0], {'mode':'block', 'scores': None, 'explain': []}
    # spectral
    scores = [(m, policy.score_move(game.board, p, m)) for m in moves]
    m_best, s_best = max(scores, key=lambda t: t[1])
    # feature contributions
    b = game.board.copy(); b[m_best]=p
    phi = features_from_board(b)
    contrib = policy.w * phi
    labels = [f'X[{i//3},{i%3}]' for i in range(9)] + [f'O[{i//3},{i%3}]' for i in range(9)]
    idxs = np.argsort(-np.abs(contrib))[:topk]
    explain = [(labels[i], float(phi[i]), float(policy.w[i]), float(contrib[i])) for i in idxs]
    return m_best, {'mode':'spectral', 'scores': scores, 'explain': explain}

def visualize_one_game(policy: SpectralPolicy3x3, opponent, first: str='policy', topk:int=5, seed:int=0):
    random.seed(seed)
    g=TicTacToe3x3()
    players = [policy, opponent] if first=='policy' else [opponent, policy]
    turn=0
    print('\n=== Visualizing one game ===')
    while not g.is_terminal():
        cur = players[0]
        p = g.current_player
        you = (cur is policy)
        print(f"\nTurn {turn} | Player: {'Policy' if you else type(cur).__name__} ({'X' if p==1 else 'O'})")
        print(fmt_board(g.board))
        if you:
            # Show score grid
            S = score_grid(policy, g.board, p)
            print('Spectral scores (higher is better):')
            for r in range(3):
                print(' '.join(f"{S[r,c]:6.3f}" if not np.isnan(S[r,c]) else '  ----' for c in range(3)))
            m, info = choose_move_and_explain(policy, g, topk=topk)
            if info['mode']!='spectral':
                print(f"Decision: {info['mode']} at cell {m}")
            else:
                print(f"Decision: spectral argmax at cell {m}")
                print('Top feature contributions [feature, phi, weight, phi*weight]:')
                for name,phi_i,w_i,c_i in info['explain']:
                    print(f"  {name:7s}  phi={phi_i:+.3f}  w={w_i:+.4f}  contrib={c_i:+.4f}")
        else:
            m = cur.choose_move(g)
        g.make_move(m)
        players=[players[1], players[0]]
        turn+=1
    print('\nFinal board:')
    print(fmt_board(g.board))
    w = g.check_winner()
    print('Result:', 'X wins' if w==1 else ('O wins' if w==-1 else 'Draw'))
    return w

def visualize_first_outcomes(policy: SpectralPolicy3x3, opponent_kind: str, max_seeds: int, topk:int=5, start_seed:int=0):
    print(f"\n=== Searching first W/D/L examples vs {opponent_kind.capitalize()} (up to {max_seeds} seeds, start_seed={start_seed}) ===")
    opp = RandomPlayer3x3() if opponent_kind=='random' else PerfectPlayer3x3()
    first_seen = {'W': None, 'D': None, 'L': None}
    for s in range(start_seed, start_seed+max_seeds):
        # Try policy first perspective
        random.seed(s)
        g=TicTacToe3x3()
        res = play_game(policy, opp)
        tag = 'W' if res==1 else ('D' if res==0 else 'L')
        if first_seen[tag] is None:
            first_seen[tag] = ('policy', s)
        # If all found, stop
        if all(v is not None for v in first_seen.values()):
            break
    # Visualize in order W, D, L if exist
    order = ['W','D','L']
    labels = {'W': 'Win', 'D': 'Draw', 'L': 'Loss'}
    for k in order:
        item = first_seen[k]
        if item is None:
            continue
        who, seed = item
        print(f"\n--- First {labels[k]} example (seed={seed}) ---")
        visualize_one_game(policy, opp, first='policy', topk=topk, seed=seed)

# ---------- Perfect player (minimax) ----------
class PerfectPlayer3x3:
    def __init__(self):
        self.cache={}
    def eval(self, b):
        g=TicTacToe3x3(); g.board=b.copy()
        return g.check_winner()
    def minimax(self, b, maxing):
        t=tuple(b)
        if t in self.cache: return self.cache[t]
        g=TicTacToe3x3(); g.board=b.copy()
        w=self.eval(b)
        if w!=0: self.cache[t]=w; return w
        if 0 not in b: self.cache[t]=0; return 0
        moves=[i for i,v in enumerate(b) if v==0]
        if maxing:
            v=-2
            for m in moves:
                b[m]=1; v=max(v, self.minimax(b, False)); b[m]=0
            self.cache[t]=v; return v
        else:
            v=2
            for m in moves:
                b[m]=-1; v=min(v, self.minimax(b, True)); b[m]=0
            self.cache[t]=v; return v
    def choose_move(self, game: TicTacToe3x3):
        moves=game.get_valid_moves()
        best=moves[0]
        if game.current_player==1:
            bv=-2
            for m in moves:
                game.board[m]=1; v=self.minimax(game.board, False); game.board[m]=0
                if v>bv: bv=v; best=m
        else:
            bv=2
            for m in moves:
                game.board[m]=-1; v=self.minimax(game.board, True); game.board[m]=0
                if v<bv: bv=v; best=m
        return best

# ---------- Evaluation ----------
class RandomPlayer3x3:
    def choose_move(self, game):
        return random.choice(game.get_valid_moves())

def play_game(p1, p2):
    g=TicTacToe3x3()
    players=[p1,p2]
    while not g.is_terminal():
        m=players[0].choose_move(g)
        g.make_move(m)
        players=[players[1], players[0]]
    return g.check_winner()


def main():
    parser = argparse.ArgumentParser(description='Deterministic spectral policy (3x3) using DCT features + ridge regression.')
    parser.add_argument('--lam', type=float, default=None, help='Ridge lambda. If omitted, selected by k-fold CV.')
    parser.add_argument('--cv-k', type=int, default=5, help='K for cross-validation.')
    parser.add_argument('--lam-grid', type=str, default='1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1', help='Comma-separated lambda grid for CV.')
    parser.add_argument('--save-weights', type=str, default='', help='Optional path to save learned weights as JSON.')
    parser.add_argument('--sym-avg', action='store_true', help='Use symmetry-averaged DCT features instead of raw features.')
    parser.add_argument('--no-play', action='store_true', help='Skip gameplay evaluation.')
    parser.add_argument('--no-safety', action='store_true', help='Disable safety overrides (no immediate win/block).')
    parser.add_argument('--show-game', type=str, default='', choices=['', 'random', 'perfect'], help='Visualize one game vs chosen opponent.')
    parser.add_argument('--topk', type=int, default=5, help='Top-K feature contributions to display for chosen moves.')
    parser.add_argument('--games-random', type=int, default=50, help='Number of games vs Random to play in evaluation block.')
    parser.add_argument('--games-perfect', type=int, default=50, help='Number of games vs Perfect to play in evaluation block (alternating first).')
    parser.add_argument('--no-examples', action='store_true', help='Disable visualization of first win/draw/loss examples.')
    parser.add_argument('--seed', type=int, default=None, help='Set global random seed for reproducibility.')
    parser.add_argument('--self-play', type=int, default=0, help='Number of self-play games (policy vs itself).')
    args = parser.parse_args()

    print('Building dataset from exhaustive tree...')
    X,y,meta = build_dataset_with_sym_avg(sym_avg=args.sym_avg)
    print('Samples:', len(y), '| Dim:', X.shape[1], '| Mode:', 'sym-avg' if args.sym_avg else 'standard')

    chosen_lam = args.lam
    if chosen_lam is None:
        lam_list = [float(s) for s in args.lam_grid.split(',')]
        print(f'Running {args.cv_k}-fold CV over lambdas: {lam_list}')
        best_lam, lam_scores = kfold_cv_lambda(X,y,meta, lam_list=lam_list, k=args.cv_k, seed=42)
        print('CV scores:')
        for lam, sc in sorted(lam_scores.items(), key=lambda kv: kv[0]):
            print(f'  lam={lam:g}: acc={sc:.4f}')
        print(f'Best lambda: {best_lam:g}')
        chosen_lam = best_lam

    # Fit final
    w = fit_ridge(X,y,lam=chosen_lam)
    print(f'\nChosen lambda: {chosen_lam:g}')
    print('\nWeights (DCT X then DCT O):')
    for i,val in enumerate(w):
        ch = 'X' if i<9 else 'O'
        idx = i if i<9 else i-9
        u,v = divmod(idx,3)
        print(f'  {ch}[{u},{v}]: {val:.4f}')
    if args.save_weights:
        out = {
            'lambda': chosen_lam,
            'weights': w.tolist(),
            'feature_order': [f'X[{i//3},{i%3}]' for i in range(9)] + [f'O[{i//3},{i%3}]' for i in range(9)],
            'description': 'DCT-II 3x3 coefficients for X then O channels.'
        }
        Path(args.save_weights).write_text(json.dumps(out, indent=2), encoding='utf-8')
        print(f'Saved weights to {args.save_weights}')
    # Training accuracy: does argmax reproduce optimal move?
    # For each original state, compare best-scored move with label 1.0
    correct=0; total=0
    by_state = {}
    for (s,m,p), yy, phi in zip(meta, y, X):
        by_state.setdefault((s,p), []).append((m, yy, phi))
    for key, lst in by_state.items():
        # choose move with highest predicted score
        scores=[(float(w@phi), m, yy) for (m,yy,phi) in lst]
        best_m = max(scores, key=lambda t: t[0])[1]
        # check if that move had label 1
        is_correct = any((m==best_m and label>0.5) for (_, m, label) in scores)
        correct += 1 if is_correct else 0
        total += 1
    print(f"\nOptimal-move reproduction: {correct}/{total} = {100*correct/total:.1f}% of states")

    # Additional metrics: tie-aware accuracy and random baselines
    opt_map = build_optimal_moves_map()
    # Compute predictions per state
    preds: Dict[Tuple[Tuple[int,...],int], int] = {}
    valids_count: Dict[Tuple[Tuple[int,...],int], int] = {}
    for key, lst in by_state.items():
        scores=[(float(w@phi), m) for (m,yy,phi) in lst]
        best_m = max(scores, key=lambda t: t[0])[1]
        preds[key] = best_m
        valids_count[key] = len(lst)
    # Tie-aware accuracy: predicted in any optimal move
    tie_correct = sum(1 for key, pm in preds.items() if pm in opt_map.get(key, {None}))
    print(f"Tie-aware reproduction (any optimal): {tie_correct}/{total} = {100*tie_correct/total:.1f}% of states")
    # Random baselines
    avg_rand_canonical = sum(1.0/valids_count[k] for k in preds.keys())/total
    avg_rand_tie = sum((len(opt_map.get(k,set()))/valids_count[k]) for k in preds.keys())/total
    print(f"Random baseline (canonical): {100*avg_rand_canonical:.1f}% | Random baseline (any optimal): {100*avg_rand_tie:.1f}%")

    # --- Gameplay evaluation and visualization ---
    if not args.no_play:
        # Set seed if provided
        if args.seed is not None:
            random.seed(args.seed)
        else:
            random.seed(0)
        # Play matches
        policy = SpectralPolicy3x3(w, sym_avg=args.sym_avg, safety=(not args.no_safety))
        # vs Random
        wns=drw=los=0
        for _ in range(args.games_random):
            r=play_game(policy, RandomPlayer3x3())
            if r==1: wns+=1
            elif r==0: drw+=1
            else: los+=1
        line_random = f"Vs Random ({args.games_random}): W:{wns} D:{drw} L:{los}"
        # vs Perfect (alternating first)
        wns=drw=los=0
        for i in range(args.games_perfect):
            if i%2==0:
                r=play_game(policy, PerfectPlayer3x3())
            else:
                r=play_game(PerfectPlayer3x3(), policy)
                r = -r  # invert perspective
            if r==1: wns+=1
            elif r==0: drw+=1
            else: los+=1
        line_perfect = f"Vs Perfect ({args.games_perfect} alt): W:{wns} D:{drw} L:{los}"
        print(f"\n{line_random}\n{line_perfect}")
        # Self-play
        if args.self_play > 0:
            wns=drw=los=0
            for i in range(args.self_play):
                # alternate first
                if i%2==0:
                    r=play_game(policy, policy)
                else:
                    r=play_game(policy, policy)
                if r==1: wns+=1
                elif r==0: drw+=1
                else: los+=1
            print(f"Self-play ({args.self_play}): W:{wns} D:{drw} L:{los}")
        if not args.no_examples:
            # Visualize first W/D/L examples vs Random
            visualize_first_outcomes(policy, 'random', max_seeds=args.games_random, topk=args.topk, start_seed=(args.seed or 0))
            # Visualize one (draw) example vs Perfect
            visualize_first_outcomes(policy, 'perfect', max_seeds=1, topk=args.topk, start_seed=(args.seed or 0))

    # Visualization of one game if requested
    if args.show_game:
        policy = SpectralPolicy3x3(w, sym_avg=args.sym_avg, safety=(not args.no_safety))
        opp = RandomPlayer3x3() if args.show_game=='random' else PerfectPlayer3x3()
        # Show policy first vs requested opponent
        visualize_one_game(policy, opp, first='policy', topk=args.topk, seed=(args.seed or 0))

if __name__=='__main__':
    main()
