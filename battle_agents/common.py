# ttt/common.py
from __future__ import annotations
import json, math
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Iterable, Set

# =========================
# Data structures
# =========================

@dataclass
class StateN3:
    id: str
    board: List[int]          # len=9, row-major, values: +1(X), -1(O), 0(empty)
    turn: int                 # whose turn: +1 / -1
    legal: List[int]          # legal positions (0..8)
    terminal: bool = False
    per_move: Dict[str, Dict[str, int]] = None  # per-move stats (wins/draws/losses)

@dataclass
class StateN4:
    id: str
    board: List[int]          # len=16
    turn: int
    legal: List[int]
    terminal: bool = False

# =========================
# Normalization: accept "X"/"O"/"." or 1/-1/0
# =========================

_STR2VAL = {
    'X': +1, 'x': +1, '+1': +1, '1': +1,
    'O': -1, 'o': -1, '-1': -1,
    '.': 0, '0': 0, '': 0, ' ': 0
}

def _norm_player(x: Any) -> int:
    if isinstance(x, int):
        if x in (+1, -1): return x
        raise ValueError(f"turn must be +1/-1, got {x}")
    if isinstance(x, str):
        if x in ('X','x','+1','1'): return +1
        if x in ('O','o','-1'):     return -1
    raise ValueError(f"turn must be +1/-1 or 'X'/'O', got {x}")

def _norm_cell(v: Any) -> int:
    if isinstance(v, int):
        if v in (-1, 0, +1): return v
        raise ValueError(f"board cell must be -1/0/+1, got {v}")
    s = str(v).strip()
    return _STR2VAL.get(s, 0)

def _norm_board(arr: List[Any]) -> List[int]:
    return [_norm_cell(v) for v in arr]

def _compat_id(r: Dict[str, Any]) -> str:
    return r.get('id') or r.get('state_id')

# =========================
# Loading (flexible parsing, ignore extra fields, normalize values)
# =========================

def load_n3_exhaustive(path: str) -> List[StateN3]:
    with open(path) as f:
        raw = json.load(f)
    out: List[StateN3] = []
    for r in raw:
        out.append(StateN3(
            id=_compat_id(r),
            board=_norm_board(r['board']),
            turn=_norm_player(r['turn']),
            legal=r.get('legal', []),
            terminal=r.get('terminal', False),
            per_move=r.get('per_move', {})
        ))
    return out

def load_n4_states(path: str) -> List[StateN4]:
    with open(path) as f:
        raw = json.load(f)
    out: List[StateN4] = []
    for r in raw:
        out.append(StateN4(
            id=_compat_id(r),
            board=_norm_board(r['board']),
            turn=_norm_player(r['turn']),
            legal=r.get('legal', []),
            terminal=r.get('terminal', False)
        ))
    return out

def load_n4_best(path: str) -> Dict[str, List[int]]:
    """
    Supported formats:
    1) dict: { state_id: [best_moves...] }
    2) dict: { state_id: {"best_set":[...]} }
    3) list: [ {"id"/"state_id":..., "best_set":[...]}, ... ]
    4) list: [ {"id"/"state_id":..., "best":[...]}, ... ]
    """
    with open(path) as f:
        raw = json.load(f)

    out: Dict[str, List[int]] = {}

    # Format 1/2: dict type
    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, dict):
                bs = v.get('best_set') or v.get('best') or v.get('moves')
            else:
                bs = v
            if bs is None:
                continue
            out[str(k)] = [int(x) for x in bs]
        return out

    # Format 3/4: list type
    if isinstance(raw, list):
        for r in raw:
            sid = _compat_id(r)
            if not sid:
                continue
            bs = r.get('best_set') or r.get('best') or r.get('moves')
            if bs is None:
                # Some files may contain {"top1": 5}; this is not a set, skip or wrap
                top1 = r.get('top1')
                if top1 is not None:
                    out[str(sid)] = [int(top1)]
                continue
            out[str(sid)] = [int(x) for x in bs]
        return out

    raise ValueError(f"Unrecognized n4_best format in {path}: type={type(raw)}")

# =========================
# Relative encoding
# =========================

def encode_relative_board(board: List[int] | List[Any], turn: int | str) -> List[int]:
    """
    Convert board to perspective of SELF = +1.
    Input:
        board: length N*N, elements in {+1,-1,0} or {'X','O','.'}
        turn:  current player {+1/-1 or 'X'/'O'}
    Output:
        Same length array in {+1,-1,0}
    """
    b = board if isinstance(board[0], int) else _norm_board(board)
    t = turn if isinstance(turn, int) else _norm_player(turn)
    if t == +1:
        return b[:]  # SELF is +1
    return [(-x) for x in b]  # If it's -1's turn, flip signs

# =========================
# Direction / window extraction (3x3 local)
# =========================

_OFF = '#'  # out of bounds
_EMP = 0    # center is forced to 0 (current move)

def _get_cell(rel_board: List[int], N: int, r: int, c: int):
    if r < 0 or c < 0 or r >= N or c >= N:
        return _OFF
    return rel_board[r * N + c]

def _line3(rel_board: List[int], N: int, r: int, c: int, dr: int, dc: int) -> Tuple[Any, Any, Any]:
    a = _get_cell(rel_board, N, r - dr, c - dc)
    b = _EMP
    d = _get_cell(rel_board, N, r + dr, c + dc)
    return (a, b, d)

def _window3(rel_board: List[int], N: int, r: int, c: int):
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                out.append(_EMP)
            else:
                out.append(_get_cell(rel_board, N, r + dr, c + dc))
    return tuple(out)

def extract_four_line3_n3(rel_board: List[int], pos: int):
    N = 3
    r, c = divmod(pos, N)
    yield _line3(rel_board, N, r, c, 0, 1)
    yield _line3(rel_board, N, r, c, 1, 0)
    yield _line3(rel_board, N, r, c, 1, 1)
    yield _line3(rel_board, N, r, c, 1, -1)

def extract_window3_n3(rel_board: List[int], pos: int):
    N = 3
    r, c = divmod(pos, N)
    return _window3(rel_board, N, r, c)

def extract_four_line3_n4(rel_board: List[int], pos: int):
    N = 4
    r, c = divmod(pos, N)
    yield _line3(rel_board, N, r, c, 0, 1)
    yield _line3(rel_board, N, r, c, 1, 0)
    yield _line3(rel_board, N, r, c, 1, 1)
    yield _line3(rel_board, N, r, c, 1, -1)

def extract_window3_n4(rel_board: List[int], pos: int):
    N = 4
    r, c = divmod(pos, N)
    return _window3(rel_board, N, r, c)

# =========================
# Laplace-smoothed logit
# =========================

def laplace_logit(mean_p: float, cnt: int, tau: int = 10, prior: float = 0.5) -> float:
    if cnt < 0:
        raise ValueError("cnt must be non-negative")
    mp = max(min(mean_p, 1.0 - 1e-9), 1e-9)
    p = (cnt * mp + tau * prior) / (cnt + tau) if (cnt + tau) > 0 else prior
    p = max(min(p, 1.0 - 1e-12), 1e-12)
    return math.log(p / (1.0 - p))



def pretty_board(board: List[int], N: int) -> str:
    mp = {+1: 'X', -1: 'O', 0: '.'}
    rows = []
    for r in range(N):
        rows.append(' '.join(mp[board[r*N + c]] for c in range(N)))
    return '\n'.join(rows)

# =========================
# N=4 immediate win / must-block detection
# =========================

def _n4_all_lines() -> List[Tuple[int, int, int, int]]:
    """Return all 4-in-a-row lines (rows/cols/diagonals) on a 4x4 board."""
    lines: List[Tuple[int, int, int, int]] = []
    N = 4
    # rows
    for r in range(N):
        base = r * N
        lines.append((base + 0, base + 1, base + 2, base + 3))
    # cols
    for c in range(N):
        lines.append((c, c + 4, c + 8, c + 12))
    # diags
    lines.append((0, 5, 10, 15))
    lines.append((3, 6, 9, 12))
    return lines

_N4_LINES = _n4_all_lines()

def _is_win_n4(board: List[int], player: int) -> bool:
    for a, b, c, d in _N4_LINES:
        s = board[a] + board[b] + board[c] + board[d]
        if s == 4 * player:
            return True
    return False

def _will_win_if_play_n4(board: List[int], move: int, player: int) -> bool:
    if board[move] != 0:
        return False
    tmp = board[:]
    tmp[move] = player
    return _is_win_n4(tmp, player)

def n4_must_win_moves(board: List[int], player: int, legal: Iterable[int]) -> Set[int]:
    """Return all legal moves that give the current player an immediate win."""
    out: Set[int] = set()
    for m in legal:
        if _will_win_if_play_n4(board, m, player):
            out.add(int(m))
    return out

def n4_must_block_moves(board: List[int], player: int, legal: Iterable[int]) -> Set[int]:
    """If opponent has an immediate winning move, these are the legal positions that must be played to block."""
    opp = -player
    opp_wins: Set[int] = set()
    for m in range(16):
        if board[m] == 0 and _will_win_if_play_n4(board, m, opp):
            opp_wins.add(m)
    return {int(m) for m in legal if int(m) in opp_wins}

def check_winner(board: List[str], n: int, k: int) -> str:
    """
    Check if there is a winner on an N×N board with K in a row required.
    
    Args:
        board: List of 'X', 'O', ' ' representing the board
        n: board size (NxN)
        k: win length (K in a row)
        
    Returns:
        'X', 'O', or None if no winner
    """
    # Check rows
    for r in range(n):
        for c in range(n - k + 1):
            if board[r*n+c] != ' ':
                if all(board[r*n+c+i] == board[r*n+c] for i in range(k)):
                    return board[r*n+c]
                    
    # Check columns
    for c in range(n):
        for r in range(n - k + 1):
            if board[r*n+c] != ' ':
                if all(board[(r+i)*n+c] == board[r*n+c] for i in range(k)):
                    return board[r*n+c]
                    
    # Check diagonals
    for r in range(n - k + 1):
        for c in range(n - k + 1):
            if board[r*n+c] != ' ':
                if all(board[(r+i)*n+(c+i)] == board[r*n+c] for i in range(k)):
                    return board[r*n+c]
                    
    # Check anti-diagonals
    for r in range(n - k + 1):
        for c in range(k - 1, n):
            if board[r*n+c] != ' ':
                if all(board[(r+i)*n+(c-i)] == board[r*n+c] for i in range(k)):
                    return board[r*n+c]
                    
    return None

def print_board(board: List[str], n: int):
    """
    Pretty-print the board to console.
    """
    print("  " + " ".join([f"{i}" for i in range(n)]))
    for r in range(n):
        print(f"{r} " + " ".join([board[r*n+c] for c in range(n)]))
