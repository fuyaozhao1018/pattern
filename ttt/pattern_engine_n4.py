# ttt/pattern_engine_n4.py
from __future__ import annotations
import argparse, json, os, statistics
from multiprocessing import Pool
from typing import Any, Dict, Iterable, Tuple, List

from ttt.common import (
    load_n4_states, encode_relative_board,
    extract_four_line3_n4, extract_window3_n4,
    n4_must_win_moves, n4_must_block_moves
)

# ------------ Scoring engine ------------
class PatternEngine:
    def __init__(self, dir_lib=None, win_lib=None,
                 lam_win: float = 0.3,
                 log_w_pos: dict[str, float] | None = None,
                 lam_prior: float = 0.5,
                 clip_logit: float | None = None):
        self.dir_lib = dir_lib or {}
        self.win_lib = win_lib or {}
        self.lam_win = float(lam_win)
        self.log_w_pos = log_w_pos  # {str(pos): log_w}
        self.lam_prior = float(lam_prior)
        self.clip_logit = float(clip_logit) if clip_logit is not None else None

    def _clip(self, v: float) -> float:
        if self.clip_logit is None:
            return v
        if v > self.clip_logit:
            return self.clip_logit
        if v < -self.clip_logit:
            return -self.clip_logit
        return v

    def score_rel(self, rel_board, pos: int) -> Tuple[float, Dict[str, float]]:
        s = 0.0
        comp: Dict[str, float] = {'dir': 0.0, 'win': 0.0, 'prior': 0.0}

        # Directional 3-cell pattern library (four directions)
        if self.dir_lib:
            acc = 0.0
            for tline in extract_four_line3_n4(rel_board, pos):
                rec = self.dir_lib.get(str(tuple(tline)))
                if rec:
                    acc += float(rec['logit'])
            acc = self._clip(acc)
            s += acc
            comp['dir'] = acc

        # 3×3 window pattern library
        if self.win_lib:
            wkey = str(tuple(extract_window3_n4(rel_board, pos)))
            rec = self.win_lib.get(wkey)
            if rec:
                val = self.lam_win * float(rec['logit'])
                val = self._clip(val)
                s += val
                comp['win'] = val

        # Positional prior
        if self.log_w_pos is not None:
            k = str(pos)
            v = self.log_w_pos.get(k, self.log_w_pos.get(pos))
            if v is not None:
                val = self.lam_prior * float(v)
                val = self._clip(val)
                s += val
                comp['prior'] = val

        return s, comp

# top1 tie-breaking: inner 2×2 (best) -> corner -> edge -> index
def pos_priority(idx: int) -> int:
    r, c = divmod(idx, 4)
    if (1 <= r <= 2) and (1 <= c <= 2): return 0
    if (r in (0, 3)) and (c in (0, 3)): return 1
    return 2

# ------------ Global state for multiprocessing (spawn-friendly) ------------
_G_ENGINE: PatternEngine | None = None
_G_DELTA: float = 0.0
_G_DELTA_MODE: str = 'fixed'  # 'fixed' or 'adaptive'
_G_K: float = 0.6             # k for adaptive mode
_G_SPREAD: str = 'top3'       # 'top3' or 'mad'
_G_NO_SCORES: bool = False
_G_OVERRIDES: bool = False
_G_STD_MODE: str = 'none'     # 'none' or 'zscore'


def _worker_init(engine_payload: Dict[str, Any], delta: float, no_scores: bool,
                 delta_mode: str, k: float, spread: str, overrides: bool,
                 std_mode: str):
    """Initialize engine in worker process to avoid re-deserializing large dicts."""
    global _G_ENGINE, _G_DELTA, _G_NO_SCORES, _G_DELTA_MODE, _G_K, _G_SPREAD, _G_OVERRIDES, _G_STD_MODE
    _G_DELTA = float(delta)
    _G_NO_SCORES = bool(no_scores)
    _G_DELTA_MODE = delta_mode
    _G_K = float(k)
    _G_SPREAD = spread
    _G_OVERRIDES = bool(overrides)
    _G_STD_MODE = std_mode

    dir_lib = engine_payload.get("dir_lib") or {}
    win_lib = engine_payload.get("win_lib") or {}
    log_w_pos = engine_payload.get("log_w_pos")  # None or dict
    lam_win = float(engine_payload.get("lam_win", 0.3))
    lam_prior = float(engine_payload.get("lam_prior", 0.5))
    clip_logit = engine_payload.get("clip_logit")
    _G_ENGINE = PatternEngine(dir_lib=dir_lib, win_lib=win_lib,
                              lam_win=lam_win, log_w_pos=log_w_pos, lam_prior=lam_prior,
                              clip_logit=clip_logit)


def _compute_adaptive_delta(values: List[float]) -> float:
    if not values:
        return _G_DELTA
    vmax = max(values)
    if _G_SPREAD == 'mad' and len(values) >= 3:
        med = statistics.median(values)
        devs = [abs(v - med) for v in values]
        spread = statistics.median(devs)
    else:
        top3 = sorted(values, reverse=True)[:3]
        med_top3 = statistics.median(top3)
        spread = vmax - med_top3
    return max(0.0, _G_K * spread)


def _score_one(state) -> Tuple[str, Dict[str, Any]]:
    """Worker function: score a single state and return (state_id, record)."""
    assert _G_ENGINE is not None
    legal = state.legal or []
    if not legal:
        return state.id, {'best_set': [], 'top1': None, 'scores': {}}

    rel = encode_relative_board(state.board, state.turn)
    # Component-wise scores
    scores: Dict[int, float] = {}
    comp_scores: Dict[int, Dict[str, float]] = {}
    for m in legal:
        sc, comp = _G_ENGINE.score_rel(rel, m)
        scores[m] = sc
        comp_scores[m] = comp

    # Optional normalization: per-component z-score over legal moves, then recombine with weights
    if _G_STD_MODE == 'zscore' and comp_scores:
        def zmap(vals: List[float]) -> List[float]:
            if not vals:
                return []
            mean = sum(vals) / len(vals)
            var = sum((v - mean) * (v - mean) for v in vals) / len(vals)
            std = (var ** 0.5) if var > 1e-12 else 1.0
            return [(v - mean) / std for v in vals]

        order = list(legal)
        dir_arr = [comp_scores[m].get('dir', 0.0) for m in order]
        win_arr = [comp_scores[m].get('win', 0.0) for m in order]
        pri_arr = [comp_scores[m].get('prior', 0.0) for m in order]

        z_dir = zmap(dir_arr)
        z_win = zmap(win_arr)
        z_pri = zmap(pri_arr)

        # Linear combination with the same weights as original scoring
        new_scores = {}
        for i, m in enumerate(order):
            new_scores[m] = z_dir[i] + (_G_ENGINE.lam_win * z_win[i]) + (_G_ENGINE.lam_prior * z_pri[i])
        scores = new_scores

    # Adaptive delta threshold
    if _G_DELTA_MODE == 'adaptive':
        values = list(scores.values())
        delta = _compute_adaptive_delta(values)
    else:
        delta = _G_DELTA

    # Override rules: must-win / must-block
    overrides_set: set[int] = set()
    if _G_OVERRIDES:
        mw = n4_must_win_moves(state.board, state.turn, legal)
        mb = n4_must_block_moves(state.board, state.turn, legal)
        overrides_set = set(mw) | set(mb)

    mval = max(scores.values())
    best = [m for m, v in scores.items() if v >= mval - delta]
    # Always add override moves into best_set
    if overrides_set:
        for m in overrides_set:
            if m not in best:
                best.append(m)

    # top1 tie-breaker: first by override flag, then by positional priority, then by index
    def _tie_key(x: int):
        return (0 if x in overrides_set else 1, pos_priority(x), x)

    top1 = sorted(best, key=_tie_key)[0] if best else None

    if _G_NO_SCORES:
        return state.id, {'best_set': best, 'top1': top1}
    else:
        # Also output component scores for diagnostics
        return state.id, {
            'best_set': best,
            'top1': top1,
            'scores': {str(k): float(v) for k, v in scores.items()},
            'components': {str(k): comp_scores[k] for k in comp_scores}
        }

# ------------ Main entrypoint ------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n4_states', required=True, help='data/n4_exhaustive_states.json')
    ap.add_argument('--dir_lib', required=True, help='directional pattern library JSON or {} for empty')
    ap.add_argument('--win_lib', required=True, help='3×3 window pattern library JSON or {} for empty')
    ap.add_argument('--lam_win', type=float, default=0.3, help='weight for window library (logit coefficient)')
    ap.add_argument('--out_json', required=True, help='output predictions JSON path')
    ap.add_argument('--pos_priors', default=None, help='out/pos_priors_n4.json (optional positional priors)')
    ap.add_argument('--lam_prior', type=float, default=0.5, help='weight for positional prior (log coefficient)')
    ap.add_argument('--delta', type=float, default=0.05, help='tie threshold: score >= max - delta enters best_set')
    ap.add_argument('--delta_mode', choices=['fixed','adaptive'], default='fixed', help='threshold mode: fixed or adaptive')
    ap.add_argument('--delta_k', type=float, default=0.6, help='adaptive scaling factor k (recommended 0.5–0.8)')
    ap.add_argument('--delta_spread', choices=['top3','mad'], default='top3', help='spread metric: max-med(top3) or MAD')
    ap.add_argument('--overrides', action='store_true', help='enable must-win / must-block overrides')
    ap.add_argument('--clip_logit', type=float, default=None, help='symmetric clipping for logits to [-c, +c]')
    ap.add_argument('--std_mode', choices=['none','zscore'], default='none', help='component normalization: none or zscore')
    ap.add_argument('--workers', type=int, default=1, help='number of worker processes (recommend = CPU cores)')
    ap.add_argument('--chunk', type=int, default=50000, help='chunksize for Pool.imap_unordered')
    ap.add_argument('--no_scores', action='store_true', help='do not write per-move scores to reduce I/O')
    args = ap.parse_args()

    # Load pattern libraries
    with open(args.dir_lib) as f:
        dir_lib = json.load(f)
    with open(args.win_lib) as f:
        win_lib = json.load(f)

    # Load positional priors (optional)
    log_w_pos = None
    if args.pos_priors:
        with open(args.pos_priors) as f:
            pri = json.load(f)
        # Normalize to string keys
        if isinstance(pri, dict) and 'log_w_pos' in pri:
            log_w_pos = {str(k): float(v) for k, v in pri['log_w_pos'].items()}
        else:
            log_w_pos = {str(k): float(v) for k, v in pri.items()}

    states = load_n4_states(args.n4_states)

    # Engine payload for worker initialization
    engine_payload = {
        "dir_lib": dir_lib,
        "win_lib": win_lib,
        "log_w_pos": log_w_pos,
        "lam_win": args.lam_win,
        "lam_prior": args.lam_prior,
        "clip_logit": args.clip_logit,
    }

    out: Dict[str, Any] = {}

    if args.workers and args.workers > 1:
        with Pool(processes=args.workers,
                  initializer=_worker_init,
                  initargs=(engine_payload, args.delta, args.no_scores,
                            args.delta_mode, args.delta_k, args.delta_spread, args.overrides, args.std_mode)) as pool:
            for sid, rec in pool.imap_unordered(_score_one, states, chunksize=max(1, int(args.chunk))):
                out[sid] = rec
    else:
        # Single-process path: initialize engine in main process and run sequentially
        _worker_init(engine_payload, args.delta, args.no_scores,
                     args.delta_mode, args.delta_k, args.delta_spread, args.overrides, args.std_mode)
        for st in states:
            sid, rec = _score_one(st)
            out[sid] = rec

    os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(out, f)
    print('Wrote', args.out_json)

if __name__ == '__main__':
    main()
