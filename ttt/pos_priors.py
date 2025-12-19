# ttt/pos_priors.py
from __future__ import annotations
import argparse, json, math, os
from collections import defaultdict
from ttt.common import load_n3_exhaustive, encode_relative_board

# Geometric mean to avoid extreme values dominating multiplicative aggregation
def geom_mean(xs):
    xs = [max(min(x, 1 - 1e-9), 1e-9) for x in xs]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else 1.0

# Linearly interpolate a curve of length N_from to length N_to in index-normalized space u = i/(N-1)
def lin_interp_curve(vals_from, N_from, N_to):
    out = []
    for j in range(N_to):
        u = j / (N_to - 1) if N_to > 1 else 0.0
        x = u * (N_from - 1)
        i0, i1 = int(math.floor(x)), min(int(math.ceil(x)), N_from - 1)
        if i0 == i1:
            out.append(vals_from[i0])
        else:
            t = x - i0
            out.append((1 - t) * vals_from[i0] + t * vals_from[i1])
    return out

# Shell index r defined by Manhattan distance from geometric center ((N-1)/2, (N-1)/2), rounded
# N=3 -> r ∈ {0,1,2}; N=4 -> r ∈ {0,1,2,3}
def shell_index_from_center(idx: int, N: int) -> int:
    r, c = divmod(idx, N)
    cx, cy = (N - 1) / 2.0, (N - 1) / 2.0
    dist = abs(r - cx) + abs(c - cy)
    return int(round(dist))

def orbit_class_N3(idx):
    r, c = divmod(idx, 3)
    if r == 1 and c == 1: return 'center'
    if (r in (0, 2)) and (c in (0, 2)): return 'corner'
    return 'edge'

def orbit_class_N4(idx):
    r, c = divmod(idx, 4)
    if (1 <= r <= 2) and (1 <= c <= 2): return 'center'  # inner 2×2
    if (r in (0, 3)) and (c in (0, 3)): return 'corner'
    return 'edge'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n3', required=True, help='path to data/n3_exhaustive.json (states file)')
    ap.add_argument('--lambda_draw', type=float, default=0.5)
    ap.add_argument('--out_json', required=True, help='output path, e.g., out/pos_priors_n4.json')
    args = ap.parse_args()

    n3 = load_n3_exhaustive(args.n3)

    # 1) Aggregate Peff on N=3 for each family (Peff = (win + λ·draw) / total)
    fam_orbit = defaultdict(list)  # {'center':[], 'edge':[], 'corner':[]}
    fam_row   = defaultdict(list)  # {0:[],1:[],2:[]}
    fam_col   = defaultdict(list)  # {0:[],1:[],2:[]}
    fam_shell = defaultdict(list)  # {0:[],1:[],2:[]}

    for S in n3:
        rel = encode_relative_board(S.board, S.turn)
        for m in S.legal:
            rec = S.per_move[str(m)]
            tot = rec['wins'] + rec['draws'] + rec['losses']
            if tot == 0:
                continue
            peff = (rec['wins'] + args.lambda_draw * rec['draws']) / tot
            r3, c3 = divmod(m, 3)
            fam_row[r3].append(peff)
            fam_col[c3].append(peff)
            fam_orbit[orbit_class_N3(m)].append(peff)
            fam_shell[shell_index_from_center(m, 3)].append(peff)  # r=0..2

    # 2) Normalize geometric means within each family (mean ≈ 1 in multiplicative space)
    def norm_geom_mean(dct, keys):
        vals, out = [], {}
        for k in keys:
            v = geom_mean(dct[k]) if dct.get(k) else 1.0
            out[k] = v
            vals.append(v)
        g = geom_mean(vals) if vals else 1.0
        for k in out:
            out[k] /= g if g > 0 else 1.0
        return out

    w_orbit3 = norm_geom_mean(fam_orbit, ['center','edge','corner'])
    w_row3   = norm_geom_mean(fam_row,   [0,1,2])
    w_col3   = norm_geom_mean(fam_col,   [0,1,2])
    w_shell3 = norm_geom_mean(fam_shell, [0,1,2])

    # 3) Upsample to N=4: interpolate row/column/shell weights from 3→4; orbit is categorical mapping
    w_row4   = lin_interp_curve([w_row3[0], w_row3[1], w_row3[2]], 3, 4)
    w_col4   = lin_interp_curve([w_col3[0], w_col3[1], w_col3[2]], 3, 4)
    w_shell4 = lin_interp_curve([w_shell3[0], w_shell3[1], w_shell3[2]], 3, 4)  # r=0..3
    w_orbit4 = {'center': w_orbit3['center'], 'edge': w_orbit3['edge'], 'corner': w_orbit3['corner']}

    # 4) Build per-position log prior on N=4 (multiplicative → sum in log space)
    log_w_pos = {}
    for p in range(16):
        r, c   = divmod(p, 4)
        orbit  = orbit_class_N4(p)
        shell4 = shell_index_from_center(p, 4)  # 0..3
        w = w_orbit4[orbit] * w_row4[r] * w_col4[c] * w_shell4[shell4]
        w = max(w, 1e-9)             # numeric floor to avoid log(0)
        log_w_pos[str(p)] = math.log(w)  # string keys for robustness

    os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump({
            'log_w_pos': log_w_pos,
            'row4': w_row4,
            'col4': w_col4,
            'shell4': w_shell4,
            'orbit4': w_orbit4
        }, f)
    print('Wrote', args.out_json)

if __name__ == '__main__':
    main()
