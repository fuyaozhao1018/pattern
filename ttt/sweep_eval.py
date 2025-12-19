# ttt/sweep_eval.py
from __future__ import annotations
import argparse, json, os, itertools, subprocess, tempfile, csv, sys, math
from typing import List, Dict, Any

PY = sys.executable  # rely on current interpreter

ENGINE_MOD = 'ttt.pattern_engine_n4'
EVAL_MOD = 'ttt.quick_eval'

PARAM_GRID_DEFAULT = {
    'delta_mode': ['adaptive'],
    'delta_k': [0.4, 0.6, 0.8],
    'delta_spread': ['top3'],
    'overrides': [True, False],
    'clip_logit': [None, 2.0, 3.0],
    'std_mode': ['none', 'zscore'],
}

def run_engine(args, param_set, out_dir) -> str:
    out_name = (
        f"preds_d{param_set['delta_mode']}" +
        f"_k{param_set['delta_k']}" +
        f"_sp{param_set['delta_spread']}" +
        ("_ov" if param_set['overrides'] else "") +
        (f"_clip{param_set['clip_logit']}" if param_set['clip_logit'] is not None else "") +
        f"_std{param_set['std_mode']}" + ".json"
    )
    out_path = os.path.join(out_dir, out_name)

    cmd = [PY, '-m', ENGINE_MOD,
           '--n4_states', args.n4_states,
           '--dir_lib', args.dir_lib,
           '--win_lib', args.win_lib,
           '--lam_win', str(args.lam_win),
           '--out_json', out_path,
           '--delta', str(args.delta),
           '--delta_mode', param_set['delta_mode'],
           '--delta_k', str(param_set['delta_k']),
           '--delta_spread', param_set['delta_spread'],
           '--std_mode', param_set['std_mode']]
    if args.pos_priors:
        cmd += ['--pos_priors', args.pos_priors, '--lam_prior', str(args.lam_prior)]
    if param_set['overrides']:
        cmd.append('--overrides')
    if param_set['clip_logit'] is not None:
        cmd += ['--clip_logit', str(param_set['clip_logit'])]
    if args.workers > 1:
        cmd += ['--workers', str(args.workers), '--chunk', str(args.chunk)]

    subprocess.run(cmd, check=True)
    return out_path


def run_eval(args, preds_path) -> Dict[str, Any]:
    cmd = [PY, '-m', EVAL_MOD,
           '--n4_states', args.n4_states,
           '--n4_best', args.n4_best,
           '--preds', preds_path,
           '--k', str(args.sample_k),
           '--seed', str(args.seed),
           '--use_best_set']
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    lines = [l.strip() for l in out.stdout.splitlines() if l.strip()]
    if len(lines) >= 2:
        header = lines[0].split('\t')
        values = lines[1].split('\t')
        return dict(zip(header, values))
    return {}


def expand_grid(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid.keys())
    combos = []
    for vals in itertools.product(*(grid[k] for k in keys)):
        combos.append({k: v for k, v in zip(keys, vals)})
    return combos


def main():
    ap = argparse.ArgumentParser(description='Grid sweep quick-win parameters and evaluate hit-any.')
    ap.add_argument('--n4_states', required=True)
    ap.add_argument('--n4_best', required=True)
    ap.add_argument('--dir_lib', required=True)
    ap.add_argument('--win_lib', required=True)
    ap.add_argument('--pos_priors', default=None)
    ap.add_argument('--lam_win', type=float, default=0.5)
    ap.add_argument('--lam_prior', type=float, default=0.5)
    ap.add_argument('--delta', type=float, default=0.05)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--chunk', type=int, default=50000)
    ap.add_argument('--sample_k', type=int, default=100000)
    ap.add_argument('--seed', type=int, default=2025)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--out_csv', required=True)
    ap.add_argument('--grid_json', default=None, help='Optional custom grid JSON file overriding defaults.')
    args = ap.parse_args()

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    if args.grid_json:
        custom = json.load(open(args.grid_json))
        grid = custom
    else:
        grid = PARAM_GRID_DEFAULT

    runs = expand_grid(grid)
    rows: List[Dict[str, Any]] = []

    for ps in runs:
        preds_path = run_engine(args, ps, args.out_dir)
        metrics = run_eval(args, preds_path)
        row = {**ps, **metrics, 'preds_path': os.path.basename(preds_path)}
        rows.append(row)
        print('[sweep]', row)

    # Write CSV
    headers = sorted({k for r in rows for k in r.keys()})
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print('[done] wrote', args.out_csv)

if __name__ == '__main__':
    main()
