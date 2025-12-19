# ttt/prep_patterns_from_n3.py
from __future__ import annotations
import argparse, json, os
from collections import defaultdict
from ttt.common import (
    load_n3_exhaustive, encode_relative_board,
    extract_four_line3_n3, extract_window3_n3, laplace_logit
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n3', required=True, help='path to data/n3_exhaustive.json (states)')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--lambda_draw', type=float, default=0.5)
    ap.add_argument('--tau', type=int, default=10, help='Laplace smoothing strength')
    ap.add_argument('--use_best_only', action='store_true',
                    help='only use moves in n3_best for building pattern libs')
    ap.add_argument('--n3_best', default=None, help='path to data/n3_best_lambdaXX.json')
    ap.add_argument('--strict_best', action='store_true',
                    help='with --use_best_only: require --n3_best, else error')
    args = ap.parse_args()

    # --- load n3 dataset ---
    n3 = load_n3_exhaustive(args.n3)

    # --- load best map if provided ---
    best_map = None
    if args.n3_best:
        with open(args.n3_best) as f:
            best_map = json.load(f)
    elif args.use_best_only and args.strict_best:
        raise ValueError('use_best_only 已开启，但未提供 --n3_best 且 strict_best 要求严格只用 best。')

    if args.use_best_only and best_map is None:
        print('[WARN] use_best_only 开启但未提供 --n3_best，已回退为“所有合法步”。'
              ' 这会稀释模板，建议提供 n3_best 或加 --strict_best 防止回退。')

    # --- accumulators: key -> [count, sum_peff] ---
    dir_stats = defaultdict(lambda: [0, 0.0])  # 四方向三元组
    win_stats = defaultdict(lambda: [0, 0.0])  # 3x3 窗口九元组

    used, skipped, states_seen = 0, 0, 0

    # --- scan all states ---
    for S in n3:
        states_seen += 1
        rel = encode_relative_board(S.board, S.turn)

        # 选择用于建库的动作集合
        if args.use_best_only and best_map is not None:
            # 容错：某些 state_id 可能不在 best_map
            moves = [int(x) for x in best_map.get(S.id, [])]
            if not moves:
                # 如果该状态缺 best，回退为空集（只跳过这个状态）
                continue
        else:
            moves = S.legal  # 回退策略：使用所有合法步

        for m in moves:
            rec = S.per_move.get(str(m))
            if not rec:
                skipped += 1
                continue
            tot = rec['wins'] + rec['draws'] + rec['losses']
            if tot == 0:
                skipped += 1
                continue

            peff = (rec['wins'] + args.lambda_draw * rec['draws']) / tot

            # 四方向三元组
            for tline in extract_four_line3_n3(rel, m):
                key = str(tuple(tline))
                dir_stats[key][0] += 1
                dir_stats[key][1] += peff

            # 3x3 窗口九元组
            wkey = str(tuple(extract_window3_n3(rel, m)))
            win_stats[wkey][0] += 1
            win_stats[wkey][1] += peff

            used += 1

    # --- pack with Laplace logit (using tau) ---
    def pack(stats):
        out = {}
        for k, (cnt, sum_peff) in stats.items():
            mean_p = sum_peff / cnt
            out[k] = {
                'cnt': cnt,
                'mean_peff': mean_p,
                'logit': laplace_logit(mean_p, cnt, tau=args.tau)
            }
        return out

    dir_lib = pack(dir_stats)
    win_lib = pack(win_stats)

    # --- save ---
    os.makedirs(args.out_dir, exist_ok=True)
    tag = '_best' if args.use_best_only else ''
    p1 = os.path.join(args.out_dir, f'dir_lib_lambda{args.lambda_draw:.2f}{tag}.json')
    p2 = os.path.join(args.out_dir, f'win_lib_lambda{args.lambda_draw:.2f}{tag}.json')
    with open(p1, 'w') as f:
        json.dump(dir_lib, f)
    with open(p2, 'w') as f:
        json.dump(win_lib, f)

    print('Wrote:', p1)
    print('Wrote:', p2)
    print(f'[prep] states_seen={states_seen}, samples_used={used}, skipped={skipped}, '
          f'dir_keys={len(dir_lib)}, win_keys={len(win_lib)}, '
          f'lambda_draw={args.lambda_draw}, tau={args.tau}, best_only={args.use_best_only}')

if __name__ == '__main__':
    main()
