# ttt/quick_eval.py
from __future__ import annotations
import argparse, json, random, os, csv
from typing import Any, Dict, List, Set, Tuple

from ttt.common import load_n4_states, load_n4_best

def to_str(k: Any) -> str:
    try: return str(int(k))
    except Exception: return str(k)

def as_int_set(xs: Any) -> Set[int]:
    if xs is None: return set()
    try: return set(int(x) for x in xs)
    except Exception: return set()

def load_preds(path: str) -> Dict[str, dict]:
    raw = json.load(open(path))
    out: Dict[str, dict] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            out[to_str(k)] = v if isinstance(v, dict) else {"best_set": v}
        return out
    if isinstance(raw, list):
        for r in raw:
            sid = r.get("id") or r.get("state_id")
            if sid is None: continue
            out[to_str(sid)] = r
        return out
    raise ValueError(f"Unrecognized preds format: {type(raw)}")

def eval_hit_any(ids: List[str], gold: Dict[str, List[int]], preds: Dict[str, dict], use_best_set: bool):
    total = 0; hit = 0
    for sid in ids:
        gset = as_int_set(gold.get(sid))
        if not gset: continue
        total += 1
        p = preds.get(sid, {})
        if use_best_set:
            pset = as_int_set(p.get('best_set') or p.get('best') or p.get('pred_best'))
            hit += 1 if len(pset & gset) > 0 else 0
        else:
            t = p.get('top1')
            try: top1 = int(t) if t is not None else None
            except Exception: top1 = None
            hit += 1 if (top1 is not None and top1 in gset) else 0
    acc = (hit / total) if total > 0 else 0.0
    return {'num': total, 'hits': hit, 'accuracy_hit_any': round(acc, 6)}

def eval_full(ids: List[str], gold: Dict[str, List[int]], preds: Dict[str, dict]):
    # macro
    mp = mr = mj = 0.0; mhit = mtop1 = 0; n = 0
    # micro
    inter_tot = pred_tot = gold_tot = 0
    for sid in ids:
        gset = as_int_set(gold.get(sid))
        if not gset: continue
        n += 1
        p = preds.get(sid, {})
        pset = as_int_set(p.get('best_set') or p.get('best') or p.get('pred_best'))
        t = p.get('top1')
        try: top1 = int(t) if t is not None else None
        except Exception: top1 = None

        inter = len(pset & gset)
        union = len(pset | gset)
        prec = (inter/len(pset)) if len(pset)>0 else 0.0
        rec  = (inter/len(gset)) if len(gset)>0 else 0.0
        jac  = (inter/union) if union>0 else 0.0
        hit_any = 1 if inter>0 else 0
        top1_hit = 1 if (top1 is not None and top1 in gset) else 0

        mp += prec; mr += rec; mj += jac; mhit += hit_any; mtop1 += top1_hit
        inter_tot += inter; pred_tot += len(pset); gold_tot += len(gset)

    if n == 0:
        return {'num_evaluated_states': 0}

    macro = {
        'num_evaluated_states': n,
        'precision_macro': round(mp/n,6),
        'recall_macro': round(mr/n,6),
        'jaccard_macro': round(mj/n,6),
        'hit_any_macro': round(mhit/n,6),
        'top1_macro': round(mtop1/n,6),
    }
    micro_prec = (inter_tot/pred_tot) if pred_tot>0 else 0.0
    micro_rec  = (inter_tot/gold_tot) if gold_tot>0 else 0.0
    micro_f1   = (2*micro_prec*micro_rec/(micro_prec+micro_rec)) if (micro_prec+micro_rec)>0 else 0.0
    micro = {
        'precision_micro': round(micro_prec,6),
        'recall_micro': round(micro_rec,6),
        'f1_micro': round(micro_f1,6),
        'gold_total': int(gold_tot),
        'pred_total': int(pred_tot),
        'inter_total': int(inter_tot),
    }
    return {'macro': macro, 'micro': micro}

def main():
    ap = argparse.ArgumentParser(description="Quick random-sample evaluation over multiple prediction files.")
    ap.add_argument('--n4_states', required=True)
    ap.add_argument('--n4_best', required=True)
    ap.add_argument('--preds', nargs='+', required=True, help='one or more preds json files')
    ap.add_argument('--k', type=int, default=100000, help='sample size (after filtering gold-nonempty)')
    ap.add_argument('--rate', type=float, default=None, help='sample rate (alternative to --k)')
    ap.add_argument('--seed', type=int, default=2025)
    ap.add_argument('--use_best_set', action='store_true', help='for hit-any: use predicted best_set (else use top1)')
    ap.add_argument('--full_metrics', action='store_true', help='report precision/recall/jaccard too')
    ap.add_argument('--out_csv', default=None, help='optional: write a summary table csv for all preds')
    args = ap.parse_args()

    random.seed(args.seed)
    states = load_n4_states(args.n4_states)
    gold = load_n4_best(args.n4_best)
    all_ids = [to_str(s.id) for s in states if len(as_int_set(gold.get(to_str(s.id))))>0]

    if args.rate is not None:
        k = max(1, int(len(all_ids)*args.rate))
    else:
        k = min(args.k, len(all_ids))
    sample_ids = random.sample(all_ids, k)

    rows = []
    for p in args.preds:
        preds = load_preds(p)
        name = os.path.basename(p)
        if args.full_metrics:
            res = eval_full(sample_ids, gold, preds)
            if 'macro' in res:
                row = {
                    'preds': name,
                    'sample_n': res['macro']['num_evaluated_states'],
                    'hit_any_macro': res['macro']['hit_any_macro'],
                    'top1_macro': res['macro']['top1_macro'],
                    'precision_macro': res['macro']['precision_macro'],
                    'recall_macro': res['macro']['recall_macro'],
                    'jaccard_macro': res['macro']['jaccard_macro'],
                    'precision_micro': res['micro']['precision_micro'],
                    'recall_micro': res['micro']['recall_micro'],
                    'f1_micro': res['micro']['f1_micro'],
                }
            else:
                row = {'preds': name, 'sample_n': 0}
        else:
            res = eval_hit_any(sample_ids, gold, preds, args.use_best_set)
            row = {'preds': name, 'sample_n': res['num'], 'accuracy_hit_any': res['accuracy_hit_any']}
        rows.append(row)

    # print nice table
    headers = list(rows[0].keys()) if rows else []
    print('\t'.join(headers))
    for r in rows:
        print('\t'.join(str(r[h]) for h in headers))

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
        with open(args.out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            for r in rows:
                w.writerow(r)

if __name__ == '__main__':
    main()
