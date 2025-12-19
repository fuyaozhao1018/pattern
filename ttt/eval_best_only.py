# ttt/eval_best_only.py
from __future__ import annotations
import argparse, csv, json, os
from typing import Dict, List, Any, Tuple, Set

from ttt.common import load_n4_states, load_n4_best

def to_str_key(k: Any) -> str:
    try:
        return str(int(k))
    except Exception:
        return str(k)

def load_preds(path: str) -> Dict[str, Any]:
    """
    Load prediction file. Supported formats:
    - { state_id: {"best_set":[...], "top1":..., "scores":{...}}, ... }
    - list[{"id"/"state_id":..., "best_set":[...], "top1":...}, ...]
    """
    with open(path) as f:
        raw = json.load(f)
    out: Dict[str, Any] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            out[to_str_key(k)] = v
        return out
    if isinstance(raw, list):
        for r in raw:
            sid = r.get('id') or r.get('state_id')
            if sid is None: 
                continue
            out[to_str_key(sid)] = r
        return out
    raise ValueError(f"Unrecognized preds format in {path}: {type(raw)}")

def as_int_set(xs: Any) -> Set[int]:
    if xs is None: return set()
    try:
        return set(int(x) for x in xs)
    except Exception:
        return set()

def metrics_for_pair(pset: Set[int], gset: Set[int], top1: int | None) -> Tuple[float,float,float,int,int,int]:
    """Return (precision, recall, jaccard, hit_any, top1_hit, inter)."""
    inter = len(pset & gset)
    union = len(pset | gset)
    # gold is non-empty here; main loop guarantees gset is non-empty
    precision = (inter / len(pset)) if len(pset) > 0 else 0.0
    recall    = (inter / len(gset)) if len(gset) > 0 else 0.0
    jaccard   = (inter / union)     if union > 0   else 0.0
    hit_any   = 1 if inter > 0 else 0
    top1_hit  = 1 if (top1 is not None and top1 in gset) else 0
    return precision, recall, jaccard, hit_any, top1_hit, inter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n4_states', required=True)
    ap.add_argument('--n4_best',   required=True)
    ap.add_argument('--preds',     required=True)
    ap.add_argument('--out_csv',   required=True)
    ap.add_argument('--out_summary_json', default=None,
                    help='Optional: write global metrics JSON (default: metrics_summary.json in the same dir as CSV)')
    args = ap.parse_args()

    states = load_n4_states(args.n4_states)
    gold_best_map = load_n4_best(args.n4_best)
    preds_map = load_preds(args.preds)

    # state index, support string ids
    states_by_id: Dict[str, Any] = {to_str_key(s.id): s for s in states}

    # macro aggregators
    rows = []
    macro_sum_p = macro_sum_r = macro_sum_j = 0.0
    macro_sum_hit = macro_sum_top1 = 0
    n_eval = 0

    # micro aggregators
    micro_pred_total = 0
    micro_gold_total = 0
    micro_inter_total = 0

    for sid, st in states_by_id.items():
        gset = as_int_set(gold_best_map.get(sid))
        # skip states with empty gold set (terminal / no legal moves / no "best next move")
        if len(gset) == 0:
            continue

        pred_entry = preds_map.get(sid)
        if pred_entry is None:
            pset = set()
            top1 = None
        else:
            best_set = pred_entry.get('best_set') or pred_entry.get('pred_best') or pred_entry.get('best')
            pset = as_int_set(best_set)
            t = pred_entry.get('top1')
            try:
                top1 = int(t) if t is not None else None
            except Exception:
                top1 = None

        p, r, j, hit_any, top1_hit, inter = metrics_for_pair(pset, gset, top1)

        rows.append({
            'state_id': sid,
            'gold_best': sorted(list(gset)),
            'pred_best': sorted(list(pset)),
            'top1': top1 if top1 is not None else '',
            'intersection': inter,
            'precision': round(p, 6),
            'recall': round(r, 6),
            'jaccard': round(j, 6),
            'hit_any': hit_any,
            'top1_hit': top1_hit
        })

        macro_sum_p += p
        macro_sum_r += r
        macro_sum_j += j
        macro_sum_hit += hit_any
        macro_sum_top1 += top1_hit
        n_eval += 1

        micro_inter_total += inter
        micro_pred_total  += len(pset)
        micro_gold_total  += len(gset)

    # write CSV
    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'state_id','gold_best','pred_best','top1',
            'intersection','precision','recall','jaccard','hit_any','top1_hit'
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # compute and write summary
    if n_eval == 0:
        print('[eval] No evaluable states (gold best is empty). Check whether n4_best matches n4_states.')
        return

    macro = {
        'num_evaluated_states': n_eval,
        'precision_macro': round(macro_sum_p / n_eval, 6),
        'recall_macro':    round(macro_sum_r / n_eval, 6),
        'jaccard_macro':   round(macro_sum_j / n_eval, 6),
        'hit_any_macro':   round(macro_sum_hit / n_eval, 6),
        'top1_macro':      round(macro_sum_top1 / n_eval, 6)
    }

    micro_prec = (micro_inter_total / micro_pred_total) if micro_pred_total > 0 else 0.0
    micro_rec  = (micro_inter_total / micro_gold_total) if micro_gold_total > 0 else 0.0
    micro_f1   = (2*micro_prec*micro_rec / (micro_prec+micro_rec)) if (micro_prec+micro_rec) > 0 else 0.0
    micro = {
        'precision_micro': round(micro_prec, 6),
        'recall_micro':    round(micro_rec, 6),
        'f1_micro':        round(micro_f1, 6),
        'gold_total':      int(micro_gold_total),
        'pred_total':      int(micro_pred_total),
        'inter_total':     int(micro_inter_total)
    }

    summary = {'macro': macro, 'micro': micro}
    out_summary = args.out_summary_json
    if not out_summary:
        base_dir = os.path.dirname(args.out_csv) or '.'
        out_summary = os.path.join(base_dir, 'metrics_summary.json')
    with open(out_summary, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"[eval] wrote CSV: {args.out_csv}")
    print(f"[eval] summary: {summary}")

if __name__ == '__main__':
    main()
