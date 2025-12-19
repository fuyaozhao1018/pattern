# ttt/relabel_gold.py
from __future__ import annotations
import argparse, json
from typing import Any, Dict, List


def to_str(k: Any) -> str:
    try:
        return str(int(k))
    except Exception:
        return str(k)


def load_json(path: str):
    with open(path, 'r') as f:
        return json.load(f)


def load_gold(path: str) -> Dict[str, List[int]]:
    raw = load_json(path)
    out: Dict[str, List[int]] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, dict):
                bs = v.get('best_set') or v.get('best') or v.get('moves')
            else:
                bs = v
            if bs is None:
                continue
            out[to_str(k)] = [int(x) for x in bs]
        return out
    if isinstance(raw, list):
        for r in raw:
            sid = r.get('id') or r.get('state_id')
            if sid is None:
                continue
            bs = r.get('best_set') or r.get('best') or r.get('moves')
            if bs is None:
                top1 = r.get('top1')
                if top1 is not None:
                    out[to_str(sid)] = [int(top1)]
                continue
            out[to_str(sid)] = [int(x) for x in bs]
        return out
    raise ValueError(f"Unrecognized gold format in {path}: type={type(raw)}")


def load_preds(path: str) -> Dict[str, dict]:
    raw = load_json(path)
    out: Dict[str, dict] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            out[to_str(k)] = v if isinstance(v, dict) else {"best_set": v}
        return out
    if isinstance(raw, list):
        for r in raw:
            sid = r.get("id") or r.get("state_id")
            if sid is None:
                continue
            out[to_str(sid)] = r
        return out
    raise ValueError(f"Unrecognized preds format: {type(raw)}")


def estimate_eps_by_quantile(gold: Dict[str, List[int]], preds: Dict[str, dict], q: float) -> float:
    margins: List[float] = []
    for sid, g in gold.items():
        p = preds.get(sid)
        if not p:
            continue
        scores = p.get('scores') or {}
        if not scores:
            continue
        vals = sorted([float(scores.get(str(m), float('-inf'))) for m in g], reverse=True)
        if len(vals) >= 2 and vals[0] != float('-inf') and vals[1] != float('-inf'):
            margins.append(vals[0] - vals[1])
    if not margins:
        raise ValueError("No margin samples found. Ensure preds contain 'scores' for gold moves.")
    margins.sort()
    idx = max(0, min(len(margins)-1, int(q*(len(margins)-1))))
    return float(margins[idx])


def soften_gold(gold: Dict[str, List[int]], preds: Dict[str, dict], eps: float) -> Dict[str, List[int]]:
    new_gold: Dict[str, List[int]] = {}
    for sid, _ in gold.items():
        p = preds.get(sid)
        if not p:
            new_gold[sid] = sorted(set(gold[sid]))
            continue
        scores: Dict[str, float] = { str(k): float(v) for k, v in (p.get('scores') or {}).items() }
        if not scores:
            new_gold[sid] = sorted(set(gold[sid]))
            continue
        best_move, best_score = None, float('-inf')
        for mk, sc in scores.items():
            if sc > best_score:
                best_move, best_score = mk, sc
        keep = [int(mk) for mk, sc in scores.items() if sc >= best_score - eps]
        if not keep:
            keep = sorted(set(gold[sid]))
        else:
            keep = sorted(set(keep))



            
        new_gold[sid] = keep
    return new_gold


def summarize_sizes(gold: Dict[str, List[int]]):
    sizes = [len(v) for v in gold.values()]
    sizes.sort()
    def q(p: float):
        idx = max(0, min(len(sizes)-1, int(p*(len(sizes)-1))))
        return sizes[idx]
    return {
        "n": len(sizes),
        "mean": (sum(sizes)/len(sizes)) if sizes else 0.0,
        "median": q(0.5) if sizes else 0,
        "p25": q(0.25) if sizes else 0,
        "p75": q(0.75) if sizes else 0,
    }


def main():
    ap = argparse.ArgumentParser(description="Relabel gold best-set by epsilon or margin quantile")
    ap.add_argument("--n4_best", required=True)
    ap.add_argument("--preds", required=True, help="preds with 'scores' per move")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--eps", type=float, help="absolute epsilon threshold")
    group.add_argument("--q", type=float, help="use quantile of gold margins to set epsilon, e.g., 0.75")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gold = load_gold(args.n4_best)
    preds = load_preds(args.preds)
    eps = args.eps if args.eps is not None else estimate_eps_by_quantile(gold, preds, args.q)
    new_gold = soften_gold(gold, preds, eps)

    with open(args.out, "w") as f:
        json.dump(new_gold, f, ensure_ascii=False)

    print(json.dumps({
        "eps_used": eps,
        "orig_sizes": summarize_sizes(gold),
        "new_sizes": summarize_sizes(new_gold)
    }, ensure_ascii=False))

if __name__ == "__main__":
    main()
