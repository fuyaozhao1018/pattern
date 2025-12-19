# ttt/derive_n3_best.py
from __future__ import annotations
import argparse, json, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n3', required=True)
    ap.add_argument('--lambda_draw', type=float, default=0.5)
    ap.add_argument('--out', required=True)
    ap.add_argument('--eps', type=float, default=1e-12)
    args = ap.parse_args()

    with open(args.n3,'r') as f:
        states=json.load(f)

    out={}
    for r in states:
        m2p={}
        mmax=None
        for m in r['legal']:
            rec=r['per_move'][str(m)]
            tot=rec['wins']+rec['draws']+rec['losses']
            p=(rec['wins']+args.lambda_draw*rec['draws'])/tot if tot>0 else 0
            m2p[m]=p
            mmax=p if mmax is None else max(mmax,p)
        best=[int(m) for m,p in m2p.items() if p>=mmax-args.eps]
        out[r['id']]=best

    os.makedirs(os.path.dirname(args.out) or '.',exist_ok=True)
    with open(args.out,'w') as f: json.dump(out,f)
    print('Wrote',args.out)

if __name__=='__main__':
    main()
