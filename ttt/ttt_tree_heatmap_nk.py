import argparse
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

from tic_tac_toe_nk import TicTacToe, X, O

Key = Tuple[Tuple[int, ...], int]  # (board_flat_tuple, player_to_move)

@dataclass
class Node:
    key: Key
    moves: List[Tuple[int, int]] = field(default_factory=list)
    children: Dict[Tuple[int,int], Key] = field(default_factory=dict)
    counts: Optional[Tuple[int,int,int,int]] = None  # (x_win, o_win, draw, total)
    depth: int = 0

class GameTree:
    def _init_(self, n: int, k: Optional[int]=None, mode: str="exact", mc_rollouts: int=200, seed: int=0):
        self.n = n
        self.k = k if k is not None else n
        self.mode = mode
        self.mc_rollouts = mc_rollouts
        self.rng = np.random.default_rng(seed)
        self.nodes: Dict[Key, Node] = {}

    def _key_of(self, g: TicTacToe) -> Key:
        return (g.get_board_hash(), g.current_player)

    def _ensure_node(self, g: TicTacToe, depth: int) -> Node:
        key = self._key_of(g)
        nd = self.nodes.get(key)
        if nd is None:
            nd = Node(key=key, moves=g.get_valid_moves(), depth=depth)
            self.nodes[key] = nd
        return nd

    def _terminal_counts(self, g: TicTacToe) -> Tuple[int,int,int,int]:
        w = g.check_winner()
        if w is None:
            raise RuntimeError("Not terminal")
        if w == X:
            return (1,0,0,1)
        elif w == O:
            return (0,1,0,1)
        else:
            return (0,0,1,1)

    def _count_exact(self, g: TicTacToe, depth: int=0) -> Tuple[int,int,int,int]:
        """Exhaustive enumeration from this node; memoized by state+player."""
        key = self._key_of(g)
        nd = self._ensure_node(g, depth)
        if nd.counts is not None:
            return nd.counts
        w = g.check_winner()
        if w is not None:
            nd.counts = self._terminal_counts(g)
            return nd.counts
        xw=ow=dr=tt=0
        nd.moves = g.get_valid_moves()
        nd.children = {}
        for (r,c) in nd.moves:
            g.make_move(r, c)
            child_key = self._key_of(g)
            nd.children[(r,c)] = child_key
            cxw,cow,cdr,ctt = self._count_exact(g, depth+1)
            g.undo_move(r, c)
            xw += cxw; ow += cow; dr += cdr; tt += ctt
        nd.counts = (xw, ow, dr, tt)
        return nd.counts

    def build_exact(self) -> Key:
        g = TicTacToe(self.n, self.k)
        root_key = self._key_of(g)
        self._count_exact(g, 0)
        return root_key

    # ---- JSON export ----
    def key_to_str(self, key: Key) -> str:
        board_flat, player = key
        sym = {1:'X', -1:'O', 0:'.'}
        board_str = ''.join(sym[v] for v in board_flat)
        pch = 'X' if player == 1 else 'O'
        return f"{board_str}|{pch}"

    def dump_json(self, root_key: Key, out_path: Path) -> None:
        data = {
            "n": self.n,
            "k": self.k,
            "root": self.key_to_str(root_key),
            "num_nodes": len(self.nodes),
            "nodes": {}
        }
        for key, nd in self.nodes.items():
            kstr = self.key_to_str(key)
            node_obj = {
                "depth": nd.depth,
                "moves": [[r, c] for (r, c) in nd.moves],
                "children": { f"{r},{c}": self.key_to_str(child_key) for (r,c), child_key in nd.children.items() },
                "counts": None
            }
            if nd.counts is not None:
                xw, ow, dr, tt = nd.counts
                node_obj["counts"] = {"x_win": xw, "o_win": ow, "draw": dr, "total": tt}
            data["nodes"][kstr] = node_obj
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ---- Heatmap helpers ----
    def _softmax_board(self, values: Dict[Tuple[int,int], float]) -> np.ndarray:
        """Return an n x n matrix with softmax over valid cells; invalid cells set to 0."""
        mat = np.zeros((self.n, self.n), dtype=float)
        if not values:
            return mat
        vec = np.array([values[mv] for mv in values])
        # Softmax (temperature=1)
        e = np.exp(vec - vec.max())
        probs = e / e.sum()
        for (mv, p) in zip(values.keys(), probs):
            r,c = mv
            mat[r,c] = p
        return mat

    def heatmap_for_node(self, key: Key, out_png: Path, title: str="", annotate: str="none") -> None:
        """For a node, compute per-move win-rate for CURRENT player at that node, then softmax to board.
        annotate: 'none' | 'winrate' | 'softmax' | 'both'
        """
        nd = self.nodes[key]
        board_flat, player = nd.key
        vals: Dict[Tuple[int,int], float] = {}
        for mv, child_key in nd.children.items():
            cxw, cow, cdr, ctt = self.nodes[child_key].counts
            if ctt == 0:
                v = 0.0
            else:
                v = (cxw / ctt) if player == X else (cow / ctt)
            vals[mv] = v

        mat = self._softmax_board(vals)

        plt.figure(figsize=(max(4, self.n*0.9), max(4, self.n*0.9)))
        im = plt.imshow(mat, origin='upper', interpolation='nearest')
        plt.title(title if title else f"N={self.n},K={self.k} softmax(win-rate) @depth={nd.depth}")
        plt.xlabel("col"); plt.ylabel("row")
        plt.xticks(range(self.n)); plt.yticks(range(self.n))
        plt.colorbar(im, fraction=0.046, pad=0.04)

        ann = annotate.lower()
        if ann in ("winrate", "softmax", "both"):
            for r in range(self.n):
                for c in range(self.n):
                    if (r, c) in vals:
                        if ann == "winrate":
                            txt = f"{vals[(r,c)]*100:.0f}%"
                        elif ann == "softmax":
                            txt = f"{mat[r,c]:.2f}"
                        else:
                            txt = f"{vals[(r,c)]*100:.0f}%\n{mat[r,c]:.2f}"
                        plt.text(c, r, txt, ha='center', va='center')

        plt.tight_layout()
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close()

    def export_layer_heatmaps(self, root_key: Key, out_dir: Path, topk: Optional[int]=None, annotate: str="none") -> List[Path]:
        """
        Export heatmaps for tree layer 0 (root) and layer 1 (each child of root).
        If topk is provided, only export for top-k moves (by softmax prob at root).
        Returns list of saved image paths.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = []

        # Root heatmap
        root_png = out_dir / "layer0_root.png"
        self.heatmap_for_node(root_key, root_png, title="Layer0 (root): softmax(win-rate)", annotate=annotate)
        saved.append(root_png)

        # Decide which children to include for layer 1
        root = self.nodes[root_key]
        vals: Dict[Tuple[int,int], float] = {}
        for mv, child_key in root.children.items():
            cxw, cow, cdr, ctt = self.nodes[child_key].counts
            if ctt == 0:
                v = 0.0
            else:
                v = (cxw/ctt) if root.key[1] == X else (cow/ctt)
            vals[mv] = v
        mv_sorted = sorted(vals.items(), key=lambda x: x[1], reverse=True)
        if topk is not None:
            mv_sorted = mv_sorted[:topk]

        for mv, _ in mv_sorted:
            child_key = root.children[mv]
            rc = f"r{mv[0]}c{mv[1]}"
            png = out_dir / f"layer1_after_{rc}.png"
            self.heatmap_for_node(child_key, png, title=f"Layer1 after move {mv}: softmax(win-rate)", annotate=annotate)
            saved.append(png)

        return saved

def main():
    ap = argparse.ArgumentParser(description="Tree-based heatmaps for first two layers (softmax over win-rates)")
    ap.add_argument("--n", type=int, required=True, help="board size N")
    ap.add_argument("--k", type=int, help="win length K (default: N)")
    ap.add_argument("--outdir", type=str, required=True, help="output directory for heatmaps")
    ap.add_argument("--topk", type=int, default=None, help="export only top-K children heatmaps at layer1")
    ap.add_argument("--annotate", type=str, choices=["none","winrate","softmax","both"], default="none",
                    help="annotate per-cell text: winrate(胜率)/softmax/both/none")
    ap.add_argument("--dump-json", type=str, default=None, help="path to write full game tree as JSON")
    args = ap.parse_args()

    n = args.n
    k = args.k if args.k is not None else n
    outdir = Path(args.outdir)

    gt = GameTree(n=n, k=k, mode="exact")
    root_key = gt.build_exact()

    paths = gt.export_layer_heatmaps(root_key, out_dir=outdir, topk=args.topk, annotate=args.annotate)
    for p in paths:
        print(p.resolve())

    if args.dump_json:
        gt.dump_json(root_key, Path(args.dump_json))
        print(Path(args.dump_json).resolve())

if _name_ == "_main_":
    main()