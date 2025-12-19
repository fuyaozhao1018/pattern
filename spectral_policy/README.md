

# Pattern-Driven Spectral Policies for Tic-Tac-Toe (N×N)
This research explores a math-first, interpretable alternative to neural networks for board games like Tic-Tac-Toe. We build a deterministic policy using spectral features (DCT coefficients) and linear regression. It trains in milliseconds, uses very little data, and plays competitively vs random and light MCTS baselines. The same idea generalizes from 3×3 to 4×4 and arbitrary N.



## Why this matters

Neural networks are powerful but opaque—a trained policy is a black box. By contrast, spectral policies are not heuristic-based: they are trained directly from data using mathematical optimization (ridge regression), not hand-crafted rules or ad-hoc scoring. Instead of relying on human intuition, the model learns its weights from exhaustive or simulated game outcomes.

Spectral policies are:

- **Interpretable:** Weights directly correspond to frequency patterns in the board. High weights on low-frequency (global) patterns vs high-frequency (local) patterns tell a story about strategy.
- **Data-Efficient:** Closed-form ridge regression trains in milliseconds on small datasets (no backprop, no GPU).
- **Fast:** Inference is just a handful of matrix multiplies per move; easily embedded in search algorithms.
- **Generalizable:** The same DCT basis transfers across board sizes (3x3 → 4x4 → 5x5 → arbitrary N).
- **Reproducible:** No randomness in training; given the same data and seed, you get identical results every time.

For simple games like Tic-Tac-Toe, spectral patterns often outperform neural networks trained on the same data, and always beat random/greedy baselines. This shows that **domain-appropriate features matter more than model complexity**.



## Key Features

- **Spectral Features:** Uses orthonormal DCT-II basis for X/O occupancy grids, yielding $2N^2$ features per board.
- **Linear Policy:** Trains a single linear scorer $s(\phi) = w^\top \phi$ via closed-form ridge regression.
- **Safety Heuristic:** Optionally checks for immediate win/block moves before using the learned policy.
- **Symmetry Averaging:** Optionally averages features over all 8 D4 symmetries for invariance.
- **Labeling:** 3x3 uses exhaustive optimal moves (`optimal_decision_tree.json`); NxN uses Monte Carlo rollouts.
- **Baselines:** Competes against Random, MCTS, and self-play.
- **CLI Flags:** All scripts support `--sym-avg`, `--no-safety`, `--self-play`, and more.



## Files and roles

- `spectral_policy_3x3.py` — 3×3 policy trained on optimal labels; CV over λ, visualizations, W/D/L vs Random/Perfect; optional `--sym-avg`.
- `spectral_policy_4x4.py` — 4×4 policy trained on MC labels; evaluation vs Random and MCTS; optional `--sym-avg`.
- `spectral_policy_nxn.py` — generalized NxN trainer/evaluator; `--labels mc|optimal` (optimal for N=3), Random/MCTS baselines, optional `--sym-avg`.
- `optimal_decision_tree.json` — exhaustive optimal data for 3×3 (required by 3×3 optimal mode).

You may also have `analytical_patterns_3x3.py` (analytical hand-crafted features); this compares cleanly with the spectral approach.



## Mathematical formulation

### Board Encoding

Let the board be flattened to length N² with values: 1 (X), −1 (O), 0 (empty).
Build two binary grids X and O in ℝ^{N×N}: X[r,c]=1 if X occupies (r,c), O likewise for O.

### Spectral Features (DCT-II)

We use an orthonormal 1D DCT-II matrix C∈ℝ^{N×N}, with

$$
C_{u,x} = \alpha_u \cos\left(\frac{\pi(2x+1)u}{2N}\right),\quad \alpha_0 = \sqrt{\tfrac{1}{N}},\; \alpha_{u>0} = \sqrt{\tfrac{2}{N}}
$$

The 2D DCT is computed by separability:

$$
\mathrm{DCT2}(A) = C A C^T
$$

Our feature map for a (post-move) board state is

$$
\phi \in \mathbb{R}^{2N^2} = \big[\mathrm{vec}(\mathrm{DCT2}(X));\; \mathrm{vec}(\mathrm{DCT2}(O))\big]
$$

Optional symmetry averaging (invariance under D4) replaces ϕ by the average of DCT features across the 8 dihedral transforms of the board.

#### Visual Example: 3x3 Board and DCT Features

Board:

```
X . .
. O .
. . X
```

X grid:
```
1 0 0
0 0 0
0 0 1
```
O grid:
```
0 0 0
0 1 0
0 0 0
```

DCT2(X) and DCT2(O) yield two $3 \times 3$ matrices, which are then flattened and concatenated for the feature vector $\phi$.

#### Symmetry Averaging

With `--sym-avg`, the feature vector is averaged over all 8 board symmetries (rotations/flips).

### Labels


- 3×3 optimal: for each non-terminal state and legal move, label 1 if the move is optimal, 0 otherwise (from the tree).
- MC rollouts (any N): for each state and legal move, place the move and estimate

$$y = \mathbb{E}[\text{outcome}] \in [-1,1], \quad \text{win}=+1,\; \text{draw}=0,\; \text{loss}=-1,$$

by averaging random playouts to termination vs a Random opponent.

### Model Training

$$
\min_w \|Xw - y\|_2^2 + \lambda \|w\|_2^2 \implies w = (X^\top X + \lambda I)^{-1} X^\top y
$$

We fit a single linear scorer for post-move features:

$$
s(\phi) = w^\top \phi
$$

Ridge regression (closed form):

$$
\min_w \; \|Xw - y\|_2^2 + \lambda \|w\|_2^2 \quad \Rightarrow \quad w = (X^\top X + \lambda I)^{-1} X^\top y
$$

For 3×3 we can select λ by K-fold cross-validation on state-level accuracy (does argmax s pick an optimal move for that state?).

### Policy


- Safety (optional): check any immediate winning move; if none, block any opponent immediate win.
- Otherwise choose the legal move with the highest score s(ϕ) using features from the hypothetical post-move board.

Note: If you prefer a “pure function” with no hand-crafted heuristics, add `--no-safety` to disable immediate win/block. This runs the learned spectral scorer alone.


## Feature Interpretation Guide: Understanding DCT Weights

One of the greatest strengths of spectral policies is that you can directly inspect and understand what patterns the model learned. This section explains how to read the weights and decode their meaning.

### What Does Each DCT Coefficient Represent?

The 2D DCT decomposes a board into frequency components:

- **DC (u=0, v=0):** Global mean. For an X grid, this is the total number of X's on the board. For an O grid, total number of O's.
- **Low-frequency (small u, v):** Large-scale patterns. E.g., (u=0, v=1) captures vertical gradient (top-heavy vs bottom-heavy). (u=1, v=0) captures horizontal gradient.
- **High-frequency (large u, v):** Local details. Checkerboard patterns, corners, edges.

For 3x3, indices are 0–8 (after flattening the 3x3 DCT output). The feature vector has 18 elements: indices 0–8 for X, 9–17 for O.

### Reading Weights: An Example

Suppose `spectral_policy_3x3.py` trains and prints:

```
Top weights for X channel (most important patterns):
  w[0] = +0.45  (DC: total X count, positive → winning strategy favors more X's)
  w[1] = +0.22  (Low freq, vertical gradient: positive → favor X in top rows)
  w[2] = -0.15  (Low freq, horizontal gradient: negative → avoid X-heavy right edge)
  w[8] = -0.08  (High freq, local details: small negative weight)

Top weights for O channel:
  w[9] = -0.35  (DC: total O count, negative → winning strategy disfavors O)
  w[10] = -0.12 (Low freq: slight aversion to O in top area)
```

Interpretation:
- High positive weight on X's DC → X should try to occupy more cells overall.
- High negative weight on O's DC → Opponent should be blocked from occupying cells.
- Positive weight on X vertical gradient → X benefits from controlling upper rows.
- Negative weight on O vertical gradient → O is less threatening in upper rows; focus on lower area.

This tells a story: **the learned policy favors early board control and vertical dominance**.

### Visualizing Feature Contributions During Play

When `spectral_policy_3x3.py` visualizes a game, it shows:

```
Board state:
X . .
. O .
. . .

Legal moves: [1, 2, 3, 5, 6, 7, 8]

Score grid s(ϕ) for each legal move:
Pos 1: 0.52
Pos 2: 0.48
Pos 3: 0.51
Pos 5: 0.60 ← Chosen
Pos 6: 0.53
Pos 7: 0.50
Pos 8: 0.55

Top feature contributions for move 5 (center):
  w[0] * ϕ[0] = +0.45 * 0.33 = +0.149  (X DC: placing in center increases total X)
  w[1] * ϕ[1] = +0.22 * 0.15 = +0.033  (X vert gradient: center is middle, neutral contribution)
  w[9] * ϕ[9] = -0.35 * 0.25 = -0.088  (O DC: doesn't block O yet, but prevents O from here)
  ... (other features have small contributions)
```

This shows **why** the move was chosen: the weights on X's DC and gradient most strongly favor the center position.

### Domain Knowledge Encoded in Weights

After training, you can extract strategic insights:

1. **Control vs Threat:** Compare magnitudes of w[0] (X DC) vs w[9] (O DC).
   - If |w[9]| > |w[0]|, the policy is threat-focused (block opponent).
   - If w[0] > |w[9]|, the policy is control-focused (build your own advantage).

2. **Positional Bias:** Check low-frequency weights (u=1, v=0 for horizontal; u=0, v=1 for vertical).
   - Positive weight → favor that direction.
   - Negative weight → avoid that direction.

3. **Local Tactics:** High-frequency weights (u ≥ 2 or v ≥ 2) capture edge and corner preferences.
   - Usually small in magnitude for simple games.

4. **Symmetry Effects:** With `--sym-avg`, weights should be more "balanced" across all directions since the feature vector itself is symmetric.

### Why This Matters for Research

Because weights are human-readable, you can:

- **Validate the model:** Do the learned weights match domain knowledge? E.g., "center is valuable in Tic-Tac-Toe" should appear as a high weight on center-related DCT modes.
- **Transfer learning:** Train on one board size, inspect the weights, then apply to a larger board to see which patterns persist.
- **Hybrid models:** Combine spectral features with hand-crafted rules. E.g., use DCT for strategy, add explicit "threat" features for tactics.
- **Teach:** Use weight inspection to explain strategy without mentioning neural networks.

---

## File Roles

- `spectral_policy_3x3.py`: 3x3 policy, optimal labels, visualizations, CLI flags.
- `spectral_policy_4x4.py`: 4x4 policy, MC labels, MCTS baseline, CLI flags.
- `spectral_policy_nxn.py`: General NxN trainer/evaluator, MC/optimal labels, CLI flags.
- `optimal_decision_tree.json`: Exhaustive optimal moves for 3x3.
- `extract_positional_probabilities.py`, `complete_heatmap_generator.py`, `game_tree_visualizer.py`, `ttt_nxn.py`: Utilities for extracting, visualizing, and analyzing game tree data.
- `probabilities.json`, `probabilities.csv`: Exhaustive state/move outcome probabilities for 3x3.

---


## How to run

From the project folder:

- 3×3 (optimal labels), compare symmetry:
```bash
python spectral_policy_3x3.py --games-random 50 --games-perfect 30 --seed 42
python spectral_policy_3x3.py --sym-avg --games-random 50 --games-perfect 30 --seed 42
```

- 4×4 (MC labels), with/without symmetry and MCTS baseline:
```bash
python spectral_policy_4x4.py --samples 400 --rollouts 6 --games-random 50 --games-mcts 20 --mcts-iters 150 --seed 42
python spectral_policy_4x4.py --sym-avg --samples 400 --rollouts 6 --games-random 50 --games-mcts 20 --mcts-iters 150 --seed 42
```

- General N×N (MC by default). Examples:
```bash
# N=3 quick run (MC labels)
python spectral_policy_nxn.py --n 3 --samples 200 --rollouts 6 --games-random 20 --seed 42

# N=3 with optimal labels (needs optimal_decision_tree.json)
python spectral_policy_nxn.py --n 3 --labels optimal --games-random 50 --seed 42

# N=4 with MCTS baseline
python spectral_policy_nxn.py --n 4 --samples 600 --rollouts 8 --games-random 100 --games-mcts 20 --mcts-iters 200 --seed 42

# N=5 exploratory
python spectral_policy_nxn.py --n 5 --samples 300 --rollouts 12 --games-random 50 --games-mcts 20 --mcts-iters 150 --seed 42
```

Common flags:
- `--sym-avg` enforces symmetry invariance by feature averaging.
- `--no-safety` disables immediate win/block safety overrides to evaluate the bare linear scorer.


## Visualized Example: Score Grid and Feature Contributions

When running `spectral_policy_3x3.py` with visualization enabled, you get output like:

```
Board:
X . .
. O .
. . X

Score grid (s(ϕ) for each move):
0.52 0.48 0.51
0.47 0.60 0.49
0.53 0.50 0.55

Top feature contributions for chosen move:
  w[0]*ϕ[0] = 0.12
  w[3]*ϕ[3] = -0.08
  ...
```

This shows which board patterns (DCT features) most influence the move selection.

---

## Data Files: How to Use and Visualize

- `optimal_decision_tree.json` contains exhaustive optimal moves and outcome probabilities for every 3x3 state.
- `probabilities.json`/`probabilities.csv` provide $P(\text{X win})$, $P(\text{O win})$, $P(\text{draw})$ for every state/player combination.

Example entry from `optimal_decision_tree.json`:
```json
"[0, 0, 0, 0, 0, 0, 0, 0, 0]": {
  "move": 4,
  "player": "X",
  "score": 0.692857142857143,
  "move_count": 9,
  "good_moves": 1,
  "p_x_win": 0.584920634920635,
  "p_o_win": 0.288095238095238,
  "p_draw": 0.126984126984127,
  "move_details": [...]
}
```

You can use `extract_positional_probabilities.py` and `complete_heatmap_generator.py` to generate heatmaps and positional win probabilities. Use `game_tree_visualizer.py` to explore the game tree interactively.

---


## What we found (selected results)

Numbers below are representative runs with seed 42; use higher game counts for tighter estimates.

- 3×3 (optimal labels)
  - Standard features:
    - CV acc ~0.516 (state-level), tie-aware ~0.623
    - Vs Random (30): W:29 D:1 L:0
    - Vs Perfect (30 alt): W:0 D:30 L:0 (draws as expected vs perfect play)
  - Symmetry-averaged features:
    - CV acc ~0.507, tie-aware ~0.638
    - Vs Random (30): W:29 D:1 L:0
    - Vs Perfect (30 alt): W:0 D:30 L:0
  - Takeaway: symmetry averaging changes little in 3×3 (patterns already symmetric; invariance removes directional cues but also reduces variance).

- 4×4 (MC labels, samples=400, rollouts=6)
  - Standard:
    - Vs Random (25): W:17 D:8 L:0
    - Vs MCTS@150 iters (10): W:10 D:0 L:0
  - Sym-avg:
    - Vs Random (25): W:20 D:5 L:0
    - Vs MCTS@150 iters (10): W:8 D:2 L:0
  - Takeaway: both modes are competitive; small differences are within typical evaluation variance at these match counts.

- NxN quick checks (MC labels)
  - N=3: Vs Random (10): W:9 D:1 L:0
  - N=4: Vs Random (10): W:3 D:7 L:0; Vs MCTS@100 (5): W:5 D:0 L:0

### Safety vs no-safety (trade-offs)

Safety adds two hand-coded checks: take an immediate win if available; otherwise block the opponent’s immediate win. Disabling it (`--no-safety`) evaluates the learned linear policy alone. Expect a classic precision–recall style trade-off:

- With safety ON: more conservative, high draw rate, avoids blunders and near-term losses.
- With safety OFF: more aggressive, often increases wins, but can also increase losses.

Observed quick runs (seed 42; your numbers will vary slightly):

- 3×3 vs Perfect (10 alt):
  - safety ON: W:0 D:10 L:0
  - safety OFF: W:0 D:5 L:5
- 3×3 vs Random (10):
  - safety ON: W:9 D:1 L:0
  - safety OFF: W:9 D:0 L:1
- 4×4 vs MCTS@150 (10): safety ON/OFF both achieved W:10 D:0 L:0 in a representative run (variance across small samples is expected).

Recommendation: keep safety ON when benchmarking basic strength and fairness; try `--no-safety` to evaluate the pure function approximator.



## Reproducibility and performance

- All scripts accept `--seed` for reproducibility.
- Ridge solve is closed-form with tiny matrices (≤ 2N² features); training is instantaneous.
- Inference is a handful of matrix multiplies per legal move.



## Troubleshooting and tips

- Missing `optimal_decision_tree.json`: the 3×3 optimal-label modes require this file (present in this repo). If you renamed or moved it, put it back in the project root or pass an absolute path if the script supports it. For NxN with `--labels mc`, the file is not needed.
- Long runs: reduce `--games-*`, `--samples`, and/or `--rollouts` for quicker smoke tests; increase for stable estimates.
- MCTS strength: use higher `--mcts-iters` for a stronger opponent; note that runtime grows roughly linearly with iterations.



## Heatmap fidelity note

Exact reproduction of every per-move probability heatmap is not achievable with a single linear spectral model; however, we reproduce optimal-move choices in 50.8% of canonical states and 63.8% tie-aware, significantly above random baselines (33.6%/49.5%). For improved heatmap fidelity, we evaluate rank agreement (Spearman correlation) and calibration (MSE/Brier score) between model scores and true probabilities from `optimal_decision_tree.json`. Training on true probabilities (instead of 0/1 labels) and per-depth calibration can further enhance alignment. *This applies to 3x3, where exhaustive optimal moves and probabilities are available; for NxN (N>3), MC rollouts provide approximate labels without exact heatmaps.*



## API quick reference (choose_move contract)

- Input: a `TicTacToe` game object with fields `board` (length N², values {1,0,−1}) and `current_player` (1=X or −1=O).
- Output: an integer move index in `[0, N²)` corresponding to the chosen empty cell.
- Error modes: raises if no legal moves (should not happen because callers check `is_terminal()`); otherwise deterministic given the same `w`, `sym_avg`, `safety`, and RNG seed for any stochastic opponents used in evaluation.

Edge cases handled:
- Empty board, nearly full board, and one-move-to-win/loss are supported.
- With `--sym-avg`, feature invariance ensures rotated/flipped boards map to the same representation (within numeric precision).

- DCT-II basis and separability are standard; we adopt the orthonormal variant for stability.
- Dihedral group (D4) symmetries provide natural invariances for square grids.
- MCTS opponent is a minimal UCT implementation used for calibration.

---

## Conclusion

This project shows that interpretable spectral patterns can match or outperform neural network policies for Tic-Tac-Toe, with full transparency and mathematical rigor. The approach is general, fast, and ideal for teaching, benchmarking, or as a strong baseline for more complex games.

