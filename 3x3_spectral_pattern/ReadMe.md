# Spectral Policy for Tic-Tac-Toe (3x3 and NxN)

## The following goals are met but the professor wants something more general pattern.

1. **No heuristics**: All tactical knowledge emerges from data + learned weights
2. **Pure math**: Only DCT, polynomial expansion, ridge regression, deterministic scoring
3. **Self-contained**: No external tic-tac-toe libraries; all logic explicit
4. **Reproducible**: Full control over randomness via seed; no non-deterministic operations
5. **Interpretable**: Weights can be inspected; features are mathematical, not learned embeddings

## Overview

Trying to build a deterministic tic-tac-toe policy that works for both 3x3 (exhaustive supervision) and generic NxN boards (sampled minimax supervision). The policy uses only interpretable mathematical components:

1. **Pure spectral (DCT) features** - Orthonormal Discrete Cosine Transform of board occupancy
2. **Polynomial feature expansion** - Deterministic monomial expansion for interaction terms
3. **Ridge regression** - Closed-form solution from minimax-labeled data
4. **Deterministic move selection** - Linear scorer with tie-breaking by cell index
5. **Safety override** - Immediate win/block detection (optional)

## Math

### Spectral Features (18-dimensional base)

For each board state and candidate move, we compute 2N² DCT coefficients:
- N² coefficients for X occupancy DCT
- N² coefficients for O occupancy DCT

For 3x3 boards, the DCT basis is orthonormal with explicit normalization constants:

```
α_u = sqrt(1/3) if u=0 else sqrt(2/3)
C[u,x] = α_u * cos(π(2x+1)u / 6)
Features = [DCT(X_board), DCT(O_board)] vectorized
```

### Polynomial Expansion

For polynomial order k, we systematically generate all monomials of degree ≤ k:
- Order 1: Base features (18 dimensions)
- Order 4: All degree-1 through degree-4 monomials (7,314 dimensions for 3x3)

Each monomial is computed deterministically as a product of base features.

### Ridge Regression Training

Given training data X (samples × features) and minimax labels y:

```
w = argmin_w ||Xw - y||² + λ||w||²
w = (X'X + λI)^(-1) X'y   [closed-form]
```

**For 3x3 (exhaustive):**
- Lambda: 0.0001 (minimal regularization, full data)
- Samples: 19,121 (after block oversampling)
- Training source: Exhaustive minimax tree from optimal_decision_tree.json

### Dataset Labels

**Binary classification**: 1.0 for optimal moves, 0.0 for suboptimal moves

Optimal moves are determined by:
- **3x3**: Tree's pre-computed move_details scores (probability-based ranking)

**Block oversampling**: Optimal moves that block opponent immediate wins are duplicated in the training set, increasing their relative weight during ridge regression. (this might be heuristic?)

## Code Structure

### Core Classes

**TicTacToeNxN**
- Generic tic-tac-toe board logic for any size N
- Methods: `get_valid_moves()`, `check_winner()`, `is_terminal()`

**PolynomialFeatureMap**
- Deterministic expansion from base features to polynomial features
- Uses combinations_with_replacement for systematic monomial generation
- Methods: `transform_vector()`, `transform_matrix()`

**SpectralPolicy**
- Evaluates board states via linear scoring: score = w · φ(state, move)
- Optional safety override for immediate win/block detection
- Deterministic tie-breaking by cell index
- Method: `choose_move()` returns highest-scored valid move

**Perfect3x3Agent**
- Minimax solver for 3x3 evaluation (no caching issues for small tree)
- Used to test policy performance against optimal play

**RandomAgent**
- Uniformly random move selection
- Uses global random module for reproducibility

### Training Functions

**build_exhaustive_dataset(data_fraction, seed)**
- Loads exhaustive minimax tree for 3x3
- Extracts binary labels from tree's move_details scores
- Applies block oversampling
- Downsamples by unique (state, player) pairs if data_fraction < 1.0

**build_sampled_dataset(size, target_states, max_empties, seed)**
- Randomly generates non-terminal NxN board states
- Computes optimal moves via minimax with caching
- Creates training samples for all valid moves (1.0 if optimal, 0.0 otherwise)

**fit_ridge(X, y, lam)**
- Solves (X'X + λI)w = X'y via numpy.linalg.solve()
- Returns weight vector w

### Evaluation Functions

**run_matches(size, agent, opponent, games, swap_first)**
- Alternates first player if swap_first=True
- Returns (wins, draws, losses) from agent's perspective

**run_self_play(size, agent, games)**
- Plays agent against itself
- Both players use identical policy

## Usage Examples

### 3x3 with Cached Weights (Recommended)

```bash
python spectral_policy_minimal.py --size 3 --poly-order 4
```

On first run: Trains and saves weights to `best_poly4_weights.json`. Subsequent runs load cached weights instantly.

Results (100 games each, seed 123):
- Vs Random: W:97 D:3 L:0 (policy always goes first)
- Vs Perfect 3x3 (alternating): W:0 D:100 L:0 (never loses)
- Self-play (alternating): W:0 D:100 L:0 (all draws when both optimal)

### 3x3 Retrain from Scratch

```bash
rm best_poly4_weights.json
python spectral_policy_minimal.py --size 3 --poly-order 4 --seed 42
```

### 4x4 with Sampled States

```bash
python spectral_policy_minimal.py --size 4 --poly-order 3 --sample-states 1000
```

Note: 4x4 uses lower poly-order (3) to manage memory. Results generalize from 3x3 pattern but are not exhaustively verified.

### Disable Safety Override (Pure Learning)

```bash
python spectral_policy_minimal.py --no-safety --random-games 50
```

Without safety, policy occasionally loses to fork attacks (depends on learned weights capturing subtle tactical patterns).

### Data Fraction Testing

```bash
python spectral_policy_minimal.py --data-fraction 0.5 --random-games 100
```

Tests policy trained on 50% of unique states. Results show pattern robustness at different data densities.

## Command-Line Arguments

```
--size INT                  Board size (default: 3)
--poly-order INT            Polynomial expansion order (default: 4)
--lam FLOAT                 Ridge lambda (default: 1e-4 for 3x3, 0.05 for NxN)
--data-fraction FLOAT       Fraction of unique states to keep (default: 1.0)
--sample-states INT         States to sample for NxN training (default: 500)
--max-empties INT           Maximum empty cells when sampling (default: 6)
--random-games INT          Evaluation games vs random opponent (default: 50)
--perfect-games INT         Evaluation games vs perfect 3x3 opponent (default: 50)
--self-play INT             Self-play games (default: 50)
--seed INT                  Random seed for reproducibility
--save-weights PATH         Save learned weights to JSON file
--load-weights PATH         Load pre-trained weights from JSON file
--no-safety                 Disable immediate win/block safety override
--max-weight-print INT      Weights to display (default: 10, use -1 for all)
```

## Key Results (3x3, seed 123)

### Dataset Statistics
- Total samples: 19,121 (after block oversampling)
- Positive labels (optimal moves): 9,705
- Negative labels: 9,416
- Base dimension: 18 (pure DCT)
- Polynomial dimension: 7,314 (poly-order 4)

### Learned Weights (First 10)
```
w[0]: +0.2510   (X[0,0] DCT coefficient)
w[1]: +0.0000   (X[0,1] DCT coefficient)
w[2]: -0.2863   (X[0,2] DCT coefficient)
w[3]: +0.0000   (X[1,0] DCT coefficient)
w[4]: +0.0000   (X[1,1] DCT coefficient)
w[5]: +0.0000   (X[1,2] DCT coefficient)
w[6]: -0.2863   (X[2,0] DCT coefficient)
w[7]: +0.0000   (X[2,1] DCT coefficient)
w[8]: +0.8010   (X[2,2] DCT coefficient)
w[9]: +0.1648   (O[0,0] DCT coefficient)
... (7304 more polynomial interaction terms)
```

## Related Files

- `spectral_policy_3x3.py` - Original full-featured implementation with tactical atoms, safety tuning, etc.
- `optimal_decision_tree.json` - Exhaustive minimax tree for 3x3 (from game_tree_visualizer.py)
- `best_poly4_weights.json` - Cached poly-order-4 weights for instant 3x3 inference
