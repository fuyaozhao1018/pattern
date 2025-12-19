# Pattern-Based Tic-Tac-Toe / Connect-K Engine

A modular framework for exhaustive search, pattern-based evaluation, and multi-agent battles across N×N Connect-K environments.

This repository implements a research-oriented system for generalized Tic-Tac-Toe using:

- Exhaustive game-tree enumeration (3×3, 4×4)
- Local 4×4 minimax probability databases
- Pattern-based scoring engines
- Enhanced tactical agents
- Multi-agent battle scripts (vs AlphaZero, minimax, random)

The codebase supports both research experiments and direct agent execution via command-line interfaces.

---

## Key Features

### 4×4 Exhaustive Database

- Generates all legal states and per-move (win, draw, loss) statistics.
- Stored in JSON or SQLite formats.
- Used as a local oracle for 9×9 evaluation windows.

### N=4 Pattern Engine

Learns and uses:

- Directional 3-cell patterns
- 3×3 window templates
- Spatial priors (center, edge, corner)

Provides:

- Logit-based move scoring
- Optional z-score normalization
- Fixed or adaptive thresholds
- Must-win / must-block overrides
- Parallel evaluation support

### 9×9 Pattern Agent

Performs:

- Sliding 4×4 window evaluation on the 9×9 board
- Aggregation of win/draw/loss probabilities
- Three perspectives:
  - Full board
  - Self-only
  - Opponent-only

Selects moves via a weighted scoring function.

### Enhanced Strategic Agent

Adds tactical reasoning on top of pattern scores:

- Immediate win and immediate block
- Double threats
- Open-(K−1) and open-(K−2) extensions
- Edge traps and low-information window penalties

### Battle Environment

Supports automated tournaments:

- Against AlphaZero-based agents
- Against strategic or convolutional agents
- Against minimax or random agents

Loss games can be extracted, saved, and replayed for analysis.

---

## Repository Structure

```text
pattern/
│
├── battle_agents/
│   ├── battle_alphazero_vs_enhanced.py
│   ├── enhanced_alphazero_counter.py
│   ├── n9_exhaustive_pattern_agent.py
│   └── n4_exhaustive_agent_db.py
│
├── ttt/
│   ├── common.py
│   ├── pattern_engine_n4.py
│   ├── pos_priors.py
│   ├── gen_exhaustive.py
│   ├── gen_exhaustive_full.py
│   ├── gen_exhaustive_n9.py
│   ├── eval_best_only.py
│   ├── n9_exhaustive_pattern_agent.py
│   ├── n9_enhanced_convolution_agent.py
│   ├── n9_strategic_battle.py
│   ├── replay_loss_games.py
│   └── quick_eval.py
│
├── data/
│   ├── n4_full/
│   └── n4_dual/
│
└── out/
```
## Installation

```bash
git clone https://github.com/<your-username>/pattern.git
cd pattern
pip install -r requirements.txt
```


## Quick Start

1. **Battle: AlphaZero vs Enhanced Pattern Agent**

    ```bash
    python battle_agents/battle_alphazero_vs_enhanced.py \
      --model output_tictac/models/7_best_model.pt \
      --games 20 \
      --mcts-sims 200
    ```

    Outputs:

    - `battle_agents/enhanced_lost_games.txt`
    - `battle_agents/enhanced_won_games_sample.txt`

2. **Generate a 4×4 Exhaustive Dataset**

    ```bash
    python ttt/gen_exhaustive.py \
      --N 4 \
      --K 4 \
      --lambda_draw 0.5 \
      --out_dir data/n4_full/
    ```

3. **Evaluate Predicted Best Moves**

    ```bash
    python ttt/eval_best_only.py \
      --n4_states data/n4_full/n4_exhaustive_states.json \
      --n4_best  data/n4_full/n4_exhaustive_best.json \
      --preds    out/predicted.json \
      --out_csv  out/eval_results.csv
    ```

4. **Replay Loss Games (9×9)**

    ```bash
    python ttt/replay_loss_games_9x9.py
    ```