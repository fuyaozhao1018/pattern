# AlphaZero vs Enhanced – Project Overview

## Test Results

**Tested against best model (model7) with 200 MCTS simulations**: Enhanced agent lost to AlphaZero's best model. See loss game logs in `enhanced_lost_games.txt` for detailed analysis.

## Directory Structure

- `battle_agents/`
  - `battle_alphazero_vs_enhanced.py`: Battle script that loads AlphaZero model and runs matches against Enhanced agent. Supports `--games`, `--mcts-sims`, `--size` parameters. Outputs battle logs and results.
  - `enhanced_alphazero_counter.py`: Enhanced agent core. Implements comprehensive scoring with strategy weights, center opening, center distance priority, edge/corner penalty, and 4×4 pattern search evaluation (including immediate threats, forcing moves, double threats).
  - `n9_exhaustive_pattern_agent.py`: 9×9 pattern agent. Core function `evaluate_move_in_windows` scans all 4×4 windows covering a move, aggregating probabilities and features from three perspectives (full/self/opponent). Integrates tactical flags: immediate win/block, creating open-(K−1)/(K−2) with edge trap detection, double threat count.
  - `n4_exhaustive_agent_db.py`: 4×4 exhaustive probability data access layer. Prioritizes JSON (`probs_json`), fallback to SQLite. Provides `get_move_probs` and `get_move`.
  - `common.py`: Common utilities for board display and winner checking.


- `data/`
  - `n4_full/n4_exhaustive.db`: Legacy 4×4 exhaustive probability database (SQLite). Current implementation prefers JSON probability files (exported by generator scripts).

## Enhanced Algorithm Overview

- Goal: Counter AlphaZero on 9×9 board with Connect-4 rules (K=4).
- Framework:
  - First move: always center. Tie-break by Manhattan distance to center.
  - Use 4×4 exhaustive probabilities for "local window" evaluation of each legal move (all 4×4 windows covering that move).
  - Three-perspective scoring:
    - Full board (actual win/draw/loss probability, most important)
    - Self-only pieces (offensive potential)
    - Opponent-only pieces (defensive awareness)
  - Integrate tactical flags in 4×4 evaluation:
    - Immediate win (immediate_win)
    - Blocks opponent immediate win (blocks_opp_immediate) & opponent has immediate (opp_has_immediate)
    - Creates open-(K−1)/open-(K−2) (creates_open_k1/k2), with edge trap detection (edge_trapped) to avoid "dead-end corners"
    - Double threat count (double_threat_count)
  - Scoring combines:
    - Large weights reward immediate wins & successful opponent blocks; penalty for failure to block.
    - Probability terms: full-view win rate bonus, loss rate penalty; self-view offensive bonus; opponent-view win rate as defensive penalty.
    - Window coverage count bonus; low-info window penalties (0/1/2 pieces - 0 largest, 2 smallest).
    - Forcing move rewards: double threats, non-edge-trapped open-(K−1)/(K−2).
    - Center preference & center distance tie-break, small edge/corner penalties.

## 4×4 Probability Database Source

We tested two versions of 4×4 exhaustive data:

1. **Illegal minimax version** (`data/n4_full/n4_exhaustive.db`):
   - Complete minimax search including illegal states
   - 34GB SQLite database
   - More comprehensive but includes non-legal game positions

2. **Legal dual exhaustive version** (`data/n4_dual/n4_exhaustive_probs.json`):
   - Only legal game states from actual play
   - 8GB JSON file (compressed in `n4_dual.tar.gz`)
   - More realistic but smaller state space

- Format:
  - JSON (recommended): List or dict format, records `board` (len 16, char array), `turn` ('X'/'O'), `per_move` (dict, keys 0..15 local indices, values contain `win_prob/draw_prob/loss_prob`).
  - SQLite (legacy): Binary database format.
- Loading: `n4_exhaustive_agent_db.py` prioritizes JSON; fallback to SQLite if missing. `path_resolver.py` handles file location.


## Strategy Rationale

- Use 4×4 local probabilities for "local aggregation" scoring of 9×9 board, balancing offense and defense.
- Push key tactics (immediate win/block, creating open lines, double threats) down into 4×4 search evaluation, avoiding redundant checks in scoring phase; scoring only consumes these features.
- Strengthen defense: prioritize blocking open lines and immediate wins; avoid edge/corner "dead-end" forcing moves.
- Robustness:
  - Penalty for 4×4 windows with 0/1/2 pieces, reducing overoptimistic scoring in low-info regions.
  - Center and inner ring priority, reducing ineffective edge/corner moves.

## Loss Analysis & Logs

- Output & Recording:
  - After running `battle_alphazero_vs_enhanced.py`, games lost by Enhanced are written to `battle_agents/enhanced_lost_games.txt`.
  - Sample of won games written to `battle_agents/enhanced_won_games_sample.txt`.
- Common failure causes (for debugging and tuning):
  - AlphaZero forcing sequences lead to high window scores but overall position disadvantage; increase `opp_next_win` penalty or boost defense weights.
  - High proportion of low-info windows (0/1/2 pieces), reducing scoring robustness; increase `trivial_penalty` or expand tier coefficient gap.
  - Edge trap filtering incomplete; increase `edge_penalty` or tighten boundary extension conditions.
  - Open-(K−1)/(K−2) preference too strong in losing positions; reduce `forcing_threat`.

## Quick Start

```bash
# From project root src directory
python3 battle_agents/battle_alphazero_vs_enhanced.py \
  --model output_tictac/models/7_best_model.pt \
  --games 10 \
  --mcts-sims 200 \
  --verbose
```

- After running, check:
  - Loss logs: `battle_agents/enhanced_lost_games.txt`
  - Win samples: `battle_agents/enhanced_won_games_sample.txt`

## Tuning Tips (Optional)

- Stronger defense: increase `pattern_defense` or `opp_next_win` to boost blocking priority.
- Reduce forcing: decrease `forcing_threat` and open-(K−1)/(K−2) bonuses.
- Robust scoring: increase `trivial_penalty` (maintain 0>1>2 tier coefficients).
- Reduce edge/corner: increase `edge_penalty`.

---
For detailed generation process or data format examples, add export documentation to generator scripts or include sample JSON files in `data/` directory.
