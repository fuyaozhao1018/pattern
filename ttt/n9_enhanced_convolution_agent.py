# ttt/n9_enhanced_convolution_agent.py
"""
Enhanced 9×9 Connect-4 agent using 4×4 exhaustive data as convolution kernels.

Enhancement over basic ConvolutionAgent:
  1. Must-win rule: If any move leads to immediate victory, take it
  2. Must-block rule: If opponent has a winning move, block it
  3. Convolution evaluation: Use 4×4 exhaustive data to score remaining moves
  4. Fallback heuristics: If convolution returns all zeros, use center preference

This combines tactical awareness with strategic evaluation!
"""
from __future__ import annotations
import json
import random
from typing import Dict, List, Tuple, Optional

# Import base functions from n9_convolution_agent
from ttt.n9_convolution_agent import (
    N9, N4, K,
    new_board_9x9, legal_moves_9x9, switch_turn,
    check_winner_9x9, is_full_9x9,
    extract_4x4_window, normalize_board_4x4, global_to_local_move,
    RandomAgent9x9, SmartAgent9x9, play_game_9x9
)


class EnhancedConvolutionAgent:
    """
    Enhanced convolution agent with tactical rules.
    
    Priority order:
      1. Must-win: Take immediate winning move
      2. Must-block: Block opponent's winning move
      3. Convolution: Use 4×4 exhaustive data to evaluate
      4. Fallback: Center preference if convolution fails
    """
    
    def __init__(self, probs_file: str, lambda_draw: float = 0.3, rng: random.Random = None):
        self.lambda_draw = lambda_draw
        self.rng = rng or random.Random()
        self.epsilon = 1e-12
        
        # Load 4×4 exhaustive probability map
        print(f"Loading 4×4 exhaustive data from {probs_file}...")
        with open(probs_file, 'r') as f:
            states = json.load(f)
        
        # Build state lookup by (board, turn)
        self.state_map: Dict[Tuple[Tuple[str, ...], str], Dict] = {}
        for st in states:
            key = (tuple(st['board']), st['turn'])
            self.state_map[key] = {'per_move': st.get('per_move', {})}
        
        print(f"Loaded {len(self.state_map)} 4×4 states.")
        print("Enhanced mode: Tactical rules enabled (must-win, must-block)")
    
    def _is_winning_move(self, board: List[str], move: int, player: str) -> bool:
        """Check if making this move leads to immediate victory for player."""
        board[move] = player
        result = check_winner_9x9(board)
        board[move] = ' '
        return result == player
    
    def _evaluate_convolution(self, board_9x9: List[str], move: int, turn: str) -> float:
        """
        Evaluate a move by aggregating scores from all 4×4 windows containing it.
        """
        global_r = move // N9
        global_c = move % N9
        
        total_score = 0.0
        num_windows = 0
        
        # Enumerate all 4×4 windows that include this position
        for window_r in range(max(0, global_r - N4 + 1), min(global_r + 1, N9 - N4 + 1)):
            for window_c in range(max(0, global_c - N4 + 1), min(global_c + 1, N9 - N4 + 1)):
                # Extract 4×4 window
                window = extract_4x4_window(board_9x9, window_r, window_c)
                
                # Normalize
                norm_window, norm_turn = normalize_board_4x4(window, turn)
                
                # Look up in exhaustive data
                key = (tuple(norm_window), norm_turn)
                if key not in self.state_map:
                    continue
                
                state = self.state_map[key]
                per_move = state.get('per_move', {})
                
                # Convert global move to local 4×4 move
                local_move = global_to_local_move(move, window_r, window_c)
                if local_move is None:
                    continue
                
                # Get score for this move in this window
                move_str = str(local_move)
                if move_str in per_move:
                    stats = per_move[move_str]
                    win_prob = stats.get('win_prob', 0.0)
                    draw_prob = stats.get('draw_prob', 0.0)
                    score = win_prob + self.lambda_draw * draw_prob
                    total_score += score
                    num_windows += 1
        
        # Average score across all windows
        if num_windows == 0:
            return 0.0
        return total_score / num_windows
    
    def _heuristic_score(self, board: List[str], move: int, turn: str) -> float:
        """
        Fallback heuristic when convolution cannot evaluate.
        
        Scoring:
          - Center positions (3,3) to (5,5): higher score
          - Adjacent to own pieces: bonus
        """
        r, c = move // N9, move % N9
        score = 0.0
        
        # Center preference: distance from (4,4)
        dist_from_center = abs(r - 4) + abs(c - 4)
        score += (8 - dist_from_center) * 2.0  # max 16 for center, 0 for corners
        
        # Bonus for adjacency to own pieces
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < N9 and 0 <= nc < N9:
                if board[nr * N9 + nc] == turn:
                    score += 5.0
        
        return score
    
    def select_move(self, board_9x9: List[str], turn: str) -> int:
        """
        Select best move with enhanced tactical rules.
        
        Algorithm:
          1. Check for immediate winning moves
          2. Check for opponent's winning moves (must block)
          3. Use convolution evaluation for all legal moves
          4. If all convolution scores are 0, use heuristic
          5. Random tie-breaking among best moves
        """
        legal = legal_moves_9x9(board_9x9)
        if not legal:
            return 0
        
        opponent = switch_turn(turn)
        
        # Priority 1: Must-win
        for move in legal:
            if self._is_winning_move(board_9x9, move, turn):
                return move
        
        # Priority 2: Must-block
        for move in legal:
            if self._is_winning_move(board_9x9, move, opponent):
                return move
        
        # Priority 3: Convolution evaluation
        move_scores = []
        for move in legal:
            score = self._evaluate_convolution(board_9x9, move, turn)
            move_scores.append((move, score))
        
        # Find max convolution score
        max_conv_score = max(score for _, score in move_scores)
        
        # Priority 4: If convolution fails (all zeros), use heuristic
        if max_conv_score <= self.epsilon:
            move_scores = []
            for move in legal:
                score = self._heuristic_score(board_9x9, move, turn)
                move_scores.append((move, score))
            max_conv_score = max(score for _, score in move_scores)
        
        # Collect all moves with best score (within epsilon)
        best_moves = [move for move, score in move_scores 
                      if abs(score - max_conv_score) <= self.epsilon]
        
        # Random tie-breaking
        return self.rng.choice(best_moves)


if __name__ == '__main__':
    # Quick test to compare basic vs enhanced
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m ttt.n9_enhanced_convolution_agent <path_to_n4_probs.json>")
        sys.exit(1)
    
    probs_file = sys.argv[1]
    
    rng = random.Random(2025)
    enhanced_agent = EnhancedConvolutionAgent(probs_file, lambda_draw=0.3, rng=rng)
    random_agent = RandomAgent9x9(rng)
    
    print("\n" + "="*80)
    print("Testing Enhanced Convolution Agent")
    print("="*80)
    print("\nPlaying 20 games: EnhancedConvolutionAgent (X) vs Random (O)")
    
    wins = draws = losses = 0
    
    for i in range(20):
        outcome, _ = play_game_9x9(enhanced_agent, random_agent, record_history=False)
        if outcome == 'X':
            wins += 1
        elif outcome == 'D':
            draws += 1
        else:
            losses += 1
        print(f"Game {i+1:2d}: {outcome}", end='  ')
        if (i+1) % 5 == 0:
            print()
    
    print(f"\nResults: W:{wins} D:{draws} L:{losses}")
    print(f"Win Rate: {wins/20:.1%}")
