# ttt/n9_strategic_convolution_agent.py
"""
Strategic 9×9 Connect-4 agent with shape-aware opening strategy.

Key improvements over enhanced agent:
  1. Shape-aware heuristic: Rewards forming consecutive patterns
  2. Open-ended lines: Prioritizes lines that can extend in both directions
  3. Two-in-a-row bonus: Early game formation of potential threats
  4. Three-in-a-row bonus: Mid-game tactical positioning

This addresses the "random opening" problem by giving the agent
strategic direction even when convolution data doesn't apply.
"""
from __future__ import annotations
import json
import random
from typing import Dict, List, Tuple, Optional

# Import base functions
from ttt.n9_convolution_agent import (
    N9, N4, K,
    new_board_9x9, legal_moves_9x9, switch_turn,
    check_winner_9x9, is_full_9x9,
    extract_4x4_window, normalize_board_4x4, global_to_local_move,
    RandomAgent9x9, SmartAgent9x9, play_game_9x9
)


class StrategicConvolutionAgent:
    """
    Strategic convolution agent with shape-aware opening play.
    
    Priority order:
      1. Must-win: Take immediate winning move
      2. Must-block: Block opponent's winning move
      3. Convolution: Use 4×4 exhaustive data to evaluate
      4. Strategic heuristic: Form consecutive patterns (not just center play)
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
        print("Strategic mode: Shape-aware heuristics enabled")
    
    def _is_winning_move(self, board: List[str], move: int, player: str) -> bool:
        """Check if making this move leads to immediate victory for player."""
        board[move] = player
        result = check_winner_9x9(board)
        board[move] = ' '
        return result == player
    
    def _count_consecutive(self, board: List[str], r: int, c: int, dr: int, dc: int, player: str) -> int:
        """
        Count consecutive pieces of 'player' starting from (r,c) in direction (dr,dc).
        Returns count including the starting position if it's the player's piece.
        """
        count = 0
        nr, nc = r, c
        while 0 <= nr < N9 and 0 <= nc < N9 and board[nr * N9 + nc] == player:
            count += 1
            nr += dr
            nc += dc
        return count
    
    def _evaluate_line_potential(self, board: List[str], r: int, c: int, dr: int, dc: int, player: str) -> float:
        """
        Evaluate the potential of a line passing through (r,c) in direction (dr,dc).
        
        Returns high score if:
          - Placing here creates 2-in-a-row or 3-in-a-row
          - Line is "open-ended" (has space to grow in both directions)
          - Line can potentially reach K=4 consecutive
        """
        # Count consecutive in both directions from this position
        forward = self._count_consecutive(board, r + dr, c + dc, dr, dc, player)
        backward = self._count_consecutive(board, r - dr, c - dc, -dr, -dc, player)
        
        # Total consecutive if we place here
        total_consecutive = forward + backward + 1
        
        # Check if line can extend to K=4
        # Count total spaces (including empty) in a K-sized window
        forward_spaces = 0
        nr, nc = r, c
        for _ in range(K):
            if 0 <= nr < N9 and 0 <= nc < N9:
                if board[nr * N9 + nc] in (' ', player):
                    forward_spaces += 1
                else:
                    break
            else:
                break
            nr += dr
            nc += dc
        
        backward_spaces = 0
        nr, nc = r - dr, c - dc
        for _ in range(K - 1):  # -1 because we already counted (r,c)
            if 0 <= nr < N9 and 0 <= nc < N9:
                if board[nr * N9 + nc] in (' ', player):
                    backward_spaces += 1
                else:
                    break
            else:
                break
            nr -= dr
            nc -= dc
        
        total_spaces = forward_spaces + backward_spaces
        
        # Can't possibly win in this direction
        if total_spaces < K:
            return 0.0
        
        # Score based on consecutive count
        # 3-in-a-row: huge bonus (one move from winning)
        # 2-in-a-row: good bonus (building threats)
        # 1-in-a-row with open space: small bonus
        score = 0.0
        if total_consecutive == 3:
            score += 100.0  # One away from winning!
        elif total_consecutive == 2:
            score += 20.0   # Building a threat
        else:
            score += 2.0    # Starting a line
        
        # Open-ended bonus: both sides can extend
        if forward_spaces >= 2 and backward_spaces >= 2:
            score *= 1.5
        
        return score
    
    def evaluate_move(self, board_9x9: List[str], move: int, turn: str) -> float:
        """
        Evaluate a move using convolution of 4×4 windows.
        
        For each 4×4 window containing the move, we look up the position's
        probability from the exhaustive 4×4 data and aggregate.
        
        Returns aggregated score with penalties for:
        1. Empty windows (no pieces = no battle happening)
        2. Sparse windows (very few pieces = low activity)
        """
        row, col = move // N9, move % N9
        
        total_score = 0.0
        count = 0
        
        # Enumerate all possible 4×4 windows that include this move
        for start_r in range(max(0, row - N4 + 1), min(row + 1, N9 - N4 + 1)):
            for start_c in range(max(0, col - N4 + 1), min(col + 1, N9 - N4 + 1)):
                # Extract 4×4 window
                window = []
                for r in range(start_r, start_r + N4):
                    for c in range(start_c, start_c + N4):
                        window.append(board_9x9[r * N9 + c])
                
                # Count pieces in this window
                x_count = window.count('X')
                o_count = window.count('O')
                total_pieces = x_count + o_count
                
                # Empty window penalty: if no pieces at all, this window is irrelevant
                if total_pieces == 0:
                    # Completely empty 4×4 window - no battle here, skip it
                    continue
                
                # Check balance: exhaustive data only has ±1 piece difference
                piece_diff = abs(x_count - o_count)
                if piece_diff > 1:
                    # Unbalanced window - not in exhaustive data
                    # This happens in 9×9 when window crosses different battle zones
                    # Skip it because lookup will fail anyway
                    continue
                
                # Sparse window weight: if very few pieces, reduce weight
                window_weight = 1.0
                if total_pieces == 1:
                    window_weight = 0.3  # Only 1 piece in 4×4, not much happening
                elif total_pieces == 2:
                    window_weight = 0.6  # 2 pieces, some activity
                # 3+ pieces: full weight (1.0)
                
                # Determine local position within this 4×4 window
                local_row = row - start_r
                local_col = col - start_c
                local_pos = local_row * N4 + local_col
                
                # Lookup probability from 4×4 exhaustive data
                window_score = self._lookup_position_score(window, local_pos, turn)
                total_score += window_score * window_weight
                count += 1
        
        if count == 0:
            # All windows are empty - this move is in a barren area
            return 0.0
        
        avg_score = total_score / count
        return avg_score
    
    def _lookup_position_score(self, window: List[str], local_pos: int, turn: str) -> float:
        """
        Look up the score for a position in a 4×4 window from exhaustive data.
        """
        # Normalize window to canonical form
        normalized_board, normalized_turn = normalize_board_4x4(window, turn)
        # For now, assume no transformation, so position stays the same
        normalized_pos = local_pos
        
        # Lookup in state map
        key = (tuple(normalized_board), normalized_turn)
        if key not in self.state_map:
            return 0.0
        
        per_move = self.state_map[key]['per_move']
        pos_str = str(normalized_pos)
        if pos_str not in per_move:
            return 0.0
        
        data = per_move[pos_str]
        p_win = data.get('p_win', 0.0)
        p_draw = data.get('p_draw', 0.0)
        
        return p_win + self.lambda_draw * p_draw
    
    def _get_position_weight(self, board: List[str], r: int, c: int) -> float:
        """
        Calculate position weight based on distance from action center.
        
        When there are pieces on the board, moves closer to existing pieces
        should be preferred. This prevents playing in irrelevant corners.
        
        Returns: weight in range (0.1, 1.0]
        """
        # Find center of mass of all existing pieces
        piece_positions = []
        for idx, cell in enumerate(board):
            if cell != ' ':
                pr, pc = idx // N9, idx % N9
                piece_positions.append((pr, pc))
        
        if not piece_positions:
            # Empty board: all positions equally good
            return 1.0
        
        # Calculate center of mass
        center_r = sum(pr for pr, _ in piece_positions) / len(piece_positions)
        center_c = sum(pc for _, pc in piece_positions) / len(piece_positions)
        
        # Calculate distance from this move to center of mass
        dist = abs(r - center_r) + abs(c - center_c)  # Manhattan distance
        
        # Apply penalty based on distance
        # dist=0: weight=1.0
        # dist=4: weight=0.7
        # dist=8: weight=0.4
        # dist=12+: weight=0.1 (minimum)
        max_dist = 12.0
        weight = 1.0 - (dist / max_dist) * 0.9
        weight = max(0.1, weight)  # Ensure minimum weight
        
        return weight
    
    def _strategic_heuristic(self, board: List[str], move: int, turn: str) -> float:
        """
        Strategic heuristic with shape awareness.
        
        Evaluates all 4 directions (horizontal, vertical, 2 diagonals) and
        rewards moves that create consecutive patterns.
        
        This is the key improvement over the basic heuristic!
        """
        r, c = move // N9, move % N9
        score = 0.0
        
        # Center preference (baseline positioning)
        dist_from_center = abs(r - 4) + abs(c - 4)
        score += (8 - dist_from_center) * 1.0  # Reduced weight compared to line potential
        
        # Evaluate all 4 directions for line potential
        directions = [
            (0, 1),   # Horizontal
            (1, 0),   # Vertical
            (1, 1),   # Diagonal \
            (1, -1),  # Diagonal /
        ]
        
        for dr, dc in directions:
            line_score = self._evaluate_line_potential(board, r, c, dr, dc, turn)
            score += line_score
        
        # Also consider blocking opponent's lines (defensive)
        opponent = switch_turn(turn)
        defensive_score = 0.0
        for dr, dc in directions:
            opp_line_score = self._evaluate_line_potential(board, r, c, dr, dc, opponent)
            defensive_score += opp_line_score * 0.5  # 50% weight of offensive score
        
        score += defensive_score
        
        return score
    
    def _get_opening_region_moves(self, board: List[str]) -> List[int]:
        """
        Get legal moves restricted to the effective opening region.
        
        For N×N board with K-in-a-row to win:
          - Effective region: rows/cols in range [K-1, N-K]
          - For 9×9 with K=4: [3, 5] → center 3×3 region
          
        Rationale: Placing in corners/edges wastes the first move's
        strategic advantage, as those positions can't form winning lines
        in all directions.
        """
        # Count each player's pieces to determine if it's opening phase
        x_count = sum(1 for cell in board if cell == 'X')
        o_count = sum(1 for cell in board if cell == 'O')
        
        # Only apply restriction when BOTH players have made ≤ 1 move
        # (i.e., we're in the first 2 moves of the game total)
        if x_count > 1 or o_count > 1:
            return legal_moves_9x9(board)
        
        # Calculate effective region boundaries
        min_pos = K - 1  # For K=4: min_pos=3
        max_pos = N9 - K  # For K=4, N9=9: max_pos=5
        
        # Filter legal moves to those in the center region
        opening_moves = []
        for move in legal_moves_9x9(board):
            r, c = move // N9, move % N9
            if min_pos <= r <= max_pos and min_pos <= c <= max_pos:
                opening_moves.append(move)
        
        # Fallback: if somehow no moves in region (shouldn't happen), use all legal
        return opening_moves if opening_moves else legal_moves_9x9(board)
    
    def select_move(self, board_9x9: List[str], turn: str) -> int:
        """
        Select best move with strategic awareness.
        
        Algorithm:
          1. (Opening) Restrict to effective center region [K-1, N-K]
          2. Check for immediate winning moves (must-win)
          3. Check for opponent's winning moves (must-block)
          4. Use convolution evaluation for all legal moves
          5. If all convolution scores are 0, use STRATEGIC heuristic (not just center)
          6. Random tie-breaking among best moves
        """
        # Use opening region restriction for early game
        legal = self._get_opening_region_moves(board_9x9)
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
            score = self.evaluate_move(board_9x9, move, turn)
            move_scores.append((move, score))
        
        # Find max convolution score
        max_conv_score = max(score for _, score in move_scores)
        
        # Priority 4: If convolution fails (all zeros), use STRATEGIC heuristic
        if max_conv_score <= self.epsilon:
            move_scores = []
            for move in legal:
                score = self._strategic_heuristic(board_9x9, move, turn)
                move_scores.append((move, score))
            max_conv_score = max(score for _, score in move_scores)
        
        # Collect all moves with best score (within epsilon)
        best_moves = [move for move, score in move_scores 
                      if abs(score - max_conv_score) <= self.epsilon]
        
        # Random tie-breaking
        return self.rng.choice(best_moves)


if __name__ == '__main__':
    # Quick test
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m ttt.n9_strategic_convolution_agent <path_to_n4_probs.json>")
        sys.exit(1)
    
    probs_file = sys.argv[1]
    
    rng = random.Random(2025)
    strategic_agent = StrategicConvolutionAgent(probs_file, lambda_draw=0.3, rng=rng)
    random_agent = RandomAgent9x9(rng)
    
    print("\n" + "="*80)
    print("Testing Strategic Convolution Agent")
    print("="*80)
    print("\nPlaying 20 games: StrategicConvolutionAgent (X) vs Random (O)")
    
    wins = draws = losses = 0
    
    for i in range(20):
        outcome, _ = play_game_9x9(strategic_agent, random_agent, record_history=False)
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
