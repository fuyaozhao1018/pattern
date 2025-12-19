# ttt/compare_opening_strategies.py
"""
Compare opening strategies across different 9×9 Connect-4 agents:
  1. Enhanced Agent: Tactical rules + basic heuristic (center + adjacency)
  2. Strategic Agent: Tactical rules + shape-aware heuristic + opening region restriction
  3. Random Agent: Baseline

Key comparison:
  - Enhanced: Can randomly pick edges/corners in opening
  - Strategic: Restricted to center [K-1, N-K] = [3,5] region (3×3 center)
  
We'll visualize first moves AND show statistical distribution over 1000 games.
"""
import sys
import random
from ttt.n9_convolution_agent import (
    new_board_9x9, legal_moves_9x9, N9, K
)
from ttt.n9_enhanced_convolution_agent import EnhancedConvolutionAgent
from ttt.n9_strategic_convolution_agent import StrategicConvolutionAgent


def format_board_simple(board):
    """Simple board formatting for quick view."""
    lines = []
    lines.append("  " + " ".join(str(i) for i in range(N9)))
    for r in range(N9):
        row = board[r*N9:(r+1)*N9]
        row_str = " ".join('.' if c == ' ' else c for c in row)
        lines.append(f"{r} {row_str}")
    return "\n".join(lines)


def show_opening_moves(agent, agent_name, probs_file):
    """Show first 5 moves of an agent starting from empty board."""
    print("\n" + "="*80)
    print(f"{agent_name} - Opening 5 Moves")
    print("="*80)
    
    board = new_board_9x9()
    turn = 'X'
    
    for step in range(5):
        if isinstance(agent, str) and agent == 'random':
            # Random baseline
            legal = legal_moves_9x9(board)
            move = random.choice(legal)
        else:
            move = agent.select_move(board, turn)
        
        r, c = move // N9, move % N9
        board[move] = turn
        
        print(f"\nStep {step+1}: {turn} plays at ({r},{c}) [pos {move}]")
        print(format_board_simple(board))
        
        turn = 'O' if turn == 'X' else 'X'
    
    print()


def analyze_opening_distribution(agent, agent_name, num_games=1000):
    """
    Statistical analysis: where does agent place the first move?
    
    Categorizes into 3 regions:
      - Center [3,5]: Optimal opening region (K-1 to N-K)
      - Inner [2,6]: Acceptable but less optimal
      - Outer [0-1,7-8]: Poor (edges/corners)
    """
    center_count = 0
    inner_count = 0
    outer_count = 0
    
    for _ in range(num_games):
        board = new_board_9x9()
        
        if isinstance(agent, str) and agent == 'random':
            legal = legal_moves_9x9(board)
            move = random.choice(legal)
        else:
            move = agent.select_move(board, 'X')
        
        r, c = move // N9, move % N9
        
        # Categorize
        if 3 <= r <= 5 and 3 <= c <= 5:
            center_count += 1
        elif 2 <= r <= 6 and 2 <= c <= 6:
            inner_count += 1
        else:
            outer_count += 1
    
    return {
        'name': agent_name,
        'center': center_count,
        'inner': inner_count,
        'outer': outer_count,
        'total': num_games
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m ttt.compare_opening_strategies <path_to_n4_probs.json>")
        print("\nExample:")
        print("  python -m ttt.compare_opening_strategies data/n4_dual/n4_exhaustive_probs.json")
        sys.exit(1)
    
    probs_file = sys.argv[1]
    
    print("\n" + "="*80)
    print("Opening Strategy Comparison - 9×9 Connect-4")
    print("="*80)
    print(f"Board size: {N9}×{N9}, K-in-a-row: {K}")
    print(f"Optimal opening region: [{K-1}, {N9-K}] = [3, 5] (3×3 center)")
    print("="*80)
    
    # Initialize agents
    enhanced = EnhancedConvolutionAgent(probs_file, lambda_draw=0.3, rng=random.Random(2025))
    strategic = StrategicConvolutionAgent(probs_file, lambda_draw=0.3, rng=random.Random(2025))
    
    # Part 1: Visualize opening moves
    print("\n" + "#"*80)
    print("# PART 1: Opening Move Visualization (First 5 Moves)")
    print("#"*80)
    
    show_opening_moves(enhanced, "Enhanced Agent (NO opening restriction)", probs_file)
    show_opening_moves(strategic, "Strategic Agent (WITH opening restriction [3,5])", probs_file)
    
    random.seed(2025)
    show_opening_moves('random', "Random Agent (baseline)", probs_file)
    
    # Part 2: Statistical distribution
    print("\n\n" + "#"*80)
    print("# PART 2: Statistical Opening Distribution (1000 games each)")
    print("#"*80)
    print()
    
    # Reset random seeds
    enhanced.rng = random.Random(2025)
    strategic.rng = random.Random(2025)
    random.seed(2025)
    
    results = [
        analyze_opening_distribution(enhanced, "Enhanced Agent", 1000),
        analyze_opening_distribution(strategic, "Strategic Agent", 1000),
        analyze_opening_distribution('random', "Random Agent", 1000)
    ]
    
    # Display table
    print(f"{'Agent':<30} {'Center [3,5]':>15} {'Inner [2,6]':>15} {'Outer [0-1,7-8]':>15}")
    print("-" * 80)
    
    for r in results:
        center_pct = r['center'] / r['total'] * 100
        inner_pct = r['inner'] / r['total'] * 100
        outer_pct = r['outer'] / r['total'] * 100
        
        print(f"{r['name']:<30} {r['center']:>6} ({center_pct:5.1f}%)  "
              f"{r['inner']:>6} ({inner_pct:5.1f}%)  "
              f"{r['outer']:>6} ({outer_pct:5.1f}%)")
    
    print("\n" + "="*80)
    print("Key Findings:")
    print("="*80)
    print(f"✓ Strategic Agent: {results[1]['center']/10:.1f}% opening moves in optimal [3,5] region")
    print(f"  - This ensures first moves are in positions that can form lines in ALL directions")
    print(f"  - No wasted moves on edges/corners in opening phase")
    print()
    print(f"✗ Enhanced Agent: {results[0]['outer']/10:.1f}% opening moves in outer edges")
    print(f"  - Can randomly pick suboptimal positions (corners/edges)")
    print(f"  - No opening strategy, just heuristic scoring")
    print()
    print("Region Definitions:")
    print("  - Center [3,5]: Can form 4-in-a-row in any direction (optimal)")
    print("  - Inner [2,6]: Limited flexibility but still reasonable")
    print("  - Outer [0-1,7-8]: Edges/corners - poor opening choice")
    print("="*80)


if __name__ == '__main__':
    main()
