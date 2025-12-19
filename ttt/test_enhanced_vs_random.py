"""
Test Enhanced AlphaZero Counter Agent vs Random Player

Tests the enhanced agent with:
- Double threat detection
- Open-4/Open-3 recognition
- 4-layer decision priority
- Aggressive defensive weights
"""

import sys
import os
import random
import time
from typing import List, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ttt.enhanced_alphazero_counter import AlphaZeroCounterAgent


class RandomAgent:
    """Simple random player for testing"""
    
    def get_move(self, board: List[str], turn: str) -> int:
        """Pick a random legal move"""
        legal_moves = [i for i, cell in enumerate(board) if cell == ' ']
        return random.choice(legal_moves) if legal_moves else -1


def check_winner(board: List[str], N: int = 9, K: int = 4) -> str:
    """Check if there's a winner"""
    # Check rows
    for r in range(N):
        for c in range(N - K + 1):
            window = [board[r * N + c + i] for i in range(K)]
            if window[0] != ' ' and all(cell == window[0] for cell in window):
                return window[0]
    
    # Check columns
    for c in range(N):
        for r in range(N - K + 1):
            window = [board[(r + i) * N + c] for i in range(K)]
            if window[0] != ' ' and all(cell == window[0] for cell in window):
                return window[0]
    
    # Check diagonals (top-left to bottom-right)
    for r in range(N - K + 1):
        for c in range(N - K + 1):
            window = [board[(r + i) * N + (c + i)] for i in range(K)]
            if window[0] != ' ' and all(cell == window[0] for cell in window):
                return window[0]
    
    # Check anti-diagonals (top-right to bottom-left)
    for r in range(N - K + 1):
        for c in range(K - 1, N):
            window = [board[(r + i) * N + (c - i)] for i in range(K)]
            if window[0] != ' ' and all(cell == window[0] for cell in window):
                return window[0]
    
    return None


def play_game(agent: AlphaZeroCounterAgent, random_agent: RandomAgent, 
              agent_first: bool = True, verbose: bool = False) -> str:
    """
    Play one game
    
    Returns:
        'X' - agent wins
        'O' - random wins
        'draw' - draw
    """
    N = agent.N
    K = agent.K
    board = [' '] * (N * N)
    
    agent_symbol = 'X' if agent_first else 'O'
    random_symbol = 'O' if agent_first else 'X'
    
    current_player = 'X'
    move_count = 0
    max_moves = N * N
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Game Start: Agent={agent_symbol}, Random={random_symbol}")
        print(f"{'='*60}")
    
    while move_count < max_moves:
        if current_player == agent_symbol:
            # Agent's turn
            move = agent.get_move(board, current_player, verbose=verbose)
        else:
            # Random's turn
            move = random_agent.get_move(board, current_player)
            if verbose:
                print(f"Random plays: {move}")
        
        if move == -1:
            break
        
        board[move] = current_player
        move_count += 1
        
        # Check winner
        winner = check_winner(board, N, K)
        if winner:
            if verbose:
                print(f"\n{'='*60}")
                print(f"Winner: {winner}")
                print(f"{'='*60}")
            return winner
        
        # Switch player
        current_player = 'O' if current_player == 'X' else 'X'
    
    if verbose:
        print(f"\n{'='*60}")
        print("Game ended in draw")
        print(f"{'='*60}")
    
    return 'draw'


def run_test_series(num_games: int = 100, board_size: int = 9, connect_k: int = 4):
    """
    Run a series of games
    
    Args:
        num_games: Number of games to play
        board_size: Board size (9, 11, 13, 15)
        connect_k: Number to connect (4 or 5)
    """
    print("=" * 80)
    print(f"Enhanced AlphaZero Counter Agent vs Random Player")
    print("=" * 80)
    print(f"Board Size: {board_size}×{board_size}")
    print(f"Win Condition: {connect_k}-in-a-row")
    print(f"Number of Games: {num_games}")
    print(f"Agent plays as X (first) and O (second) alternately")
    print("=" * 80)
    
    # Initialize agents
    print("\nInitializing agents...")
    agent = AlphaZeroCounterAgent(board_size=board_size)
    agent.K = connect_k  # Override K if needed
    random_agent = RandomAgent()
    
    print(f"✓ Enhanced Agent loaded")
    print(f"✓ Random Agent ready")
    
    # Statistics
    results = {
        'agent_wins': 0,
        'random_wins': 0,
        'draws': 0,
        'agent_wins_as_first': 0,
        'agent_wins_as_second': 0
    }
    
    start_time = time.time()
    
    print(f"\nStarting {num_games} games...")
    print("-" * 80)
    
    for i in range(num_games):
        agent_first = (i % 2 == 0)
        verbose = (i < 2)  # First 2 games verbose
        
        result = play_game(agent, random_agent, agent_first, verbose)
        
        agent_symbol = 'X' if agent_first else 'O'
        
        if result == agent_symbol:
            results['agent_wins'] += 1
            if agent_first:
                results['agent_wins_as_first'] += 1
            else:
                results['agent_wins_as_second'] += 1
        elif result == 'draw':
            results['draws'] += 1
        else:
            results['random_wins'] += 1
        
        # Progress
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            games_per_sec = (i + 1) / elapsed
            win_rate = results['agent_wins'] / (i + 1) * 100
            print(f"Progress: {i+1}/{num_games} games | "
                  f"Win Rate: {win_rate:.1f}% | "
                  f"Speed: {games_per_sec:.1f} games/s")
    
    elapsed_time = time.time() - start_time
    
    # Print results
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    
    total_games = num_games
    agent_wins = results['agent_wins']
    random_wins = results['random_wins']
    draws = results['draws']
    
    print(f"\nTotal Games: {total_games}")
    print(f"Time Elapsed: {elapsed_time:.2f}s")
    print(f"Speed: {total_games / elapsed_time:.2f} games/s")
    print()
    
    print(f"Agent Wins:  {agent_wins:4d} / {total_games} ({agent_wins/total_games*100:.1f}%)")
    print(f"  - As First:  {results['agent_wins_as_first']:4d}")
    print(f"  - As Second: {results['agent_wins_as_second']:4d}")
    print(f"Random Wins: {random_wins:4d} / {total_games} ({random_wins/total_games*100:.1f}%)")
    print(f"Draws:       {draws:4d} / {total_games} ({draws/total_games*100:.1f}%)")
    print()
    
    # Win rate analysis
    win_rate = agent_wins / total_games * 100
    
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    if win_rate >= 95:
        print("✅ EXCELLENT: Agent dominates with >95% win rate!")
    elif win_rate >= 90:
        print("✅ GREAT: Agent performs very well with >90% win rate!")
    elif win_rate >= 80:
        print("✓ GOOD: Agent performs well with >80% win rate")
    elif win_rate >= 70:
        print("⚠ OK: Agent has decent performance with >70% win rate")
    else:
        print("❌ POOR: Agent needs improvement (<70% win rate)")
    
    print()
    
    if random_wins > 0:
        print(f"⚠ Warning: Random won {random_wins} games!")
        print("  Enhanced agent should never lose to random player")
    else:
        print("✅ Perfect: No losses to random player!")
    
    print()
    
    if draws > total_games * 0.3:
        print(f"⚠ High draw rate ({draws/total_games*100:.1f}%)")
        print("  Consider more aggressive play")
    else:
        print(f"✓ Draw rate is acceptable ({draws/total_games*100:.1f}%)")
    
    print("=" * 80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Enhanced Agent vs Random')
    parser.add_argument('--games', type=int, default=100, help='Number of games to play')
    parser.add_argument('--size', type=int, default=9, help='Board size (9, 11, 13, 15)')
    parser.add_argument('--connect', type=int, default=4, help='Connect-K (4 or 5)')
    
    args = parser.parse_args()
    
    run_test_series(num_games=args.games, board_size=args.size, connect_k=args.connect)
