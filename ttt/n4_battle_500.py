"""
Simple 4×4 Exhaustive Agent vs Random - 500 Games
Quick battle test using the database version
"""

import random
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ttt.n4_exhaustive_agent_db import ExhaustiveAgentDB, check_winner, is_full


def play_game(agent: ExhaustiveAgentDB, agent_color: str) -> dict:
    """Play one game, return result"""
    board = [' '] * 16
    turn = 'X'
    moves = 0
    
    while True:
        winner = check_winner(board)
        if winner:
            return {'winner': winner, 'moves': moves}
        if is_full(board):
            return {'winner': 'draw', 'moves': moves}
        
        # Get move
        if turn == agent_color:
            move = agent.get_move(board, turn)
        else:
            legal = [i for i in range(16) if board[i] == ' ']
            move = random.choice(legal)
        
        board[move] = turn
        moves += 1
        turn = 'O' if turn == 'X' else 'X'


def main():
    print("="*60)
    print("4×4 Exhaustive Agent vs Random - 500 Games")
    print("Using FULL 43M exhaustive database (all states)")
    print("="*60)
    
    # Load agent with FULL database
    print("Loading agent...")
    agent = ExhaustiveAgentDB('data/n4_full/n4_exhaustive.db')
    
    # Play 500 games
    results = {'wins': 0, 'draws': 0, 'losses': 0, 'total_moves': 0}
    
    print(f"\nPlaying 500 games...")
    start = time.time()
    
    for i in range(500):
        agent_color = 'X' if i % 2 == 0 else 'O'
        result = play_game(agent, agent_color)
        
        results['total_moves'] += result['moves']
        
        if result['winner'] == agent_color:
            results['wins'] += 1
        elif result['winner'] == 'draw':
            results['draws'] += 1
        else:
            results['losses'] += 1
        
        # Progress
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/500 - W/D/L: {results['wins']}/{results['draws']}/{results['losses']}")
    
    elapsed = time.time() - start
    
    # Results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Time: {elapsed:.1f}s ({500/elapsed:.1f} games/s)")
    print(f"Avg moves: {results['total_moves']/500:.1f}")
    print()
    print(f"Wins:   {results['wins']}/500 ({results['wins']/5:.1f}%)")
    print(f"Draws:  {results['draws']}/500 ({results['draws']/5:.1f}%)")
    print(f"Losses: {results['losses']}/500 ({results['losses']/5:.1f}%)")
    print()
    
    if results['losses'] == 0:
        print("✓ PERFECT! Agent never lost!")
    if results['wins'] == 500:
        print("✓ 100% WIN RATE!")
    
    print("="*60)


if __name__ == '__main__':
    main()
