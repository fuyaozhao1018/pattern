#!/usr/bin/env python3
"""
Battle: AlphaZero (trained model) vs Enhanced Pattern Agent

This script loads your trained AlphaZero model and tests it against
the enhanced exhaustive pattern agent.

Usage:
    python ttt/battle_alphazero_vs_enhanced.py --model path/to/model.pth --games 100

Requirements:
    - Trained AlphaZero model (.pth file)
    - 4×4 exhaustive database (data/n4_full/n4_exhaustive.db)
"""
import sys
import os
import argparse
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from ttt.enhanced_alphazero_counter import EnhancedAlphaZeroCounter
from ttt.common import check_winner, print_board

# Import AlphaZero model components
try:
    from src.model import NeuralNetwork
    from src.mcts import MonteCarloTreeSearch
    from src.value_policy_function import ValuePolicyNetwork
    from src.game import TicTacToe
except ImportError as e:
    print(f"Error importing AlphaZero components: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


class AlphaZeroAgent:
    """Wrapper for trained AlphaZero model"""
    
    def __init__(self, model_path: str, board_size: int = 9, win_length: int = 4, 
                 num_simulations: int = 100):
        """
        Initialize AlphaZero agent with trained model
        
        Args:
            model_path: Path to trained model .pt file
            board_size: Size of board (default 9 for 9×9)
            win_length: Pieces in a row to win (default 4)
            num_simulations: MCTS simulations per move (default 100)
        """
        self.N = board_size
        self.K = win_length
        self.num_simulations = num_simulations
        
        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"AlphaZero using device: {self.device}")
        
        # Initialize game (使用你项目中的TicTacToe类)
        self.game = TicTacToe()
        
        # Load model using ValuePolicyNetwork (按照test_bot_vs_bot.py的方式)
        print(f"Loading AlphaZero model from {model_path}...")
        try:
            vpn = ValuePolicyNetwork(model_path)
            self.policy_value_network = vpn.get_vp
            print(f"Model loaded successfully")
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise
        
        # Initialize MCTS (使用你项目中的MCTS类)
        self.mcts = MonteCarloTreeSearch(self.game, self.policy_value_network)
        
        print(f"AlphaZero ready (MCTS simulations: {num_simulations})")
    
    def get_move(self, board: list, turn: str, verbose: bool = False) -> int:
        """
        Get AlphaZero's move using MCTS
        
        Args:
            board: Current board state (list of 81 elements, 'X'/'O'/' ')
            turn: Current player ('X' or 'O')
            verbose: Print debug info
            
        Returns:
            Selected move position (0-80)
        """
        # Convert from 'X'/'O'/' ' format to 1/-1/0 format (AlphaZero's format)
        state = np.zeros(self.N * self.N, dtype=np.float32)
        for i in range(self.N * self.N):
            if board[i] == 'X':
                state[i] = 1
            elif board[i] == 'O':
                state[i] = -1
            else:
                state[i] = 0
        
        # Determine current player (1 for X, -1 for O)
        player = 1 if turn == 'X' else -1
        
        # Convert to canonical form (current player's perspective)
        canonical_state = state * player
        
        # Run MCTS to get visit counts
        visit_counts = self.mcts.search(canonical_state, num_searches=self.num_simulations)
        
        # Get legal moves
        legal_moves = [i for i in range(self.N * self.N) if board[i] == ' ']
        
        if verbose:
            # Show top 5 moves by visit count
            visit_list = [(i, visit_counts[i]) for i in legal_moves if visit_counts[i] > 0]
            visit_list.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\nAlphaZero MCTS top 5 moves:")
            for move, count in visit_list[:5]:
                r, c = move // self.N, move % self.N
                print(f"  Pos {move} ({r},{c}): visits={int(count)}")
        
        # Select move with highest visit count
        best_move = max(legal_moves, key=lambda m: visit_counts[m])
        
        if verbose:
            r, c = best_move // self.N, best_move % self.N
            print(f"🤖 AlphaZero selected: {best_move} ({r},{c}), visits={int(visit_counts[best_move])}")
        
        return best_move


def play_game(alphazero: AlphaZeroAgent, enhanced: EnhancedAlphaZeroCounter,
              alphazero_first: bool, verbose: bool = False) -> str:
    """
    Play one game between AlphaZero and Enhanced agent
    
    Args:
        alphazero: AlphaZero agent
        enhanced: Enhanced pattern agent
        alphazero_first: True if AlphaZero plays first (X), False if second (O)
        verbose: Print game moves
        
    Returns:
        Winner: 'alphazero', 'enhanced', or 'draw'
    """
    board = [' '] * (alphazero.N * alphazero.N)
    N = alphazero.N
    K = alphazero.K
    
    # Determine who plays X and O
    if alphazero_first:
        x_player = alphazero
        o_player = enhanced
        x_name = "AlphaZero"
        o_name = "Enhanced"
    else:
        x_player = enhanced
        o_player = alphazero
        x_name = "Enhanced"
        o_name = "AlphaZero"
    
    current_player = 'X'
    move_count = 0
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Game Start: {x_name}=X, {o_name}=O")
        print(f"{'='*60}\n")
    
    while True:
        # Get current agent
        if current_player == 'X':
            agent = x_player
            agent_name = x_name
        else:
            agent = o_player
            agent_name = o_name
        
        # Get move
        move = agent.get_move(board, current_player, verbose=verbose)
        
        # Make move
        board[move] = current_player
        move_count += 1
        
        if verbose:
            r, c = move // N, move % N
            print(f"{agent_name} plays: {move} ({r},{c})")
            print_board(board, N)
            print()
        
        # Check winner
        winner = check_winner(board, N, K)
        if winner:
            if verbose:
                print(f"{'='*60}")
                print(f"Winner: {winner}")
                print(f"{'='*60}\n")
            
            if (winner == 'X' and alphazero_first) or (winner == 'O' and not alphazero_first):
                return 'alphazero'
            else:
                return 'enhanced'
        
        # Check draw
        if move_count >= N * N:
            if verbose:
                print(f"{'='*60}")
                print(f"Draw")
                print(f"{'='*60}\n")
            return 'draw'
        
        # Switch player
        current_player = 'O' if current_player == 'X' else 'X'


def run_battle(model_path: str, num_games: int = 100, board_size: int = 9,
               win_length: int = 4, mcts_sims: int = 100, db_path: str = None):
    """
    Run a battle between AlphaZero and Enhanced agent
    
    Args:
        model_path: Path to trained AlphaZero model
        num_games: Number of games to play
        board_size: Board size
        win_length: Pieces in a row to win
        mcts_sims: MCTS simulations per move for AlphaZero
        db_path: Path to exhaustive database (optional)
    """
    print("="*80)
    print(" "*20 + "AlphaZero vs Enhanced Pattern Agent Battle")
    print("="*80)
    print(f"Board Size: {board_size}×{board_size}")
    print(f"Win Condition: {win_length}-in-a-row")
    print(f"Number of Games: {num_games}")
    print(f"AlphaZero MCTS simulations: {mcts_sims}")
    print(f"AlphaZero and Enhanced agents alternate first/second player")
    print("="*80)
    print()
    
    # Initialize agents
    print("Initializing agents...")
    
    # AlphaZero
    alphazero = AlphaZeroAgent(model_path, board_size, win_length, mcts_sims)
    
    # Enhanced agent
    print("\nLoading Enhanced Pattern Agent...")
    if db_path:
        enhanced = EnhancedAlphaZeroCounter(board_size=board_size, 
                                           window_size=4,
                                           db_path=db_path)
    else:
        enhanced = EnhancedAlphaZeroCounter(board_size=board_size, window_size=4)
    print("✓ Enhanced Agent loaded")
    
    print("\nStarting battle...")
    print("-"*80)
    
    # Statistics
    results = {
        'alphazero': 0,
        'enhanced': 0,
        'draw': 0
    }
    alphazero_first_wins = 0
    alphazero_second_wins = 0
    enhanced_first_wins = 0
    enhanced_second_wins = 0
    
    start_time = time.time()
    
    # Play games
    for i in range(num_games):
        # Alternate who goes first
        alphazero_first = (i % 2 == 0)
        
        # Show verbose output for first 2 games only
        verbose = (i < 2)
        
        # Play game
        winner = play_game(alphazero, enhanced, alphazero_first, verbose)
        
        # Update statistics
        results[winner] += 1
        if winner == 'alphazero':
            if alphazero_first:
                alphazero_first_wins += 1
            else:
                alphazero_second_wins += 1
        elif winner == 'enhanced':
            if not alphazero_first:
                enhanced_first_wins += 1
            else:
                enhanced_second_wins += 1
        
        # Progress update every 10 games
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed
            az_rate = results['alphazero'] / (i + 1) * 100
            print(f"Progress: {i+1}/{num_games} games | "
                  f"AlphaZero: {az_rate:.1f}% | "
                  f"Speed: {speed:.1f} games/s")
    
    # Final results
    elapsed = time.time() - start_time
    print()
    print("="*80)
    print(" "*30 + "FINAL RESULTS")
    print("="*80)
    print()
    print(f"Total Games: {num_games}")
    print(f"Time Elapsed: {elapsed:.2f}s")
    print(f"Speed: {num_games/elapsed:.2f} games/s")
    print()
    print(f"AlphaZero Wins:   {results['alphazero']:3d} / {num_games} "
          f"({results['alphazero']/num_games*100:.1f}%)")
    print(f"  - As First:     {alphazero_first_wins}")
    print(f"  - As Second:    {alphazero_second_wins}")
    print(f"Enhanced Wins:    {results['enhanced']:3d} / {num_games} "
          f"({results['enhanced']/num_games*100:.1f}%)")
    print(f"  - As First:     {enhanced_first_wins}")
    print(f"  - As Second:    {enhanced_second_wins}")
    print(f"Draws:            {results['draw']:3d} / {num_games} "
          f"({results['draw']/num_games*100:.1f}%)")
    print()
    print("="*80)
    print(" "*30 + "ANALYSIS")
    print("="*80)
    
    # Analysis
    az_winrate = results['alphazero'] / num_games
    if az_winrate >= 0.60:
        print(f"✅ AlphaZero DOMINATES with {az_winrate*100:.1f}% win rate!")
    elif az_winrate >= 0.50:
        print(f"✅ AlphaZero has the edge with {az_winrate*100:.1f}% win rate")
    elif az_winrate >= 0.40:
        print(f"⚠️ Close match! AlphaZero at {az_winrate*100:.1f}%")
    else:
        print(f"❌ Enhanced agent dominates! AlphaZero only {az_winrate*100:.1f}%")
    
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Battle: AlphaZero vs Enhanced Pattern Agent')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained AlphaZero model (.pt file, NOT .pth)')
    parser.add_argument('--games', type=int, default=500,
                       help='Number of games to play (default: 500)')
    parser.add_argument('--size', type=int, default=9,
                       help='Board size (default: 9 for 9×9)')
    parser.add_argument('--connect', type=int, default=4,
                       help='Pieces in a row to win (default: 4)')
    parser.add_argument('--mcts-sims', type=int, default=100,
                       help='MCTS simulations per move for AlphaZero (default: 100)')
    parser.add_argument('--db', type=str, default=None,
                       help='Path to exhaustive database (optional, default: data/n4_full/n4_exhaustive.db)')
    
    args = parser.parse_args()
    
    # Run battle
    run_battle(
        model_path=args.model,
        num_games=args.games,
        board_size=args.size,
        win_length=args.connect,
        mcts_sims=args.mcts_sims,
        db_path=args.db
    )
