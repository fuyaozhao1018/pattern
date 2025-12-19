#!/usr/bin/env python3
"""
Usage:
  python -m ttt.extract_n9_losses \
    --input out/runs/n9_battle_lambda0.3_500games.json \
    --output out/runs/n9_lambda0.3_losses.json
"""
import argparse
import json
import os


def extract_losses_from_battle_results(input_file: str, output_file: str) -> None:
    """Extract loss games from battle results and save to separate file."""
    
    # Load the battle results
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Extract losses
    all_loss_games = []
    lambda_draw = None
    
    for result in data.get('results', []):
        lambda_draw = result.get('lambda_draw', 0.0)
        
        # Extract from vs_random
        if 'vs_random' in result and 'loss_games' in result['vs_random']:
            loss_games = result['vs_random']['loss_games']
            for game in loss_games:
                game['opponent'] = 'random'
                game['lambda_draw'] = lambda_draw
            all_loss_games.extend(loss_games)
        
        # Extract from vs_smart
        if 'vs_smart' in result and 'loss_games' in result['vs_smart']:
            loss_games = result['vs_smart']['loss_games']
            for game in loss_games:
                game['opponent'] = 'smart'
                game['lambda_draw'] = lambda_draw
            all_loss_games.extend(loss_games)
        
        # Extract from vs_minimax
        if 'vs_minimax' in result and 'loss_games' in result['vs_minimax']:
            loss_games = result['vs_minimax']['loss_games']
            for game in loss_games:
                game['opponent'] = 'minimax'
                game['lambda_draw'] = lambda_draw
            all_loss_games.extend(loss_games)
    
    # Save to output file
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump({
            'lambda_draw': lambda_draw,
            'total_losses': len(all_loss_games),
            'loss_games': all_loss_games
        }, f, indent=2)
    
    print(f"Extracted {len(all_loss_games)} loss games from {input_file}")
    print(f"Saved to: {output_file}")
    
    # Print summary
    opponents = {}
    for game in all_loss_games:
        opp = game.get('opponent', 'unknown')
        opponents[opp] = opponents.get(opp, 0) + 1
    
    print("\nLoss breakdown by opponent:")
    for opp, count in sorted(opponents.items()):
        print(f"  vs {opp}: {count} losses")


def main():
    parser = argparse.ArgumentParser(description="Extract loss games from battle results.")
    parser.add_argument('--input', required=True, help='Input battle results JSON file')
    parser.add_argument('--output', required=True, help='Output losses JSON file')
    args = parser.parse_args()
    
    extract_losses_from_battle_results(args.input, args.output)


if __name__ == '__main__':
    main()
