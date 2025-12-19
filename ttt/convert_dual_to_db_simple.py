import json
import sqlite3
import sys

def convert_dual_to_db(json_file, db_file):
    """Convert n4_dual JSON format to SQLite database."""
    
    # Create database
    print(f"Creating database {db_file}...")
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # Create table with same schema as n4_full
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS states (
            board TEXT PRIMARY KEY,
            best_move INTEGER
        )
    ''')
    
    # Create index for faster lookups
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_board ON states(board)')
    
    # Read entire JSON file
    print(f"Loading JSON from {json_file}...")
    with open(json_file, 'r') as f:
        data_list = json.load(f)
    
    print(f"Found {len(data_list)} states")
    
    # Process in batches
    batch_size = 10000
    states = []
    
    for idx, data in enumerate(data_list):
        # Convert board from list to string
        board_str = ''.join(data['board'])
        
        # Find best move (highest win_prob, then highest draw_prob)
        best_move = None
        best_score = (-1, -1, 1)  # (win_prob, draw_prob, loss_prob)
        
        for move_str, stats in data['per_move'].items():
            score = (stats['win_prob'], stats['draw_prob'], -stats['loss_prob'])
            if score > best_score:
                best_score = score
                best_move = int(move_str)
        
        states.append((board_str, best_move))
        
        # Insert batch
        if len(states) >= batch_size:
            cursor.executemany(
                'INSERT INTO states (board, best_move) VALUES (?, ?)',
                states
            )
            conn.commit()
            print(f"Processed {idx + 1}/{len(data_list)} states...")
            states = []
    
    # Insert remaining states
    if states:
        cursor.executemany(
            'INSERT INTO states (board, best_move) VALUES (?, ?)',
            states
        )
        conn.commit()
        print(f"Processed {len(data_list)}/{len(data_list)} states")
    
    # Get database stats
    cursor.execute('SELECT COUNT(*) FROM states')
    count = cursor.fetchone()[0]
    
    conn.close()
    
    print(f"\nConversion complete!")
    print(f"Database: {db_file}")
    print(f"Total states: {count:,}")

if __name__ == '__main__':
    json_file = 'data/n4_dual/n4_exhaustive_probs.json'
    db_file = 'data/n4_dual/n4_exhaustive_dual.db'
    
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    if len(sys.argv) > 2:
        db_file = sys.argv[2]
    
    convert_dual_to_db(json_file, db_file)
