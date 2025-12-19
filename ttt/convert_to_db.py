"""
Convert exhaustive JSON to SQLite database for efficient lookup
"""

import json
import sqlite3
import time


def create_db(json_file: str, db_file: str):
    """
    Convert JSON to SQLite database
    
    Args:
        json_file: Path to exhaustive JSON file
        db_file: Path to output SQLite database
    """
    print(f"Creating database {db_file} from {json_file}...")
    
    # Create database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # Create table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS states (
            board TEXT NOT NULL,
            turn TEXT NOT NULL,
            per_move TEXT,
            best_move INTEGER,
            probability_map TEXT,
            PRIMARY KEY (board, turn)
        )
    ''')
    
    # Create index for faster lookup
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_board_turn ON states(board, turn)')
    
    print("Reading JSON file...")
    start = time.time()
    
    count = 0
    batch = []
    batch_size = 10000
    
    with open(json_file, 'r') as f:
        # Skip opening bracket
        line = f.readline()
        if line.strip() != '[':
            raise ValueError("Expected JSON array")
        
        for line in f:
            line = line.strip()
            if line in [']', '']:
                continue
            
            # Remove trailing comma
            if line.endswith(','):
                line = line[:-1]
            
            state = json.loads(line)
            
            batch.append((
                state['board'],
                state['turn'],
                json.dumps(state['per_move']),
                state['best_move'],
                json.dumps(state['probability_map'])
            ))
            
            count += 1
            
            if len(batch) >= batch_size:
                cursor.executemany(
                    'INSERT OR REPLACE INTO states VALUES (?, ?, ?, ?, ?)',
                    batch
                )
                batch = []
                
                if count % 1000000 == 0:
                    conn.commit()
                    elapsed = time.time() - start
                    rate = count / elapsed
                    print(f"  Processed {count:,} states ({rate:.0f} states/s)")
        
        # Insert remaining batch
        if batch:
            cursor.executemany(
                'INSERT OR REPLACE INTO states VALUES (?, ?, ?, ?, ?)',
                batch
            )
    
    conn.commit()
    elapsed = time.time() - start
    
    print(f"\nDatabase created successfully!")
    print(f"Total states: {count:,}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Rate: {count/elapsed:.0f} states/s")
    
    # Get database size
    cursor.execute("SELECT COUNT(*) FROM states")
    db_count = cursor.fetchone()[0]
    print(f"Database contains: {db_count:,} states")
    
    conn.close()


if __name__ == '__main__':
    create_db(
        'data/n4_full/n4_exhaustive_probs_full.json',
        'data/n4_full/n4_exhaustive.db'
    )
