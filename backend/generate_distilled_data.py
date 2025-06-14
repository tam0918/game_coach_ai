import os
import requests
import json
import pandas as pd
import random
import time
from tqdm import tqdm

# --- Configuration ---
API_ENDPOINT = "http://127.0.0.1:5000/generate_llm_analysis"
OUTPUT_DIR = "backend/training_data"
OUTPUT_CSV_PATH = os.path.join(OUTPUT_DIR, "llm_distilled_training_data.csv")
NUM_GAME_STATES_TO_GENERATE = 1000 # Start with a small number for testing
SYMBOLS = ["🍎", "🍐", "🍊", "🍋", "🍌", "🍉", "🍇", "🍓", "🍒", "🍑", "🥭", "🍍"]

def generate_random_game_state():
    """
    Creates a randomized, plausible game state to be sent to the LLM for analysis.
    This simulates various scenarios a player might encounter.
    """
    # --- Simulate Game Settings ---
    collection_slot_capacity = 7
    
    # --- Simulate Collection Slot ---
    num_items_in_slot = random.randint(0, collection_slot_capacity - 2)
    collection_slot = [
        {"id": f"slot-tile-{i}", "symbol": random.choice(SYMBOLS)}
        for i in range(num_items_in_slot)
    ]
    
    # --- Simulate Board Tiles ---
    num_board_tiles = random.randint(30, 80)
    board_tiles = [
        {
            "id": f"board-tile-{i}",
            "symbol": random.choice(SYMBOLS),
            "layer": random.randint(0, 4)
        }
        for i in range(num_board_tiles)
    ]
    
    # --- Simulate Possible Moves ---
    num_possible_moves = random.randint(3, 8)
    # Ensure possible moves are a subset of board tiles
    possible_moves = random.sample(board_tiles, min(num_possible_moves, len(board_tiles)))

    # --- Construct Final State Object ---
    game_state_representation = {
        "game_state": {
            "collection_slot_contents": [tile['symbol'] for tile in collection_slot],
            "collection_slot_capacity": collection_slot_capacity,
            "board_tile_count": len(board_tiles),
            "accessible_tile_count": len(possible_moves),
            "board_tiles_by_symbol": {s: sum(1 for t in board_tiles if t['symbol'] == s) for s in SYMBOLS}
        },
        "possible_moves": possible_moves
    }
    
    return game_state_representation

def main():
    """
    Main function to generate data, call the LLM API, and save the results.
    """
    print("--- Starting High-Quality Data Generation using LLM Distillation ---")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_evaluations = []
    
    print(f"Generating {NUM_GAME_STATES_TO_GENERATE} game states for analysis...")
    
    for _ in tqdm(range(NUM_GAME_STATES_TO_GENERATE), desc="Analyzing Game States"):
        try:
            # 1. Generate a random game scenario
            payload = generate_random_game_state()
            
            # Create a lookup for move details
            moves_lookup = {move['id']: move for move in payload['possible_moves']}

            # 2. Call the LLM analysis endpoint
            response = requests.post(API_ENDPOINT, json=payload, timeout=180) # 3-minute timeout for LLM
            
            if response.status_code == 200:
                results = response.json()
                if "evaluations" in results and isinstance(results["evaluations"], list):
                    # 3. Add game state AND move-specific context to each evaluation
                    for eval_data in results["evaluations"]:
                        move_id = eval_data.get("move_id")
                        move_details = moves_lookup.get(move_id)

                        # --- Data Enrichment Step ---
                        if move_details:
                            eval_data['move_symbol'] = move_details['symbol']
                            eval_data['move_layer'] = move_details['layer']
                        else:
                            eval_data['move_symbol'] = "UNKNOWN"
                            eval_data['move_layer'] = -1

                        # Flatten the game state into the record
                        eval_data.update(payload["game_state"])
                        all_evaluations.append(eval_data)
                else:
                    print(f"WARNING: API response for a state was malformed. Response: {results}")

            else:
                print(f"ERROR: Received status code {response.status_code} from API.")
                print(f"Response Body: {response.text}")
            
            # Respectful delay to not overwhelm the local server or external APIs
            time.sleep(1)

        except requests.exceptions.RequestException as e:
            print(f"ERROR: An exception occurred while calling the API: {e}")
            continue # Skip this state and move to the next
        except Exception as e:
            print(f"An unexpected error occurred in the loop: {e}")
            continue

    if not all_evaluations:
        print("\n--- No evaluations were collected. Halting. ---")
        print("Please ensure the backend server is running and the /generate_llm_analysis endpoint is working correctly.")
        return

    # 4. Convert to a DataFrame and save
    df = pd.DataFrame(all_evaluations)
    
    # Reorder columns for better readability
    preferred_order = [
        'move_id', 'move_symbol', 'move_layer', 'score', 'category', 'rationale', 
        'collection_slot_contents', 'collection_slot_capacity',
        'board_tile_count', 'accessible_tile_count'
    ]
    # Add the symbol counts to the end
    other_cols = [col for col in df.columns if col not in preferred_order]
    df = df[preferred_order + other_cols]

    df.to_csv(OUTPUT_CSV_PATH, index=False)
    
    print("\n--- Data Generation Complete ---")
    print(f"✅ Successfully generated and saved {len(df)} move evaluations.")
    print(f"Data saved to: {OUTPUT_CSV_PATH}")
    print("\n--- Sample of Generated Data ---")
    print(df.head().to_string())

if __name__ == "__main__":
    main() 