import requests
import json

# The URL of the Flask server's endpoint
API_URL = "http://localhost:5001/evaluate-move"

# This is sample data that our RL Agent (Tier 1) would generate.
# It represents a specific game state and the move the player made.
# You can modify this data to test different scenarios.
sample_move_data = {
    "game_state": {
        "collection_fill_ratio": 0.57,  # e.g., 4 out of 7 slots are filled
        "board_tile_count": 80,
        "available_symbols": {
            "bamboo": 15,
            "dragon": 10,
            "flower": 25,
            "wind": 30
        },
        "blocking_relationships": 120, # Number of tiles blocking others
        "deadlock_probability": 0.1 # A pre-calculated risk metric
    },
    "current_move": {
        "tile_id": "T123-bamboo",
        "type": "select_tile",
        "tile_layer": 2,
        "is_potential_match": True # The selected tile could form a match in the collection
    }
}

def test_evaluation_endpoint():
    """
    Sends a sample move to the evaluation server and prints the response.
    """
    print("Sending sample move data to the evaluation server...")
    print(f"Data: {json.dumps(sample_move_data, indent=2)}")
    
    try:
        # The request body must match what the server expects: {"user_input_text": ...}
        response = requests.post(API_URL, json={"user_input_text": sample_move_data})
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Parse the JSON response from the server
        evaluation_result = response.json()
        
        print("\n--- Evaluation Result Received ---")
        print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
        print("----------------------------------\n")
        
    except requests.exceptions.RequestException as e:
        print(f"\n[ERROR] Could not connect to the server at {API_URL}.")
        print("Please ensure the Flask server (app.py) is running in a separate terminal.")
        print(f"Details: {e}")
    except json.JSONDecodeError:
        print("\n[ERROR] Failed to parse the server's response.")
        print(f"Raw response: {response.text}")

if __name__ == "__main__":
    test_evaluation_endpoint() 