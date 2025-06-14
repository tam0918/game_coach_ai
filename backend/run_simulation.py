import requests
import json
import csv
import os
import argparse
import time
import random
from simulation.game_logic import GameState, generate_board_for_difficulty, DIFFICULTY_LEVELS
from simulation.heuristic_agent import HeuristicAgent

# The URL of the Flask server we are running separately
API_URL = "http://localhost:5001/evaluate-move"
OUTPUT_CSV_FILE = "backend/simulation_results.csv"

# Custom JSON encoder to handle dataclasses
class DataclassJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)

def write_to_csv(data_row: dict):
    """Appends a row of data to the CSV file."""
    file_exists = os.path.isfile(OUTPUT_CSV_FILE)
    
    with open(OUTPUT_CSV_FILE, 'a', newline='', encoding='utf-8') as csvfile:
        # Use a flexible fieldnames list based on the first data written in a session, or a predefined full set
        fieldnames = data_row.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists or os.path.getsize(OUTPUT_CSV_FILE) == 0:
            writer.writeheader()
            
        writer.writerow(data_row)

def play_full_game(difficulty: str, game_id: int):
    """
    Plays a full game from start to finish for a given difficulty,
    using a weighted random choice for exploration, evaluating every move,
    and saving the results.
    """
    print(f"\n{'='*20} STARTING GAME {game_id} (Difficulty: {difficulty.upper()}) {'='*20}")
    
    # Map difficulty to numeric value
    difficulty_map = {'easy': 1, 'medium': 2, 'hard': 3, 'hell': 4}
    difficulty_numeric = difficulty_map[difficulty]
    
    game_state = generate_board_for_difficulty(difficulty)
    agent = HeuristicAgent()
    turn_number = 0

    while True:
        turn_number += 1
        print(f"\n--- Turn {turn_number} ---")

        # 1. Check for game over condition before making a move
        is_over, result = game_state.is_game_over()
        if is_over:
            print(f"Game Over! Result: {result.upper()}")
            break

        # 2. Get all possible moves and their heuristic scores
        possible_moves = agent.calculate_raw_agent_moves(game_state)

        if not possible_moves:
            # This case is also a loss condition, caught by is_game_over, but as a safeguard:
            print("No possible moves left. Game Over!")
            break

        # This list provides crucial context for the LLM
        accessible_symbols_on_board = [move['tile'].symbol for move in possible_moves]

        # 3. Exploration: Choose a move using weighted random selection
        scores = [move['score'] for move in possible_moves]
        # Normalize scores to be non-negative for weighting
        min_score = min(scores)
        weights = [(score - min_score + 1) for score in scores] # +1 to avoid all zero weights
        
        move_to_play = random.choices(possible_moves, weights=weights, k=1)[0]
        
        print(f"Selected move (out of {len(possible_moves)}): {move_to_play['description']} (Score: {move_to_play['score']})")

        # 4. Look ahead: Simulate the chosen move to check for immediate loss
        hypothetical_state = GameState(
            board_tiles=[t for t in game_state.board_tiles], # Shallow copy
            collection_tiles=[t for t in game_state.collection_tiles],
            difficulty_settings=game_state.difficulty_settings
        )
        hypothetical_tile = next(t for t in hypothetical_state.board_tiles if t.id == move_to_play['tile'].id)
        hypothetical_state.apply_move(hypothetical_tile)
        is_immediate_loss, _ = hypothetical_state.is_game_over()
        move_outcome = "immediate_loss" if is_immediate_loss else "unknown"
        
        # Convert outcome to binary value (0: loss, 1: win/ongoing)
        win_loss = 0 if move_outcome == "immediate_loss" else 1
        
        # Calculate collection fullness (0-7 scale)
        collection_fullness = int(round(len(game_state.collection_tiles) / game_state.collection_capacity * 7))
        
        # Calculate the number of matching tiles to selected tile in the accessible list
        matching_tiles_count = accessible_symbols_on_board.count(move_to_play['tile'].symbol)
        
        # Calculate matching tiles by layer
        # We would need access to more detailed board state to fully implement this

        # 5. Evaluate the CHOSEN move with the LLM, providing the crucial new context
        payload = {
            "game_state": {
                "difficulty": difficulty,
                "difficulty_numeric": difficulty_numeric,  # New numeric encoding of difficulty
                "turn_number": turn_number,
                "collection_state_before_move": [t.symbol for t in game_state.collection_tiles],
                "accessible_board_symbols": accessible_symbols_on_board,
                "collection_fill_ratio": len(game_state.collection_tiles) / game_state.collection_capacity,
                "collection_fullness": collection_fullness,  # New 0-7 scale of queue fullness
                "board_tile_count": len([t for t in game_state.board_tiles if not t.is_collected and not t.is_matched]),
                "accessible_tile_count": len(possible_moves),
                "matching_tiles_count": matching_tiles_count,  # New count of matching tiles
            },
            "current_move": {
                "tile_symbol": move_to_play['tile'].symbol,
                "tile_layer": move_to_play['tile'].layer,
                "heuristic_score": move_to_play['score'],
                "is_best_heuristic_move": move_to_play['score'] == scores[0], # Is it the top move?
                "unblocks_potential": move_to_play['unblocks'],
                "move_outcome_if_known": move_outcome,
                "win_loss": win_loss  # New binary encoding of outcome
            }
        }

        try:
            response = requests.post(API_URL, json={"user_input_text": payload})
            response.raise_for_status()
            evaluation = response.json()
            
            # 6. Save the state, the move, AND the evaluation to CSV
            flat_data = {
                'game_id': game_id,
                'turn_number': turn_number,
                'difficulty': difficulty,
                'difficulty_numeric': difficulty_numeric,  # New numeric feature
                'gs_collection_fill_ratio': payload['game_state']['collection_fill_ratio'],
                'collection_fullness': collection_fullness,  # New 0-7 scale feature
                'gs_board_tile_count': payload['game_state']['board_tile_count'],
                'gs_accessible_tile_count': payload['game_state']['accessible_tile_count'],
                'gs_accessible_symbols': json.dumps(payload['game_state']['accessible_board_symbols']), # Store as JSON string
                'matching_tiles_count': matching_tiles_count,  # New count of matching tiles
                'move_tile_symbol': payload['current_move']['tile_symbol'],
                'move_tile_layer': payload['current_move']['tile_layer'],
                'move_heuristic_score': payload['current_move']['heuristic_score'],
                'move_is_best_heuristic': payload['current_move']['is_best_heuristic_move'],
                'move_unblocks_potential': payload['current_move']['unblocks_potential'],
                'move_outcome': move_outcome,
                'win_loss': win_loss,  # New binary outcome
                'llm_classification': evaluation.get('classification'),
                'llm_quality_score': evaluation.get('evaluation', {}).get('quality_score'),
                'llm_strategy_score': evaluation.get('evaluation', {}).get('strategy'),
                'llm_risk_score': evaluation.get('evaluation', {}).get('risk'),
                'llm_efficiency_score': evaluation.get('evaluation', {}).get('efficiency'),
                'llm_combo_potential_score': evaluation.get('evaluation', {}).get('combo_potential')
            }
            write_to_csv(flat_data)
            print(f"LLM classified as '{evaluation.get('classification')}'. Result saved.")
        
        except requests.exceptions.RequestException as e:
            print(f"\n[ERROR] API request failed: {e}. Aborting this game.")
            break # Stop this game and move to the next
            
        # 7. Apply the move to the REAL game state
        game_state.apply_move(move_to_play['tile'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run full, multi-difficulty self-play simulations for Triple Tile.")
    parser.add_argument(
        '--num_games_per_difficulty', 
        type=int, 
        default=5,
        help='Number of full games to play for each difficulty level.'
    )
    args = parser.parse_args()

    difficulties_to_cycle = ['easy', 'medium', 'hard', 'hell']
    total_games_played = 0
    
    for i in range(args.num_games_per_difficulty):
        print(f"\n{'#'*20} STARTING CYCLE {i+1}/{args.num_games_per_difficulty} {'#'*20}")
        for difficulty in difficulties_to_cycle:
            total_games_played += 1
            play_full_game(difficulty, game_id=total_games_played)
            time.sleep(1) # Small delay between games
            
    print(f"\nSimulation finished. Played a total of {total_games_played} games. All data saved in {OUTPUT_CSV_FILE}") 