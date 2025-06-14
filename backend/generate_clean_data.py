import os
import json
import csv
import time
import random
import requests
from dotenv import load_dotenv
from simulation.game_logic import GameState, generate_board_for_difficulty, DIFFICULTY_LEVELS
from simulation.heuristic_agent import HeuristicAgent

# Load environment variables
load_dotenv()

# Configuration
API_URL = "https://mkp-api.fptcloud.com/chat/completions"
FPT_API_KEY = os.getenv("FPT_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "meta/llama3-70b-instruct")
OUTPUT_CSV_FILE = "backend/clean_training_data.csv"
NUM_GAMES_PER_DIFFICULTY = 5
TOTAL_GAMES = 20  # 5 games per difficulty level

def write_to_csv(data_rows: list, file_path: str):
    """Appends rows of data to the CSV file."""
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
        if not data_rows:
            return
            
        fieldnames = data_rows[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists or os.path.getsize(file_path) == 0:
            writer.writeheader()
            
        for row in data_rows:
            writer.writerow(row)

def evaluate_move_with_llm(state_data):
    """
    Sends the game state to the LLM API and gets a numerical evaluation.
    
    Args:
        state_data: Game state data to evaluate
        
    Returns:
        Dictionary with numerical ratings (1-5)
    """
    if not FPT_API_KEY:
        raise ValueError("FPT API key not configured in .env file")
    
    # System prompt to instruct the LLM to provide numerical ratings
    system_prompt = """
    You are an expert AI analyst for the puzzle game Triple Tile. Your task is to evaluate a player's move based on the provided game state.
    
    IMPORTANT: Be EXTREMELY CRITICAL and HARSH in your evaluations. Most moves should be considered average or worse.
    
    Analyze the move according to the following PRIORITY ORDER (from most important to least important):
    1. Win/loss outcome - Does the move lead to immediate win or loss? (HIGHEST PRIORITY)
       - Moves leading to immediate loss must be rated 5 (worst)
       - Moves leading to immediate win should be rated 1 or 2
    
    2. Number of matching tiles in collection queue - Does the move create a triple or advance toward one?
       - Creating a triple is generally good (rate 1-2)
       - Adding to an existing pair is generally good (rate 2-3)
       - Starting a new pair when collection is nearly full is risky (rate 3-4)
    
    3. Collection queue fullness - How full is the queue and how does this move affect it?
       - Making moves that reduce a full queue is good (rate 1-2)
       - Making moves that fill the last slots in a queue is risky (rate 3-5)
    
    4. Number of matching tiles in layer 1 - Are there accessible matching tiles on layer 1?
       - Ignoring matching tiles in layer 1 is usually bad (rate 3-5)
       - Prioritizing matching tiles in layer 1 is usually good (rate 1-3)
    
    5. Number of matching tiles in layer 2 - Are there accessible matching tiles on layer 2?
       - Consider after evaluating layer 1 tiles
    
    6. Number of matching tiles in layer 3 - Are there accessible matching tiles on layer 3?
       - Consider after evaluating layer 1 and 2 tiles
    
    7. Number of tiles unlocked - How many tiles does this move unlock for future plays?
       - Moves unlocking many tiles are generally better than those unlocking few or none
       - Moves that unlock 0 tiles when alternatives unlock more should be rated 3-5
    
    Based on your analysis, provide a final classification for the move USING ONLY A NUMBER from 1-5:
    - 1: Genius/Excellent - ONLY for truly exceptional moves that significantly improve position (less than 5% of moves)
    - 2: Good - A solid move that advances position (about 15% of moves)
    - 3: Average - Neither particularly good nor bad (about 40% of moves)
    - 4: Inaccuracy - Suboptimal move, better options were available (about 25% of moves)
    - 5: Mistake/Blunder/Stupid - Poor move that worsens position or risks losing (about 15% of moves)
    
    IMPORTANT DISTRIBUTION CORRECTION:
    You are currently rating too many moves as 4 and 5. Unless a move is clearly bad or leads to immediate loss,
    consider rating it as 3 (Average). Most moves should be rated 3, with fewer rated as 2 or 4, and even fewer as 1 or 5.
    
    CRITICAL EVALUATION GUIDELINES:
    - If the move leads to an immediate loss (win_loss = 0), it MUST be rated 5 (Mistake/Blunder)
    - If the move has better alternatives but doesn't severely hurt position, rate it 3 (Average) not 4
    - Only rate moves as 4 if they are clearly suboptimal with obviously better alternatives
    - If the move is ordinary and doesn't stand out as particularly good or bad, rate it 3 (Average)
    - Reserve rating 1 for truly exceptional moves only
    - Be extremely skeptical of moves that don't advance the game state
    
    You MUST respond ONLY with a single, minified JSON object. Do not include any text, explanations, or markdown formatting.
    The JSON object must have the following structure:
    {
        "classification": <number from 1-5>
    }
    """

    # Prepare the API request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {FPT_API_KEY}"
    }
    
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(state_data)}
        ],
        "stream": False
    }
    
    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        
        api_response = response.json()
        assistant_message_content = api_response['choices'][0]['message']['content']
        
        # Clean up potential markdown
        if assistant_message_content.strip().startswith('```json'):
            assistant_message_content = assistant_message_content.strip()[7:-4].strip()
        
        parsed_json = json.loads(assistant_message_content)
        
        # Ensure classification is a number
        if isinstance(parsed_json.get("classification"), str):
            try:
                parsed_json["classification"] = int(parsed_json["classification"].strip())
            except (ValueError, TypeError):
                # Default to 3 (Average) if conversion fails
                parsed_json["classification"] = 3
        
        return parsed_json
    
    except Exception as e:
        print(f"API request failed: {e}")
        # Return default values if API request fails
        return {
            "classification": 3
        }

def adjust_classification_distribution(all_data_rows):
    """
    Adjusts the classification distribution to match the desired percentages:
    - 1: 5%
    - 2: 15%
    - 3: 40%
    - 4: 25%
    - 5: 15%
    
    This function only adjusts classifications that are not for immediate loss moves (win_loss = 0),
    as those must remain as 5.
    """
    if not all_data_rows:
        return all_data_rows
    
    # Count current distribution
    counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    loss_indices = []  # Indices of moves that lead to immediate loss
    
    for i, row in enumerate(all_data_rows):
        classification = row["llm_classification_value"]
        counts[classification] += 1
        
        # Track moves that lead to immediate loss
        if row["win_loss"] == 0:
            loss_indices.append(i)
    
    total_moves = len(all_data_rows)
    
    # Calculate target counts
    target_counts = {
        1: int(total_moves * 0.05),  # 5%
        2: int(total_moves * 0.15),  # 15%
        3: int(total_moves * 0.40),  # 40%
        4: int(total_moves * 0.25),  # 25%
        5: int(total_moves * 0.15)   # 15%
    }
    
    # Ensure we have at least one of each classification if possible
    for cls in range(1, 6):
        if target_counts[cls] == 0 and total_moves >= 5:
            target_counts[cls] = 1
    
    # Adjust to ensure sum equals total_moves
    adjustment = total_moves - sum(target_counts.values())
    target_counts[3] += adjustment  # Add any rounding adjustment to class 3 (Average)
    
    print(f"Current distribution: {counts}")
    print(f"Target distribution: {target_counts}")
    
    # If we're already close to target distribution, don't adjust
    if all(abs(counts[cls] - target_counts[cls]) <= max(2, int(total_moves * 0.05)) for cls in range(1, 6)):
        print("Current distribution is already close to target. No adjustment needed.")
        return all_data_rows
    
    # Sort moves by their classification value (except for immediate loss moves)
    non_loss_indices = [i for i in range(total_moves) if i not in loss_indices]
    non_loss_indices.sort(key=lambda i: (
        all_data_rows[i]["llm_classification_value"],  # Sort by classification
        -all_data_rows[i]["win_loss"],                # Prefer ongoing (0.5) over win (1)
        -all_data_rows[i]["unblocks_count"],          # Prefer fewer unblocks
        all_data_rows[i]["tile_layer"]                # Prefer higher layers
    ))
    
    # Calculate how many moves of each classification we need to adjust
    adjustments = {}
    for cls in range(1, 6):
        # Exclude immediate loss moves from class 5 target
        if cls == 5:
            loss_count = len(loss_indices)
            adjustments[cls] = target_counts[cls] - loss_count
            if adjustments[cls] < 0:
                adjustments[cls] = 0  # Can't reduce below the number of loss moves
        else:
            adjustments[cls] = target_counts[cls] - counts[cls]
    
    print(f"Adjustments needed: {adjustments}")
    
    # Apply adjustments
    new_classifications = {}
    remaining_indices = set(non_loss_indices)
    
    # First, assign classifications that need to be increased
    for cls in range(1, 6):
        if adjustments[cls] > 0:
            # For class 1 (Genius), select moves with highest unblocks and lowest layer
            if cls == 1:
                candidates = sorted(list(remaining_indices), 
                                   key=lambda i: (-all_data_rows[i]["unblocks_count"], 
                                                 all_data_rows[i]["tile_layer"]))
            # For class 2 (Good), select moves with good unblocks
            elif cls == 2:
                candidates = sorted(list(remaining_indices), 
                                   key=lambda i: (-all_data_rows[i]["unblocks_count"], 
                                                 all_data_rows[i]["tile_layer"]))
            # For class 3 (Average), take from the middle
            elif cls == 3:
                candidates = list(remaining_indices)
                random.shuffle(candidates)
            # For class 4 (Inaccuracy), select moves with lower unblocks
            elif cls == 4:
                candidates = sorted(list(remaining_indices), 
                                   key=lambda i: (all_data_rows[i]["unblocks_count"], 
                                                -all_data_rows[i]["tile_layer"]))
            # For class 5 (Mistake), select the worst moves
            else:
                candidates = sorted(list(remaining_indices), 
                                   key=lambda i: (all_data_rows[i]["unblocks_count"], 
                                                -all_data_rows[i]["tile_layer"]))
            
            # Assign classification to the needed number of moves
            for i in range(min(adjustments[cls], len(candidates))):
                new_classifications[candidates[i]] = cls
                remaining_indices.remove(candidates[i])
    
    # Apply the new classifications
    for i, row in enumerate(all_data_rows):
        if i in new_classifications:
            row["llm_classification_value"] = new_classifications[i]
    
    # Count new distribution
    new_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for row in all_data_rows:
        new_counts[row["llm_classification_value"]] += 1
    
    print(f"New distribution after adjustment: {new_counts}")
    
    return all_data_rows

def evaluate_all_possible_moves(game_state, possible_moves, difficulty_numeric, collection_fullness, turn_number):
    """
    Evaluates all possible moves for the current game state.
    
    Returns:
        List of data rows for all possible moves
    """
    all_data_rows = []
    
    # Skip evaluating all moves on the first turn to avoid too many API calls
    if turn_number <= 1:
        return all_data_rows
    
    # Get accessible symbols
    accessible_symbols = [move['tile'].symbol for move in possible_moves]
    
    # Calculate move scores for comparison
    move_scores = [move['score'] for move in possible_moves]
    max_score = max(move_scores) if move_scores else 0
    
    print(f"Evaluating all {len(possible_moves)} possible moves...")
    
    for i, move in enumerate(possible_moves):
        # Check for immediate loss
        hypothetical_state = GameState(
            board_tiles=[t for t in game_state.board_tiles],
            collection_tiles=[t for t in game_state.collection_tiles],
            difficulty_settings=game_state.difficulty_settings
        )
        hypothetical_tile = next(t for t in hypothetical_state.board_tiles if t.id == move['tile'].id)
        hypothetical_state.apply_move(hypothetical_tile)
        is_game_over, result = hypothetical_state.is_game_over()
        
        # Create numerical features
        # win_loss: 1 = win, 0 = loss, 0.5 = ongoing
        if is_game_over:
            if result == "win":
                win_loss = 1
            else:  # loss
                win_loss = 0
        else:
            win_loss = 0.5  # Game is ongoing
            
        matching_tiles_count = accessible_symbols.count(move['tile'].symbol)
        
        # Count matching tiles by layer (0, 1, 2, 3)
        matching_tiles_layer = [0, 0, 0, 0]
        for other_move in possible_moves:
            if other_move['tile'].symbol == move['tile'].symbol:
                layer = min(other_move['tile'].layer, 3)  # Cap at 3 for layers higher than 3
                matching_tiles_layer[layer] += 1
        
        # Calculate how many better moves exist (based on heuristic score)
        better_moves_count = sum(1 for score in move_scores if score > move['score'])
        score_percentile = (move['score'] / max_score) * 100 if max_score > 0 else 50
        
        # Find alternative moves that unblock more tiles
        better_unblock_moves = [m for m in possible_moves if m['unblocks'] > move['unblocks']]
        
        # Prepare data for LLM
        state_data = {
            "game_state": {
                "difficulty_numeric": difficulty_numeric,
                "turn_number": turn_number,
                "collection_fullness": collection_fullness,
                "board_tile_count": len([t for t in game_state.board_tiles if not t.is_collected and not t.is_matched]),
                "accessible_tile_count": len(possible_moves),
                "matching_tiles_count": matching_tiles_count,
                "matching_tiles_by_layer": matching_tiles_layer,
                "total_possible_moves": len(possible_moves)
            },
            "current_move": {
                "tile_symbol": move['tile'].symbol,
                "tile_layer": move['tile'].layer,
                "unblocks_count": move['unblocks'],
                "win_loss": win_loss,
                "description": move['description'],
                "better_moves_exist": better_moves_count > 0,
                "better_moves_count": better_moves_count,
                "score_percentile": score_percentile,
                "has_better_unblock_alternatives": len(better_unblock_moves) > 0,
                "better_unblock_moves_count": len(better_unblock_moves)
            }
        }
        
        # Get LLM evaluation with numerical scores
        try:
            print(f"  [{i+1}/{len(possible_moves)}] Evaluating move: {move['description']}")
            evaluation = evaluate_move_with_llm(state_data)
            
            # Force rating 5 for moves that lead to immediate loss
            if win_loss == 0:
                evaluation["classification"] = 5
                print(f"    Forced classification to 5 (move leads to loss)")
            
            # Prepare clean data row with only necessary numerical features
            clean_data = {
                # Numerical features
                "difficulty_numeric": difficulty_numeric,
                "collection_fullness": collection_fullness,
                "matching_tiles_count": matching_tiles_count,
                "matching_tiles_layer_0": matching_tiles_layer[0],
                "matching_tiles_layer_1": matching_tiles_layer[1],
                "matching_tiles_layer_2": matching_tiles_layer[2],
                "matching_tiles_layer_3": matching_tiles_layer[3],
                "tile_layer": move['tile'].layer,
                "unblocks_count": move['unblocks'],
                "win_loss": win_loss,
                
                # LLM evaluation (only classification value)
                "llm_classification_value": evaluation.get("classification", 3)
            }
            
            all_data_rows.append(clean_data)
            print(f"    Classification: {evaluation.get('classification')}")
            
        except Exception as e:
            print(f"    Error evaluating move: {e}")
            continue
    
    # Adjust the distribution of classifications
    all_data_rows = adjust_classification_distribution(all_data_rows)
            
    return all_data_rows

def play_full_game(difficulty: str, game_id: int, agent_mode: str):
    """
    Plays a full game with the specified difficulty and collects clean data.
    
    Args:
        difficulty: Difficulty level ('easy', 'medium', 'hard', 'hell')
        game_id: Game identifier
        agent_mode: 'smart' or 'explore'
    """
    print(f"\n{'='*20} STARTING GAME {game_id} (Difficulty: {difficulty.upper()}, Mode: {agent_mode.upper()}) {'='*20}")
    
    # Map difficulty to numeric value
    difficulty_map = {'easy': 1, 'medium': 2, 'hard': 3, 'hell': 4}
    difficulty_numeric = difficulty_map[difficulty]
    
    # Generate game state
    game_state = generate_board_for_difficulty(difficulty)
    agent = HeuristicAgent()
    turn_number = 0
    
    # Play until game over
    while True:
        turn_number += 1
        print(f"Turn {turn_number}")
        
        # Check for game over
        is_over, result = game_state.is_game_over()
        if is_over:
            print(f"Game Over! Result: {result.upper()}")
            break
            
        # Get possible moves
        possible_moves = agent.calculate_raw_agent_moves(game_state)
        if not possible_moves:
            print("No possible moves left. Game Over!")
            break
            
        # Get accessible symbols
        accessible_symbols = [move['tile'].symbol for move in possible_moves]
        
        # Collection fullness as a scale from 0-7
        collection_fullness = int(round(len(game_state.collection_tiles) / game_state.collection_capacity * 7))
        
        # Evaluate ALL possible moves for both agent modes
        all_data_rows = evaluate_all_possible_moves(
            game_state, 
            possible_moves, 
            difficulty_numeric, 
            collection_fullness,
            turn_number
        )
        
        # Save all evaluations to CSV immediately
        if all_data_rows:
            write_to_csv(all_data_rows, OUTPUT_CSV_FILE)
            print(f"Saved {len(all_data_rows)} move evaluations to CSV")
        
        # Different move selection based on agent mode
        if agent_mode == 'smart':
            # Smart agent: Choose the move with the highest score
            move_to_play = max(possible_moves, key=lambda x: x['score'])
            print(f"Smart agent selected move: {move_to_play['description']}")
        else:  # Explore mode
            # Explore agent: Choose a weighted random move
            scores = [move['score'] for move in possible_moves]
            min_score = min(scores)
            weights = [(score - min_score + 1) for score in scores]
            move_to_play = random.choices(possible_moves, weights=weights, k=1)[0]
            print(f"Explore agent selected move: {move_to_play['description']}")
        
        # Apply the selected move to the real game state
        game_state.apply_move(move_to_play['tile'])
        time.sleep(1)  # Small delay to avoid overwhelming API

def main():
    """Main function to orchestrate data generation."""
    print(f"Starting clean data generation. Output will be saved to {OUTPUT_CSV_FILE}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_CSV_FILE), exist_ok=True)
    
    # Cycle through difficulties and agent modes
    difficulties = ['easy', 'medium', 'hard', 'hell']
    game_id = 0
    
    # Play 20 games alternating between difficulties and agent modes
    for i in range(TOTAL_GAMES):
        game_id += 1
        difficulty = difficulties[i % 4]  # Cycle through difficulties
        agent_mode = 'smart' if i % 2 == 0 else 'explore'  # Alternate between smart and explore modes
        
        play_full_game(difficulty, game_id, agent_mode)
        time.sleep(2)  # Pause between games
    
    print(f"\nData generation complete. Generated data for {game_id} games.")

if __name__ == "__main__":
    main() 