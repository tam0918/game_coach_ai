import os
import json
import requests
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import joblib
import pandas as pd
import numpy as np

from predictor import Predictor
from rationale_predictor import RationalePredictor

# --- App Initialization ---
load_dotenv()
app = Flask(__name__)
CORS(app)

# --- FPT API Configuration (for future training) ---
FPT_API_KEY = os.getenv("FPT_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
FPT_API_URL = "https://mkp-api.fptcloud.com/chat/completions"

# --- Scikit-learn Model Predictor Initialization ---
sklearn_predictor = None
try:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    # The model path from train_improved.py
    model_path = os.path.join(_script_dir, 'models_3_classes_manual_balance', 'move_classifier_3_classes_manual_balance.joblib')
    
    print(f"Attempting to load scikit-learn model from: {model_path}")
    if os.path.exists(model_path):
        sklearn_predictor = joblib.load(model_path)
        print("✅ Scikit-learn predictor loaded successfully!")
    else:
        print(f"🔴 WARNING: Scikit-learn model not found at {model_path}. The '/predict_sklearn' endpoint will be unavailable.")
        sklearn_predictor = None

except Exception as e:
    print(f"🔴 WARNING: An unexpected error occurred during scikit-learn model loading: {e}")
    traceback.print_exc()
    sklearn_predictor = None

# --- Local Model Predictor Initialization ---
predictor = None
try:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(_script_dir, 'trainer', 'saved_model_torch')
    
    print(f"Attempting to load model from: {model_dir}")
    predictor = Predictor(model_dir_path=model_dir)
    print("✅ Predictor for local inference loaded successfully!")

except FileNotFoundError:
    print("🔴 WARNING: Local model not found. The '/predict' endpoint will be unavailable.")
    print("Please ensure the model is trained and saved in 'backend/trainer/saved_model_torch/'.")
    predictor = None
except Exception as e:
    print(f"🔴 WARNING: An unexpected error occurred during local model loading: {e}")
    traceback.print_exc()
    predictor = None

# --- Rationale Model Predictor Initialization ---
rationale_predictor = None
try:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    rationale_model_path = os.path.join(_script_dir, 'distilled_model_v2_rationale', 'rationale_predictor.joblib')
    
    print(f"Attempting to load rationale predictor model from: {rationale_model_path}")
    
    # We'll always create a RationalePredictor instance
    # If the model file doesn't exist, it will create a fallback model
    rationale_predictor = RationalePredictor(rationale_model_path)
    
    if hasattr(rationale_predictor, 'using_fallback') and rationale_predictor.using_fallback:
        print("⚠️ Using fallback rationale predictor model. The '/predict_rationale' endpoint will still be available but with limited functionality.")
    else:
        print("✅ Rationale predictor loaded successfully!")

except Exception as e:
    print(f"🔴 WARNING: An unexpected error occurred during rationale model loading: {e}")
    traceback.print_exc()
    print("⚠️ Attempting to create a fallback rationale predictor...")
    
    try:
        # Try to create with a non-existent path to trigger fallback mode
        fallback_path = os.path.join(_script_dir, 'fallback_model.joblib')
        rationale_predictor = RationalePredictor(fallback_path)
        print("✅ Fallback rationale predictor created successfully!")
    except Exception as fallback_err:
        print(f"🔴 CRITICAL: Could not create fallback rationale predictor: {fallback_err}")
        traceback.print_exc()
        rationale_predictor = None


# --- Helper Functions ---
def map_frontend_fields_to_model_fields(frontend_data):
    """
    Maps field names from the frontend format to the format expected by the model.
    Handles both field renaming and default value insertion.
    Also applies necessary numerical encodings for the model.
    """
    field_mapping = {
        # Direct renames
        "combo_count": "move_combo_count",
        "gs_cleared_tiles_count": "gs_cleared_tile_count",
        "gs_remaining_tiles_count": "gs_board_tile_count",
        "gs_accessible_tiles_count": "gs_accessible_tile_count",
        "gs_inaccessible_tiles_count": "gs_inaccessible_tile_count",
        "gs_is_game_over_on_next_move": "gs_is_game_over",
    }
    
    # Create a new dictionary with mapped field names
    model_data = {}
    
    # Copy fields as-is or rename them according to mapping
    for key, value in frontend_data.items():
        model_key = field_mapping.get(key, key)  # Use mapped name if it exists, otherwise use as-is
        model_data[model_key] = value
    
    # Add required default fields if missing
    required_fields_with_defaults = {
        "game_id": f"game-{hash(str(frontend_data))}",
        "gs_collection_fill_ratio": 0.5,  # Default if not provided
        "move_heuristic_score": 50,       # Default score
        "move_tile_layer": 1,             # Default layer
        "move_unblocks_potential": 0,     # Default unblocking potential
    }
    
    for key, default_value in required_fields_with_defaults.items():
        if key not in model_data:
            model_data[key] = default_value
    
    # Convert difficulty to numerical encoding if present as string
    if 'difficulty' in model_data and isinstance(model_data['difficulty'], str):
        difficulty_mapping = {'easy': 1, 'medium': 2, 'hard': 3, 'hell': 4}
        model_data['difficulty_numeric'] = difficulty_mapping.get(model_data['difficulty'].lower(), 2)
    
    # Calculate collection fullness (0-7 scale) if not present
    if 'collection_fullness' not in model_data and 'gs_collection_fill_ratio' in model_data:
        fill_ratio = model_data['gs_collection_fill_ratio']
        model_data['collection_fullness'] = int(round(fill_ratio * 7))
    
    # Calculate win/loss binary feature if outcome info is available
    if 'win_loss' not in model_data and 'move_outcome' in model_data:
        model_data['win_loss'] = 0 if model_data['move_outcome'] == 'immediate_loss' else 1
    
    # Count matching tiles if we have accessible symbols and selected tile
    if 'gs_accessible_symbols' in model_data and 'move_tile_symbol' in model_data:
        try:
            if isinstance(model_data['gs_accessible_symbols'], str):
                accessible_symbols = json.loads(model_data['gs_accessible_symbols'])
            else:
                accessible_symbols = model_data['gs_accessible_symbols']
                
            tile_symbol = model_data['move_tile_symbol']
            model_data['matching_tiles_count'] = accessible_symbols.count(tile_symbol)
        except (json.JSONDecodeError, TypeError, AttributeError):
            model_data['matching_tiles_count'] = 0
    
    # Add necessary emoji symbol fields expected by the model
    # These are created during training from gs_accessible_symbols JSON field
    common_fruit_emojis = ["🍎", "🍐", "🍊", "🍋", "🍌", "🍉", "🍇", "🍓", "🍒", 
                          "🍑", "🥭", "🍍", "🥥", "🥝", "🍅", "🍆", "🥑"]
    
    for emoji in common_fruit_emojis:
        emoji_key = f"sym_count_{emoji}"
        if emoji_key not in model_data:
            model_data[emoji_key] = 0
    
    # Add layer-based features
    for layer in range(4):  # Layers 0-3
        layer_key = f"matching_tiles_layer_{layer}"
        if layer_key not in model_data:
            model_data[layer_key] = 0
            
    # For debugging - we should see the exact data being sent to the model
    print("\n--- Data being sent to model ---")
    for key in sorted(model_data.keys()):
        print(f"{key}: {model_data[key]}")
    print("-------------------------------\n")
    
    return model_data


# --- API Endpoints ---

@app.route('/predict', methods=['POST'])
def predict_move():
    """
    (NEW) Receives game state data, uses the PRE-LOADED LOCAL model to get an
    evaluation, and returns it. This is used by the frontend for real-time analysis.
    """
    if predictor is None:
        return jsonify({"error": "Local model is not loaded. The server cannot make local predictions."}), 503

    try:
        request_data = request.get_json()
        if not request_data:
             return jsonify({"error": "Invalid input. No JSON data received."}), 400
        print("📥 Received prediction request with data:", json.dumps(request_data, indent=2))
    except Exception as e:
        return jsonify({"error": f"Invalid JSON format: {e}"}), 400
    
    try:
        # Map frontend field names to the format expected by the model
        model_ready_data = map_frontend_fields_to_model_fields(request_data)
        
        # Make prediction using the processed data
        prediction_result = predictor.predict(model_ready_data)
        
        # Add additional context to help interpret the prediction
        classification_desc = {
            1: "Brilliant move that significantly improves the position or solves a complex situation",
            2: "Solid, logical move that advances your position",
            3: "Average move, neither particularly good nor bad",
            4: "Suboptimal move, a better option was available",
            5: "Poor move that worsens your position or risks losing the game"
        }
        
        if 'classification_value' in prediction_result:
            class_val = prediction_result['classification_value']
            prediction_result['classification_description'] = classification_desc.get(
                class_val, "No description available"
            )
        
        return jsonify(prediction_result), 200
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal error occurred during prediction.", "details": str(e)}), 500

@app.route('/predict_sklearn', methods=['POST'])
def predict_move_sklearn():
    """
    Receives game state data, uses the pre-loaded scikit-learn model to get an
    evaluation, and returns it as one of three classes (0: Good, 1: Average, 2: Bad).
    """
    if sklearn_predictor is None:
        return jsonify({"error": "Scikit-learn model is not loaded. Cannot make predictions."}), 503

    try:
        request_data = request.get_json()
        if not request_data:
             return jsonify({"error": "Invalid input. No JSON data received."}), 400
        print("📥 Received scikit-learn prediction request:", json.dumps(request_data, indent=2))
    except Exception as e:
        return jsonify({"error": f"Invalid JSON format: {e}"}), 400

    try:
        # We can't be certain about the exact features without the CSV header.
        # As a robust workaround, we'll try to extract features that are commonly
        # available in the frontend and likely used in training.
        
        feature_dict = {
            'move_unblocks_potential': request_data.get('unblocks', 0),
            'move_tile_layer': request_data.get('layer', 0),
            'gs_collection_fill_ratio': request_data.get('collectionFillRatio', 0.0),
            'move_heuristic_score': request_data.get('score', 0),
            'gs_board_tile_count': request_data.get('boardTileCount', 0),
            'gs_accessible_tile_count': request_data.get('accessibleTileCount', 0),
            'difficulty_numeric': {'easy': 1, 'medium': 2, 'hard': 3, 'hell': 4}.get(request_data.get('difficulty', 'medium'), 2),
            'gs_is_game_over': 1 if request_data.get('isGameOver', False) else 0,
            'collection_fullness': int(request_data.get('collectionFillRatio', 0.0) * 7),
            'matching_tiles_in_collection': request_data.get('matchingTilesInCollection', 0),
            'matching_tiles_on_board': request_data.get('matchingTilesOnBoard', 0)
        }
        
        # Create a DataFrame with a single row for the prediction.
        # The order of columns must match the training data.
        # Based on typical feature importance, we'll assume a plausible order.
        # This is the most fragile part of the implementation due to the missing CSV.
        feature_names = [
            'move_heuristic_score', 'move_unblocks_potential', 'move_tile_layer', 
            'gs_collection_fill_ratio', 'gs_board_tile_count', 'gs_accessible_tile_count', 
            'difficulty_numeric', 'gs_is_game_over', 'collection_fullness',
            'matching_tiles_in_collection', 'matching_tiles_on_board'
        ]
        
        # Create a DataFrame with all possible feature names and fill with defaults
        input_df = pd.DataFrame([feature_dict], columns=feature_names)
        
        print("\n--- Data being sent to scikit-learn model ---")
        print(input_df.to_string())
        print("---------------------------------------------\n")
        
        # The pipeline in the model handles scaling
        prediction = sklearn_predictor.predict(input_df)
        prediction_proba = sklearn_predictor.predict_proba(input_df)

        # Map numeric prediction to a meaningful label
        class_mapping = {0: 'Good', 1: 'Average', 2: 'Bad'}
        prediction_label = class_mapping.get(prediction[0], "Unknown")

        response = {
            "prediction": int(prediction[0]),
            "prediction_label": prediction_label,
            "probabilities": prediction_proba[0].tolist()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"An error occurred during scikit-learn prediction: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal error occurred during prediction.", "details": str(e)}), 500

@app.route('/generate_llm_analysis', methods=['POST'])
def generate_llm_analysis():
    """
    (NEW - Distillation Pipeline) Receives a full game state, including all possible moves,
    and calls an external LLM to get a multi-faceted analysis for EACH move.
    This is used to generate a high-quality dataset for training smaller, local models.
    """
    if not FPT_API_KEY or not MODEL_NAME:
        return jsonify({"error": "FPT API key or model name not configured in .env file."}), 500

    try:
        request_data = request.get_json()
        if not request_data or "game_state" not in request_data or "possible_moves" not in request_data:
             return jsonify({"error": "Invalid input. 'game_state' and 'possible_moves' are required."}), 400
        
        # This becomes the context for the LLM
        user_input_context = {
            "game_state": request_data["game_state"],
            "possible_moves": request_data["possible_moves"]
        }
        
    except Exception as e:
        return jsonify({"error": f"Invalid JSON format: {e}"}), 400

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {FPT_API_KEY}"
    }

    system_prompt = """
    You are a world-class master tactician for the tile-matching puzzle game "Triple Tile Match". Your task is to deeply analyze a given game state and evaluate every possible move.

    **Game Rules:**
    1. The goal is to clear all tiles from the board.
    2. Players select accessible (uncovered) tiles from the board to place them into a limited-capacity collection slot (usually 7 slots).
    3. When 3 identical tiles are in the collection slot, they are matched and removed, freeing up space.
    4. The game is lost if the collection slot becomes full before a match is made.

    **Your Task:**
    You will be given a JSON object containing the current `game_state` and a list of `possible_moves`.
    For EACH move in the `possible_moves` list, you must provide a detailed analysis.
    You MUST respond ONLY with a single, minified JSON object containing a key "evaluations", which is a list of JSON objects. Each object in the list must have the following structure:
    {
      "move_id": "<The id of the tile for the move>",
      "score": <A numerical score from 1.0 (terrible) to 10.0 (brilliant)>,
      "category": "<One of the following string labels: 'Brilliant', 'Good', 'Strategic', 'Inaccuracy', 'Mistake', 'Blunder'>",
      "rationale": "<A concise (under 25 words) tactical explanation of the move's opportunities and risks.>"
    }

    **Example Rationale:**
    - "Completes a critical triple, freeing up 3 slots and preventing a loss."
    - "Sets up a future match, but adds a tile that is not immediately useful."
    - "Unblocks 3 other tiles, opening up more strategic options."
    - "Wastes a slot with a tile that has no other matches available on the board."

    Analyze the entire situation holistically before providing your evaluations for each move.
    """

    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_input_context, indent=2)}
        ],
        "stream": False,
        "temperature": 0.6, # Add some creativity for rationales
        "top_p": 0.9
    }

    try:
        print("Submitting request to FPT API for analysis...")
        response = requests.post(FPT_API_URL, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        
        api_response = response.json()
        assistant_message_content = api_response['choices'][0]['message']['content']

        # Clean up potential markdown code blocks
        if assistant_message_content.strip().startswith('```json'):
            assistant_message_content = assistant_message_content.strip()[7:-4].strip()
        
        # Attempt to parse the cleaned content
        parsed_json = json.loads(assistant_message_content)

        # --- Ensure the response conforms to the API contract ---
        # The LLM sometimes returns a raw list instead of the requested JSON object.
        # We handle that case here to make the endpoint more robust.
        if isinstance(parsed_json, list):
            final_response = {"evaluations": parsed_json}
        elif isinstance(parsed_json, dict) and "evaluations" in parsed_json:
            final_response = parsed_json
        else:
            # The response is not in a format we can use.
            raise json.JSONDecodeError("LLM response was valid JSON but not the expected format (list or {'evaluations':...}).", assistant_message_content, 0)

        print("✅ Successfully received and parsed analysis from FPT API.")
        return jsonify(final_response), 200

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} - {response.text}")
        return jsonify({"error": "Failed to get a valid response from the FPT API.", "details": response.text}), 502
    except json.JSONDecodeError as json_err:
        print(f"JSON decode error: {json_err}")
        return jsonify({
            "error": "Failed to parse the LLM's response into valid JSON.",
            "raw_response": assistant_message_content
        }), 500
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred.", "details": str(e)}), 500

@app.route('/predict_rationale', methods=['POST'])
def predict_rationale():
    """
    Receives game state data, uses the rationale predictor model to get a
    score, category and rationale explanation for a move.
    """
    if rationale_predictor is None:
        return jsonify({"error": "Rationale model is not loaded. The server cannot make rationale predictions."}), 503

    try:
        request_data = request.get_json()
        if not request_data:
             return jsonify({"error": "Invalid input. No JSON data received."}), 400
        print("📥 Received rationale prediction request with data:", json.dumps(request_data, indent=2))
    except Exception as e:
        return jsonify({"error": f"Invalid JSON format: {e}"}), 400
    
    try:
        # Map frontend field names to the format expected by the model
        model_ready_data = map_frontend_fields_to_model_fields(request_data)
        
        # Make prediction using the processed data
        prediction_result = rationale_predictor.predict(model_ready_data)
        
        return jsonify(prediction_result), 200
    except Exception as e:
        print(f"An error occurred during rationale prediction: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal error occurred during rationale prediction.", "details": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Checks the status of the server and its models.
    """
    return jsonify({
        "status": "ok",
        "fpt_model_configured": bool(FPT_API_KEY and MODEL_NAME),
        "local_torch_model_loaded": predictor is not None,
        "local_sklearn_model_loaded": sklearn_predictor is not None,
        "rationale_model_loaded": rationale_predictor is not None
    })

if __name__ == '__main__':
    # Make sure to set the port and debug settings as needed
    app.run(host='0.0.0.0', port=5000, debug=True)
