import pickle
import torch
import pandas as pd
import numpy as np
import os
import sys
from os.path import dirname, abspath

# Add the trainer directory to the Python path to find data_processor.py
trainer_dir = os.path.join(dirname(abspath(__file__)), 'trainer')
sys.path.append(trainer_dir)

# Import needed classes 
from trainer.model import MultiOutputModel
from trainer.data_processor import DataProcessor

# Create an alias for the data_processor module to handle unpickling
# This allows pickle to find the module even if it was saved with a different path
sys.modules['data_processor'] = sys.modules['trainer.data_processor']

class Predictor:
    """
    A class to load a trained PyTorch model and its corresponding DataProcessor
    to make predictions on new, unseen data.
    """
    def __init__(self, model_dir_path: str):
        """
        Initializes the Predictor by loading the model and the data processor.

        Args:
            model_dir_path (str): Path to the directory containing the model
                                  and data_processor.pkl.
        """
        processor_path = os.path.join(model_dir_path, 'data_processor.pkl')
        model_path = os.path.join(model_dir_path, 'triple_tile_model.pth')

        if not os.path.exists(processor_path) or not os.path.exists(model_path):
            raise FileNotFoundError("Model or data processor not found in the specified directory.")

        # 1. Load the data processor
        print("Loading data processor...")
        with open(processor_path, 'rb') as f:
            self.processor: DataProcessor = pickle.load(f)
        print("Data processor loaded.")

        # 2. Determine model parameters from the processor
        # These are needed to reconstruct the model architecture correctly
        self.input_shape = len(self.processor.columns_after_fit)
        self.num_classes = len(self.processor.encoded_class_names)
        
        # 3. Load the PyTorch model
        print("Loading PyTorch model...")
        self.model = MultiOutputModel(self.input_shape, self.num_classes)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  # IMPORTANT: Set the model to evaluation mode
        print("PyTorch model loaded and in evaluation mode.")
        
        # 4. Store the expected column names for debugging
        self.expected_columns = self.processor.columns_after_fit
        print(f"Model expects {len(self.expected_columns)} feature columns")
        # Print first 10 and last 10 expected columns for debugging
        print("Sample expected columns: ", 
              self.expected_columns[:5], "... and ...", 
              self.expected_columns[-5:] if len(self.expected_columns) > 5 else [])
        
        # 5. Create mapping from numerical labels back to human-readable labels
        self.class_map_inverse = {
            1: "Genius/Excellent",
            2: "Good",
            3: "Average",
            4: "Inaccuracy",
            5: "Blunder/Stupid"
        }

    def predict(self, single_move_data: dict) -> dict:
        """
        Makes a prediction on a single data sample.

        Args:
            single_move_data (dict): A dictionary representing a single row of data,
                                     matching the structure of the original CSV.

        Returns:
            A dictionary containing the predicted class and scores.
        """
        try:
            # Process difficulty as numeric if it's a string
            if 'difficulty' in single_move_data and isinstance(single_move_data['difficulty'], str):
                difficulty_map = {'easy': 1, 'medium': 2, 'hard': 3, 'hell': 4}
                single_move_data['difficulty_numeric'] = difficulty_map.get(single_move_data['difficulty'].lower(), 2)
            
            # Calculate collection fullness if not provided
            if 'collection_fullness' not in single_move_data and 'gs_collection_fill_ratio' in single_move_data:
                fill_ratio = single_move_data['gs_collection_fill_ratio']
                single_move_data['collection_fullness'] = int(round(fill_ratio * 7))
            
            # Calculate win/loss if not provided
            if 'win_loss' not in single_move_data and 'move_outcome' in single_move_data:
                single_move_data['win_loss'] = 0 if single_move_data['move_outcome'] == 'immediate_loss' else 1
            
            # The processor and model expect a DataFrame, so we convert the dict
            input_df = pd.DataFrame([single_move_data])
            
            # Debug output to check column matching
            print(f"\nInput columns: {input_df.columns.tolist()}")
            
            # Create a safer version of _preprocess_features that handles missing columns
            processed_features = self._safe_preprocess_features(input_df)
            
            # Convert the processed data to a PyTorch Tensor
            input_tensor = torch.FloatTensor(processed_features.values)
            
            # Make the prediction
            with torch.no_grad(): # No need to track gradients for inference
                class_logits, reg_output = self.model(input_tensor)
                
                # Post-process the output
                class_probs = torch.softmax(class_logits, dim=1)
                predicted_class_index = torch.argmax(class_probs, dim=1).item()
                
                # Decode the class index back to its numerical value (1-5)
                predicted_class_value = self.processor.label_encoder.inverse_transform([predicted_class_index])[0]
                
                # Convert the numerical value to human-readable label
                predicted_class_name = self.class_map_inverse.get(predicted_class_value, "Average")
                
                predicted_scores_raw = reg_output.numpy().flatten()

                # Normalize scores using a sigmoid function and scale to 0-10
                # This provides a more principled scaling than simple clamping.
                normalized_scores = (1 / (1 + np.exp(-predicted_scores_raw))) * 10
                
            # Format the final result
            result = {
                'predicted_classification': predicted_class_name,
                'classification_value': int(predicted_class_value),  # Include the numerical value
                'class_confidence': float(class_probs.numpy().flatten()[predicted_class_index]),
                'predicted_scores': {
                    'quality': float(normalized_scores[0]),
                    'strategy': float(normalized_scores[1]),
                    'risk': float(normalized_scores[2]),
                    'efficiency': float(normalized_scores[3]),
                    'combo_potential': float(normalized_scores[4]),
                }
            }
            
            return result
        
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return a basic fallback prediction if something goes wrong
            return {
                'predicted_classification': 'Average',
                'classification_value': 3,  # Middle value
                'class_confidence': 0.5,
                'predicted_scores': {
                    'quality': 5.0,
                    'strategy': 5.0,
                    'risk': 5.0,
                    'efficiency': 5.0,
                    'combo_potential': 5.0,
                },
                'error': str(e)
            }
    
    def _safe_preprocess_features(self, input_df):
        """
        A safer version of the data processor's _preprocess_features method
        that handles missing columns and ensures all expected columns are present.
        """
        try:
            # First try the standard processing 
            processed = self.processor._preprocess_features(input_df, is_fitting=False)
            return processed
        except ValueError as e:
            # If we hit a feature mismatch error, try manual column alignment
            print(f"Warning: Feature mismatch error. Attempting manual alignment: {str(e)}")
            
            # Manually create a DataFrame with all expected columns
            safe_df = pd.DataFrame(0, index=[0], columns=self.expected_columns)
            
            # Process the input df to extract one-hot encoded columns
            # First handle categorical columns
            cat_cols = input_df.select_dtypes(include=['object']).columns
            if len(cat_cols) > 0:
                dummies = pd.get_dummies(input_df[cat_cols], prefix=cat_cols)
                # Add any dummy columns that match our expected columns
                for col in dummies.columns:
                    if col in self.expected_columns:
                        safe_df[col] = dummies[col].values
            
            # Then handle numeric columns
            num_cols = input_df.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                if col in self.expected_columns:
                    safe_df[col] = input_df[col].values[0] if len(input_df) > 0 else 0
            
            print(f"Manual alignment complete. Created feature dataframe with shape {safe_df.shape}")
            return safe_df

if __name__ == '__main__':
    import json
    # Example usage:
    # This demonstrates how to use the predictor. App.py will use it similarly.
    MODEL_DIRECTORY = os.path.join(dirname(abspath(__file__)), 'trainer', 'saved_model_torch')
    
    try:
        predictor = Predictor(MODEL_DIRECTORY)

        # Create a sample data point that mimics a real game state with our new numerical features
        # The structure MUST match the columns the model was trained on.
        sample_data = {
            'difficulty': 'medium',
            'difficulty_numeric': 2,
            'turn_number': 5,
            'gs_collection_fill_ratio': 0.5,
            'collection_fullness': 3,
            'gs_board_tile_count': 40,
            'gs_accessible_tile_count': 10,
            'move_tile_symbol': '🍓',
            'move_tile_layer': 1,
            'move_heuristic_score': 75,
            'move_is_best_heuristic': True,
            'move_unblocks_potential': 2,
            'move_outcome': 'unknown',
            'win_loss': 1,
            'matching_tiles_count': 2,
            'gs_accessible_symbols': json.dumps(['🍓', '🍓', '🍌', '🍉', '🍇', '🍊', '🍍', '🥝', '🍒', '🍑'])
        }

        prediction = predictor.predict(sample_data)
        
        print("\n--- Prediction Result ---")
        print(prediction)
        print("-------------------------")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}") 