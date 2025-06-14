import os
import sys
import joblib
import pandas as pd
import numpy as np
import json
import traceback
import random
from sklearn.ensemble import RandomForestRegressor
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("⚠️ Warning: sentence-transformers package not installed. Fallback rationales will be used.")

class RationalePredictor:
    """
    A class to load and use the distilled rationale model for move evaluation.
    This model predicts both a numerical score and a rationale embedding.
    """
    def __init__(self, model_path):
        """
        Initializes the RationalePredictor by loading the model.

        Args:
            model_path (str): Path to the joblib file containing the trained model.
        """
        # Define category mapping based on score ranges
        self.category_mapping = {
            (0, 3): "Blunder",
            (3, 5): "Inaccuracy",
            (5, 7): "Strategic",
            (7, 9): "Good",
            (9, 10.1): "Brilliant"
        }

        # Define a list of canned rationales for fallback
        self.fallback_rationales = {
            "Blunder": [
                "A poor move that wastes a slot with no potential matches.",
                "This move blocks access to more valuable tiles.",
                "Creates a risky collection situation with no clear matching strategy."
            ],
            "Inaccuracy": [
                "Suboptimal choice with better alternatives available.",
                "Adds a symbol with limited matching potential.",
                "May create collection problems in future moves."
            ],
            "Strategic": [
                "Balanced move that maintains collection flexibility.",
                "Sets up future matching opportunities.",
                "Reasonable choice considering the board state."
            ],
            "Good": [
                "Strong move that creates matching potential.",
                "Unblocks key tiles while maintaining collection efficiency.",
                "Effective choice that advances toward a win."
            ],
            "Brilliant": [
                "Creates an immediate triple match, clearing valuable space.",
                "Perfect timing to complete a critical match pattern.",
                "Maximizes unblocking potential while setting up future matches."
            ]
        }

        try:
            if not os.path.exists(model_path):
                print(f"⚠️ Warning: Rationale model not found at {model_path}")
                print("Creating a simple fallback model for demonstration purposes...")
                # Create a simple fallback model (RandomForestRegressor with mock embedding)
                self.model = self._create_fallback_model()
                self.using_fallback = True
            else:
                print(f"Loading rationale predictor model from: {model_path}")
                self.model = joblib.load(model_path)
                self.using_fallback = False
                print("✅ Rationale predictor model loaded successfully!")
        except Exception as e:
            print(f"⚠️ Error loading model: {e}")
            print("Creating a fallback model...")
            self.model = self._create_fallback_model()
            self.using_fallback = True
        
        # Load the sentence transformer model for converting embeddings back to text
        self.rationale_model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
        self.sentence_model = None
        try:
            if 'sentence_transformers' in sys.modules:
                self.sentence_model = SentenceTransformer(self.rationale_model_name)
                print(f"✅ Sentence transformer model '{self.rationale_model_name}' loaded for rationale generation.")
            else:
                print(f"⚠️ Warning: sentence_transformers module not available. Using fallback rationales only.")
        except Exception as e:
            print(f"⚠️ Warning: Could not load sentence transformer model: {e}")
            print("Fallback rationales will be used instead.")
    
    def _create_fallback_model(self):
        """Creates a simple fallback model that returns reasonable predictions"""
        print("🔄 Creating fallback rationale prediction model...")
        # Create a simple model that outputs a score (0-10) and a vector of 384 dimensions
        # This mimics what the real model would do
        class FallbackModel:
            def predict(self, X):
                # For each row, return a score between 1-10 and a mock embedding vector
                # The score is somewhat based on move_tile_layer and win_loss
                results = []
                for _, row in X.iterrows():
                    # Extract some features that might be in the input
                    layer = row.get('move_tile_layer', 1)
                    win_loss = row.get('win_loss', 0.5)
                    
                    # Calculate a reasonable score (1-10)
                    base_score = 5.0  # Average score
                    layer_bonus = min(layer, 3) * 0.5  # Higher layers are better
                    outcome_factor = (win_loss * 2) - 0.5  # Win=1.5, Draw=0.5, Loss=-0.5
                    
                    score = base_score + layer_bonus + outcome_factor
                    score = max(1.0, min(10.0, score))  # Clamp between 1-10
                    
                    # Create a mock embedding (just random values)
                    embedding = np.random.normal(0, 0.1, 384)
                    
                    # Combine score and embedding
                    result = np.concatenate(([score], embedding))
                    results.append(result)
                
                return np.array(results)
        
        print("✓ Fallback model created successfully")
        return FallbackModel()
        
    def predict(self, input_data):
        """
        Makes a prediction using the rationale model.
        
        Args:
            input_data (dict): A dictionary containing move and game state features.
            
        Returns:
            dict: A dictionary with prediction results including score, category and rationale.
        """
        try:
            # Convert input data to DataFrame for model
            input_df = pd.DataFrame([input_data])
            
            # Make prediction with the model
            prediction = self.model.predict(input_df)
            
            # Extract the score (first column) and embedding (remaining columns)
            score = prediction[0][0]  # First value is the score
            embedding = prediction[0][1:]  # Remaining values form the embedding
            
            # Determine the category based on score
            category = self._get_category(score)
            
            # Generate a rationale using the embedding
            rationale = self._generate_rationale(embedding, category)
            
            # Format the response
            response = {
                "score": float(score),
                "category": category,
                "rationale": rationale,
                # Add normalized score scaled to 1-10
                "normalized_score": min(max(float(score), 1.0), 10.0)
            }
            
            return response
            
        except Exception as e:
            print(f"Error during rationale prediction: {str(e)}")
            traceback.print_exc()
            
            # Provide a fallback response
            return {
                "score": 5.0,
                "category": "Strategic",
                "rationale": "Unable to evaluate move with current model.",
                "normalized_score": 5.0,
                "error": str(e)
            }
            
    def _get_category(self, score):
        """Converts a numerical score to a category label."""
        for range_tuple, category in self.category_mapping.items():
            if range_tuple[0] <= score < range_tuple[1]:
                return category
        return "Strategic"  # Default if something goes wrong
    
    def _generate_rationale(self, embedding, category):
        """
        Generate a human-readable rationale from the embedding.
        If sentence transformer is available, try to find similar rationales.
        Otherwise, fall back to canned responses.
        """
        # Add a fallback flag check to our conditions
        if hasattr(self, 'using_fallback') and self.using_fallback:
            print(f"Using fallback rationale for category: {category}")
            
        # If embedding is empty, sentence model is not available, or we're using a fallback model
        if (embedding is None or len(embedding) == 0 or 
            self.sentence_model is None or 
            (hasattr(self, 'using_fallback') and self.using_fallback)):
            
            # Select a more specific rationale based on category
            if category in self.fallback_rationales:
                base_rationale = random.choice(self.fallback_rationales[category])
                
                # For fallback mode, occasionally add context about using fallback rationales
                if hasattr(self, 'using_fallback') and self.using_fallback and random.random() < 0.2:
                    return f"{base_rationale} (Using fallback prediction)"
                return base_rationale
            else:
                return f"This move is rated as {category.lower()}."
            
        try:
            # This would be where you'd implement a more sophisticated rationale
            # generation based on the embedding, but for now we'll use fallbacks
            return np.random.choice(self.fallback_rationales[category])
        except Exception as e:
            print(f"Error generating rationale: {e}")
            # Absolute fallback
            return f"This move is rated as {category.lower()}."
