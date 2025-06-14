# Triple Tile AI Analyst Backend

This backend system powers the Triple Tile AI Analyst, providing real-time move evaluation and analysis for the Triple Tile game.

## Model Improvements

The machine learning model has been updated to use numerical feature encoding for better performance and interpretability:

- **Difficulty levels**: Encoded as 1-4 (easy, medium, hard, hell)
- **Queue fullness**: Encoded on a 0-7 scale
- **Move outcomes**: Binary 0/1 encoding (loss/win)
- **Matching tiles**: Count-based features for each layer
- **Output labels**: 1-5 scale (Genius, Good, Average, Inaccuracy, Blunder)

## Architecture

The backend consists of these main components:

1. **Flask API** (`app.py`): Provides endpoints for real-time move evaluation and health checking
2. **Predictor** (`predictor.py`): Loads and manages the trained model for inference
3. **Simulation** (`simulation/`): Contains game logic and simulation tools for data generation
4. **Trainer** (`trainer/`): Machine learning pipeline for model training and evaluation

## Getting Started

1. Create a Python virtual environment: `python -m venv .venv`
2. Activate the environment:
   - Windows: `.\.venv\Scripts\activate`
   - macOS/Linux: `source .venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Run the server: `python app.py`

The API server will start on `http://127.0.0.1:5001`.

## API Endpoints

- `/predict`: Evaluate a move using the local PyTorch model
- `/evaluate-move`: Evaluate a move using the FPT API (for training data generation)
- `/health`: Check server health status

## Training the Model

To retrain the model with new data:

1. Run simulations to generate data: `python run_simulation.py`
2. Train the model: `python trainer/train.py`

For more details on model training and numerical features, see the [trainer README](trainer/README.md). 