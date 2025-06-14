# Triple Tile Model - Numerical Feature Encoding

This directory contains the machine learning model training pipeline for the Triple Tile game AI analyst. The model has been updated to use numerical feature encoding for better performance and interpretability.

## Feature Encoding

The model now uses the following numerical encodings:

### Input Features:

1. **Difficulty**: 1-4 scale
   - 1: easy
   - 2: medium
   - 3: hard
   - 4: hell

2. **Queue Fullness**: 0-7 scale
   - Represents how full the collection queue is
   - Calculated as: `int(round(collection_fill_ratio * 7))`

3. **Win/Loss Potential**: Binary 0/1
   - 0: Immediate loss
   - 1: Win or ongoing game

4. **Matching Tiles**: Count of tiles that match the selected tile
   - For each layer (0-3), we count matching tiles

5. **Unlockable Cells**: Number of cells that can be unlocked after making the move

### Output Labels:

The model predicts a classification label on a 1-5 scale:
1. **Genius/Excellent**: Brilliant move that significantly improves position
2. **Good**: Solid, logical move that advances position
3. **Average**: Neither particularly good nor bad move
4. **Inaccuracy**: Suboptimal move, better options available
5. **Mistake/Blunder/Stupid**: Poor move that worsens position or risks losing

## Training Improvements

The updated training pipeline includes:
- Better metrics tracking (accuracy, confusion matrix)
- Training history visualization
- Support for numerical encodings
- Improved model evaluation metrics

## Files

- `data_processor.py`: Handles data preprocessing and feature engineering
- `model.py`: Defines the neural network architecture
- `train.py`: Main script for training the model
- `saved_model_torch/`: Directory containing trained model and preprocessing components

## Running the Training

To train the model:

```
cd backend
python trainer/train.py
```

The training process will:
1. Load data from `simulation_results.csv`
2. Process features with numerical encodings
3. Train the model for the configured number of epochs
4. Save the trained model and data processor to `saved_model_torch/`
5. Generate performance visualizations

## Model Usage

The trained model is used by the `predictor.py` module in the backend to evaluate game moves in real-time. 