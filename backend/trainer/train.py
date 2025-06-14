import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, mean_absolute_error, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from data_processor import DataProcessor
from model import MultiOutputModel

# --- Configuration ---
# Build paths relative to the script's location for robustness
_script_dir = os.path.dirname(os.path.abspath(__file__))
# The CSV is in the 'backend' folder, one level up from 'trainer'
DATA_PATH = os.path.join(_script_dir, '..', 'simulation_results.csv') 
MODEL_SAVE_DIR = os.path.join(_script_dir, 'saved_model_torch')
PROCESSOR_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'data_processor.pkl')
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'triple_tile_model.pth')
EPOCHS = 500
BATCH_SIZE = 32
LEARNING_RATE = 0.001
# Weight for combining the two loss functions
# loss = (loss_weight * classification_loss) + (1 - loss_weight) * regression_loss
LOSS_WEIGHT = 0.6 

def plot_training_history(losses, val_losses=None, save_path=None):
    """Plot the training loss history"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    if val_losses:
        plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss History')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Plot a confusion matrix for the classification results"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    """
    Main function to orchestrate the data processing, model training,
    evaluation, and saving using PyTorch.
    """
    # --- 1. Load and Process Data ---
    print("Step 1: Loading and processing data...")
    processor = DataProcessor(DATA_PATH)
    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = processor.get_processed_data()
    
    # Print info about the data
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Classification labels: {np.unique(y_class_train)}")
    
    # Map classification labels back to human-readable form for interpretability
    class_map_inverse = {
        1: "Genius/Excellent",
        2: "Good",
        3: "Average",
        4: "Inaccuracy",
        5: "Blunder/Stupid"
    }
    
    # Map the numeric labels back to their human-readable forms
    readable_class_names = [class_map_inverse[label] if label in class_map_inverse else f"Class {label}" 
                            for label in sorted(np.unique(y_class_train))]
    print(f"Classes: {readable_class_names}")
    
    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train.values)
    X_test_tensor = torch.FloatTensor(X_test.values)
    y_class_train_tensor = torch.LongTensor(y_class_train)
    y_class_test_tensor = torch.LongTensor(y_class_test)
    y_reg_train_tensor = torch.FloatTensor(y_reg_train)
    y_reg_test_tensor = torch.FloatTensor(y_reg_test)

    # Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_class_train_tensor, y_reg_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_class_test_tensor, y_reg_test_tensor)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print("Data processed and loaded into PyTorch DataLoaders.")
    print(f"Input feature shape: {X_train_tensor.shape[1]}")
    print(f"Number of classes: {len(processor.encoded_class_names)}")

    # --- 2. Create the Model, Loss Functions, and Optimizer ---
    print("\nStep 2: Creating the neural network model...")
    model = MultiOutputModel(
        input_shape=X_train.shape[1],
        num_classes=len(processor.encoded_class_names)
    )
    
    # Loss functions
    classification_loss_fn = nn.CrossEntropyLoss()
    regression_loss_fn = nn.MSELoss() # Mean Squared Error
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Model, losses, and optimizer created.")
    print(model)

    # --- 3. Train the Model ---
    print(f"\nStep 3: Training the model for {EPOCHS} epochs...")
    train_losses = []
    val_losses = []
    
    for epoch in range(EPOCHS):
        # --- Training phase ---
        model.train() # Set model to training mode
        total_loss = 0
        for batch_X, batch_y_class, batch_y_reg in train_loader:
            # Forward pass
            class_pred, reg_pred = model(batch_X)
            
            # Calculate loss
            class_loss = classification_loss_fn(class_pred, batch_y_class)
            reg_loss = regression_loss_fn(reg_pred, batch_y_reg)
            
            # Combine losses
            loss = (LOSS_WEIGHT * class_loss) + ((1 - LOSS_WEIGHT) * reg_loss)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # --- Validation phase ---
        model.eval()  # Set model to evaluation mode
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y_class, batch_y_reg in test_loader:
                # Forward pass
                class_pred, reg_pred = model(batch_X)
                
                # Calculate loss
                class_loss = classification_loss_fn(class_pred, batch_y_class)
                reg_loss = regression_loss_fn(reg_pred, batch_y_reg)
                
                # Combine losses
                loss = (LOSS_WEIGHT * class_loss) + ((1 - LOSS_WEIGHT) * reg_loss)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(test_loader)
        val_losses.append(avg_val_loss)
        
        # Print update every 10 epochs
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    print("Model training complete.")

    # Plot training history
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    plot_path = os.path.join(MODEL_SAVE_DIR, 'training_history.png')
    plot_training_history(train_losses, val_losses, save_path=plot_path)

    # --- 4. Evaluate the Model ---
    print("\nStep 4: Evaluating the model on the test set...")
    model.eval() # Set model to evaluation mode
    all_class_preds = []
    all_class_true = []
    all_reg_preds = []
    all_reg_true = []

    with torch.no_grad(): # No need to calculate gradients during evaluation
        for batch_X, batch_y_class, batch_y_reg in test_loader:
            class_pred, reg_pred = model(batch_X)
            
            # Store predictions and true labels
            all_class_preds.extend(torch.argmax(class_pred, axis=1).numpy())
            all_class_true.extend(batch_y_class.numpy())
            all_reg_preds.extend(reg_pred.numpy())
            all_reg_true.extend(batch_y_reg.numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_class_true, all_class_preds)
    mae = mean_absolute_error(np.array(all_reg_true), np.array(all_reg_preds))

    print("\n--- Evaluation Results ---")
    print(f"Classification Accuracy: {accuracy:.4f}")
    print(f"Regression Mean Absolute Error (MAE): {mae:.4f}")
    
    # Get the class labels from the encoder
    class_names = [class_map_inverse[idx] for idx in processor.encoded_class_names]
    
    # Generate and print the classification report
    report = classification_report(all_class_true, all_class_preds, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    # Plot confusion matrix
    cm_path = os.path.join(MODEL_SAVE_DIR, 'confusion_matrix.png')
    plot_confusion_matrix(all_class_true, all_class_preds, class_names, save_path=cm_path)
    print("--------------------------")

    # --- 5. Save the Model and Processor ---
    print("\nStep 5: Saving the trained model and data processor...")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # Save the model's state dictionary
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    # Save the data processor object
    with open(PROCESSOR_SAVE_PATH, 'wb') as f:
        pickle.dump(processor, f)
        
    print(f"Model state dict saved to: {MODEL_SAVE_PATH}")
    print(f"Data processor saved to: {PROCESSOR_SAVE_PATH}")
    print(f"Training history plot saved to: {plot_path}")
    print(f"Confusion matrix plot saved to: {cm_path}")

if __name__ == '__main__':
    main() 