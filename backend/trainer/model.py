import torch
import torch.nn as nn

class MultiOutputModel(nn.Module):
    """
    Creates a multi-output neural network model using PyTorch.
    This model is designed to handle numerical features and predict both
    classification (1-5 scale) and regression outputs.
    """
    def __init__(self, input_shape: int, num_classes: int, num_regression_outputs: int = 5):
        """
        Initializes the model layers.

        Args:
            input_shape (int): The number of input features.
            num_classes (int): The number of classes for the classification task (1-5).
            num_regression_outputs (int): The number of outputs for the regression task.
        """
        super(MultiOutputModel, self).__init__()

        # --- Shared Hidden Layers ---
        self.shared_layers = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # --- Task-specific Output Heads ---

        # 1. Classification Head (1-5 scale)
        self.classification_head = nn.Linear(32, num_classes)

        # 2. Regression Head (scores for various aspects)
        self.regression_head = nn.Linear(32, num_regression_outputs)

    def forward(self, x: torch.Tensor):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor containing numerical features.

        Returns:
            A tuple containing the classification and regression outputs.
        """
        # Pass input through shared layers
        shared_features = self.shared_layers(x)

        # Get outputs from each head
        classification_output = self.classification_head(shared_features)
        regression_output = self.regression_head(shared_features)

        return classification_output, regression_output

if __name__ == '__main__':
    # Example of creating the model
    DUMMY_INPUT_SHAPE = 150
    DUMMY_NUM_CLASSES = 5  # Now using 1-5 scale
    
    model = MultiOutputModel(DUMMY_INPUT_SHAPE, DUMMY_NUM_CLASSES)
    print(model)

    # Example of a forward pass
    # Create a dummy input tensor
    dummy_input = torch.randn(1, DUMMY_INPUT_SHAPE) # Batch size of 1
    
    # Get model outputs
    classification_out, regression_out = model(dummy_input)
    
    print("\n--- Example Forward Pass ---")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Classification output shape: {classification_out.shape}")
    print(f"Regression output shape: {regression_out.shape}")
    print("--------------------------") 