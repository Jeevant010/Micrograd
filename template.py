"""
Template module for Micrograd project.
Contains base templates and utilities for the project.
"""

class ModelTemplate:
    """Base template class for neural network models."""
    
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        
    def forward(self, x):
        """Forward pass through the model."""
        raise NotImplementedError("Subclasses must implement forward method")
    
    def backward(self):
        """Backward pass for gradient computation."""
        raise NotImplementedError("Subclasses must implement backward method")


class TrainerTemplate:
    """Base template class for model trainers."""
    
    def __init__(self, model, learning_rate=0.01):
        self.model = model
        self.learning_rate = learning_rate
        self.losses = []
        
    def train_step(self, inputs, targets):
        """Perform a single training step."""
        raise NotImplementedError("Subclasses must implement train_step method")
    
    def train(self, dataset, epochs=10):
        """Train the model on the dataset."""
        for epoch in range(epochs):
            total_loss = 0
            for inputs, targets in dataset:
                loss = self.train_step(inputs, targets)
                total_loss += loss
            
            avg_loss = total_loss / len(dataset)
            self.losses.append(avg_loss)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return self.losses


def create_config(model_type='basic', **kwargs):
    """Create configuration dictionary for models."""
    config = {
        'model_type': model_type,
        'learning_rate': kwargs.get('learning_rate', 0.01),
        'epochs': kwargs.get('epochs', 10),
        'batch_size': kwargs.get('batch_size', 32),
    }
    return config
