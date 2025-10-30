# Research Folder

This folder contains research code and experiments for the Micrograd project.

## Contents

### dataset.py
Contains dataset classes and utilities for loading and managing training data:
- `GenerativeDataset`: Dataset for sequential generative model training
- `TextDataset`: Dataset for text-based language model training
- Utilities for loading, saving, and creating datasets

### trainer.py
Implements training logic for generative models:
- `GenerativeModelTrainer`: Trainer class for generative models
- `SimpleGenerativeModel`: Basic generative model implementation
- `run_training_experiment()`: Function to run complete training experiments

## Usage

### Creating and Using Datasets

```python
from dataset import load_dataset, save_dataset

# Create a generative dataset
dataset = load_dataset('generative', sequence_length=10)

# Create a text dataset
text_dataset = load_dataset('text', vocab_size=1000)

# Save dataset
save_dataset(dataset, 'my_dataset.json')
```

### Training a Model

```python
from trainer import run_training_experiment

# Run a training experiment
trainer, model = run_training_experiment(
    dataset_type='generative',
    epochs=20,
    learning_rate=0.01
)
```

### Running from Command Line

```bash
# Run dataset examples
python dataset.py

# Run training experiment
python trainer.py
```

## Extending the Code

To add your own models and experiments:
1. Create new model classes inheriting from `SimpleGenerativeModel`
2. Extend `GenerativeModelTrainer` with custom training logic
3. Add new dataset types to `dataset.py` as needed
