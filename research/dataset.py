"""
Dataset module for generative model training.
Provides sample datasets and data loading utilities.
"""

import json
import random


class GenerativeDataset:
    """Dataset class for generative model training."""
    
    def __init__(self, data=None, sequence_length=10):
        self.sequence_length = sequence_length
        self.data = data if data is not None else self._generate_sample_data()
        
    def _generate_sample_data(self):
        """Generate sample sequential data for training."""
        # Generate synthetic sequences for demonstration
        sequences = []
        for i in range(100):
            # Create simple numerical sequences
            start = random.uniform(0, 10)
            step = random.uniform(0.1, 1.0)
            sequence = [start + step * j for j in range(self.sequence_length)]
            sequences.append(sequence)
        
        return sequences
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a data sample by index."""
        sequence = self.data[idx]
        # For generative models, input is sequence[:-1], target is sequence[1:]
        inputs = sequence[:-1]
        targets = sequence[1:]
        return inputs, targets
    
    def __iter__(self):
        """Iterator over dataset."""
        for i in range(len(self)):
            yield self[i]


class TextDataset:
    """Text dataset for language model training."""
    
    def __init__(self, texts=None, vocab_size=1000):
        self.vocab_size = vocab_size
        self.texts = texts if texts is not None else self._generate_sample_texts()
        self.vocab = self._build_vocab()
        
    def _generate_sample_texts(self):
        """Generate sample text data."""
        sample_texts = [
            "The quick brown fox jumps over the lazy dog",
            "Machine learning is transforming artificial intelligence",
            "Neural networks learn patterns from data",
            "Gradient descent optimizes model parameters",
            "Backpropagation computes gradients efficiently",
        ]
        return sample_texts
    
    def _build_vocab(self):
        """Build vocabulary from texts."""
        vocab = {}
        idx = 0
        for text in self.texts:
            for word in text.lower().split():
                if word not in vocab:
                    vocab[word] = idx
                    idx += 1
        return vocab
    
    def tokenize(self, text):
        """Convert text to token indices."""
        words = text.lower().split()
        return [self.vocab.get(word, 0) for word in words]
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        """Get a tokenized text sample."""
        text = self.texts[idx]
        tokens = self.tokenize(text)
        if len(tokens) > 1:
            inputs = tokens[:-1]
            targets = tokens[1:]
        else:
            inputs = tokens
            targets = tokens
        return inputs, targets


def load_dataset(dataset_type='generative', **kwargs):
    """Load a dataset for training.
    
    Args:
        dataset_type: Type of dataset ('generative' or 'text')
        **kwargs: Additional arguments for dataset initialization
    
    Returns:
        Dataset object
    """
    if dataset_type == 'generative':
        sequence_length = kwargs.get('sequence_length', 10)
        return GenerativeDataset(sequence_length=sequence_length)
    elif dataset_type == 'text':
        vocab_size = kwargs.get('vocab_size', 1000)
        texts = kwargs.get('texts', None)
        return TextDataset(texts=texts, vocab_size=vocab_size)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def save_dataset(dataset, filepath):
    """Save dataset to a JSON file."""
    data_dict = {
        'type': type(dataset).__name__,
        'data': dataset.data if hasattr(dataset, 'data') else dataset.texts,
        'config': {}
    }
    
    if isinstance(dataset, GenerativeDataset):
        data_dict['config']['sequence_length'] = dataset.sequence_length
    elif isinstance(dataset, TextDataset):
        data_dict['config']['vocab_size'] = dataset.vocab_size
        data_dict['config']['vocab'] = dataset.vocab
    
    with open(filepath, 'w') as f:
        json.dump(data_dict, f, indent=2)
    
    print(f"Dataset saved to {filepath}")


def load_dataset_from_file(filepath):
    """Load dataset from a JSON file."""
    with open(filepath, 'r') as f:
        data_dict = json.load(f)
    
    dataset_type = data_dict['type']
    config = data_dict['config']
    
    if dataset_type == 'GenerativeDataset':
        dataset = GenerativeDataset(
            data=data_dict['data'],
            sequence_length=config.get('sequence_length', 10)
        )
    elif dataset_type == 'TextDataset':
        dataset = TextDataset(
            texts=data_dict['data'],
            vocab_size=config.get('vocab_size', 1000)
        )
        dataset.vocab = config.get('vocab', {})
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    print(f"Dataset loaded from {filepath}")
    return dataset


if __name__ == '__main__':
    # Example usage
    print("Creating sample generative dataset...")
    gen_dataset = load_dataset('generative', sequence_length=10)
    print(f"Generated dataset with {len(gen_dataset)} samples")
    
    print("\nCreating sample text dataset...")
    text_dataset = load_dataset('text')
    print(f"Text dataset with {len(text_dataset)} samples")
    print(f"Vocabulary size: {len(text_dataset.vocab)}")
    
    # Save sample datasets
    save_dataset(gen_dataset, 'sample_generative_dataset.json')
    save_dataset(text_dataset, 'sample_text_dataset.json')
