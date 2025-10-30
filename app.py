"""
Flask API application for Micrograd.
Provides endpoints for model training and inference.
"""

from flask import Flask, request, jsonify
from template import ModelTemplate, TrainerTemplate, create_config
import json

app = Flask(__name__)


@app.route('/')
def index():
    """Root endpoint - API information."""
    return jsonify({
        'name': 'Micrograd API',
        'version': '0.1.0',
        'description': 'A small autograd engine and neural network API',
        'endpoints': {
            '/': 'API information',
            '/health': 'Health check',
            '/config': 'Get/create model configuration',
            '/train': 'Train a model',
            '/predict': 'Make predictions'
        }
    })


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'service': 'micrograd-api'})


@app.route('/config', methods=['GET', 'POST'])
def config():
    """Get or create model configuration."""
    if request.method == 'POST':
        data = request.get_json()
        model_type = data.get('model_type', 'basic')
        learning_rate = data.get('learning_rate', 0.01)
        epochs = data.get('epochs', 10)
        batch_size = data.get('batch_size', 32)
        
        config = create_config(
            model_type=model_type,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size
        )
        
        return jsonify({'config': config, 'status': 'created'})
    
    # GET request - return default config
    default_config = create_config()
    return jsonify({'config': default_config})


@app.route('/train', methods=['POST'])
def train():
    """Train a model endpoint."""
    try:
        data = request.get_json()
        
        # Extract parameters
        model_type = data.get('model_type', 'basic')
        learning_rate = data.get('learning_rate', 0.01)
        epochs = data.get('epochs', 10)
        
        # Simulate training
        result = {
            'status': 'success',
            'message': 'Model training initiated',
            'model_type': model_type,
            'learning_rate': learning_rate,
            'epochs': epochs,
            'note': 'This is a placeholder - connect actual training logic'
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions with a trained model."""
    try:
        data = request.get_json()
        
        # Extract input data
        inputs = data.get('inputs', [])
        
        # Validate inputs
        if not inputs:
            return jsonify({'status': 'error', 'message': 'No input data provided'}), 400
        
        if not isinstance(inputs, list):
            return jsonify({'status': 'error', 'message': 'Inputs must be a list'}), 400
        
        # Validate input types (expecting numerical data)
        try:
            # Attempt to convert to floats to validate
            _ = [float(x) for x in inputs]
        except (ValueError, TypeError):
            return jsonify({'status': 'error', 'message': 'All inputs must be numeric'}), 400
        
        # Simulate prediction
        result = {
            'status': 'success',
            'predictions': [0.5] * len(inputs),  # Placeholder
            'note': 'This is a placeholder - connect actual model inference'
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'status': 'error', 'message': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'status': 'error', 'message': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(debug=True)
