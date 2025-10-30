# Micrograd

A small autograd engine and neural network library with API support.

## Project Structure

```
Micrograd/
├── setup.py           # Project setup and dependencies
├── template.py        # Base templates for models and trainers
├── main.py            # API server entry point
├── app.py             # Flask API implementation
└── research/          # Research code and experiments
    ├── README.md      # Research folder documentation
    ├── dataset.py     # Dataset utilities for generative models
    └── trainer.py     # Generative model training logic
```

## Features

- **Autograd Engine**: Automatic differentiation for gradient computation
- **Neural Network Library**: Template-based model building
- **REST API**: Flask-based API for model training and inference
- **Research Tools**: Dataset utilities and training experiments

## Installation

```bash
pip install -e .
```

## Usage

### Running the API Server

```bash
python main.py
```

The API will be available at `http://localhost:5000` with the following endpoints:

- `GET /` - API information
- `GET /health` - Health check
- `GET/POST /config` - Get or create model configuration
- `POST /train` - Train a model
- `POST /predict` - Make predictions

### Running Research Experiments

```bash
cd research
python trainer.py
```

## Configuration

Set environment variables to configure the API:

- `DEBUG=true` - Enable debug mode
- `HOST=0.0.0.0` - Set host address
- `PORT=5000` - Set port number

## Development

See the `research/` folder for experimental code and model training utilities.

## License

See LICENSE file for details.
