"""
Main entry point for the Micrograd API server.
This module initializes and runs the Flask application.
"""

from app import app
import os

def setup_api():
    """Setup and configure the API server."""
    # Configuration
    app.config['DEBUG'] = os.getenv('DEBUG', 'False').lower() == 'true'
    app.config['HOST'] = os.getenv('HOST', '0.0.0.0')
    app.config['PORT'] = int(os.getenv('PORT', 5000))
    
    # Additional API configurations
    app.config['JSON_SORT_KEYS'] = False
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max request size
    
    print("API server configured successfully")
    return app


if __name__ == '__main__':
    api = setup_api()
    
    host = api.config['HOST']
    port = api.config['PORT']
    debug = api.config['DEBUG']
    
    print(f"Starting Micrograd API server on {host}:{port}")
    print(f"Debug mode: {debug}")
    
    api.run(host=host, port=port, debug=debug)
