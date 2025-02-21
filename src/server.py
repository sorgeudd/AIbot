import os
import sys
# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import logging
from src.ai_model.ai_service import AIService
from src.config import Config
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    # Initialize Flask app with the correct template folder
    app = Flask(__name__, template_folder='templates')
    CORS(app)

    # Initialize config
    try:
        config = Config.load()
        # Only initialize AI service if OpenAI key is present
        if os.getenv('OPENAI_API_KEY'):
            ai_service = AIService(config)
        else:
            logger.warning("OpenAI API key not found. AI features will be limited.")
            ai_service = None
    except Exception as e:
        logger.error(f"Error initializing config: {e}")
        config = None
        ai_service = None

    # Simulation state
    simulation_state = {
        'active': False,
        'progress': 0,
        'status': 'idle'
    }

    @app.route('/')
    def index():
        try:
            logger.info("Rendering index template")
            return render_template('index.html')
        except Exception as e:
            logger.error(f"Error rendering template: {e}")
            return str(e), 500

    @app.route('/api/simulation/start', methods=['POST'])
    def start_simulation():
        try:
            simulation_state['active'] = True
            simulation_state['progress'] = 0
            simulation_state['status'] = 'running'
            logger.info("Simulation started")
            return jsonify(simulation_state)
        except Exception as e:
            logger.error(f"Error starting simulation: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/simulation/status', methods=['GET'])
    def get_simulation_status():
        try:
            if simulation_state['active']:
                simulation_state['progress'] = min(1.0, simulation_state['progress'] + 0.1)
            return jsonify(simulation_state)
        except Exception as e:
            logger.error(f"Error getting simulation status: {e}")
            return jsonify({"error": str(e)}), 500

    return app

if __name__ == '__main__':
    # Get port from environment or default to 3000
    port = int(os.environ.get('PORT', 3000))
    # Use 0.0.0.0 to make it accessible from Replit
    logger.info(f"Starting Flask server on port {port}")
    app = create_app()
    app.run(host='0.0.0.0', port=port, debug=True)