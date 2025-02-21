from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import os
import logging
from src.ai_model.ai_service import AIService
from src.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize AI service
config = Config.load()
ai_service = AIService(config)

# Simulation state
simulation_state = {
    'active': False,
    'progress': 0,
    'status': 'idle'
}

@app.route('/')
def index():
    return render_template('index.html')

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

@app.route('/api/simulation/stop', methods=['POST'])
def stop_simulation():
    try:
        simulation_state['active'] = False
        simulation_state['status'] = 'stopped'
        logger.info("Simulation stopped")
        return jsonify(simulation_state)
    except Exception as e:
        logger.error(f"Error stopping simulation: {e}")
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3001))
    app.run(host='0.0.0.0', port=port, debug=True)