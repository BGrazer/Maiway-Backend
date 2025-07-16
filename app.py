import sys
import os
from flask import Flask
from flask_cors import CORS

# Add routing directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'routing')))

# Import Blueprints
from rfr import rfr_bp
from chatbot import chatbot_bp
from routing.routing import routing_bp

# Initialize Flask App
app = Flask(__name__)
CORS(app)

# Register Blueprints
app.register_blueprint(rfr_bp)
app.register_blueprint(chatbot_bp)
app.register_blueprint(routing_bp, url_prefix='/routing')

if __name__ == '__main__':
    print(f"\n🚀 Combined backend running at: http://0.0.0.0:5001\n")
    app.run(host='0.0.0.0', port=5001)