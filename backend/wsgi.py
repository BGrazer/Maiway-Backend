from flask import Flask
from flask_cors import CORS

# Import the app creation functions from your scripts
from rfr import create_app as create_rfr_app
from chatbot import create_app as create_chatbot_app

# Create the main app
app = Flask(__name__)
CORS(app)

# Create and register the blueprints
rfr_app = create_rfr_app()
chatbot_app = create_chatbot_app()

app.register_blueprint(rfr_app.blueprints['rfr'])
app.register_blueprint(chatbot_app.blueprints['chatbot'])

@app.route('/')
def index():
    return "Maiway-Backend is running!"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)