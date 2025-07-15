# chatbot.py

import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from chatbot_model import ChatbotModel

# Initialize Flask App
app = Flask(__name__)
CORS(app)

# Initialize Chatbot
chatbot = ChatbotModel()

@app.route('/chat', methods=['POST'])
def chat():
    if request.json is None:
        return jsonify({"error": "Request body must be JSON"}), 400

    user_message = request.json.get('message')
    if not user_message:
        return jsonify({"error": "No 'message' key provided in JSON body or message is empty"}), 400

    response = chatbot.get_response(user_message)
    return jsonify({"response": response})

@app.route('/dynamic_suggestions', methods=['GET'])
def get_dynamic_suggestions():
    try:
        query = request.args.get('query', '')
        if not query:
            return jsonify({"suggestions": []})
        suggestions = chatbot.get_matching_questions(query)
        return jsonify({"suggestions": suggestions})
    except Exception as e:
        return jsonify({"error": f"An unexpected server error occurred: {e}"}), 500

@app.route('/admin/add_faq', methods=['POST'])
def add_faq():
    if request.json is None:
        return jsonify({"error": "Request body must be JSON for FAQ addition"}), 400

    data = request.json
    question = data.get('question')
    answer = data.get('answer')

    if not question or not answer:
        return jsonify({"error": "Both 'question' and 'answer' are required."}), 400

    success = chatbot.add_faq(question, answer)
    if success:
        return jsonify({"message": "FAQ added and chatbot knowledge base updated."}), 200
    else:
        return jsonify({"message": "FAQ (or similar question) already exists."}), 200

@app.route('/admin/reload_chatbot', methods=['POST'])
def reload_chatbot():
    chatbot.reload_data()
    return jsonify({"message": "Chatbot data reloaded successfully."})

@app.route('/data/faq_data.json')
def serve_faq_data():
    return send_from_directory(os.path.join(app.root_path, 'data'), 'faq_data.json')

if __name__ == '__main__':
    print(f"\n🤖 Chatbot backend running at: http://0.0.0.0:5001\n")
    app.run(host='0.0.0.0', port=5001)
