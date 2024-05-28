from flask import Flask, request, jsonify
from datetime import datetime
import socket
import json
import argparse
import os
from gradientai import Gradient

app = Flask(__name__)
base_model = None
token = 'gFUwiECcl52VEZOUZE6RYALLu4QjwX2P'
workspace_id = '93af8759-ef81-4b11-8ed7-363ff2efc1cb_workspace'

os.environ['GRADIENT_ACCESS_TOKEN'] = token
os.environ['GRADIENT_WORKSPACE_ID'] = workspace_id

def prepareLlamaBot():
    global base_model
    gradient = Gradient()
    base_model = gradient.get_base_model(base_model_slug="llama3-8b-chat")

@app.route('/')
def index():
    return "Welcome to the Flask API!"

@app.route('/blogs', methods=['GET'])
def get_blogs():
    try:
        with open('blogs.json', 'r') as file:
            blogs = json.load(file)
        return jsonify(blogs)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    global base_model, user_message, chat_history
    data = request.get_json()

    if 'studentMessage' not in data or not isinstance(data['studentMessage'], str):
        return jsonify({'error': 'studentMessage must be a string'}), 400

    if 'difficultyLevel' not in data or not isinstance(data['difficultyLevel'], str):
        return jsonify({'error': 'difficultyLevel must be a string'}), 400
    
    if 'targetLanguage' not in data or not isinstance(data['targetLanguage'], str):
        return jsonify({'error': 'targetLanguage must be a string'}), 400

    if 'chatHistory' not in data or not isinstance(data['chatHistory'], list):
        return jsonify({'error': 'chatHistory must be a list'}), 400

    if not all(isinstance(item, dict) and 'Student' in item and 'Tutor' in item for item in data['chatHistory']):
        return jsonify({'error': 'chatHistory must be a list of dictionaries with keys Student and Tutor'}), 400

    student_message = data['studentMessage']
    chat_history = data['chatHistory']
    difficulty_level = data['difficultyLevel']
    target_language = data['targetLanguage']
    chat_history_str = '\n'.join([f"{item['Student']} - {item['Tutor']}" for item in chat_history])

    QUERY = f"""
    [INST]
    YOU ARE A VIRTUAL TUTOR HELPING A STUDENT LEARN {target_language.upper()}. 
    USE THE FOLLOWING GUIDELINES FOR THE CONVERSATION:
    1. CONSIDER THE CHAT HISTORY AND THE LATEST MESSAGE FROM THE USER.
    2. HOLD A CONTEXT-AWARE CONVERSATION.
    3. ADJUST THE DIFFICULTY OF THE CONVERSATION BASED ON THE USER'S PERFORMANCE.
    4. PROVIDE CONSTRUCTIVE FEEDBACK ON GRAMMAR, VOCABULARY, AND PRONUNCIATION.
    5. GIVE ENCOURAGEMENT AND SUGGEST IMPROVEMENTS.
    6. THE CURRENT DIFFICULTY LEVEL BY THE USER IS: {difficulty_level}.
    7. FOCUS ONLY ON READING AND WRITING.
    
    GIVEN THE CHAT HISTORY:
    {chat_history_str}
    
    AND THE LATEST MESSAGE FROM USER:
    {student_message}
    
    RESPOND TO THE USER, CONTINUE THE CONVERSATION APPROPRIATELY, ADJUST THE DIFFICULTY AS PER THE USER RESPONSE AS CURRENT DIFFICULTY LEVEL IS {difficulty_level}, AND PROVIDE FEEDBACK. KEEP YOUR RESPONSE SHORT
    [/INST]
    """

    response = base_model.complete(query=QUERY, max_generated_token_count=500).generated_output

    return jsonify({'message': response}), 200

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000, help='Specify the port number')
    args = parser.parse_args()

    port_num = args.port
    prepareLlamaBot()
    print(f"App running on port {port_num}")
    app.run(port=port_num)
