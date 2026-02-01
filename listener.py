from flask import Flask, request, jsonify
import math
from datetime import datetime

app = Flask(__name__)

@app.route('/click', methods=['POST'])
def click_feedback():
    data = request.json
    
    click_x = data.get('x')
    click_y = data.get('y')
    drawing_age = data.get('drawingAge')
    drawing_id = data.get('drawingId')
    timestamp = data.get('timestamp')
    button = data.get('button')

    print(f"Click received: ({click_x}, {click_y})")
    print(f"Drawing age: {drawing_age}ms, ID: {drawing_id}, Button: {button}")
    
    # MUST return a response!
    return jsonify({
        'status': 'received',
        'click_x': click_x,
        'click_y': click_y,
    }), 200

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'Server is running'}), 200

if __name__ == '__main__':
    print("Starting Click Feedback Server on port 3001...")
    print("Waiting for click feedback...\n")
    app.run(host='0.0.0.0', port=3001, debug=True)