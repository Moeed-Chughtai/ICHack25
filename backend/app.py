from flask import Flask, send_file, request, jsonify
from sentiment_analysis_charting.sentiment_chart import generate_spider_chart
from flask_cors import CORS
import json
import os




app = Flask(__name__)
CORS(app)


@app.route('/api/sentiment-chart', methods=['GET'])
def check():
    return jsonify({"message": "Hello, World!"})


JSON_FILE = "../frontend/src/data/Sessions.json"

@app.route('/api/sentiment-chart', methods=['GET'])
def sentiment_chart():
    """
    API endpoint to generate and return a spider chart
    for sentiment analysis of a specific session.
    """
    session_id = request.args.get("session_id")  # Get session ID from query parameter
    if not session_id:
        return jsonify({"error": "Missing session_id parameter"}), 400

    # Check if the JSON file exists
    if not os.path.exists(JSON_FILE):
        return jsonify({"error": f"JSON file not found at {JSON_FILE}"}), 500

    # Load the session data
    try:
        with open(JSON_FILE, "r") as file:
            data = json.load(file)

        # Retrieve the specific session data
        session = data["sessions"].get(session_id)
        if not session:
            return jsonify({"error": f"Session ID {session_id} not found"}), 404

        # Extract student sentiment distribution
        sentiments = session["sentiment_analysis"]["student_emotion_distribution"]

        # Generate the spider chart
        chart_path = generate_spider_chart(sentiments)
        return send_file(chart_path, mimetype="image/png")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/submit', methods=['POST'])
def submit():
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, "r") as file:
            sessions = json.load(file)
    else:
        sessions = {"sessions": {}}

    data = request.json

    # Generate a new session ID
    new_id = str(len(sessions["sessions"]) + 1)

    # Add new session
    sessions["sessions"][new_id] = {
        "title": data.get("title"),
        "subject": data.get("subject"),
        "date": data.get("date"),
        "time": data.get("time")
    }

    with open(JSON_FILE, "w") as file:
        json.dump(sessions, file, indent=4)

    return jsonify({"message": "Session added successfully", "sessions": sessions}), 201







# from confidence_measurement.gcpstt import analyze_speech
# import os

# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the upload directory exists

# @app.route('/api/analyze-speech', methods=['POST'])
# def analyze_speech_endpoint():
#     """
#     Receives an audio file from the frontend, saves it temporarily,
#     and processes it with the `analyze_speech` function.
#     """
#     if 'file' not in request.files:
#         return jsonify({"error": "No file provided"}), 400

#     file = request.files['file']

#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400

#     # Save the file temporarily
#     file_path = os.path.join(UPLOAD_FOLDER, file.filename)
#     file.save(file_path)

#     # Process the audio file using the analyze_speech function
#     result = analyze_speech(file_path, cleanup=True)

#     return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
