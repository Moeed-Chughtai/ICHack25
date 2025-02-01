from flask import Flask, send_file, request, jsonify
from sentiment_analysis_charting.sentiment_chart import generate_spider_chart
from flask_cors import CORS
import json
import os

app = Flask(__name__)
CORS(app)

@app.route('/api/sentiment-chart', methods=['GET'])
def sentiment_chart():
    chart_path = generate_spider_chart()  # Generate the chart
    return send_file(chart_path, mimetype='image/png')

JSON_FILE = "../frontend/src/data/Sessions.json"

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


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
