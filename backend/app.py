from flask import Flask, send_file
from sentiment_analysis_charting.sentiment_chart import generate_spider_chart

app = Flask(__name__)

@app.route('/api/sentiment-chart', methods=['GET'])
def sentiment_chart():
    chart_path = generate_spider_chart()  # Generate the chart
    return send_file(chart_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
