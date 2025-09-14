from flask import Flask, request, render_template_string
from transformers import pipeline
import os

app = Flask(__name__)

# Load sentiment model
sentiment_model = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")

HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
    <title>MCA E-consultation Sentiment Analysis</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .result { padding: 20px; margin: 20px 0; border-radius: 8px; }
        .positive { background-color: #d4edda; color: #155724; }
        .negative { background-color: #f8d7da; color: #721c24; }
        textarea { width: 100%; padding: 10px; border-radius: 4px; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
    </style>
</head>
<body>
    <h2>MCA E-consultation Sentiment Analysis</h2>
    <form method="post">
        <textarea name="comment" rows="6" placeholder="Enter consultation comment here..."></textarea><br><br>
        <button type="submit">Analyze Sentiment</button>
    </form>
    {% if result %}
        <div class="result {{ result.label.lower() }}">
            <strong>Sentiment:</strong> {{ result.label }}<br>
            <strong>Confidence:</strong> {{ result.score }}%
        </div>
    {% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        comment = request.form['comment']
        prediction = sentiment_model(comment)
        result = {
            'label': prediction['label'],
            'score': round(prediction['score'] * 100, 2)
        }
    return render_template_string(HTML_TEMPLATE, result=result)

if __name__ == '__main__':
    # Use PORT environment variable for Render deployment
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
