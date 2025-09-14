from flask import Flask, request, render_template_string
from transformers import pipeline

app = Flask(__name__)

# Use lighter DistilBERT model for sentiment analysis
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1  # Use CPU
)

HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
    <title>MCA Sentiment Analysis</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .result { padding: 20px; margin: 20px 0; border-radius: 8px; }
        .POSITIVE { background-color: #d4edda; color: #155724; }
        .NEGATIVE { background-color: #f8d7da; color: #721c24; }
        textarea { width: 100%; padding: 10px; border-radius: 4px; font-size: 16px; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
    </style>
</head>
<body>
    <h2>MCA E-Consultation Sentiment Analysis</h2>
    <form method="post">
        <textarea name="comment" rows="6" placeholder="Enter consultation comment here..."></textarea><br><br>
        <button type="submit">Analyze Sentiment</button>
    </form>
    {% if result %}
        <div class="result {{ result.label }}">
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
        prediction = sentiment_model(comment)[0]
        result = {
            'label': prediction['label'],  # POSITIVE or NEGATIVE
            'score': round(prediction['score'] * 100, 2)
        }
    return render_template_string(HTML_TEMPLATE, result=result)

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
