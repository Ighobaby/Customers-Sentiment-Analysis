from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import pandas as pd
import io
import pickle

app = Flask(__name__)
CORS(app)

# Load the pre-trained model and vectorizer
with open('svc_model.pkl', 'rb') as model_file:
    model, vectorizer = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/osagie')
def osagie():
    return render_template('osagie.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.is_json:
        data = request.get_json()
        text = data.get('text', '')
        if not text:
            return jsonify({"error": "Empty text provided"}), 400
        prediction = predict_sentiment(text)
        return jsonify({"prediction": prediction})
    
    elif 'review' in request.form:
        text = request.form['review']
        if not text:
            return jsonify({"error": "Empty review provided"}), 400
        prediction = predict_sentiment(text)
        return jsonify({"prediction": prediction})
    
    elif 'file' in request.files:
        file = request.files['file']
        try:
            df = pd.read_csv(file)
        except Exception as e:
            return jsonify({"error": "Invalid file format. Please upload a CSV file."}), 400
        
        if 'Review' not in df.columns:
            return jsonify({"error": "CSV must contain a 'Review' column."}), 400
        
        reviews = df['Review'].astype(str)
        predictions = reviews.apply(lambda x: predict_sentiment(x))

        output = io.StringIO()
        predictions.to_csv(output, header=True, index=False)
        output.seek(0)

        response = Response(output, mimetype='text/csv')
        response.headers['Content-Disposition'] = 'attachment; filename=predictions.csv'
        response.headers['X-Graph-Exists'] = 'false'
        return response
    
    else:
        return jsonify({"error": "Invalid request"}), 400

def predict_sentiment(text):
    # Ensure the text is vectorized correctly using the loaded vectorizer
    X = vectorizer.transform([text])
    
    # Predict sentiment
    prediction = model.predict(X)[0]
    
    # Map numeric prediction back to the sentiment label
    if prediction == 1:
        return "Positive"
    elif prediction == 0:
        return "Negative"
    else:
        return "Neutral"

if __name__ == '__main__':
    app.run(debug=True)
