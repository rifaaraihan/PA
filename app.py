from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def fakenews_analysis():
    if request.method == 'GET':
        return render_template("index.html")
    elif request.method == 'POST':
        print(dict(request.form))
        selected_features = dict(request.form).values()
        model, std_scaler = joblib.load(
            "model-development/bernoulli_nb.pkl")
        selected_features = std_scaler.transform([selected_features])
        print(selected_features)
        result = model.predict(selected_features)
        fakenews = {
            's': 'Valid',
            's': 'Hoax',
            
        }
        result = fakenews
        return render_template('index.html', result=result)
    else:
        return "Unsupported Request Method"


if __name__ == '__main__':
    app.run(port=5000, debug=True)
