from flask import Flask, render_template, url_for, request
import joblib
import numpy
app = Flask(__name__)
model = joblib.load("modelpred.pkl")
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict",methods=['POST'])
def predict():
    input_features = [i for i in request.form.values()]
    input_numpy = [numpy.array(input_features)]
    prediction = model.predict(input_numpy)
    return render_template("index.html",prediction_text="Predicted Salary is:{}".format(prediction[0]))

if __name__ == "__main__":
    app.run(debug=True)