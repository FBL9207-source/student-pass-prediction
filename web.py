
from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the preprocessing and prediction pipeline
with open("model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    form_data = pd.DataFrame([request.form.to_dict()])
    prediction = model.predict(form_data)[0]

    if prediction == "yes":
        result_text = "PASS"
        result_class = "pass"
    else:
        result_text = "FAIL"
        result_class = "fail"

    return render_template(
        "result.html",
        result_text=result_text,
        result_class=result_class
    )


if __name__ == "__main__":
    app.run(debug=True)
