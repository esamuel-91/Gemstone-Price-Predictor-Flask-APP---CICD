from flask import Flask,render_template,request
from src.logger import logger
from src.pipeline.prediction_pipeline import PredictionPipeline,Custom_Data
import numpy as np

application = Flask(__name__)
app = application

@app.route("/")
def homepage():
    return render_template("index.html")
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("form.html")

    try:
        # Safely convert numeric fields
        try:
            log_carat = float(request.form.get("log_carat"))
            volume = float(request.form.get("volume"))
            depth = float(request.form.get("depth"))
            table = float(request.form.get("table"))
        except (ValueError, TypeError):
            logger.error("Invalid numeric input received.")
            return render_template("form.html", error="Invalid input: Please enter valid numeric values"), 400

        # Create data object
        data = Custom_Data(
            log_carat=log_carat,
            volume=volume,
            depth=depth,
            table=table,
            cut=request.form.get("cut"),
            color=request.form.get("color"),
            clarity=request.form.get("clarity")
        )

        final_new_data = data.gather_data_as_dataframe()

        prediction_pipeline = PredictionPipeline()
        log_pred = prediction_pipeline.predict(final_new_data)

        # Your model predicts log(price)
        # So you reverse using expm1:

        final_price = np.expm1(log_pred[0])
        result = round(float(final_price), 2)

        logger.info(f"Prediction Successful: Log Value: {log_pred[0]} and final result: {result}")

        return render_template("result.html", final_result=result)

    except Exception as e:
        logger.error(f"Unexpected error in prediction: {e}")
        return render_template("form.html", error="Something went wrong. Please try again."), 500
        

if __name__ == "__main__":
     app.run(host="0.0.0.0",port=5001,debug=True)