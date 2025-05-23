from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline


application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    try:
        if request.method=='GET':
            return render_template('home.html', results = None)
        else:
            data=CustomData(
                age = request.form.get('age'),
                sex = request.form.get('sex'),
                bmi = float(request.form.get('bmi')),
                smoker_binary = int(request.form.get('smoker_binary')),
                region = request.form.get('region')
            )

            pred_df = data.get_data_as_data_frame()
            print(pred_df)

            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            return render_template('home.html', results = results[0])
    except Exception as e:
        import traceback
        print("==== ERROR OCCURRED DURING PREDICTION ====")
        print(e)
        traceback.print_exc()
        return f"Internal server error: {e}", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)