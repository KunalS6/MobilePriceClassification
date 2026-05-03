from flask import Flask, render_template, request
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return render_template('index.html')


# Prediction route
@app.route('/predicdata', methods=['GET', 'POST'])
def predict_datapoint():

    if request.method == 'GET':
        return render_template('home.html')

    try:
        data = CustomData(
            battery_power=int(request.form.get('battery_power')),
            blue=int(request.form.get('blue')),
            clock_speed=float(request.form.get('clock_speed')),
            dual_sim=int(request.form.get('dual_sim')),
            fc=int(request.form.get('fc')),
            four_g=int(request.form.get('four_g')),
            int_memory=int(request.form.get('int_memory')),
            m_dep=int(request.form.get('m_dep')),
            mobile_wt=int(request.form.get('mobile_wt')),
            n_cores=int(request.form.get('n_cores')),
            pc=int(request.form.get('pc')),
            px_height=int(request.form.get('px_height')),
            px_width=int(request.form.get('px_width')),
            ram=int(request.form.get('ram')),
            sc_h=int(request.form.get('sc_h')),
            sc_w=int(request.form.get('sc_w')),
            talk_time=int(request.form.get('talk_time')),
            three_g=int(request.form.get('three_g')),
            touch_screen=int(request.form.get('touch_screen')),
            wifi=int(request.form.get('wifi'))
        )

        # Convert to DataFrame
        pred_df = data.get_data_as_data_frame()

        # Prediction
        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(pred_df)

        # Map result to readable label
        price_map = {
            0: "Low Cost 📱",
            1: "Mid Range 📱",
            2: "High Range 📱",
            3: "Premium 📱"
        }

        final_result = price_map.get(result[0], "Unknown")

        return render_template('home.html', results=final_result)

    except Exception as e:
        return f"Error occurred: {e}"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)