from flask import Flask, render_template
import joblib
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

app = Flask(__name__)

# Load model
try:
    model = joblib.load('model.pkl')
except FileNotFoundError:
    print("❌ Error: First run train_model.py")
    exit(1)

def get_sensor_data():
    return {
        'rainfall': 85,    # mm
        'water_level': 5.3 # meters
    }

@app.route('/')
def dashboard():
    data = get_sensor_data()
    risk = model.predict([[data['rainfall'], data['water_level']]])[0]
    
    dates = [datetime.now() + timedelta(days=i) for i in range(7)]
    fig = px.line(
        x=dates, 
        y=[5.1, 5.7, 6.2, 6.9, 6.5, 5.8, 5.2],
        title='7-Day Forecast'
    )
    
    return render_template(
        'dashboard.html',
        plot=fig.to_html(full_html=False),
        risk_level="HIGH" if risk == 1 else "LOW",
        data=data
    )

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Changed port to avoid conflicts
    