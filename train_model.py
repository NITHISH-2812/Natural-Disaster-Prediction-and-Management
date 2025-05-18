import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

data = {
    'rainfall': [120, 90, 60, 30, 150, 80, 200],
    'water_level': [5.2, 4.1, 3.8, 2.9, 6.8, 5.0, 7.5],
    'flood_occurred': [1, 1, 0, 0, 1, 0, 1]
}
df = pd.DataFrame(data)
model = RandomForestClassifier(n_estimators=100)
model.fit(df[['rainfall', 'water_level']], df['flood_occurred'])
joblib.dump(model, 'flood_model.pkl')
print("✅ Model trained and saved!")