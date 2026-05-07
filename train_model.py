import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

df = pd.read_csv("dataset.csv", sep=';')

X = df[['suhu','kelembaban','lux','ph','suhu_air','fase']]
y = df['durasi']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, "modelfinal.pkl")

print("Model baru dengan fase berhasil dibuat!")