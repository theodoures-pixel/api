from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(
    title="Machine Learning Puspita Farm",
    description="API prediksi durasi pemompaan nutrisi melon berbasis Random Forest",
    version="2.0.0"
)

# ─────────────────────────────────────────
# CORS — izinkan semua origin (dev lokal)
# ─────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# Load model sekali saat startup
# ─────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "modelfinal.pkl")
model = joblib.load(MODEL_PATH)

# ─────────────────────────────────────────
# Schema Input (Pydantic)
# ─────────────────────────────────────────
class PredictRequest(BaseModel):
    """Input dari Backend proxy endpoint /api/melon/nutrisi-cerdas."""
    suhu: float         # Suhu udara lingkungan (°C)
    kelembaban: float   # Kelembaban udara (%)
    lux: float          # Intensitas cahaya (lux)
    ph: float           # Kadar pH larutan nutrisi
    suhu_air: float     # Suhu air larutan nutrisi (°C)
    fase: int           # Fase pertumbuhan: 1=Vegetatif, 2=Generatif 1, 3=Generatif 2

class PredictDosisDosis(BaseModel):
    """Input dari Auto-Dosing background worker (schema lama)."""
    suhu: float
    kelembaban: float
    ph: float
    tds_sekarang: float
    fase: str           # Nama fase sebagai string

# ─────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────
@app.get("/")
def home():
    return {
        "message": "ML API Puspita Farm is running",
        "version": "2.0.0",
        "endpoints": ["/predict", "/predict-dosis"]
    }

@app.post("/predict")
def predict(data: PredictRequest):
    """
    Endpoint utama: Dipanggil oleh Backend FastAPI (proxy /api/melon/nutrisi-cerdas).
    Menerima data sensor lengkap + fase integer, mengembalikan durasi pompa dalam detik.
    """
    try:
        input_data = np.array([[
            data.suhu,
            data.kelembaban,
            data.lux,
            data.ph,
            data.suhu_air,
            data.fase
        ]])

        durasi_detik = round(float(model.predict(input_data)[0]), 2)

        return {
            "status": "success",
            "durasi_pompa_detik": durasi_detik,
            # Alias backward compat
            "durasi": durasi_detik
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/predict-dosis")
def predict_dosis(data: PredictDosisDosis):
    """
    Endpoint kompatibilitas: Dipanggil oleh Auto-Dosing background worker.
    Menerima fase sebagai string, mengonversi ke integer, lalu menggunakan
    fitur default untuk lux dan suhu_air karena tidak tersedia dari worker.
    """
    try:
        # Konversi nama fase string → integer (sama dengan logika backend)
        fase_map = {"Vegetatif": 1, "Generatif 1": 2, "Generatif 2": 3}
        fase_int = fase_map.get(data.fase, 1)

        # Gunakan nilai default untuk kolom yang tidak dikirim oleh worker
        lux_default = 5000.0
        suhu_air_default = 28.0

        input_data = np.array([[
            data.suhu,
            data.kelembaban,
            lux_default,
            data.ph,
            suhu_air_default,
            fase_int
        ]])

        durasi_detik = round(float(model.predict(input_data)[0]), 2)

        # Logika bisnis: apakah pompa perlu menyala?
        pompa_status = "ON" if durasi_detik > 0 else "OFF"

        return {
            "status": "success",
            "status_pompa": pompa_status,
            "durasi_pompa_detik": durasi_detik,
            "durasi": durasi_detik,
            "target_ppm": 1500 if fase_int >= 2 else 1050
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}