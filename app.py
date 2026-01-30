from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI(title="California Housing Price Predictor")

# Load model
model = joblib.load("california_model.pkl")

@app.get("/")
def home():
    return {"message": "California Housing Model API is running"}

@app.post("/predict")
def predict_price(
    MedInc: float,
    HouseAge: float,
    AveRooms: float,
    AveBedrms: float,
    Population: float,
    AveOccup: float,
    Latitude: float,
    Longitude: float
):
    features = np.array([[ 
        MedInc, HouseAge, AveRooms, AveBedrms,
        Population, AveOccup, Latitude, Longitude
    ]])

    prediction = model.predict(features)[0]

    return {
        "input_features": {
            "MedInc": MedInc,
            "HouseAge": HouseAge,
            "AveRooms": AveRooms,
            "AveBedrms": AveBedrms,
            "Population": Population,
            "AveOccup": AveOccup,
            "Latitude": Latitude,
            "Longitude": Longitude
        },
        "predicted_house_price": round(prediction, 2),
        "message": "Price prediction successful"
    }
import os

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )
