from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

import joblib

# --- Input/Output Schemas (Pydantic v2 style) ---

class PredictRequest(BaseModel):
    # Use Field constraints instead of conlist
    # features: list[float] = Field(min_length=1, description="Feature vector for one sample")
    features: list[float] = Field(
        min_length=4,
        max_length=4,
        description="Iris features: sepal length, sepal width, petal length, petal width"
    )

class PredictResponse(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    probabilities: list[float]

# --- App lifecycle (lifespan replaces startup/shutdown events) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = joblib.load("model.joblib")
    app.state.class_names = ["setosa", "versicolor", "virginica"]

    yield
    # Shutdown: clean up if needed (close files, DB, GPU memory, etc.)
    # e.g. del app.state.model

app = FastAPI(
    title="Model Inference API",
    version="1.0.0",
    lifespan=lifespan,  # modern replacement for @app.on_event("startup")
)

@app.get("/health")
def health(request: Request):
    model_loaded = getattr(request.app.state, "model", None) is not None
    return {"status": "ok", "model_loaded": model_loaded}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request):
    model = request.app.state.model
    class_names = request.app.state.class_names

    X = [req.features]  # make it 2D

    pred = model.predict(X)[0]
    probs = model.predict_proba(X)[0]

    return PredictResponse(
        class_id=int(pred),
        class_name=class_names[pred],
        confidence=float(probs[pred]),
        probabilities=probs.tolist(),
    )

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("app:app", reload=True)