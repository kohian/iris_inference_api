from contextlib import asynccontextmanager

import gcsfs
import joblib
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

MODEL_PATH = "gs://iris-csv/model_artifacts/logreg.joblib"

class PredictRequest(BaseModel):
    features: list[float] = Field(
        min_length=4,
        max_length=4,
        description="Iris features: sepal length, sepal width, petal length, petal width",
    )


class PredictResponse(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    probabilities: list[float]


def load_model(path: str):
    fs = gcsfs.GCSFileSystem()
    with fs.open(path, "rb") as f:
        return joblib.load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model(MODEL_PATH)
    app.state.class_names = ["setosa", "versicolor", "virginica"]

    if hasattr(app.state.model, "classes_"):
        if len(app.state.class_names) != len(app.state.model.classes_):
            raise RuntimeError("class_names length does not match model classes")
    else:
        raise RuntimeError("Loaded model is not a classifier (missing classes_)")

    yield


app = FastAPI(
    title="Model Inference API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {"message": "Iris inference API is running"}


@app.get("/health")
def health(request: Request):
    model_loaded = getattr(request.app.state, "model", None) is not None
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "model_version": request.app.state.model_version,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request):
    model = request.app.state.model
    class_names = request.app.state.class_names

    if not hasattr(model, "predict_proba"):
        raise HTTPException(
            status_code=500,
            detail="Model does not support probability output",
        )

    try:
        X = [req.features]
        pred = model.predict(X)[0]
        probs = model.predict_proba(X)[0]

        pred_idx = int(pred)

        return PredictResponse(
            class_id=pred_idx,
            class_name=class_names[pred_idx],
            confidence=float(probs[pred_idx]),
            probabilities=probs.tolist(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")