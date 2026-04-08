from pydantic import BaseModel, Field


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