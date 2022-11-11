from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRFClassifier
import os

app = FastAPI(title="DDos classifier",
              description="DDos classifier", version="0.1")


class ModelParameters(BaseModel):
    Dur: float
    SrcBytes: float
    DstBytes: float
    TotBytes: float
    SrcPkts: float
    DstPkts: float
    TotPkts: float
    SrcRate: float
    DstRate: float
    Rate: float
    Min: float
    Max: float
    Sum: float
    Mean: float
    StdDev: float
    model: int


model = None


@app.on_event('startup')
@app.get("/")
async def root():
    return {"message": "Hello World"}

# call predict function


@app.post("/classify")
async def classify(model_parameters: ModelParameters):

    absolute_path = os.path.dirname(__file__)
    if model_parameters.model == 1:
        relative_path = "model/model_xgboost98_semifinal"
    elif model_parameters.model == 2:
        relative_path = "model/model_xgboost99_semifinal.txt"
    elif model_parameters.model == 3:
        relative_path = "model/model_xgboost99_semifinal_rate.txt"

    full_path = os.path.join(absolute_path, relative_path)
    model = XGBRFClassifier()
    model.load_model(relative_path)
    params = model_parameters.dict()
    params.pop('model')
    test = pd.DataFrame([params])
    # normalize data
    scaler = StandardScaler()
    test = scaler.fit_transform(test.T)
    test2 = pd.DataFrame(test.T, columns=['Dur', 'SrcBytes', 'DstBytes', 'TotBytes', 'SrcPkts', 'DstPkts',
                                          'TotPkts', 'SrcRate', 'DstRate', 'Rate', 'Min', 'Max', 'Sum', 'Mean', 'StdDev'], dtype=float)
    prediction = model.predict(test2)
    print(prediction)
    return {"class": prediction.tolist()[0]}
