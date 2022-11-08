from matplotlib.pyplot import title
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import XGBRFClassifier
from sklearn.metrics import accuracy_score
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from sklearn.preprocessing import StandardScaler

app = FastAPI(title="DDos classifier",
              description="DDos classifier", version="0.1")


class ModelParameters(BaseModel):
    param1: float
    param2: float
    param3: float
    param4: float
    param5: float
    param6: float
    param7: float
    param8: float
    param9: float
    param10: float
    param11: float
    param12: float
    param13: float
    param14: float


model = None


@app.on_event('startup')
@app.get("/")
async def root():
    return {"message": "Hello World"}

# call predict function


@app.post("/classify")
async def classify(model_parameters: ModelParameters):
    model = XGBRFClassifier()
    model.load_model(
        "/home/royer/Documents/AD2022/ddosClassifier-api/model/model_xgboost99.txt")
    test = pd.DataFrame([model_parameters.dict()])
    # class 0
    #test = [[1, 144, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 0, 144, 0, 0.0, 0.0]]
    # class 1
    #test = [[29, 4118, 4.959919999999999, 4.959919999999999, 0.0, 4.959919999999999,4.959919999999999, 4.959919999999999, 29, 0, 4118, 0, 5.645251999999999, 0.0]]
    # class 2
    #test = [[24, 3696, 4.881948, 4.881948, 0.0, 4.881948, 4.881948,4.881948, 24, 0, 3696, 0, 4.711233999999999, 0.0]]
    test2 = pd.DataFrame(test, columns=['TotPkts', 'TotBytes', 'Dur', 'Mean', 'StdDev', 'Sum', 'Min',
                                        'Max', 'SrcPkts', 'DstPkts', 'SrcBytes', 'DstBytes', 'SrcRate', 'DstRate'], dtype=float)
    prediction = model.predict(test2)
    print(prediction)
    return {"class": prediction.tolist()}
