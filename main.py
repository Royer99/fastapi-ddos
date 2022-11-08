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

    test = [[1, 144, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 0, 144, 0, 0.0, 0.0]]
    test2 = pd.DataFrame(test, columns=['TotPkts', 'TotBytes', 'Dur', 'Mean', 'StdDev', 'Sum', 'Min',
                                        'Max', 'SrcPkts', 'DstPkts', 'SrcBytes', 'DstBytes', 'SrcRate', 'DstRate'], dtype=float)
    #prediction = model.predict(model_parameters)
    prediction = model.predict(test2)
    print(prediction)
    #json_compatible_item_data = jsonable_encoder(prediction)
    # return JSONResponse(content=json_compatible_item_data)
    return {"class": prediction.tolist()}
