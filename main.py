from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRFClassifier
import os
import pickle
from sklearn.model_selection import train_test_split
import joblib
# import tensorflow as tf
# import keras
import numpy as np

app = FastAPI(title="DDos classifier",
              description="DDos classifier", version="0.1")


class ModelParameters(BaseModel):
    Dur: float
    SrcBytes: float
    DstBytes: float
    SrcPkts: float
    DstPkts: float
    SrcRate: float
    DstRate: float
    Rate: float
    Min: float
    Max: float
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
        relative_path = "model/xgboost_udp.sav"
        # relative_path = "model/gru84"

    full_path = os.path.join(absolute_path, relative_path)
    # model = joblib.load(full_path)
    # model = keras.models.load_model(full_path)

    # model = pickle.load(open(full_path, 'rb'))
    # pickle.load(open(full_path, 'rb'))
    model = XGBRFClassifier()
    model.load_model(full_path)

    # model = joblib.load(full_path)
    params = model_parameters.dict()
    params.pop('model')

    # # normalize data
    scalerpath = os.path.join(
        absolute_path, "model/scaler_udp.sav")
    scaler = joblib.load(scalerpath)

    test = pd.DataFrame(params, index=[0])
    # test = pd.DataFrame(test, columns=['Dur', 'SrcBytes', 'DstBytes', 'SrcPkts','DstPkts', 'SrcRate', 'DstRate', 'Rate', 'Min', 'Max', 'Mean', 'StdDev'])
    print(test)
    test = test.reindex(columns=[" Flow Duration", " Fwd Header Length", " Bwd Header Length", " Total Fwd Packets", " Total Backward Packets",
                                 "Fwd Packets/s", " Bwd Packets/s", " Flow Packets/s", " Flow IAT Max", " Flow IAT Min", " Flow IAT Mean", " Flow IAT Std"])
    test = scaler.transform(test)
    print(test)
    prediction = model.predict(test)
    print(prediction)
    result = (prediction)
    print(result)
    # print(np.argmax(result))
    # res = np.argmax(result)
    # if(res == 0):
    #     res = 1
    # else:
    #     res = -1
    # print(res)
    # return {"class": res}
    return {"class": prediction.tolist()[0]}
