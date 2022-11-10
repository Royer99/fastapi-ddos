import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRFClassifier
import numpy as np

# load model
model = XGBRFClassifier()
model.load_model('model/model_xgboost99_semifinal_rate_noscale.txt')
test = [[4.8986410000000005, 4004, 0, 4004, 26, 0, 26, 5.103456, 0.0, 5.103456,
         4.8986410000000005, 4.8986410000000005, 4.8986410000000005, 4.8986410000000005, 0.0]]
# normalize data
print(test)
#scaler = StandardScaler()
#test = scaler.fit_transform(test)
print(test)
test2 = pd.DataFrame(test, columns=['Dur', 'SrcBytes', 'DstBytes', 'TotBytes', 'SrcPkts', 'DstPkts',
                                    'TotPkts', 'SrcRate', 'DstRate', 'Rate', 'Min', 'Max', 'Sum', 'Mean', 'StdDev'], dtype=float)
print(test2)
prediction = model.predict(test2)
print(prediction)
