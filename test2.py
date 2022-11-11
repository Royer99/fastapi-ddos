import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle

# load model
model = pickle.load(
    open('model/basic_rf.txt', 'rb'))
test = [[29, 4466, 4.9363339999999996, 4.9363339999999996, 0.0, 4.9363339999999996,
         4.9363339999999996, 4.9363339999999996, 29, 0, 4466, 0, 5.672225, 5.672225, 0.0]]

'''
test2 = pd.DataFrame(test, columns=['Dur', 'SrcBytes', 'DstBytes', 'TotBytes', 'SrcPkts', 'DstPkts',
                                    'TotPkts', 'SrcRate', 'DstRate', 'Rate', 'Min', 'Max', 'Sum', 'Mean', 'StdDev'], dtype=float)
'''

print(test)
prediction = model.predict(test)
print(prediction)
