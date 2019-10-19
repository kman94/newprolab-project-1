#!/opt/anaconda/envs/bd9/bin/python

import sys, os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import dill
import json
import re
from urllib.parse import urlparse
from urllib.request import urlretrieve, unquote

from tqdm import tqdm
tqdm().pandas()


model_file = "./newprolab-project-1/model.dill"
pipeline, enc = dill.load(open(model_file, 'rb'))

columns=['gender','age','uid','user_json']

df = pd.read_table(
    sys.stdin, 
    sep='\t', 
    header=None, 
    names=columns
)

y_pred = pipeline.predict(df)

answer = enc.inverse_transform(y_pred)

df['gender'] = answer[:,0]
df['age'] = answer[:,1]

output = df[['uid', 'gender', 'age']]
output.sort_values(by='uid',axis = 0, ascending = True, inplace = True)
sys.stdout.write(output.to_json(orient='records'))