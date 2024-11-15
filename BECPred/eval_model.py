import os
import numpy as np
import pandas as pd
import torch
import logging
import random
import pkg_resources
import sklearn

from rxnfp.models import SmilesClassificationModel
logger = logging.getLogger(__name__)


train_model_path =  pkg_resources.resource_filename("BECPred/model")

model = SmilesClassificationModel("bert", train_model_path, use_cuda=torch.cuda.is_available())

df = pd.read_pickle('../data/final_df_ec.pkl')
df = df.loc[df['split']=='test']
print(df[:5])
test_df = df.rxn
test_reactions = test_df.values.tolist()
y_true = df.class_id
y_true = y_true.values.tolist()
print(y_true[:5])
y_true = y_true

y_preds = model.predict(test_reactions)

y_preds = pd.Series(y_preds)
y_pred = y_preds.values.tolist()
y_pred = y_pred[0]


def f1_multiclass(y_true,y_pred):
      return sklearn.metrics.f1_score(y_true,y_pred, average='weighted')

def prec_multiclass(y_true,y_pred):
      return sklearn.metrics.precision_score(y_true,y_pred, average='weighted')

def rec_multiclass(y_true,y_pred):
      return sklearn.metrics.recall_score(y_true,y_pred, average='weighted')

# for y1_true, y1_pred in zip(y_true, y_pred):
prec=prec_multiclass(y_true,y_pred)
rec=rec_multiclass(y_true,y_pred)
acc=sklearn.metrics.accuracy_score(y_true,y_pred)
mcc=sklearn.metrics.matthews_corrcoef(y_true,y_pred)
f1=f1_multiclass(y_true,y_pred)

print(prec,rec,acc,mcc,f1)
# print(y_preds)
