# -*- coding: utf-8 -*- 
import numpy as np
import lightgbm as lgb
from datetime import datetime
import pandas as pd
import pickle
from sklearn import cross_validation
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp,space_eval,rand,Trials,partial,STATUS_OK
import gc

global X_train,Y_train,X_val,Y_val
print datetime.now()
train_data = pd.read_hdf('data/slide_data_910.hdf', 'ATEC')
print train_data.shape
print datetime.now()
val_data = pd.read_hdf('data/slide_data_11.hdf', 'ATEC')
print datetime.now()

X_train = train_data.drop('id',axis=1).drop('label',axis=1).drop('date',axis=1)
Y_train = train_data['label']
del train_data; gc.collect()
print 'train data done', datetime.now()
X_val = val_data.drop('id',axis=1).drop('label',axis=1).drop('date',axis=1)
Y_val = val_data['label']
del val_data; gc.collect()
print 'val data done', datetime.now()

def score(y,pred): 
	fpr, tpr, thresholds = roc_curve(y, pred, pos_label=1)
	score = 0.4*tpr[np.where(fpr>=0.001)[0][0]]+0.3*tpr[np.where(fpr>=0.005)[0][0]]+0.3*tpr[np.where(fpr>=0.01)[0][0]] 
	return -score

def lgbm(argsDict):
	max_depth = argsDict["max_depth"] + 1
	n_estimators  = argsDict['n_estimators'] * 10 + 1
	learning_rate = argsDict["learning_rate"] * 0.001 + 0.001
	subsample = argsDict["subsample"] * 0.1 + 0.5
	num_leaves = argsDict["num_leaves"] + 1
	print 'max_depth:',max_depth
	print 'n_estimators:',n_estimators
	print 'learning_rate:',learning_rate
	print 'subsample:',subsample
	print 'num_leaves:',num_leaves
	model = lgb.LGBMRegressor(
		boosting_type='gbdt',
		objective='regression',
		max_depth=max_depth,
		learning_rate=learning_rate,
		n_estimators=n_estimators,
		subsample=subsample,
		num_leaves=num_leaves,
		class_weight='balanced',
		random_state=2018,
		n_jobs=-1,
		silent=False
	)
	print 'start fitting'
	model.fit(X_train, Y_train)
	print 'start scoring'
	fpr_tpr_score = score(Y_val, model.predict(X_val))
	print fpr_tpr_score
	return -fpr_tpr_score
	'''metric = cross_validation.cross_val_score(model, X_train, np.log1p(Y_train), cv=5, scoring="neg_mean_squared_error")
	print -metric
	print -metric.mean()
	return -metric.mean()'''

lgbm_space = {
	"max_depth":hp.randint("max_depth",50),
	"n_estimators":hp.randint("n_estimators",100),  #[0,1,2,3,4,5] -> [50,]
	"learning_rate":hp.randint("learning_rate",10),  #[0,1,2,3,4,5] -> 0.05,0.06
	"subsample":hp.randint("subsample",5),			#[0,1,2,3] -> [0.7,0.8,0.9,1.0]
	"num_leaves":hp.randint("num_leaves",256)
}

algo = partial(tpe.suggest,n_startup_jobs=5)
best = fmin(lgbm, lgbm_space, algo=algo, max_evals=50)

print best
print lgbm(best)