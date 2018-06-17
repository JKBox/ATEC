# -*- coding: utf-8 -*- 
import pandas as pd
from datetime import datetime
<<<<<<< HEAD
from sklearn.preprocessing import MinMaxScaler
=======
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
>>>>>>> 7ed22b80562a8bf53700ecbefb58cb2f50475406
from sklearn import cross_validation
import densenet
import numpy as np

#----------Training Part------------
'''
data = pd.read_hdf('data/atec_anti_fraud_train.hdf', 'ATEC')
data = data.fillna(0)

data = data.drop(data[data.label==-1].index)	#丢弃不确定类别样本

X_data = data.drop('id',axis=1).drop('label',axis=1).drop('date',axis=1)
Y_data = data['label']


X_train, X_val, y_train, y_val = cross_validation.train_test_split(X_data,Y_data,test_size=0.2,random_state=0)
print 'split done...'
a, X_val, c, y_val = cross_validation.train_test_split(X_val,y_val,test_size=0.01,random_state=0)

#scaler = MinMaxScaler()
#X_train = scaler.fit_transform(X_train)
#X_val = scaler.fit_transform(X_val)
#print 'scale done...'

X_train = np.array(X_train).reshape((-1, 297, 1))
X_val = np.array(X_val).reshape((-1, 297, 1))
densenet.train(X_train, y_train, X_val, y_val)
'''

#----------Predict Part--------------
<<<<<<< HEAD

weight_file = 'output/99.ckpt'
=======
>>>>>>> 7ed22b80562a8bf53700ecbefb58cb2f50475406
data = pd.read_hdf('data/atec_anti_fraud_test_a.hdf', 'ATEC')
data = data.fillna(0)

X_data = data.drop('id', axis=1).drop('date', axis=1)
id_series = data['id']

batch_size = 4096
def generatebatch(X,n_examples, batch_size):
	for batch_i in range((n_examples // batch_size)+1):
		print batch_i, (n_examples // batch_size)
		start = batch_i*batch_size
		end = start + batch_size
		batch_xs = X[start:end]
		yield batch_xs, batch_i # 生成每一个batch

X_data = np.array(X_data).reshape((-1, 297, 1))
predict_result = np.empty(0)
<<<<<<< HEAD
for batch_xs, batch_i in generatebatch(X_data, X_data.shape[0], batch_size):
	result = densenet.predict(batch_xs, weight_file)
=======
print predict_result
for batch_xs, batch_i in generatebatch(X_data, X_data.shape[0], batch_size):
	result = densenet.predict(batch_xs)
>>>>>>> 7ed22b80562a8bf53700ecbefb58cb2f50475406
	predict_result = np.append(predict_result, np.array(result))

frame = [id_series, pd.Series(predict_result, name='score')]
submission = pd.concat(frame, axis=1)
<<<<<<< HEAD
submission.to_csv('output/score_29.csv', index=False)
=======
submission.to_csv('output/submission.csv', index=False)
>>>>>>> 7ed22b80562a8bf53700ecbefb58cb2f50475406
