# -*- coding: utf-8 -*- 
import numpy as np
import lightgbm as lgb
from datetime import datetime
import pandas as pd
import pickle
from sklearn import cross_validation
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

def score(y,pred): 
	fpr, tpr, thresholds = roc_curve(y, pred, pos_label=1)
	plt.plot(fpr, tpr)
	plt.xlabel('FPR')
	plt.ylabel('TPR')
	plt.grid(True)
	plt.show()
	score = 0.4*tpr[np.where(fpr>=0.001)[0][0]]+0.3*tpr[np.where(fpr>=0.005)[0][0]]+0.3*tpr[np.where(fpr>=0.01)[0][0]] 
	return score

def slide_window(data):
	window_sizes = [10, 20, 35, 45, 50, 60, 70]
	X = data.copy()
	X_diff = data.diff(1, axis=1).fillna(0)
	for window_size in window_sizes:
		print('Window size is:', window_size)
		len_ = int(data.shape[1] / window_size)

		for i in range(len_):
			# Original
			tmp = X.iloc[:,i*window_size:(i+1)*window_size].copy() 
			# First Order
			tmp_diff = X_diff.iloc[:,i*window_size:(i+1)*window_size].copy()
			#original
			X[str(i)+'_'+str(window_size)+'_mean'] = tmp.mean(axis=1).values
			X[str(i)+'_'+str(window_size)+'_max'] = tmp.max(axis=1).values#train.iloc[:,i*window_size:(i+1)*window_size].max(axis=1)
			X[str(i)+'_'+str(window_size)+'_min'] = tmp.min(axis=1).values#train.iloc[:,i*window_size:(i+1)*window_size].min(axis=1)
			X[str(i)+'_'+str(window_size)+'_var'] = tmp.var(axis=1).values #train.iloc[:,i*window_size:(i+1)*window_size].var(axis=1)
			
			X[str(i)+'_'+str(window_size)+'_median'] = tmp.median(axis=1).values#train.iloc[:,i*window_size:(i+1)*window_size].min(axis=1)
			X[str(i)+'_'+str(window_size)+'_sum'] = tmp.sum(axis=1).values #train.iloc[:,i*window_size:(i+1)*window_size].var(axis=1)
			X[str(i)+'_'+str(window_size)+'_skew'] = tmp.skew(axis=1).values#train.iloc[:,i*window_size:(i+1)*window_size].min(axis=1)
			X[str(i)+'_'+str(window_size)+'_kurt'] = tmp.kurt(axis=1).values #train.iloc[:,i*window_size:(i+1)*window_size].var(axis=1)

			X[str(i)+'_'+str(window_size)+'_range'] = X[str(i)+'_'+str(window_size)+'_max'] - X[str(i)+'_'+str(window_size)+'_min']
			#X[str(i)+'_'+str(window_size)+'_argmax'] = tmp.idxmax(axis=1).values #apply(lambda x:int(x.split('_')[-1])).values
			#X[str(i)+'_'+str(window_size)+'_argmin'] = tmp.idxmin(axis=1).values #.apply(lambda x:int(x.split('_')[-1])).values
			#diff
			X[str(i)+'_'+str(window_size)+'_diffmean'] = tmp_diff.mean(axis=1).fillna(0).values #train.iloc[:,i*window_size:(i+1)*window_size].var(axis=1)
			X[str(i)+'_'+str(window_size)+'_diffvar'] = tmp_diff.var(axis=1).fillna(0).values #train.iloc[:,i*window_size:(i+1)*window_size].var(axis=1)
			X[str(i)+'_'+str(window_size)+'_diffmax'] = tmp_diff.max(axis=1).fillna(0).values #train.iloc[:,i*window_size:(i+1)*window_size].var(axis=1)
			
			X[str(i)+'_'+str(window_size)+'_diffmin'] = tmp_diff.min(axis=1).fillna(0).values #train.iloc[:,i*window_size:(i+1)*window_size].var(axis=1)
			X[str(i)+'_'+str(window_size)+'_diffskew'] = tmp_diff.skew(axis=1).fillna(0).values #train.iloc[:,i*window_size:(i+1)*window_size].var(axis=1)
			X[str(i)+'_'+str(window_size)+'_diffkurt'] = tmp_diff.kurt(axis=1).fillna(0).values #train.iloc[:,i*window_size:(i+1)*window_size].var(axis=1)

			X[str(i)+'_'+str(window_size)+'_diffrange'] = X[str(i)+'_'+str(window_size)+'_diffmax'] - X[str(i)+'_'+str(window_size)+'_diffmin']
			#X[str(i)+'_'+str(window_size)+'_diffargmax'] = tmp_diff.idxmax(axis=1).values #.apply(lambda x:int(x.split('_')[-1])).values
			#X[str(i)+'_'+str(window_size)+'_diffargmin'] = tmp_diff.idxmin(axis=1).values #.apply(lambda x:int(x.split('_')[-1])).values
	return X

def lgb_train(X_train, y_train, param):
	print 'Training lgbm model...'
	model = lgb.LGBMRegressor(
		boosting_type='gbdt',
		objective='regression',
		max_depth=param['max_depth'],
		num_leaves=param['num_leaves'],
		learning_rate=param['learning_rate'],
		subsample=param['subsample'],
		n_estimators=param['n_estimators'],
		class_weight='balanced',
		random_state=2018,
		n_jobs=128,
		silent=False
	)
	model.fit(X_train, y_train)
	return model

def train(savefile):
	#data = pd.read_hdf('data/atec_anti_fraud_train.hdf', 'ATEC')
	data = pd.read_csv('data/train_910.csv')
	val = pd.read_csv('data/val_11.csv')
	#-----------fillna(0)----------
	#data = data.fillna(0)
	#------------------------------
	#------------中位数-------------
	for i in range(1, 298):
		data['f'+str(i)] = data['f'+str(i)].fillna(data['f'+str(i)].median())
		val['f'+str(i)] = val['f'+str(i)].fillna(val['f'+str(i)].median())
		print i
	#------------------------------
	drop_fea = []
	for i in range(36, 48):
		drop_fea.append('f'+str(i))
	data = data.drop(drop_fea, axis=1)
	val = val.drop(drop_fea, axis=1)

	data = data.drop(data[data.label==-1].index)	#丢弃不确定类别样本
	val = val.drop(val[val.label==-1].index)	#丢弃不确定类别样本

	#------------------下采样-----------------
	#pos_data = data[data.label==1]
	#neg_data = data[data.label==0]
	#neg_data = neg_data[0:20000]
	#data = pos_data.append(neg_data, ignore_index=True)
	#-----------------------------------------
	#------------------上采样------------------
	'''pos_data = data[data.label==1]
	neg_data = data[data.label==0]

	neg_data.to_csv('data/neg_data.csv', index=False)
	pos_data.to_csv('data/pos_data.csv', index=False)

	data = neg_data.append(pos_data, ignore_index=True)
	for i in range(0, 40):
		print i
		data = data.append(pos_data, ignore_index=True)'''
	#-----------------------------------------

	X_data = data.drop('id',axis=1).drop('label',axis=1).drop('date',axis=1)
	Y_data = data['label']
	X_val_data = val.drop('id',axis=1).drop('label',axis=1).drop('date',axis=1)
	Y_val_data = val['label']

	#slide_window
	X_data = slide_window(X_data)
	X_val_data = slide_window(X_val_data)

	X_data['id'] = data['id']
	X_data['date'] = data['date']
	X_data['label'] = Y_data
	X_data.to_hdf('data/slide_data_910.hdf', 'ATEC')
	X_data = X_data.drop('id',axis=1).drop('label',axis=1).drop('date',axis=1)

	X_val_data['id'] = val['id']
	X_val_data['date'] = val['date']
	X_val_data['label'] = Y_val_data
	X_val_data.to_hdf('data/slide_data_11.hdf', 'ATEC')
	X_val_data = X_val_data.drop('id',axis=1).drop('label',axis=1).drop('date',axis=1)

	import gc
	del data; gc.collect()
	del val; gc.collect()

	#X_train, X_val, y_train, y_val = cross_validation.train_test_split(X_data,Y_data,test_size=0.2,random_state=0)
	#print 'split done...'

	#training part
	lgb_param = {'max_depth':10,
				 'num_leaves':50,
				 'learning_rate':0.1,
				 'subsample':0.6,
				 'n_estimators':30}
	lgb_model = lgb_train(X_data, Y_data, lgb_param)
	pickle.dump(lgb_model, open(savefile, 'wb'))


	#predict part
	y = lgb_model.predict(X_val_data)
	print score(Y_val_data, y)

def predict(modelfile, savefile):
	weight_file = modelfile
	f=open(weight_file,'r')
	lgb_model = pickle.load(f)

	data = pd.read_hdf('data/atec_anti_fraud_test_a.hdf', 'ATEC')
	#------------中位数填充-------------
	for i in range(1, 298):
		data['f'+str(i)] = data['f'+str(i)].fillna(data['f'+str(i)].median())
		print i, data['f'+str(i)].median()
	#----------------------------------

	X_data = data.drop('id', axis=1).drop('date', axis=1)
	id_series = data['id']

	import gc
	del data; gc.collect()

	X_data = slide_window(X_data)

	predict_result = lgb_model.predict(X_data)

	predict_result[np.where(predict_result<0)] = 0
	predict_result[np.where(predict_result>1)] = 1

	frame = [id_series, pd.Series(predict_result, name='score')]
	submission = pd.concat(frame, axis=1)
	submission.to_csv(savefile, index=False)

def local_tableau():
	'''pos_data = pd.read_csv('data/pos_data.csv')
	pos_data = pos_data.drop('label',axis=1).drop('date',axis=1)

	col = ['x']
	for i in range(1, 298):
		col.append(str(i))

	pos_data.columns = col
	pos_data = pos_data[0:100].T
	print pos_data

	pos_data.to_csv('data/pos_data_T.csv')'''

	neg_data = pd.read_csv('data/neg_data.csv')
	neg_data = neg_data.drop('label',axis=1).drop('date',axis=1)

	neg_data = neg_data[0:100]

	col = ['x']
	for i in range(1, 298):
		col.append(str(i))

	neg_data.columns = col
	neg_data = neg_data.T
	print neg_data

	neg_data.to_csv('data/neg_data_T.csv')

if __name__ == '__main__':
	train('output/lgb_ori_nosample.pkl')

	#-------test part------------
	#predict('output/lgb_sildewindow_nosample.pkl', 'output/lgb_sw_6337.csv')