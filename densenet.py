# -*- coding: utf-8 -*- 
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score,roc_curve
import pandas as pd
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'

def Conv_Block(X_input, in_channels, out_channels, block, stage):
	conv_name = 'dense_' + str(block) + '_' + str(stage)
	with tf.variable_scope(conv_name):
		filter_1 = tf.Variable(tf.random_normal([1, in_channels, 12], stddev=0.01, dtype=np.float32))
		filter_21 = tf.Variable(tf.random_normal([1, 12, 12], stddev=0.01, dtype=np.float32))
		filter_22 = tf.Variable(tf.random_normal([5, 12, 12], stddev=0.01, dtype=np.float32))
		filter_23 = tf.Variable(tf.random_normal([10, 12, 12], stddev=0.01, dtype=np.float32))
		filter_24 = tf.Variable(tf.random_normal([20, 12, 12], stddev=0.01, dtype=np.float32))
		filter_25 = tf.Variable(tf.random_normal([40, 12, 12], stddev=0.01, dtype=np.float32))
		filter_3 = tf.Variable(tf.random_normal([1, 60, out_channels], stddev=0.01, dtype=np.float32))
		out_1 = tf.nn.conv1d(X_input, filter_1, stride=1, padding='SAME')
		out_21 = tf.nn.conv1d(out_1, filter_21, stride=1, padding='SAME')
		out_22 = tf.nn.conv1d(out_1, filter_22, stride=1, padding='SAME')
		out_23 = tf.nn.conv1d(out_1, filter_23, stride=1, padding='SAME')
		out_24 = tf.nn.conv1d(out_1, filter_24, stride=1, padding='SAME')
		out_25 = tf.nn.conv1d(out_1, filter_25, stride=1, padding='SAME')
		out_2 = tf.concat([out_21, out_22, out_23, out_24, out_25], 2)
		out_3 = tf.nn.conv1d(out_2, filter_3, stride=1, padding='SAME')
	return out_3

def Dense_Block(X_input, in_channels, out_channels, block):
	conv_block_1 = Conv_Block(X_input, in_channels, out_channels, block, 0)
	concat_1 = tf.concat([X_input, conv_block_1], 2)
	conv_block_2 = Conv_Block(concat_1, in_channels+out_channels, out_channels, block, 1)
	concat_2 = tf.concat([conv_block_1, conv_block_2, X_input], 2)
	return concat_2, in_channels+2*out_channels


def Transition_Block(X_input, in_channels, out_channels, block, trans):
	block_name = 'Block_' + str(block) + '_transiton_' + str(trans)
	with tf.variable_scope(block_name):
		filters = tf.Variable(tf.random_normal([1, in_channels, out_channels], stddev=0.01, dtype=np.float32))
		conv_out = tf.nn.conv1d(X_input, filters, stride=1, padding='SAME')
	return conv_out

def Block(X_input, in_channels, out_channels, block):
	trans_1 = Transition_Block(X_input, in_channels, out_channels/2, block, 0)
	dense, out_c = Dense_Block(X_input, in_channels, out_channels, block)
	trans_2 = Transition_Block(dense, out_c, out_channels/2, block, 1)
	concat = tf.concat([trans_1, trans_2], 2)
	return concat

def train(X_data, Y_data, X_val, y_val):
	Y = OneHotEncoder().fit_transform(Y_data.reshape(-1,1)).todense() #one-hot编码
	Y_val = OneHotEncoder().fit_transform(y_val.reshape(-1,1)).todense() #one-hot编码
	batch_size = 2048
	def generatebatch(X,Y,n_examples, batch_size):
		for batch_i in range((n_examples // batch_size)):
			start = batch_i*batch_size
			end = start + batch_size
			batch_xs = X[start:end]
			batch_ys = Y[start:end]
			yield batch_xs, batch_ys, batch_i # 生成每一个batch

	tf.reset_default_graph()

	input_data = tf.placeholder(tf.float32, [None, 297, 1])
	tf_Y = tf.placeholder(tf.float32, [None, 2])
	
	#第一层conv
	filter_conv = tf.Variable(tf.random_normal([30, 1, 48], stddev=0.01, dtype=np.float32))
	first_conv = tf.nn.conv1d(input_data, filter_conv, stride=1, padding='SAME')
	#第1个Block
	out = Block(first_conv, 48, 12, 0)
	#5个Block
	for i in range(1, 6):
		out = Block(out, 12, 12, i)
	#Dense block
	dense_out, out_c = Dense_Block(out, 12, 12, 100)		#out_c=36维
	#1conv降维到1维
	filter_down = tf.Variable(tf.random_normal([1, out_c, 1], stddev=0.01, dtype=np.float32))
	down_conv = tf.nn.conv1d(dense_out, filter_down, stride=1, padding='SAME')
	#dense
	reshape = tf.reshape(down_conv, [-1, 297])
	fc = tf.layers.dense(inputs=reshape, units=200, activation=tf.nn.relu)
	pred = tf.layers.dense(inputs=fc, units=2, activation=tf.nn.softmax)

	#每30epoch学习率下降0.5倍
	global_step = tf.Variable(0, trainable=False)
	learning_rate = tf.train.exponential_decay(0.001, global_step, 400, decay_rate=0.5, staircase=True)

	#使用focal loss
	'''pred_fixed = tf.clip_by_value(pred, 1e-11, 1.0)
	FocalLoss = tf.pow(1-pred_fixed, 2)*tf.log(pred_fixed)
	loss = -tf.reduce_mean(tf_Y*FocalLoss)'''

	#加权loss
	pred_fixed = tf.clip_by_value(pred, 1e-11, 1.0)
	WeightLoss = tf.log(pred_fixed)
	weight_tf_Y = tf.constant([1,80], dtype=tf.float32, name='weight')*tf_Y
	loss = -tf.reduce_mean(weight_tf_Y*WeightLoss)


	trainstep = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

	#y_pred = tf.arg_max(pred, 1)
	y_ground = tf.arg_max(tf_Y, 1)

	#bool_pred = tf.equal(tf.arg_max(tf_Y, 1), y_pred)
	#accuracy = tf.reduce_mean(tf.cast(bool_pred,tf.float32)) # 准确率

	#merged_summary_op = tf.summary.merge_all()

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		#summary_writer = tf.summary.FileWriter('log/densenet_logs', sess.graph)

		for epoch in range(3):
			for batch_xs, batch_ys, batch_i in generatebatch(X_data, Y, Y.shape[0], batch_size):
				pl = sess.run(trainstep, feed_dict={input_data:batch_xs, tf_Y:batch_ys})
				#summary_str = sess.run(merged_summary_op, feed_dict={input_data:batch_xs, tf_Y:batch_ys})
				#summary_writer.add_summary(summary_str, batch_i)
				if batch_i % 100 == 0:
					y_p, y_t = sess.run([pred_fixed, y_ground], feed_dict={input_data:X_val,tf_Y:Y_val})
					#fpr, tpr, thresholds = roc_curve(y_t, y_p[:,1], pos_label=1)
					roc_auc = roc_auc_score(y_t, y_p[:,1])
					lss = sess.run(loss, feed_dict={input_data:X_val,tf_Y:Y_val})
					lr = sess.run(learning_rate)
					print 'iter:',epoch,batch_i,'auc:',roc_auc,'loss:',lss,'lr:',lr
		tf.train.Saver().save(sess, 'output/'+str(epoch)+'.ckpt')

def predict(X_data):
	tf.reset_default_graph()

	input_data = tf.placeholder(tf.float32, [None, 297, 1])

	#第一层conv
	filter_conv = tf.Variable(tf.random_normal([30, 1, 48], stddev=0.01, dtype=np.float32))
	first_conv = tf.nn.conv1d(input_data, filter_conv, stride=1, padding='SAME')
	#第1个Block
	out = Block(first_conv, 48, 12, 0)
	#5个Block
	for i in range(1, 6):
		out = Block(out, 12, 12, i)
	#Dense block
	dense_out, out_c = Dense_Block(out, 12, 12, 100)		#out_c=36维
	#1conv降维到1维
	filter_down = tf.Variable(tf.random_normal([1, out_c, 1], stddev=0.01, dtype=np.float32))
	down_conv = tf.nn.conv1d(dense_out, filter_down, stride=1, padding='SAME')
	#dense
	reshape = tf.reshape(down_conv, [-1, 297])
	fc = tf.layers.dense(inputs=reshape, units=200, activation=tf.nn.relu)
	pred = tf.layers.dense(inputs=fc, units=2, activation=tf.nn.softmax)
	pred_fixed = tf.clip_by_value(pred, 1e-11, 1.0)

	saver = tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess, 'output/2.ckpt')
		result = sess.run(pred_fixed, feed_dict={input_data:X_data})
		return result[:, 1]