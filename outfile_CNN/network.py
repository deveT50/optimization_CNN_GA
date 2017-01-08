#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L
import chainer.initializers
import math


class imageModel(chainer.Chain):

	#insize = 128
	insize = 124


	def __init__(self):
		initializer = chainer.initializers.HeNormal()
		w = math.sqrt(2)  # MSRA scaling
		super(imageModel, self).__init__(
			#入力チャネル,出力チャネル, フィルタサイズpx

			conv1=L.Convolution2D(1, 8, 7,wscale=w),
			conv2=L.Convolution2D(8, 16, 5,wscale=w),
			conv3=L.Convolution2D(16, 32, 3,wscale=w),
			conv4=L.Convolution2D(32, 48, 3,wscale=w),
			fc1=L.Linear(768,1,wscale=w),
			#fc2=L.Linear(100,1,wscale=w),


		)
		self.train = True
	def __call__(self, x, t, train):


		h = self.conv1(x)
		h = F.relu(h)
		h = F.max_pooling_2d(h, 3, stride=2)

		h = self.conv2(h)
		h = F.relu(h)
		h = F.average_pooling_2d(h, 3, stride=2)

		h = self.conv3(h)
		h = F.relu(h)
		h = F.average_pooling_2d(h, 3, stride=2)
		
		h = self.conv4(h)
		h = F.relu(F.dropout(h, ratio=0.5,train=train))
		h = F.average_pooling_2d(h, 3, stride=2)
		
		y=self.fc1(h)

		if train:
			return F.mean_squared_error(y, t)
		else:
			return F.mean_squared_error(y, t)



	def predict(self, x_data):
		x=x_data
		h = self.conv1(x)
		h = F.relu(h)
		h = F.max_pooling_2d(h, 3, stride=2)

		h = self.conv2(h)
		h = F.relu(h)
		h = F.average_pooling_2d(h, 3, stride=2)

		h = self.conv3(h)
		h = F.relu(h)
		h = F.average_pooling_2d(h, 3, stride=2)
		
		h = self.conv4(h)
		h = F.relu(F.dropout(h, ratio=0.5,train=False))
		h = F.average_pooling_2d(h, 3, stride=2)
		
		y=self.fc1(h)

		return y

		






