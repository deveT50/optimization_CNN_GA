#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import argparse
import sys
import os

import numpy as np
from PIL import Image
import math
import random
import six
#import cPickle as pickle
import six.moves.cPickle as pickle
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import serializers

import network
from network import imageModel

parser = argparse.ArgumentParser(description='predict object weight and container capacity in a picture')
parser.add_argument('path_obj', help='Path to obj-image txt file')#物体パスを書いたファイル
parser.add_argument('path_ctn', help='Path to container-image txt file')#容器パスを書いたファイル
parser.add_argument('--num','-n', default=10, type=int, help='whole obj num')#物体数
#parser.add_argument('--objw','-w', default=0, type=int, help='use only one type of obj orderd idx (0 means random select from various obj)')
parser.add_argument('--cidx','-idx', default=0, type=int, help='container index')#使用する容器

args = parser.parse_args()
args.cidx=args.cidx%5#containerのidxが4を超えたら0


#パラメータファイルロード
#mean_image = pickle.load(open("mean.npy", 'rb'))
#sigma_image = pickle.load(open("sigma.npy",'rb'))
mean_obj = pickle.load(open("mean_obj_w.npy", 'rb'),encoding='latin-1')
mean_ctn = pickle.load(open("mean_container_img.npy", 'rb'),encoding='latin-1')
sigma_obj = pickle.load(open("sigma_obj_w.npy",'rb'),encoding='latin-1')
sigma_ctn = pickle.load(open("sigma_container_img.npy",'rb'),encoding='latin-1')


#モデル設定
model_obj = network.imageModel()
model_ctn = network.imageModel()
serializers.load_hdf5("model_obj_w", model_obj)
serializers.load_hdf5("model_container_img", model_ctn)
cropwidth = 128 - model_obj.insize
model_obj.to_cpu()
model_ctn.to_cpu()


#画像読み込み
def read_image(path,model,mean_image,sigma_image):

	image = np.asarray(Image.open(path))
	#top = random.randint(0, cropwidth - 1)
	#left = random.randint(0, cropwidth - 1)
	top = left = cropwidth / 2
	bottom = model.insize + top
	right = model.insize + left
	image = image[top:bottom, left:right].astype(np.float32)
	#正規化
	image -= mean_image[top:bottom, left:right]
	image/=sigma_image

	return image


#物体の重さ・容量予測
def predict(path,model,mean_img,sigma_img):
	
	img = read_image(path,model,mean_img,sigma_img)
	x = np.ndarray((1,1, model.insize, model.insize), dtype=np.float32)
	x[0]=img
	x = chainer.Variable(np.asarray(x), volatile='on')
	number = imageModel.predict(model,x)

	return number.data[0][0]

if __name__ == '__main__': 

	#画像リスト作成
	i=1	
	w_offset=8#物体の重さオフセット
	c_scale=8#容器の容量定数倍
	outfile = open('in_list.txt','w')#出力ファイル名
	pick_idx_ctn=0

	#重さをファイルに書き込む
	#ファイルから画像のリスト作成
	lstw=open((args.path_obj),"r").readlines()
	for l in lstw:
		if i>args.num:
			break;
		path=os.path.join('.', l.strip().split()[0])
		weight=round(predict(path,model_obj,mean_obj,sigma_obj),0)
		outfile.write(str(i)+","+str(weight+w_offset)+"\r")
		i+=1
		print(path)
	
	#容量をファイルに書き込む
	lst=open((args.path_ctn),"r").readlines()
	path=lst[args.cidx].strip().split()[0]#引数で使用された容器を使う
	capacity=round(predict(path,model_ctn,mean_ctn,sigma_ctn),0)
	outfile.write("0"+","+str(capacity*c_scale)+"\r")
	outfile.close()
	print(path)



