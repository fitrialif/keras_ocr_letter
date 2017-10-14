#-*-coding:utf8-*-

import os
import time
import numpy as np
from datetime import datetime
from keras.callbacks import ModelCheckpoint
from util import one_hot_decoder, plot_loss_figure, load_data, get_char_set,load_img
from util import  list2str
from post_correction import  correction
from models.shallow import build_shallow
import pdb


char_set, char2idx = get_char_set('./data/')
nb_classes = len(char_set)
print ('nb_classes:', nb_classes)#607


def pred(model, X, char_set, post_correction):
	pred_res = model.predict(X)
	pred_res = [one_hot_decoder(i, char_set) for i in pred_res]
	pred_res = [list2str(i) for i in pred_res]
	# post correction
	if post_correction:
		pred_res = correction(pred_res, char_set)
	return pred_res

def infer(path):
	img_width, img_height = 64, 64
	img_channels = 1
	post_correction = False

	weights_file_path = 'models/models/2017-10-10/weights.01-0.00.hdf5'
	
	print("===========Building Model:===============")
	model = build_shallow(img_channels, img_width, img_height, nb_classes) # build CNN architecture
	model.load_weights(weights_file_path) # load trained model

	print("===========Begin Infering==============\n")
	x = []
	img = load_img(path, img_width, img_height, img_channels)
	x.append(img)
	#pdb.set_trace()
	x = np.asarray(x)
	x /= 255 # normalized
	res = pred(model, x, char_set, post_correction)
	print(res)



if __name__ == '__main__':
	infer('./data/test/12.png')
