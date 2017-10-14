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
	#pred_proba = model.predict_proba(X)
	print(X)
	#print(pred_proba)
	pred_res = [one_hot_decoder(i, char_set) for i in pred_res]
	pred_res = [list2str(i) for i in pred_res]
	# post correction
	if post_correction:
		pred_res = correction(pred_res, char_set)
	return pred_res

def test(model, test_data, char_set, post_correction):
	test_X, test_y = test_data[0], test_data[1]
	test_y = [one_hot_decoder(i, char_set) for i in test_y]
	test_y = [list2str(i) for i in test_y]
	pred_res = pred(model, test_X, char_set, post_correction)

	nb_correct = sum(pred_res[i]==test_y[i] for i in range(len(pred_res)))
	for i in range(len(pred_res)):
		if test_y[i] != pred_res[i]:
			print ('test:', test_y[i])
			print ('pred:', pred_res[i])
			pass
	print ('nb_correct: ', nb_correct)
	print ('Acurracy: ', float(nb_correct) / len(pred_res))


def main():
	img_width, img_height = 64, 64
	img_channels = 1  
	post_correction = False

	save_dir = 'models/models/' + str(datetime.now()).split('.')[0].split()[0] + '/' # model is saved corresponding to the datetime
	test_data_dir = './data/test/'
	weights_file_path = 'models/models/2017-10-11/weights.02-0.00.hdf5'

	print("===========Building Model:===============")
	model = build_shallow(img_channels, img_width, img_height, nb_classes) # build CNN architecture
	model.load_weights(weights_file_path) # load trained model

	print("=========Begin Loading Test Data:============:\n")
	test_data = load_data(test_data_dir, img_width, img_height, img_channels, char_set)
	
	print("===========Begin Testing==============\n")
	test(model, test_data, char_set, post_correction)

if __name__ == '__main__':
	main()
