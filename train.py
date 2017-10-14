#-*-coding:utf8-*-

import os
import time
from datetime import datetime
from keras.callbacks import ModelCheckpoint
from util import one_hot_decoder, plot_loss_figure, load_data, get_char_set
from util import  list2str
from post_correction import  correction
from models.shallow import build_shallow
import pdb


def train(model, batch_size, nb_epoch, save_dir, train_data, val_data, char_set):
	#pdb.set_trace()
	X_train, y_train = train_data[0], train_data[1]
	print ('X_train shape:', X_train.shape)
	print (X_train.shape[0], 'train samples')
	if os.path.exists(save_dir) == False:
		os.mkdir(save_dir)

	start_time = time.time()
	save_path = save_dir + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
	check_pointer = ModelCheckpoint(save_path,
		save_best_only=True)
	hist = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
		validation_data=val_data,
		validation_split=0.1,
		callbacks=[check_pointer],
		sample_weight=None
		)#返回记录字典，包括每一次迭代的训练误差率和验证误差率

	plot_loss_figure(hist, save_dir + str(datetime.now()).split('.')[0].split()[1]+'.png')
	print ('Training time(h):', (time.time()-start_time) / 3600)


def main():
	img_width, img_height = 64, 64
	img_channels = 1
	batch_size = 32   
	nb_epoch = 4
	post_correction = False

	save_dir = 'models/models/' + str(datetime.now()).split('.')[0].split()[0] + '/' # model is saved corresponding to the datetime
	train_data_dir = './data/train/'
	val_data_dir = './data/val/'
	char_set, char2idx = get_char_set('./data/')#charset = ['empty',...,'鸵', '豸', '山',...] char2idx = { ...,'弗': 3290, '毓': 6488,...}
	nb_classes = len(char_set)
	print ('nb_classes:', nb_classes)#607

	print("===========Building Model:===============")
	model = build_shallow(img_channels, img_width, img_height, nb_classes) # build CNN architecture
	
	print("=========Begin Loading Val Data:=============\n")
	val_data = load_data(val_data_dir, img_width, img_height, img_channels,char_set)
	
	print("=========Begin Loading Train Data:=============\n")
	train_data = load_data(train_data_dir,img_width, img_height, img_channels,char_set)
	
	print("===========Begin Training=============:\n")
	train(model, batch_size, nb_epoch, save_dir, train_data, val_data, char_set)



if __name__ == '__main__':
	main()
