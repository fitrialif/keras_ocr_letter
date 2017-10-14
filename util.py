import time
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
import keras.backend as K
import pdb
import natsort


def one_hot_encoder(data, whole_set):
	"""
	Encode the whole list, not a single record
	"""
	ret = []

	tmp = np.zeros(len(whole_set), dtype=np.int8)
	tmp[data] = 1
	ret.append(tmp)
	print("one_hot_encode:%s"%(data))
	return np.asarray(ret)


def one_hot_decoder(data, whole_set):
	ret = []
	if data.ndim == 1:
		data = np.expand_dims(data, 0)
	for probs in data:
		idx = np.argmax(probs)
		# print idx, whole_set[idx], probs[idx]
		ret.append(whole_set[idx])
	return ret


def list2str(data):
	return ''.join([i if i != 'empty' else '' for i in data])


def plot_loss_figure(history, save_path):
	train_loss = history.history['loss']
	val_loss = history.history['val_loss']
	plt.plot(train_loss, 'b', val_loss, 'r')
	plt.xlabel('train_loss: blue   val_loss: red      epoch')
	plt.ylabel('loss')
	plt.title('loss figure')
	plt.savefig(save_path)

def load_img(path, width, height, channels):
	img = Image.open(path)
	img = img.resize((width, height))
	if channels==1: # convert the image to gray scale image if it's RGB
		img = img.convert('L')
	img = np.asarray(img, dtype='float32')
	if channels > 1:
		img = np.rollaxis(img, 2, 0)
	else:
		img = np.expand_dims(img, 0)
	return img

def load_data(path,width,height,channels,char_set):
	tag = time.time()
	x = []
	y = []
	#pdb.set_trace()
	path2 = './data/'+path.split('/')[2]+'_label.txt'
	with open(path2,'r') as f:
		for line in f.readlines():
			filename = line.split()[0]
			filepath = path + filename
			pixels = load_img(filepath, width, height, channels)
			print("load image:"+filename)
			x.append(pixels)
			label = int(line.split()[1])
			y.append(label)

	x = np.asarray(x)
	x /= 255 # normalized
	y = [one_hot_encoder(i, char_set) for i in y]
	y = np.asarray(y)
	if y.shape[1] == 1: # keras bug ?
		y = y[:,0,:] 
	print ('Data loaded, spend time(m) :', (time.time()-tag)/60)
	#pdb.set_trace()
	return [x, y]

def get_char_set(file_dir):
	file_path = file_dir+'label.txt'
	ret = []
	with open(file_path) as f:
		for raw in f:
			raw = raw.strip('\r\n')
			for i in raw:
				if i not in ret:
					ret.append(i)
	char_set = list(ret)
	char2idx = dict(zip(char_set, range(len(char_set))))
	return char_set, char2idx

def categorical_accuracy_per_sequence(y_true, y_pred):
	return K.mean(K.min(K.equal(K.argmax(y_true, axis=-1),
				  K.argmax(y_pred, axis=-1)), axis=-1))


