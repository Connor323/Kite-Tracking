import numpy as np
import os.path as osp
import glob
import cv2

import keras
from keras import regularizers
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.metrics import categorical_accuracy 

NUM_PER_CLASS_TRAIN = 17641
NUM_PER_CLASS_TEST = 500
DATA_RATIO = 0.8 # train / total
IMAGE_SIZE = (51, 51)	
RETRATIN = False
MODEL_NAME = 'cnn_loc_model.h5'

class DataReader(object):
	def __init__(self, basePath):
		super(DataReader, self).__init__()
		self.x_train = []
		self.y_train = []
		self.x_test = []
		self.y_test = []

		for i, name in enumerate(["pos", "neg"]):
			path = osp.join(basePath, name, "*.png")
			fnames = glob.glob(path)
			label = np.zeros(2)
			label[i] = 1
			for fname in fnames:
				tmp = cv2.imread(fname)
				tmp = cv2.resize(tmp, IMAGE_SIZE).astype(float)
				self.x_train.append(self.preprocess(tmp))
				self.y_train.append(label)
			for fname in np.random.choice(fnames, NUM_PER_CLASS_TEST):
				tmp = cv2.imread(fname)
				tmp = cv2.resize(tmp, IMAGE_SIZE)
				self.x_test.append(self.preprocess(tmp))
				self.y_test.append(label)

		self.x_train = np.array(self.x_train)
		self.y_train = np.array(self.y_train)
		self.x_test = np.array(self.x_test)
		self.y_test = np.array(self.y_test)

	def preprocess(self, image):
		image = image / 255
		image -= image.mean()
		return image
		

# Generate dummy data
data = DataReader("samples")
print (data.x_train.shape, data.y_train.shape)

if RETRATIN:
	model = load_model(MODEL_NAME) 
else:
	model = Sequential()
	# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
	# this applies 32 convolution filters of size 3x3 each.
	model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)))
	model.add(Conv2D(32, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	# model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
	model.add(Dropout(0.5))
	model.add(Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(0.01)))

	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(data.x_train, data.y_train, batch_size=32, epochs=4, verbose=1)
metrics = model.evaluate(data.x_test, data.y_test, batch_size=32, verbose=1)
model.save(MODEL_NAME)
print ('Loss: {:.3f}, Accuracy: {:.3f}'.format(metrics[0], metrics[1]))
