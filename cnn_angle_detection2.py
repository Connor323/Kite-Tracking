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

NUM_PER_CLASS_TRAIN = 500
NUM_PER_CLASS_TEST = 300
DATA_RATIO = 0.8 # train / total
IMAGE_SIZE = (51, 51)   
RETRATIN = False
MODEL_NAME = 'cnn_model.h5'

class DataReader(object):
    def __init__(self, basePath, num_divs):
        super(DataReader, self).__init__()
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.num_class = num_divs
        self.label_per_angle = {}

        datasets = glob.glob(osp.join(basePath, "*.npz"))
        inputs, labels = np.array([]), np.array([])
        for dataset in datasets: 
            data = np.load(dataset)
            inputs = data['inputs'] if not len(inputs) else np.concatenate((inputs, data['inputs']), axis=0)
            labels = data['labels'] if not len(labels) else np.concatenate((labels, data['labels']), axis=0)
        images = []
        for img in inputs: 
            images.append(self.preprocess(img))

        for i in range(num_divs):
            self.label_per_angle[i] = np.where(labels == i)[0]
            print("i: ", len(self.label_per_angle[i]))

        images = np.array(images)
        labels = np.eye(self.num_class)[labels]

        train_idx, test_idx = np.array([]), np.array([])
        for i in range(num_divs):
            idx = self.label_per_angle[i]
            tmp = idx[np.random.randint(0, int(len(idx)*DATA_RATIO), NUM_PER_CLASS_TRAIN)]
            train_idx = tmp if not len(train_idx) else np.concatenate((train_idx, tmp), axis=0)
            tmp = idx[np.random.randint(int(len(idx)*DATA_RATIO), len(idx), NUM_PER_CLASS_TEST)]
            test_idx = tmp if not len(test_idx) else np.concatenate((test_idx, tmp), axis=0)

        self.x_train = images[train_idx]
        self.y_train = labels[train_idx]
        self.x_test = images[test_idx]
        self.y_test = labels[test_idx]

    def preprocess(self, image):
        image = image / 255
        image -= image.mean()
        return image
        

# Generate dummy data
data = DataReader("result", 8)
print (data.x_train.shape, data.y_train.shape)

if RETRATIN:
    model = load_model(MODEL_NAME) 
else:
    model = Sequential()
    # input: 51x51 images with 3 channels -> (51, 51, 3) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(data.num_class, activation='softmax', kernel_regularizer=regularizers.l2(0.01)))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(data.x_train, data.y_train, batch_size=32, epochs=10, verbose=1)
metrics = model.evaluate(data.x_test, data.y_test, batch_size=32, verbose=1)
down_case = np.where(data.y_test == 1)[1] == 1
metrics_down = model.evaluate(data.x_test[down_case], data.y_test[down_case], batch_size=32, verbose=1)
model.save(MODEL_NAME)
print ('Loss: {:.3f}, Accuracy: {:.3f}'.format(metrics[0], metrics[1]))
print ('DOWN: Loss: {:.3f}, Accuracy: {:.3f}'.format(metrics_down[0], metrics_down[1]))
