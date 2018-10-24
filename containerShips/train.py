
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default='images/images',
	help="path to input dataset")
ap.add_argument("-m", "--model", default='TestModel.model',
	help="path to output model")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# LeNet CNN class

class LeNet:

	@staticmethod

	# Function within that class

	def build(width, height, depth, classes):

		# Type of model to use

		model = Sequential()
		inputShape = (height, width, depth)

		if K.image_data_format() == 'channels_first':
			inputShape = (depth, height, width)

		# Adding layers to the CNN

		model.add(Conv2D(20, (5,5), padding='same', input_shape=inputShape))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

		model.add(Conv2D(50, (5,5), padding='same'))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

		model.add(Flatten())
		model.add(Dense(500))
		model.add(Activation('relu'))

		model.add(Dense(classes))
		model.add(Activation('softmax'))

		return model

EPOCHS = 25
INIT_LR = 1e-3
BS = 32

print('[TRAINSPOTTING] Loading images...')


# Data lists and Labels list

data = []
labels = []

# Declare x1 and x2

(x1, x2) = (0,0)

# Get the dataset

imagePaths = sorted(list(paths.list_images(args['dataset'])))


random.seed(42)
random.shuffle(imagePaths)

count = 0

# For each image in the image path

print('[TRAINSPOTTING] Starting loop ...')

for imagePath in imagePaths:
	img = cv2.imread(imagePath)
	img = cv2.resize(img, (28,28))
	img = img_to_array(img)
	data.append(img)

	# The label is the imagePath

	label = imagePath.split(os.path.sep)[-2]

	# If the label is 'Ship' it is a 1 otherwise it is a 0

	if label == 'Ships':
		label = 1
	else:
		label = 0

	# Append it onto the end of the

	labels.append(label)




data = np.array(data, dtype='float') / 255.0
labels = np.array(labels)

# This is the sklearn equivalent to the Keras datasets function

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)


aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# initialize the model
print("[TRAINSPOTTING] Building the model...")
model = LeNet.build(width=28, height=28, depth=3, classes=2)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
print("[TRAINSPOTTING] Training the network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])
