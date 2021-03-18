### https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/
import os
import sys

from keras.datasets import mnist
import matplotlib.pyplot as plt

import tensorflow as tf

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

from keras.preprocessing.image import ImageDataGenerator

script_folder = os.path.dirname(os.path.abspath(__file__))
base_folder = os.path.dirname(script_folder)

def baseline_model(num_classes):
	# create model
	model = tf.keras.Sequential()
    
    #model.add(tf.keras.layers.Input())
    
	model.add(tf.keras.layers.Conv2D(32, (5, 5), activation='relu', name = 'conv1'))
	model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name = 'max_pool1'))
	model.add(tf.keras.layers.Dropout(0.2, name = 'dropout1'))
	model.add(tf.keras.layers.Flatten(name = 'flatten1'))
	model.add(tf.keras.layers.Dense(128, activation='relu', name = 'dense1'))
	model.add(tf.keras.layers.Dense(num_classes, activation='softmax', name = 'dense2'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
    
# define the larger model
def larger_model(num_classes):
	# create model
	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Conv2D(30, (5, 5), activation='relu', name = 'conv1'))
	model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name = 'max_pool1'))
	model.add(tf.keras.layers.Conv2D(15, (3, 3), activation='relu', name = 'conv2'))
	model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name = 'max_pool2'))
	model.add(tf.keras.layers.Dropout(0.2, name = 'dropout1'))
	model.add(tf.keras.layers.Flatten(name = 'flatten1'))
	model.add(tf.keras.layers.Dense(128, activation='relu', name = 'dense1'))
	model.add(tf.keras.layers.Dense(50, activation='relu', name = 'dense2'))
	model.add(tf.keras.layers.Dense(num_classes, activation='softmax', name = 'dense3'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

	
def model_train():
	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	
	# plt.subplot(221)
	# plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
	# plt.subplot(222)
	# plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
	# plt.subplot(223)
	# plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
	# plt.subplot(224)
	# plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
	# # show the plot
	# plt.show()
	
	# reshape to be [samples][width][height][channels]
	X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
	X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
	
	# X_train = X_train.reshape(X_train.shape[0], 28, 28).astype('float32')
	# X_test = X_test.reshape(X_test.shape[0], 28, 28).astype('float32')

    # normalize inputs from 0-255 to 0-1
	X_train = X_train / 255
	X_test = X_test / 255
	
	# one hot encode outputs
	y_train = np_utils.to_categorical(y_train)
	y_test = np_utils.to_categorical(y_test)
	num_classes = y_test.shape[1]
	
	# build the model
	# model = baseline_model(num_classes)
	# model_save_folder = base_folder + '/output/mnist_models/baseline_model'
	
	model = larger_model(num_classes)
	model_save_folder = base_folder + '/output/mnist_models/larger_model'

	
	# Fit the model
	model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=200, verbose=2)
	model.save(model_save_folder)
	
	# for layer in model.layers:
	# 	print(layer.input_shape)
	# 	
	# #sys.exit(0)
    
	# Final evaluation of the model
	scores = model.evaluate(X_test, y_test, verbose=0)
	print("CNN Error: %.2f%%" % (100-scores[1]*100))
    
def classify():
	
	data_folder = base_folder + '/images_processed/pattern_match_template/sideline_gray'
	data_validation_folder = base_folder + '/images_processed/pattern_match_template/sideline_test_gray'
    ### https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
    
	# create generator
	datagen = ImageDataGenerator(
          rescale = 1./255,
          rotation_range = 20,
          vertical_flip = True
	)
    
	# prepare an iterators for each dataset
	train_it = datagen.flow_from_directory(
			data_folder,
			color_mode = 'grayscale',
			class_mode = 'categorical',
			target_size = (28, 28)
		)
		
	##############################
    
	# prepare an iterators for each dataset
	validation_it = datagen.flow_from_directory(
			data_validation_folder,
			color_mode = 'grayscale',
			class_mode = 'categorical',
			target_size = (28, 28)
		)
	
	model_folder = base_folder + '/output/mnist_models/baseline_model'
	#model_folder = base_folder + '/output/mnist_models/larger_model'
	
	model = tf.keras.models.load_model(model_folder)
	print(model.summary())
	
	model2 = tf.keras.Sequential()
	model2.add(tf.keras.layers.Input((28, 28, 1)))
	for layer in model.layers[:-1]:
		print(layer.name)
		model2.add(layer)
		
	model2.add(tf.keras.layers.Dense(5, activation='softmax'))
	model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	#model2.build((None, 28, 28, 1))
	
	print(model2.summary())
	
	#model2.fit(train_it, epochs=30)
	model2.fit(train_it, validation_data = validation_it, epochs=200)
		

    
	
def main():
    #model_train()
    classify()
    
    
if __name__ == '__main__':
    main()
