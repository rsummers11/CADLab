#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 12:50:13 2018

@author: Wes Caldwell <caldwellwg@gmail.com>

EmphNet 3D CNN class.
"""


from keras.models import Model, Sequential
from keras.layers import Conv3D, MaxPooling3D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.utils import multi_gpu_model
from scipy.ndimage import zoom
from sklearn.metrics import classification_report, confusion_matrix
from NoiseAwareLayer import NoiseAwareLayer
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import os
import re


class EmphNet:
    def __init__(self, weight_file='model/weights_40-0.56.h5'):
        """Initialize EmphNet class. 
           When using the combined model (w/ NAL), uncomment and the combined stuff,
           comment the model load, and sub in the combined weights."""


        print("Loading model...")
        self.model = self.get_model()
        self.model.load_weights(weight_file)
#        self.combined_model = Model(inputs=self.model.input, outputs=NoiseAwareLayer(name='noise')(self.model.output))
#        self.combined_model.compile(loss='categorical_crossentropy',
#                                    optimizer='adam',
#                                    metrics=['accuracy'])
#        self.combined_model.load_weights('model/combined_weights_31-0.49.h5')
        print("Model loaded!")


    def get_model(self):
        """Create the 3D CNN Model for Emphnet"""


        model = Sequential()

        # Convolutional Group 1
        model.add(Conv3D(64, (3, 3, 3),
                         name='conv1', strides=2, padding='same',
                         input_shape=(128, 128, 128, 1))) #(64, 64, 64, 64)
        model.add(BatchNormalization(name='batch_norm1'))
        model.add(Activation('relu', name='relu1'))
        model.add(MaxPooling3D(name='pool1',
                               pool_size=(2, 2, 2))) # (32, 32, 32, 64)

        # Convolution Group 2
        model.add(Conv3D(128, (3, 3, 3),
                         name='conv2', padding='same')) #(32, 32, 32, 128)
        model.add(BatchNormalization(name='batch_norm2'))
        model.add(Activation('relu', name='relu2'))
        model.add(MaxPooling3D(name='pool2',
                               pool_size=(2, 2, 2))) #(16, 16, 16, 128)

        # Convolution Group 3
        model.add(Conv3D(256, (3, 3, 3),
                         name='conv3', padding='same')) #(16, 16, 16, 256)
        model.add(BatchNormalization(name='batch_norm3'))
        model.add(Activation('relu', name='relu3'))
        model.add(MaxPooling3D(name='pool3',
                               pool_size=(2, 2, 2))) #(8, 8, 8, 256)

        # Convolution Group 4
        model.add(Conv3D(512, (3, 3, 3),
                         name='conv4', padding='same')) #(8, 8, 8, 512)
        model.add(BatchNormalization(name='batch_norm4'))
        model.add(Activation('relu', name='relu4'))
        model.add(MaxPooling3D(name='pool4',
                  pool_size=(2, 2, 2))) #(4, 4, 4, 512)

        # Convolution Group 5
        model.add(Conv3D(512, (3, 3, 3),
                         name='conv5', padding='same')) #(4, 4, 4, 512)
        model.add(BatchNormalization(name='batch_norm5'))
        model.add(Activation('relu', name='relu5'))
        model.add(MaxPooling3D(name='pool5',
                               pool_size=(2, 2, 2))) #(2, 2, 2, 512)

        # Convolution Group 6
        model.add(Conv3D(512, (2, 2, 2),
                         name='conv6', padding='valid')) #(1, 1, 1, 512)
        model.add(BatchNormalization(name='batch_norm6'))
        model.add(Activation('relu', name='relu6'))
        model.add(Dropout(0.5, name='dropout'))

        # Fully Connected & Output
        model.add(Conv3D(512, (1, 1, 1),
                         name='fc1', activation='relu')) #(1, 1, 1, 512)
        model.add(Dropout(0.5, name='dropout_fc1'))
        model.add(Conv3D(512, (1, 1, 1),
                         name='fc2', activation='relu')) #(1, 1, 1, 512)
        model.add(Dropout(0.5, name='dropout_fc2'))
        model.add(Flatten(name='flatten'))
        model.add(Dense(2, name='class', activation='softmax')) #(2)

        print(model.summary())

        # Compile model
        model = multi_gpu_model(model, gpus=4)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        return model


    def train(self, train_file, val_file, epochs=100, batch_size=16):
        n_train_files = len(open(train_file).readlines())
        n_val_files = len(open(val_file).readlines())

        print("Training...")
        model_checkpoint = ModelCheckpoint(
            'model/weights_{epoch:02d}-{val_loss:.2f}.h5',
            save_weights_only=True
        )
        csv_logger = CSVLogger(
            'emphnet.csv',
            append=True
        )
        self.model.fit_generator(
            data_generator(train_file, batch_size=batch_size),
            class_weight={0: 1.0, 1: 3.0},
            validation_data=data_generator(val_file, batch_size=batch_size),
            validation_steps=n_val_files / batch_size,
            initial_epoch=18, epochs=epochs,
            steps_per_epoch=n_train_files / batch_size,
            callbacks=[model_checkpoint, csv_logger]
        )
        print("All done!")


    def test(self, test_file, batch_size=16):
        n_test_files = len(open(test_file).readlines())

        print("Testing...")
        intermediate_model = Model(inputs=self.model.input, outputs=self.model.get_layer('sequential_1').get_layer('flatten').output)
        data = []
        for n, (X_batch, y_batch_true) in enumerate(data_generator(test_file, batch_size=batch_size)):
            if n >= n_test_files / batch_size:
                break
            y_batch_pred = intermediate_model.predict_on_batch(X_batch)
            batch_data = np.concatenate((np.expand_dims(y_batch_true.argmax(axis=1), axis=1), y_batch_pred), axis=1)
            print(batch_data)
            data.append(batch_data)
        data = np.array([result for batch in data for result in batch])
        np.save('feature_vecs.npy', data)
        # data = np.load('fold1_data.npy')
        y_pred = data[:, 1]
        y_true = data[:, 0]
        print(classification_report(y_true, y_pred >= 0.5))
        print(confusion_matrix(y_true, y_pred >= 0.5))


def data_generator(file, batch_size=1):
    """Neverending generator of data from a file"""


    regex = re.compile(r'^segmented_')

    # Load sample information and shuffle
    list_in = open(file, 'r').readlines()
    while True:
        np.random.shuffle(list_in)
        pids = []
        X = []
        y = []
        for line in list_in:
            # First get PID and file of CT scan and segmentation
            folder, label = line.strip().split(' ')
            for year in ["T0", "T1", "T2"]:
                folder = folder + "/" + year
                scaled_data = []
                # Check if there's a preprocessed numpy file
                if [f for f in os.listdir(folder) if f.endswith('.npy')]:
                     scaled_data = np.load(folder + '/scaled_data.npy')

                else:
                    try:
                        img_file_name = filter(regex.search, os.listdir(folder))[0][10:]
                    except:
                        continue

                    # Then load mask and calculate bounding box
                    mask_file = nib.load(folder + '/segmented_' + img_file_name)
                    mask_data = mask_file.get_data()
                    xmin, xmax, ymin, ymax, zmin, zmax = bbox_coords(mask_data)
                    if xmax == xmin or ymax == ymin or zmax == zmin:
                        print("PID #" + pid + " ain't right get it out of here")
                        continue

                    # Now that we have the bounding box, we can immediately crop the image file
                    # to shrink the array and save on computation time
                    img_file = nib.load(folder + '/' + img_file_name)
                    img_data = np.clip(img_file.get_data()[xmin:xmax, ymin:ymax, zmin:zmax],
                                       -1100,
                                       -500)
                    cropped_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())

                    # Now scale the cropped CT to a standard size
                    scaled_data = zoom(cropped_data, (128 / float(xmax - xmin),
                                                      128 / float(ymax - ymin),
                                                      128 / float(zmax - zmin)))

                    # Save numpy array for faster access later
                    np.save(folder + '/scaled_data.npy', scaled_data)

                # Append the prepared data to our batch
                X.append(np.transpose(scaled_data, (2, 0, 1)))
                y.append(np.eye(2)[int(label)])

                # Once the batch is the proper size, yield it and reset
                if len(X) == batch_size:
                    yield (np.reshape(X, (batch_size, 128, 128, 128, 1)),
                           np.reshape(y, (batch_size, 2)))
                    X = []
                    y = []


def bbox_coords(img):
    """Find the bounding box of a 3D segmentation mask"""


    x = np.any(img, axis=(1, 2))
    y = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    xmin, xmax = np.where(x)[0][[0, -1]]
    ymin, ymax = np.where(y)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return xmin, xmax, ymin, ymax, zmin, zmax


if __name__ == '__main__':
    """Main routine to train/test the network."""

    # Declare class
    emphnet = EmphNet()

    # Uncomment if training
    #emphnet.train(train_file='final_fold/train.lst', val_file='final_fold/val.lst')

    # Uncomment if testing
    emphnet.test(test_file='final_fold/test.lst')
