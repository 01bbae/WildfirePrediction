import numpy as np
import tensorflow as tf
import math
from pathlib import Path
import os

class WildfireSequence(tf.keras.utils.Sequence):

    def __init__(self, data_path, train=True):
        self.data_path = data_path
        self.train = train

    def __len__(self):
        # number of batches is half the number of files in the train or test directory
        if self.train: # remove 2 for train directory because we dont want to count mean and std files
            return int((len((os.listdir(self.data_path)))-2)/2)
        else:
            return int(len(os.listdir(self.data_path))/2)

    def __getitem__(self, idx):
        if self.train:
            X_train_norm_sample = np.load(os.path.join(self.data_path, "X_train_norm_sample_" + str(idx) + ".npy"))
            y_train_sample = np.load(os.path.join(self.data_path, "y_train_sample_" + str(idx) + ".npy"))
            # print("in generator X_train_norm_sample: ", X_train_norm_sample.shape)
            # print("in generator y_train_sample: ",y_train_sample.shape)

            X_train_norm_sample = np.expand_dims(X_train_norm_sample, axis=0)
            y_train_sample = np.expand_dims(y_train_sample, axis=0)

            return X_train_norm_sample, y_train_sample
        else:
            X_test_norm_sample = np.load(os.path.join(self.data_path, "X_test_norm_sample_" + str(idx) + ".npy"))
            y_test_sample = np.load(os.path.join(self.data_path, "y_test_sample_" + str(idx) + ".npy"))
            X_test_norm_sample = np.expand_dims(X_test_norm_sample, axis=0)
            y_test_sample = np.expand_dims(y_test_sample, axis=0)
            # print("in generator X_test_norm_sample: ", X_test_norm_sample.shape)
            # print("in generator y_test_sample: ",y_test_sample.shape)
            return X_test_norm_sample, y_test_sample