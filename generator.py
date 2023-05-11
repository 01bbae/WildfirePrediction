import numpy as np
import tensorflow as tf
import math

class WildfireSequence(tf.keras.utils.Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        # low = idx * self.batch_size
        # # Cap upper bound at array length; the last batch may be smaller
        # # if the total number of items is not a multiple of batch size.
        # high = min(low + self.batch_size, len(self.x))
        batch_x = self.x[idx,:,:,:,:]
        batch_y = self.y[idx,:,:,:]
        # print(batch_x.shape)
        # print(batch_y.shape)

        return np.reshape(batch_x, (self.batch_size, batch_x.shape[0], batch_x.shape[1], batch_x.shape[2], batch_x.shape[3])), np.reshape(batch_y, (self.batch_size, batch_y.shape[0], batch_y.shape[1], batch_y.shape[2]))
        # np.reshape(batch_y, (batch_size, batch_y.shape[0], batch_y.shape[1], batch_y.shape[2], batch_y.shape[3]))