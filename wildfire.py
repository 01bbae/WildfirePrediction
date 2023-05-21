import os
from pathlib import Path
import math
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import layers, models, losses, optimizers
from generator import WildfireSequence
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import warnings
import pickle
print(tf.config.list_physical_devices('GPU'))

# initialize path to data files
train_path = 'train'
test_path = 'test'

# load metadata
metadata = open("metadata.txt","r")
num_samples = int(metadata.readline())
timesteps_per_sample = int(metadata.readline())
X_train_norm_shape = []
for i in range(5):
    X_train_norm_shape.append(int(metadata.readline()))
X_test_norm_shape = []
for i in range(5):
    X_test_norm_shape.append(int(metadata.readline()))
y_train_shape = []
for i in range(4):
    y_train_shape.append(int(metadata.readline()))
y_test_shape = []
for i in range(4):
    y_test_shape.append(int(metadata.readline()))
X_train_norm_shape = tuple(X_train_norm_shape)
X_test_norm_shape = tuple(X_test_norm_shape)
y_train_shape = tuple(y_train_shape)
y_test_shape = tuple(y_test_shape)
metadata.close()

print(num_samples)
print(timesteps_per_sample)
print("X_train: ", X_train_norm_shape)
print("X_test: ", X_test_norm_shape)
print("y_train: ", y_train_shape)
print("y_test: ", y_test_shape)
print("input shape: ", X_train_norm_shape[-4:])


def build_ConvLSTM():
    convlstm = models.Sequential()
    convlstm.add(layers.Input(shape=X_train_norm_shape[-4:]))
    convlstm.add(layers.ConvLSTM2D(filters=32, kernel_size=(5,5), padding="same", data_format="channels_last", activation="relu", return_sequences=True))
    convlstm.add(layers.BatchNormalization())
    convlstm.add(layers.ConvLSTM2D(filters=16, kernel_size=(3,3), padding="same", data_format="channels_last", activation="relu", return_sequences=True))
    convlstm.add(layers.BatchNormalization())
    convlstm.add(layers.ConvLSTM2D(filters=8, kernel_size=(2,2), padding="same", data_format="channels_last", activation="relu", return_sequences=True))
    convlstm.add(layers.BatchNormalization())
    convlstm.add(layers.Conv3D(filters=1, kernel_size=(2, 2, 2), padding="same", data_format="channels_last", activation="sigmoid"))
    convlstm.compile(
        loss=losses.binary_crossentropy, optimizer=optimizers.Adam(), metrics=[tf.keras.metrics.BinaryAccuracy(), 
                                                                               tf.keras.metrics.AUC(),
                                                                               tf.keras.metrics.Recall(),
                                                                               tf.keras.metrics.Precision(),
                                                                            #    tf.keras.metrics.F1Score()
                                                                            ]
    )
    return convlstm

model = build_ConvLSTM()
print(model.summary())
epochs = 10
batch_size = timesteps_per_sample
# history = model.fit(X_train_norm, y_train, validation_data=(X_test_norm,y_test), epochs=epochs, batch_size=batch_size, verbose=True)
history = model.fit(x=WildfireSequence(train_path), validation_data=WildfireSequence(test_path, train=False), epochs=epochs, verbose=True)
model.save('model2')
# model = models.load_model('model2')

# Evaluation Metrics

results = model.evaluate(WildfireSequence(test_path, train=False))
print("test loss, test acc:", results)

# with open('/trainHistoryDict', 'wb') as file_pi:
#     pickle.dump(history.history, file_pi)

# threshold = 0.5
# y_hat = model.predict(WildfireSequence(train_path))
# print(y_hat.shape)
# print(y_train_shape)
# y_train = np.load(os.path.join(train_path, "y_train_sample_0.npy"))
# for idx in range(1, int((len(os.listdir(train_path))-2)/2)):
#     print("y_train: ", idx)
#     y_train = np.concatenate((y_train, np.load(os.path.join(train_path, "y_train_sample_" + str(idx) + ".npy"))))

# y_hat_mod = np.where(y_hat > threshold, 1,0)
# y_train = y_train.flatten()
# y_hat_mod = y_hat_mod.flatten()
# print(y_train.shape)
# print(y_train.shape)
# conf_mat = confusion_matrix(y_train, y_hat_mod)
# disp = ConfusionMatrixDisplay(conf_mat)
# disp.plot()
# print("plotted")
# plt.title('training confusion matrix')
# plt.savefig('conf_mat_train.png')
# plt.close()

# y_hat = model.predict(WildfireSequence(test_path, train=False))
# print(y_hat.shape)
# print(y_test_shape)
# y_hat_mod = np.where(y_hat > threshold, 1,0)
# y_test = np.load(os.path.join(test_path, "y_test_sample_0.npy"))
# for idx in range(1, int(len(os.listdir(test_path))/2)):
#     print("y_test: ", idx)
#     y_test = np.concatenate((y_test, np.load(os.path.join(test_path, "y_test_sample_" + str(idx) + ".npy"))))



# conf_mat = confusion_matrix(y_test.flatten(), y_hat_mod.flatten())
# disp = ConfusionMatrixDisplay(conf_mat)
# disp.plot()
# plt.title('testing confusion matrix')
# plt.savefig('conf_mat_test.png')
# plt.close()


# print model keys

print(history.history.keys())
# accuracy graph
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('accuracy.png')
plt.close()

# loss graph
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('loss.png')
plt.close()

plt.plot(history.history['auc'])
plt.plot(history.history['val_auc'])
plt.title('model AUC')
plt.ylabel('auc')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('AUC.png')
plt.close()

plt.plot(history.history['recall'])
plt.plot(history.history['val_recall'])
plt.title('model recall')
plt.ylabel('recall')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('recall.png')
plt.close()

plt.plot(history.history['precision'])
plt.plot(history.history['val_precision'])
plt.title('model precision')
plt.ylabel('precision')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('precision.png')
plt.close()


# wildfire_da = wildfire_dataset.to_array()
# print(wildfire_da.loc[:, :10])
# print(wildfire_dataset["burned_areas"])
# wf_sub_da = wildfire_dataset["burned_areas"].isel(time=slice(0,10), x=slice(0,5), y=slice(0,10))
# wf_sub_da.plot()
# plt.tight_layout()
# plt.savefig("burned_areas")
# print(wf_sub_da.to_numpy())