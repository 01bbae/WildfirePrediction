import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import layers, models, losses, optimizers
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn import preprocessing
import warnings
print(tf.config.list_physical_devices('GPU'))

# import dataset
datapath = "../wildfire_dataset.nc"
wildfire_dataset = xr.open_dataset(datapath, engine="netcdf4")
# print(wildfire_dataset)
feature_list = wildfire_dataset.data_vars
feature_nums = len(feature_list)

# maybe predict more than one thing later, but for not only try to predict burned areas
remove_label = ["burned_areas", "ignition_points", "number_of_fires", "POP_DENS_2009", "POP_DENS_2010", "POP_DENS_2011", "POP_DENS_2012", "POP_DENS_2013", "POP_DENS_2014", "POP_DENS_2015", "POP_DENS_2016", "POP_DENS_2017", "POP_DENS_2018", "POP_DENS_2019", "POP_DENS_2020", "POP_DENS_2021", "ROAD_DISTANCE"]
X_label = [label for label in feature_list if label not in remove_label]
y_label = "burned_areas"

# take the first 5 time steps for all x and y to try creating a smaller dataset
num_samples = 10
timesteps_per_sample = 5
width_limit = 100
height_limit = 100
timestep_samples = num_samples*timesteps_per_sample
dataset = wildfire_dataset.isel(time=slice(None,timestep_samples), x=slice(None,width_limit), y=slice(None,height_limit))
print(dataset)
# dataset_head = wildfire_dataset.head(indexers={"time": timestep_samples})
# print(dataset_head)

dataset_X = dataset[X_label]
dataset_y = dataset[y_label]

# Create the y into a numpy matrix of shape (time, x, y)
dataset_y_np = dataset_y.to_numpy()
dataset_y_np = np.transpose(dataset_y_np, (0,2,1))

# Create the X into a numpy matrix of shape (time, x, y)
dataset_X_np = dataset_X[list(dataset_X.data_vars)[0]].to_numpy()
dataset_X_np = np.transpose(dataset_X_np, (0,2,1))
dataset_X_np = np.expand_dims(dataset_X_np, 3)

# Takes each feature of the xarray Dataset and converts it into a DataArray 
# Also appends it into the new np array to make shape of (time x, y, features)
for index, feature in enumerate(list(dataset_X.data_vars)):
    if(index!=0):
        # Since wf_dataset_X_np is already initiaklized with the first element, skip
        new_np_arr = dataset_X[feature].to_numpy()
        if(len(new_np_arr.shape) == 2):
            # If a feature doesn't contain a time dimension (n), we extend the 2d matrix to 3d with copy of matrix n times
            # Might be able to use numpy broadcast instead
            new_np_arr = np.repeat(new_np_arr[:, :, np.newaxis], timestep_samples, axis=2)
            # Transpose feature to "time", "x", "y" format
            new_np_arr = np.transpose(new_np_arr)
        else:
            # Transpose feature to "time", "x", "y" format
            new_np_arr = np.transpose(new_np_arr, (0,2,1))
        if (np.isnan(new_np_arr).all()):
            # Precaution to alert if a feature has all NaN values
            warnings.warn(str(feature) + " feature's values are all NaNs")
        if (np.isnan(new_np_arr).any()):
            # Precaution to alert if a feature has all NaN values
            warnings.warn(str(feature) + " feature's values has NaNs")
        # new_np_arr = np.nan_to_num(new_np_arr)
        print(new_np_arr.shape)
        dataset_X_np = np.concatenate((dataset_X_np, np.expand_dims(new_np_arr, axis=3)), axis=3)
    print(feature)
    print(dataset_X_np.shape)

if 1 in dataset_y_np:
    print("Fire exists") 
print("Output classes: ", np.unique(dataset_y_np))
# class_weights = class_weight.compute_class_weight(class_weight = "balanced", classes = np.unique(wf_dataset_y_np), y = wf_dataset_y_np)
# print(class_weights)

# Create samples (samples, time, features, x, y)
# NOT IMPLEMENTED: Each samples are 4 days with 2 day overlap between each one
dataset_X_np = np.expand_dims(dataset_X_np, axis=0)
print("wf_dataset_X_np:", type(dataset_X_np))
print("wf_dataset_X_np(expand dims):", dataset_X_np)
dataset_X_np = np.reshape(dataset_X_np, (num_samples, timesteps_per_sample, dataset_X_np.shape[2], dataset_X_np.shape[3], dataset_X_np.shape[4]))
print("dataset_X_np(reshape):", type(dataset_X_np))

print("dataset_X_np.shape: ", dataset_X_np.shape)

# Deal with y labels
replacement_y_np = np.zeros(timestep_samples)
for i in range(dataset_y_np.shape[0]):
    if 1 in dataset_y_np[i]:
        replacement_y_np[i] = 1
replacement_y_np = np.reshape(replacement_y_np, (num_samples, timesteps_per_sample))

# train test split (70/30 split)
# split along axis 0
dataset_X_np_split = np.split(dataset_X_np, [7, 10])
dataset_y_np_split = np.split(replacement_y_np, [7, 10])
print("dataset_X_np_split:", type(dataset_X_np_split))
X_train = dataset_X_np_split[0]
X_test = dataset_X_np_split[1]
y_train = dataset_y_np_split[0]
y_test = dataset_y_np_split[1]

# NEW Normalize X_train and X_test
#(samples, time, channels, rows, cols)
# Loop through each feature
for i in range(X_train.shape[4]):
    print("X_train[:,:,:,;,i].shape: ", X_train[:,:,:,:,i].shape)
    print("X_test[:,:,:,;,i].shape: ", X_test[:,:,:,:,i].shape)
    
    # Replace NaNs with mean or median
    X_train[np.isnan(X_train)] = np.nanmean(X_train[:,:,:,:,i])
    X_test[np.isnan(X_test)] = np.nanmean(X_test[:,:,:,:,i])
    # X_train[np.isnan(X_train)] = np.nanmedian(X_train[:,:,:,:,i])
    # X_test[np.isnan(X_test)] = np.nanmedian(X_test[:,:,:,:,i])
    
    # Standard Scaler
    sc = StandardScaler()
    # Every X_train/test feature will be reshaped to a 2d array
    X_train_2d = X_train[:,:,:,:,i].reshape(X_train.shape[0]*X_train.shape[1], X_train.shape[2]*X_train.shape[3])
    X_test_2d = X_test[:,:,:,:,i].reshape(X_test.shape[0]*X_test.shape[1], X_test.shape[2]*X_test.shape[3])
    # Normalize
    X_train_transformed = sc.fit_transform(X_train_2d)
    X_test_transformed = sc.transform(X_test_2d)
    # Reshape to 4d (samples, time, x, y)
    X_train_transformed = X_train_transformed.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3])
    X_test_transformed = X_test_transformed.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], X_test.shape[3])
    # Store normalized feature in X_train
    X_train[:,:,:,:,i] = X_train_transformed
    X_test[:,:,:,:,i] = X_test_transformed
    

print("X_train: ", X_train.shape)
print("X_test: ", X_test.shape)
print("y_train: ", y_train.shape)
print("y_test: ", y_test.shape)
print("input shape: ", X_train.shape[-4:])

def build_ConvLSTM():
    convlstm = models.Sequential()
    convlstm.add(layers.Input(shape=X_train.shape[-4:]))
    convlstm.add(layers.ConvLSTM2D(filters=256, kernel_size=(5,5), data_format="channels_last", return_sequences=True))
    convlstm.add(layers.BatchNormalization())
    convlstm.add(layers.ConvLSTM2D(filters=128, kernel_size=(3,3), data_format="channels_last", return_sequences=True))
    convlstm.add(layers.BatchNormalization())
    convlstm.add(layers.ConvLSTM2D(filters=64, kernel_size=(2,2), data_format="channels_last", return_sequences=True))
    convlstm.add(layers.BatchNormalization())
    convlstm.add(layers.ConvLSTM2D(filters=32, kernel_size=(1,1), data_format="channels_last", return_sequences=True))
    convlstm.add(layers.Conv3D(filters=1, kernel_size=(1, width_limit-11+4, height_limit-11+4), data_format="channels_last", activation="sigmoid"))
    # https://www.baeldung.com/cs/convolutional-layer-size
    # image width/height - sum(kernel width/height) + num_kernels
    # kernel_size=(timesteps_per_sample, 80, 1246)
    convlstm.compile(
        loss=losses.binary_crossentropy, optimizer=optimizers.Adam(), metrics=[tf.keras.metrics.Accuracy()]
    )
    return convlstm

model = build_ConvLSTM()
print(model.summary())
epochs = 10
batch_size = 1
y_train_test = np.expand_dims(y_train, axis=(2,3,4))
print(y_train_test.shape)
history = model.fit(X_train, y_train_test, epochs=epochs, batch_size=batch_size, verbose=True)

# print model keys
print(history.history.keys())
# accuracy graph
plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('accuracy.png')

# loss graph
plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('loss.png')

# wildfire_da = wildfire_dataset.to_array()
# print(wildfire_da.loc[:, :10])
# print(wildfire_dataset["burned_areas"])
# wf_sub_da = wildfire_dataset["burned_areas"].isel(time=slice(0,10), x=slice(0,5), y=slice(0,10))
# wf_sub_da.plot()
# plt.tight_layout()
# plt.savefig("burned_areas")
# print(wf_sub_da.to_numpy())