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
# print(tf.config.list_physical_devices('GPU'))
print(tf.config.list_physical_devices('GPU'))

# import dataset
datapath = "../wildfire_dataset.nc"
wildfire_dataset = xr.open_dataset(datapath, engine="netcdf4")
# print(wildfire_dataset)
feature_list = wildfire_dataset.data_vars
feature_nums = len(feature_list)

# maybe predict more than one thing later, but for not only try to predict burned areas
remove_label = ["burned_areas", "ignition_points", "number_of_fires"]
X_label = [label for label in feature_list if label not in remove_label]
y_label = "burned_areas"


# for features in feature_list:
#     features_da = wildfire_dataset[features]
#     if "time" in features_da.coords:
#         # Process features that have a time dimension
#         print("time in feature: " + features)
        
#     else:
#         # Process features that don't have a time dimension
#         print("no time in feature: " + features)
    # wildfire_dataset[features].isel()

# take the first 5 time steps for all x and y to try creating a smaller dataset
num_samples = 10
timesteps_per_sample = 10
timesteps = num_samples*timesteps_per_sample
wf_dataset_head = wildfire_dataset.head(indexers={"time": timesteps})

############# Normalize Data Here ####################

wf_dataset_X = wf_dataset_head[X_label]
wf_ds_norm_X = wf_dataset_X # REPLACE LATER WITH NORMALIZED/CORRECTLY STRUCTURED DATA

wf_dataset_y = wf_dataset_head[y_label] 
wf_ds_norm_y = wf_dataset_y # REPLACE LATER WITH NORMALIZED/CORRECTLY STRUCTURED DATA

######################################################

# Create the y into a numpy matrix of shape (time, x, y)
wf_dataset_y_np = wf_dataset_y.to_numpy()
wf_dataset_y_np = np.transpose(wf_dataset_y_np, (0,2,1))

# Create the X into a numpy matrix of shape (time, x, y)
wf_dataset_X_np = wf_dataset_X[list(wf_dataset_X.data_vars)[0]].to_numpy()
wf_dataset_X_np = np.transpose(wf_dataset_X_np, (0,2,1))
wf_dataset_X_np = np.expand_dims(wf_dataset_X_np, 3)


print("data vars:", type(wf_dataset_X.data_vars))
# Takes each feature of the xarray Dataset and converts it into a DataArray 
# Also appends it into the new np array to make shape of (time x, y, features)
for index, feature in enumerate(list(wf_dataset_X.data_vars)):
    if(index!=0):
        # Since wf_dataset_X_np is already initiaklized with the first element, skip
        new_np_arr = wf_dataset_X[feature].to_numpy()
        if(len(new_np_arr.shape) == 2):
            # If a feature doesn't contain a time dimension (n), we extend the 2d matrix to 3d with copy of matrix n times
            # Might be able to use numpy broadcast instead
            new_np_arr = np.repeat(new_np_arr[:, :, np.newaxis], timesteps, axis=2)
            # Transpose feature to "time", "x", "y" format
            new_np_arr = np.transpose(new_np_arr)
        else:
            # Transpose feature to "time", "x", "y" format
            new_np_arr = np.transpose(new_np_arr, (0,2,1))
        if (np.isnan(new_np_arr).all()):
            # Precaution to alert if a feature has all NaN values
            warnings.warn(str(feature) + " feature's values are all NaNs")
        wf_dataset_X_np = np.concatenate((wf_dataset_X_np, np.expand_dims(new_np_arr, axis=3)), axis=3)
    print(feature)
    print(wf_dataset_X_np.shape)

# rearrange so that it matches what keras expects (time, features, x, y) instead of (time, x, y, features)
wf_dataset_X_np = np.moveaxis(wf_dataset_X_np, 3, 1)
print("wf_dataset_X_np.shape", wf_dataset_X_np.shape)
print("wf_dataset_y_np.shape", wf_dataset_y_np.shape)

if 1 in wf_dataset_y_np:
    print("Fire exists") 
print("Output classes: ", np.unique(wf_dataset_y_np))
# class_weights = class_weight.compute_class_weight(class_weight = "balanced", classes = np.unique(wf_dataset_y_np), y = wf_dataset_y_np)
# print(class_weights)

# Create samples (samples, time, features, x, y)
# Each samples are 4 days with 2 day overlap between each one
wf_dataset_X_np = np.expand_dims(wf_dataset_X_np, axis=0)
print("wf_dataset_X_np:", type(wf_dataset_X_np))
print("wf_dataset_X_np(expand dims):", wf_dataset_X_np)
wf_dataset_X_np = np.reshape(wf_dataset_X_np, (num_samples, timesteps_per_sample, wf_dataset_X_np.shape[2], wf_dataset_X_np.shape[3], wf_dataset_X_np.shape[4]))
print("wf_dataset_X_np(reshape):", type(wf_dataset_X_np))

print(wf_dataset_X_np.shape)

# train test split
# split along axis 0
wf_dataset_X_np_split = np.split(wf_dataset_X_np, [7, 10])
wf_dataset_y_np_split = np.split(wf_dataset_y_np, [7, 10])
print("wf_dataset_X_np_split:", type(wf_dataset_X_np_split))
X_train = wf_dataset_X_np_split[0]
X_test = wf_dataset_X_np_split[1]
y_train = wf_dataset_y_np_split[0]
y_test = wf_dataset_y_np_split[1]
print("X_train: ", X_train.shape)
print("X_test: ", X_test.shape)
print("y_train: ", y_train.shape)
print("y_test: ", y_test.shape)

# Normalize X_train and X_test
# Combine samples and time
X_train = X_train.reshape(70, 87, 1253, 983)
print("X_train: ", X_train.shape)
X_test = X_test.reshape(30, 87, 1253, 983)
print("X_test: ", X_test.shape)

# Loop through each feature
for i in range(X_train.shape[1]):
    print("X_train[:,i,:,;].shape: ", X_train[:,i,:,:].shape)
    print("X_test[:,i,:,;].shape: ", X_test[:,i,:,:].shape)
    # Standard Scaler
    sc = StandardScaler()
    # Every X_train/test feature will be reshaped to a 2d array
    X_train_2d = X_train[:, i, :, :].reshape(70, 1253*983)
    X_test_2d = X_test[:, i, :, :].reshape(30, 1253*983)
    # Normalize
    X_train_transformed = sc.fit_transform(X_train_2d)
    X_test_transformed = sc.transform(X_test_2d)
    # Reshape back to 3d
    X_train_transformed = X_train_transformed.reshape(70, 1253, 983)
    X_test_transformed = X_test_transformed.reshape(30, 1253, 983)
    # Store normalized feature in X_train
    X_train[:, i, :, :] = X_train_transformed
    X_test[:, i, :, :] = X_test_transformed

# Reshape X_train to 5d for keras
X_train = X_train.reshape(7, 10, 87, 1253, 983)
print("Shape: ", X_train.shape)
# Reshape X_test
X_test = X_test.reshape(3, 10, 87, 1253, 983)
print("Shape: ", X_test.shape)

def build_ConvLSTM():
    convlstm = models.Sequential()
    convlstm.add(layers.Input(shape=X_train.shape[-4:]))
    convlstm.add(layers.ConvLSTM2D(filters=256, kernel_size=(5,5), return_sequences=True))
    convlstm.add(layers.BatchNormalization())
    convlstm.add(layers.ConvLSTM2D(filters=128, kernel_size=(3,3), return_sequences=True))
    convlstm.add(layers.BatchNormalization())
    convlstm.add(layers.ConvLSTM2D(filters=64, kernel_size=(2,2), return_sequences=True))
    convlstm.add(layers.BatchNormalization())
    convlstm.add(layers.ConvLSTM2D(filters=32, kernel_size=(1,1), return_sequences=True))
    convlstm.add(layers.Conv3D(filters=1, kernel_size=(5, 80, 1246), activation="sigmoid"))
    convlstm.compile(
        loss=losses.binary_crossentropy, optimizer=optimizers.Adam(),
        # metrics=[tf.keras.metrics.Accuracy()]
    )
    return convlstm

model = build_ConvLSTM()
print(model.summary())
epochs = 10
batch_size = 1

model.fit(X_train, y_train, epochs=10, validation_data = (X_test, y_test))

# wf_dataset_y_np = np.expand_dims(wf_ds_norm_y_np, axis=0)

# test plots
# wildfire_da = wildfire_dataset.to_array()
# print(wildfire_da.loc[:, :10])
# print(wildfire_dataset["burned_areas"])
# wf_sub_da = wildfire_dataset["burned_areas"].isel(time=slice(0,10), x=slice(0,5), y=slice(0,10))
# wf_sub_da.plot()
# plt.tight_layout()
# plt.savefig("burned_areas")
# print(wf_sub_da.to_numpy())