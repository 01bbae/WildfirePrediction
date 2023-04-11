import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import layers, models, losses, optimizers
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
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
timesteps = 100
wf_dataset_head = wildfire_dataset.head(indexers={"time": timesteps})

############# Normalize Data Here ####################

wf_dataset_X = wf_dataset_head[X_label]
wf_ds_norm_X = wf_dataset_X # REPLACE LATER WITH NORMALIZED/CORRECTLY STRUCTURED DATA

wf_dataset_y = wf_dataset_head[y_label] 
wf_ds_norm_y = wf_dataset_y # REPLACE LATER WITH NORMALIZED/CORRECTLY STRUCTURED DATA

######################################################

# Create the y into a numpy matrix of shape (time, x, y)
wf_ds_norm_y_np = wf_ds_norm_y.to_numpy()
wf_ds_norm_y_np = np.transpose(wf_ds_norm_y_np, (0,2,1))

# Create the X into a numpy matrix of shape (time, x, y)
wf_ds_norm_X_np = wf_ds_norm_X[list(wf_ds_norm_X.data_vars)[0]].to_numpy()
wf_ds_norm_X_np = np.transpose(wf_ds_norm_X_np, (0,2,1))
wf_ds_norm_X_np = np.expand_dims(wf_ds_norm_X_np, 3)

# Takes each feature of the xarray Dataset and converts it into a DataArray 
# Also appends it into the new np array to make shape of (time x, y, features)
for index, feature in enumerate(list(wf_ds_norm_X.data_vars)):
    print(feature)
    print(wf_ds_norm_X_np.shape)
    if(index!=0):
        # Since wf_ds_norm_X_np is already initiaklized with the first element, skip
        new_np_arr = wf_ds_norm_X[feature].to_numpy()
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
        wf_ds_norm_X_np = np.concatenate((wf_ds_norm_X_np, np.expand_dims(new_np_arr, axis=3)), axis=3)

# rearrange so that it matches what keras expects (time, features, x, y) instead of (time, x, y, features)
wf_ds_norm_X_np = np.moveaxis(wf_ds_norm_X_np, 3, 1)
print("wf_ds_norm_X_np.shape", wf_ds_norm_X_np.shape)
print("wf_ds_norm_y_np.shape", wf_ds_norm_y_np.shape)

if 1 in wf_ds_norm_y_np:
    print("Fire exists") 
print("Output classes: ", np.unique(wf_ds_norm_y_np))
# class_weights = class_weight.compute_class_weight(class_weight = "balanced", classes = np.unique(wf_ds_norm_y_np), y = wf_ds_norm_y_np)
# print(class_weights)

# Create samples (samples, time, features, x, y)
# Each samples are 4 days with 2 day overlap between each one
wf_ds_norm_X_np = np.expand_dims(wf_ds_norm_X_np, axis=0)


# train test split

# ind = np.random.choice(range(wf_ds_norm_X_np.shape[0]), size=(5000,), replace=False)

# train_X, test_X, train_y, test_y = train_test_split(wf_experimental_X_np, test_size = 0.20, random_state = 1)
# print(wf_experimental_X_np.shape)
# print("train_X", train_X.shape)
# print("test_X", test_X.shape)
# print("train_y", train_y.shape)
# print("test_y", test_y.shape)


# print("X dimensions:" , wf_ds_norm_X_np.dims)
# (None, wf_ds_norm_X_np.dims["time"], wf_ds_norm_X_np.dims["x"], wf_ds_norm_X_np.dims["y"], len(wf_ds_norm_X_np.data_vars))

def build_ConvLSTM():
    print(wf_ds_norm_X_np.shape)
    convlstm = models.Sequential()
    convlstm.add(layers.Input(shape=wf_ds_norm_X_np.shape[-4:]))
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
wf_ds_norm_y_np = np.expand_dims(wf_ds_norm_y_np, axis=0)
# history = model.fit(wf_ds_norm_X_np, wf_ds_norm_y_np, epochs=epochs, verbose=True)



# test plots
# wildfire_da = wildfire_dataset.to_array()
# print(wildfire_da.loc[:, :10])
# print(wildfire_dataset["burned_areas"])
# wf_sub_da = wildfire_dataset["burned_areas"].isel(time=slice(0,10), x=slice(0,5), y=slice(0,10))
# wf_sub_da.plot()
# plt.tight_layout()
# plt.savefig("burned_areas")
# print(wf_sub_da.to_numpy())