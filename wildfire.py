import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import layers, models, losses, optimizers
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
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
timesteps = 5
wf_experimental = wildfire_dataset.head(indexers={"time": timesteps})
# print(wf_experimental)
wf_experimental_X_ds = wf_experimental[X_label]
wf_experimental_X = wf_experimental_X_ds # REPLACE LATER WITH NORMALIZED/CORRECTLY STRUCTURED DATA
# wf_experimental_X = wf_experimental_X_ds.to_numpy()
print(wf_experimental_X)
wf_experimental_y_ds = wf_experimental[y_label] 
wf_experimental_y = wf_experimental_y_ds # REPLACE LATER WITH NORMALIZED/CORRECTLY STRUCTURED DATA
# print(wf_experimental_y)

wf_experimental_y_np = wf_experimental_y.to_numpy()
wf_experimental_y_np = np.transpose(wf_experimental_y_np, (0,2,1))
print(wf_experimental_y_np.shape)

# Create the X into a numpy matrix
wf_experimental_X_np = wf_experimental_X[list(wf_experimental_X.data_vars)[0]].to_numpy()
wf_experimental_X_np = np.transpose(wf_experimental_X_np, (0,2,1))
wf_experimental_X_np = np.expand_dims(wf_experimental_X_np, 3)
for index, feature in enumerate(list(wf_experimental_X.data_vars)):
    print(feature)
    print(wf_experimental_X_np.shape)
    if(index!=0):
        # Since wf_experimental_X_np is already initiaklized with the first element, skip
        new_np_arr = wf_experimental_X[feature].to_numpy()
        if(len(new_np_arr.shape) == 2):
            # If a feature doesn't contain a time dimension (n), we extend the 2d matrix to 3d with copy of matrix n times
            # Might be able to use numpy broadcast instead
            new_np_arr = np.repeat(new_np_arr[:, :, np.newaxis], timesteps, axis=2)
            # Transpose feature to "time", "x", "y" format
            new_np_arr = np.transpose(new_np_arr)
        else:
            # Transpose feature to "time", "x", "y" format
            new_np_arr = np.transpose(new_np_arr, (0,2,1))
        print(new_np_arr.shape)
        if (np.isnan(new_np_arr).all()):
            # Precaution to alert if a feature has all NaN values
            warnings.warn(str(feature) + " feature's values are all NaNs")
        wf_experimental_X_np = np.concatenate((wf_experimental_X_np, np.expand_dims(new_np_arr, axis=3)), axis=3)

print(wf_experimental_X_np.shape)


# print("X dimensions:" , wf_experimental_X.dims)
# (None, wf_experimental_X.dims["time"], wf_experimental_X.dims["x"], wf_experimental_X.dims["y"], len(wf_experimental_X.data_vars))
def build_ConvLSTM():
    convlstm = models.Sequential()
    convlstm.add(layers.Input(shape=wf_experimental_X_np.shape))
    convlstm.add(layers.ConvLSTM2D(filters=256, kernel_size=(5,5), return_sequences=True))
    convlstm.add(layers.BatchNormalization())
    convlstm.add(layers.ConvLSTM2D(filters=128, kernel_size=(3,3), return_sequences=True))
    convlstm.add(layers.BatchNormalization())
    convlstm.add(layers.ConvLSTM2D(filters=64, kernel_size=(2,2), return_sequences=True))
    convlstm.add(layers.BatchNormalization())
    convlstm.add(layers.ConvLSTM2D(filters=32, kernel_size=(1,1), return_sequences=True))
    convlstm.add(layers.Conv3D(filters=1, kernel_size=(3, 3, 3), activation="sigmoid"))
    convlstm.compile(
        loss=losses.binary_crossentropy, optimizer=optimizers.Adam(),
    )
    return convlstm

model = build_ConvLSTM()
print(model.summary())
epochs = 20
batch_size = 1
model.fit(wf_experimental_X_np, wf_experimental_y_np, epochs=epochs, verbose=True)

# epochs = 20
# batch_size = 1


# wildfire_da = wildfire_dataset.to_array()
# print(wildfire_da.loc[:, :10])
# print(wildfire_dataset["burned_areas"])
# wf_sub_da = wildfire_dataset["burned_areas"].isel(time=slice(0,10), x=slice(0,5), y=slice(0,10))
# wf_sub_da.plot()
# plt.tight_layout()
# plt.savefig("burned_areas")
# print(wf_sub_da.to_numpy())