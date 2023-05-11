import math
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import layers, models, losses, optimizers
from tensorflow.keras.regularizers import l2
from generator import WildfireSequence
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import warnings
print(tf.config.list_physical_devices('GPU'))


print("If its the first time running this script, select n for the following question")
x=input("Use Loaded Dataset? [Y]/n: ")

if x == "n":
    # import dataset
    datapath = "../wildfire_dataset.nc"
    wildfire_dataset = xr.open_dataset(datapath, engine="netcdf4")
    # print(wildfire_dataset)
    feature_list = wildfire_dataset.data_vars
    feature_nums = len(feature_list)

    # maybe predict more than one thing later, but for not only try to predict burned areas
    remove_label = ["burned_areas", "ignition_points", "number_of_fires", 
                    "POP_DENS_2009", "POP_DENS_2010", "POP_DENS_2011", "POP_DENS_2012", 
                    "POP_DENS_2013", "POP_DENS_2014", "POP_DENS_2015", "POP_DENS_2016", 
                    "POP_DENS_2017", "POP_DENS_2018", "POP_DENS_2019", "POP_DENS_2020", 
                    "POP_DENS_2021", "ROAD_DISTANCE", "avg_d2m", "avg_sp", "fapar", "max_sp",
                    "max_rh", "max_sp", "max_t2m", "max_tp","max_u10", "max_v10", 
                    "max_wind_u10", "max_wind_v10", "min_d2m", "min_rh", "min_sp", "min_t2m",
                    "min_tp", "min_u10", "min_v10"]
    X_label = [label for label in feature_list if label not in remove_label]
    y_label = "burned_areas"
    Xy_label = X_label
    Xy_label.append(y_label)

    # Initialize paramters
    start_timestep = 0
    num_samples = 500
    timesteps_per_sample = 1
    width_limit = 100
    height_limit = 100
    timestep_samples = num_samples*timesteps_per_sample
    train_size = 0.8
    oversampling_factor = 5

    # DATA PROCESSING
    # 1) Train Test Split
    # 2) Convert train, test sets into np array
    # 3) For the train set, weight/multiply the fire dataset and combine it with the non fire dataset (Class weighting)
    # 4) Leave the test set as is (No need for class weighting because only used for testing)

    # partial width/height dataset (limit x and y coords)
    # train_dataset = wildfire_dataset.isel(time=slice(start_timestep, start_timestep+timestep_samples), x=slice(None,width_limit), y=slice(None,height_limit))
    # train_dataset = train_dataset[Xy_label]

    # full width/height dataset
    train_dataset = wildfire_dataset.isel(time=slice(start_timestep, start_timestep+timestep_samples))
    train_dataset = train_dataset[Xy_label]


    # axis_2_size is the timesteps for the dataset
    def dataset_to_np(dataset, axis_2_size):
        # # Create the X into a numpy matrix of shape (time, x, y)
        ds_np = dataset[list(dataset.data_vars)[0]].to_numpy()
        ds_np = np.transpose(ds_np, (0,2,1))
        ds_np = np.expand_dims(ds_np, 3)

        for index, feature in enumerate(list(dataset.data_vars)):
            print(feature)
            if(index > 0):
                # Since wf_dataset_X_np is already initiaklized with the first element, skip
                new_np_arr = dataset[feature].to_numpy()
                # print("new_np_arr shape: ",new_np_arr.shape)
                if(len(new_np_arr.shape) == 2):
                    # If a feature doesn't contain a time dimension (n), we extend the 2d matrix to 3d with copy of matrix n times
                    # Might be able to use numpy broadcast instead
                    new_np_arr = np.repeat(new_np_arr[:, :, np.newaxis], axis_2_size, axis=2)
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
                # print(new_np_arr.shape)
                ds_np = np.concatenate((ds_np, np.expand_dims(new_np_arr, axis=3)), axis=3)
            print(ds_np.shape)
        return ds_np

    # Convert train and test to numpy arrays (time, x, y)
    train_np = dataset_to_np(train_dataset, timestep_samples)

    # Split dataset into wildfire and non-wildfire datasets
    # get indicies of datapoints where wildfire occurred
    # set of 3 arrays where each array represents index of (time, x, y)
    wildfire_indicies = np.where(train_np[:,:,:,-1] == 1.0)
    # print(wildfire_indicies)

    # if fire occured at a timestep, put entire x,y grid into a seperate wildfire dataset
    fire_train = np.take(train_np, np.unique(wildfire_indicies[0]), axis=0)
    nonfire_train = np.delete(train_np, np.unique(wildfire_indicies[0]), axis=0)
    # print(fire_train)
    # print(fire_train.shape)
    # print(nonfire_train)
    # print(nonfire_train.shape)

    num_wildfire_samples = fire_train.shape[0]
    num_non_wildfire_samples = nonfire_train.shape[0]
    if num_wildfire_samples == 0:
            # throw error for not having fire datapoints to weight/multiply
        print("No wildfire data to use")
    else:
        print("=======================Fire Samples===================: ", num_wildfire_samples)

    print("=======================Non Fire Samples===================: ", num_non_wildfire_samples)

    # oversampling wildfire data
    ext_fire_train = np.repeat(fire_train, oversampling_factor, axis=0)


    # combine both wildfire and non wildfire train datasets
    train_dataset = np.concatenate((nonfire_train, ext_fire_train))

    X_train = train_dataset[:,:,:,:-1]
    y_train = train_dataset[:,:,:,-1]


    # Handle Test
    actual_train_size = train_dataset.shape[0]
    total_train_test_size = int(actual_train_size/train_size)
    actual_test_size = total_train_test_size-actual_train_size
    print("actual_train_size: ", actual_train_size)
    print("total_train_test_size: ", total_train_test_size)
    print("actual_test_size: ", actual_test_size)

    # full width/height test dataset
    test_dataset = wildfire_dataset.isel(time=slice(start_timestep+timestep_samples, start_timestep+timestep_samples+actual_test_size))
    test_dataset = test_dataset[Xy_label]

    # partial width/height test dataset
    # test_dataset = wildfire_dataset.isel(time=slice(start_timestep+timestep_samples, start_timestep+timestep_samples+actual_test_size), x=slice(None, width_limit), y=slice(None,height_limit))
    # test_dataset = test_dataset[Xy_label]

    test_np = dataset_to_np(test_dataset, actual_test_size)


    X_test = test_np[:,:,:,:-1]
    y_test = test_np[:,:,:,-1]


    print("X_train: ", X_train.shape)
    print("X_test: ", X_test.shape)
    print("y_train: ", y_train.shape)
    print("y_test: ", y_test.shape)

    def normalize(X_train, X_test):
        # NEW Normalize X_train and X_test
        #(samples, time, rows, cols, channels)
        # Loop through each feature
        for i in range(X_train.shape[4]):
            # print("X_train[:,:,:,;,i].shape: ", X_train[:,:,:,:,i].shape)
            # print("X_test[:,:,:,;,i].shape: ", X_test[:,:,:,:,i].shape)
            print("Normalizing " + str(i) + " out of " + str(X_train.shape[4]-1))

            # Replace NaNs with mean or median
            X_train[np.isnan(X_train)] = np.nanmean(X_train[:,:,:,:,i])
            X_test[np.isnan(X_test)] = np.nanmean(X_test[:,:,:,:,i])
            # X_train[np.isnan(X_train)] = np.nanmedian(X_train[:,:,:,:,i])
            # X_test[np.isnan(X_test)] = np.nanmedian(X_test[:,:,:,:,i])

            # Standard Scaler
            # sc = StandardScaler()
            # Every X_train/test feature will be reshaped to a 2d array
            # X_train_2d = X_train[:,:,:,:,i].reshape(X_train.shape[0]*X_train.shape[1], X_train.shape[2]*X_train.shape[3])
            # X_test_2d = X_test[:,:,:,:,i].reshape(X_test.shape[0]*X_test.shape[1], X_test.shape[2]*X_test.shape[3])
            # Normalize
            X_train_norm = X_train
            X_test_norm = X_test
            X_train_norm[:,:,:,:,i] = (X_train[:,:,:,:,i] - X_train[:,:,:,:,i].mean())/(X_train[:,:,:,:,i].std())
            X_test_norm[:,:,:,:,i] = (X_test[:,:,:,:,i] - X_train[:,:,:,:,i].mean())/(X_train[:,:,:,:,i].std())
            print("mean: ", X_train[:,:,:,:,i].mean())
            print("std: ", X_train[:,:,:,:,i].std())

        return X_train_norm, X_test_norm
    
    X_train = np.reshape(X_train, (actual_train_size, timesteps_per_sample, X_train.shape[1], X_train.shape[2], X_train.shape[3]))
    X_test = np.reshape(X_test, (actual_test_size, timesteps_per_sample, X_test.shape[1], X_test.shape[2], X_test.shape[3]))
    y_train = np.reshape(y_train, (actual_train_size, timesteps_per_sample, y_train.shape[1], y_train.shape[2]))
    y_test = np.reshape(y_test, (actual_test_size, timesteps_per_sample, y_test.shape[1], y_test.shape[2]))

    X_train_norm, X_test_norm = normalize(X_train, X_test)
    

    # save numpy array in file
    np.save("X_train_norm_500", X_train_norm)
    np.save("X_test_norm_500", X_test_norm)
    np.save("y_train_500", y_train)
    np.save("y_test_500", y_test)
else:
    X_train_norm = np.load("X_train_norm_500.npy")
    X_test_norm = np.load("X_test_norm_500.npy")
    y_train = np.load("y_train_500.npy")
    y_test = np.load("y_test_500.npy")
print("X_train: ", X_train_norm.shape)
print("X_test: ", X_test_norm.shape)
print("y_train: ", y_train.shape)
print("y_test: ", y_test.shape)
print("input shape: ", X_train_norm.shape[-4:])

# check if data still has NaNs
if np.isnan(X_train_norm).any():
    print("X_train_norm has nans")
if np.isnan(X_test_norm).any():
    print("X_test_norm has nans")
if np.isnan(y_train).any():
    print("y_train has nans")
if np.isnan(y_test).any():
    print("y_test has nans")

    # convlstm.add(layers.ConvLSTM2D(filters=128, kernel_size=(5,5), padding="same", data_format="channels_last", activation="relu", return_sequences=True))
    # convlstm.add(layers.BatchNormalization())
    # convlstm.add(layers.ConvLSTM2D(filters=64, kernel_size=(3,3), padding="same", data_format="channels_last", activation="relu", return_sequences=True))
    # convlstm.add(layers.BatchNormalization())
    
    # convlstm.add(layers.ConvLSTM2D(filters=32, kernel_size=(5,5), padding="same", data_format="channels_last", activation="relu", return_sequences=True))
    # convlstm.add(layers.BatchNormalization())
    # https://keras.io/api/layers/recurrent_layers/time_distributed/


def build_ConvLSTM():
    convlstm = models.Sequential()
    convlstm.add(layers.Input(shape=X_train_norm.shape[-4:]))
    # convlstm.add(layers.ConvLSTM2D(filters=32, kernel_size=(5,5), padding="same", data_format="channels_last", activation="relu", return_sequences=True))
    # convlstm.add(layers.BatchNormalization())
    convlstm.add(layers.ConvLSTM2D(filters=16, kernel_size=(3,3), padding="same", data_format="channels_last", activation="relu", return_sequences=True))
    convlstm.add(layers.BatchNormalization())
    convlstm.add(layers.ConvLSTM2D(filters=8, kernel_size=(2,2), padding="same", data_format="channels_last", activation="relu", return_sequences=True))
    convlstm.add(layers.BatchNormalization())
    convlstm.add(layers.Conv3D(filters=1, kernel_size=(2, 2, 2), padding="same", data_format="channels_last", activation="sigmoid"))
    convlstm.compile(
        loss=losses.binary_crossentropy, optimizer=optimizers.Adam(), metrics=[tf.keras.metrics.BinaryAccuracy()]
    )
    return convlstm

model = build_ConvLSTM()
print(model.summary())
epochs = 25
batch_size = 1
# history = model.fit(X_train_norm, y_train, validation_data=(X_test_norm,y_test), epochs=epochs, batch_size=batch_size, verbose=True)
history = model.fit(WildfireSequence(X_train_norm, y_train, 1), validation_data=(X_test_norm,y_test), epochs=epochs, batch_size=batch_size, verbose=True)

# Evaluation Metrics

threshold = 0.5

y_hat = model.predict(X_train_norm)
print(y_hat.shape)
print(y_train.shape)
y_hat_mod = np.where(y_hat > threshold, 1,0)
print(y_hat_mod.flatten().shape)
print(y_train.flatten().shape)

conf_mat = confusion_matrix(y_train.flatten(), y_hat_mod.flatten())
disp = ConfusionMatrixDisplay(conf_mat)
disp.plot()
plt.title('training confusion matrix')
plt.savefig('conf_mat_train.png')
plt.close()

y_hat = model.predict(X_test_norm)
print(y_hat.shape)
print(y_test.shape)
y_hat_mod = np.where(y_hat > threshold, 1,0)
print(y_hat_mod.flatten().shape)
print(y_test.flatten().shape)

conf_mat = confusion_matrix(y_test.flatten(), y_hat_mod.flatten())
disp = ConfusionMatrixDisplay(conf_mat)
disp.plot()
plt.title('testing confusion matrix')
plt.savefig('conf_mat_test.png')
plt.close()


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



# wildfire_da = wildfire_dataset.to_array()
# print(wildfire_da.loc[:, :10])
# print(wildfire_dataset["burned_areas"])
# wf_sub_da = wildfire_dataset["burned_areas"].isel(time=slice(0,10), x=slice(0,5), y=slice(0,10))
# wf_sub_da.plot()
# plt.tight_layout()
# plt.savefig("burned_areas")
# print(wf_sub_da.to_numpy())