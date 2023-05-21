import os
from pathlib import Path
import numpy as np
import xarray as xr
import warnings
from norm import normalize
import gc

# This is the Data Processing Module
# Run this before running the model in wildfire.py
# This module will save the datasets into files to feed to the model during training
# The generator.py is the tool that feeds the data and it does not need to be run directly


# Note that the hardware we were working with was with 2TiB of data
# so we offload train and test data into functions, save the results, 
# then we garbage collect the train data to have space for our test

# import dataset
datapath = "../wildfire_dataset.nc"

current_directory = os.getcwd()

# initialize path to save files
train_path = os.path.join(current_directory, 'train')
test_path = os.path.join(current_directory, 'test')
print(train_path)
if not os.path.exists(train_path):
    os.mkdir('train')
if not os.path.exists(test_path):
    os.mkdir('test')

wildfire_dataset = xr.open_dataset(datapath, engine="netcdf4", chunks={"time": 10})
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
# try keep timesteps_per_sample equal to oversampling factor to not have conflicts while reshaping (limitation of the code)
start_timestep = 80
num_samples = 100
timesteps_per_sample = 5
width_limit = 100
height_limit = 100
timestep_samples = num_samples*timesteps_per_sample
train_size = 0.8
oversampling_factor = 5

# # DATA PROCESSING
# # 1) Train Test Split
# # 2) Convert train, test sets into np array
# # 3) For the train set, weight/multiply the fire dataset and combine it with the non fire dataset (Class weighting)
# # 4) Leave the test set as is (No need for class weighting because only used for testing)

# # partial width/height dataset (limit x and y coords)
# # train_dataset = wildfire_dataset.isel(time=slice(start_timestep, start_timestep+timestep_samples), x=slice(None,width_limit), y=slice(None,height_limit))
# # train_dataset = train_dataset[Xy_label]

# # full width/height dataset
train_dataset = wildfire_dataset.isel(time=slice(start_timestep, start_timestep+timestep_samples))
train_dataset = train_dataset[Xy_label]


# axis_2_size is the total number of timesteps for the dataset
def dataset_to_np(dataset, timestep_samples):
    # # Create the X into a numpy matrix of shape (time, x, y)
    ds_np = dataset[list(dataset.data_vars)[0]].to_numpy()
    ds_np = np.transpose(ds_np, (0,2,1))
    ds_np = np.expand_dims(ds_np, axis=3)

    for index, feature in enumerate(list(dataset.data_vars)):
        print(feature)
        if(index > 0):
            # Since wf_dataset_X_np is already initiaklized with the first element, skip
            new_np_arr = dataset[feature].to_numpy()
            # print("new_np_arr shape: ",new_np_arr.shape)
            if(len(new_np_arr.shape) == 2):
                print("feature has no time dimension")
                # If a feature doesn't contain a time dimension (n), we extend the 2d matrix to 3d with copy of matrix n times
                # where n = timestep_samples = num_samples*timesteps_per_samples
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
            print(new_np_arr.shape)
            print(ds_np.shape)
            ds_np = np.concatenate((ds_np, np.expand_dims(new_np_arr, axis=3)), axis=3)
        print(ds_np.shape)
    return ds_np

# # Convert train and test to numpy arrays (time, x, y, features)
train_np = dataset_to_np(train_dataset, timestep_samples)

train_np = np.reshape(train_np, (num_samples, timesteps_per_sample, train_np.shape[1], train_np.shape[2], train_np.shape[3]))

# Split dataset into wildfire and non-wildfire datasets
# get indicies of datapoints where wildfire occurred
# set of 3 arrays where each array represents index of (time, x, y)
# IN FUTURE REVISIONS, OVERSAMPLE SAMPLES AND NOT THE TIMESTEPS
# this current method creates flaws in our samples later
wildfire_indicies = np.where(train_np[:,:,:,:,-1] == 1.0)

# if fire occured at a timestep, put entire x,y grid into a seperate wildfire dataset
fire_train = np.take(train_np, np.unique(wildfire_indicies[0]), axis=0)

num_oversampled_wildfire_samples = fire_train.shape[0]
original_samples = train_np.shape[0]
if num_oversampled_wildfire_samples == 0:
        # throw error for not having fire datapoints to weight/multiply
    print("No wildfire data to use")
else:
    print("=======================Fire Samples===================: ", num_oversampled_wildfire_samples)

print("=======================Original Samples===================: ", original_samples)

# oversampling wildfire data
ext_fire_train = np.repeat(fire_train, oversampling_factor, axis=0)

# combine both wildfire and non wildfire train datasets
# shuffle
train_dataset = np.concatenate((train_np, ext_fire_train))
np.random.shuffle(train_dataset)
print(train_dataset.shape)
X_train = train_dataset[:,:,:,:,:-1]
y_train = train_dataset[:,:,:,:,-1]

# not creating X_train variable to save memory
X_train_norm, X_train_mean, X_train_std = normalize(X_train)
np.save(os.path.join(train_path, "X_train_mean"), X_train_mean)
np.save(os.path.join(train_path, "X_train_std"),X_train_std)

del X_train
gc.collect()
print("gc X_train")


# check if data still has NaNs (should be removed through normalized function)
if np.isnan(X_train_norm).any():
    print("X_train_norm has nans")
if np.isnan(y_train).any():
    print("y_train has nans")

num_train_samples = X_train_norm.shape[0]

# X_train_norm = np.reshape(X_train_norm, (int(num_train_samples/timesteps_per_sample), timesteps_per_sample, X_train_norm.shape[1], X_train_norm.shape[2], X_train_norm.shape[3]))
# y_train = np.reshape(y_train, (int(num_train_samples/timesteps_per_sample), timesteps_per_sample, y_train.shape[1], y_train.shape[2]))

# save train np arrays to files by batches (axis=0)
for i in range(X_train_norm.shape[0]):
    np.save(os.path.join(train_path, "X_train_norm_sample_"+str(i)), X_train_norm[i,:,:,:,:])
for i in range(y_train.shape[0]):
    np.save(os.path.join(train_path, "y_train_sample_"+str(i)), y_train[i,:,:,:])

print('train data saved at ' + str(train_path))

X_train_norm_shape = X_train_norm.shape
y_train_shape = y_train.shape

del X_train_norm
del y_train
gc.collect()
print("gc X_train_norm")
print("gc y_train")

# ###################################################
# X_train_norm_shape = (280,5,1253,983,55)
# y_train_shape = (280,5,1253,983)
# num_train_samples = int((len(os.listdir(train_path))-2)/2)

# X_train_mean = np.load(os.path.join(train_path, "X_train_mean.npy"))
# X_train_std = np.load(os.path.join(train_path, "X_train_std.npy"))

# ###################################################

# Handle Test size
# Make sure that the test size is divisible by timesteps_per_sample to result in integer batch size
# If test size is not divisible by timesteps_per_sample, increase the test size until it is.
total_train_test_size = int(num_train_samples/train_size)
actual_test_size = (total_train_test_size-num_train_samples) + (timesteps_per_sample - ((total_train_test_size-num_train_samples) % timesteps_per_sample))
start_slice = start_timestep+timestep_samples
end_slice = start_timestep+timestep_samples+(actual_test_size*timesteps_per_sample)
timestep_samples_test = end_slice - start_slice

print("num_train_samples: ", num_train_samples)
print("actual_test_size: ", actual_test_size)
print("total_train_test_size: ", total_train_test_size)
print("start slice: ", str(start_slice))
print("end slice: ", str(end_slice))
print("total slice/timestep_samples_test: ", timestep_samples_test)

# full width/height test dataset
test_dataset = wildfire_dataset.isel(time=slice(start_slice, end_slice))
test_dataset = test_dataset[Xy_label]

# partial width/height test dataset
# test_dataset = wildfire_dataset.isel(time=slice(start_timestep+timestep_samples, start_timestep+timestep_samples+actual_test_size), x=slice(None, width_limit), y=slice(None,height_limit))
# test_dataset = test_dataset[Xy_label]

test_np = dataset_to_np(test_dataset, timestep_samples_test)
print(test_np.shape)
test_np = np.reshape(test_np, (actual_test_size, timesteps_per_sample, test_np.shape[1], test_np.shape[2], test_np.shape[3]))

np.random.shuffle(test_np)

X_test = test_np[:,:,:,:,:-1]
y_test = test_np[:,:,:,:,-1]
X_test_norm = normalize(X_test, X_train_mean, X_train_std)

del X_test
gc.collect()
print("gc X_test")

# X_test_norm = np.reshape(X_test_norm, (int(actual_test_size/timesteps_per_sample), timesteps_per_sample, X_test_norm.shape[1], X_test_norm.shape[2], X_test_norm.shape[3]))
# y_test = np.reshape(y_test, (actual_test_size, timesteps_per_sample, y_test.shape[1], y_test.shape[2]))

# check if data still has NaNs (should be removed through normalized function)
if np.isnan(X_test_norm).any():
    print("X_test_norm has nans")
if np.isnan(y_test).any():
    print("y_test has nans")


# save test np arrays to files by batches (axis=0)
for i in range(X_test_norm.shape[0]):
    np.save(os.path.join(test_path, "X_test_norm_sample_"+str(i)), X_test_norm[i,:,:,:,:])
for i in range(y_test.shape[0]):
    np.save(os.path.join(test_path, "y_test_sample_"+str(i)), y_test[i,:,:,:])


X_test_norm_shape = X_test_norm.shape
y_test_shape = y_test.shape

del X_test_norm
del y_test
gc.collect()
print("gc X_test_norm")
print("gc y_test")


# save dimension metadata about the np arrays saved
metadata = open("metadata.txt","w")
metadata.write(str(num_samples)+'\n')
metadata.write(str(timesteps_per_sample)+'\n')
for i in range(len(X_train_norm_shape)):
    metadata.write(str(X_train_norm_shape[i])+'\n')
for i in range(len(X_test_norm_shape)):
    metadata.write(str(X_test_norm_shape[i])+'\n')
for i in range(len(y_train_shape)):
    metadata.write(str(y_train_shape[i])+'\n')
for i in range(len(y_test_shape)):
    metadata.write(str(y_test_shape[i])+'\n')
metadata.close()