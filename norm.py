import numpy as np

def normalize(X, train_mean=None, train_std=None):
    # Normalize X features
    #(samples, time, rows, cols, channels)
    # Loop through each feature


    # If normalizing the training set, don't provide train_mean and train_std
    # If normalizing the testing set, provide train_mean and train_std after normalizing training set

    # Probably better to create fit_transform and transform methods but this is a quick solution
    if train_mean is None and train_std is None:
        # fit_transform() for train set
        train_mean = []
        train_std = []
        for i in range(X.shape[4]):
            print("Normalizing (fit_transform()) " + str(i) + " out of " + str(X.shape[4]-1))

            # Replace NaNs with mean or median
            X[np.isnan(X)] = np.nanmean(X[:,:,:,:,i])
            # X_train[np.isnan(X_train)] = np.nanmedian(X_train[:,:,:,:,i])

            # Normalize
            X_norm = np.empty(X.shape)
            mean = X[:,:,:,:,i].mean()
            std = X[:,:,:,:,i].std()

            X_norm[:,:,:,:,i] = (X[:,:,:,:,i] - mean)/std
            print(mean)
            print(std)
            train_mean.append(mean)
            train_std.append(std)

        return X_norm, np.array(train_mean), np.array(train_std)
    else:
        # transform() for test set
        for i in range(X.shape[4]):
            print("Normalizing (transform()) " + str(i) + " out of " + str(X.shape[4]-1))

            # Replace NaNs with mean or median
            X[np.isnan(X)] = np.nanmean(X[:,:,:,:,i])
            # X_train[np.isnan(X_train)] = np.nanmedian(X_train[:,:,:,:,i])

            # Normalize
            X_norm = np.empty(X.shape)
            X_norm[:,:,:,:,i] = (X[:,:,:,:,i] - train_mean[i])/(train_std[i])

        return X_norm