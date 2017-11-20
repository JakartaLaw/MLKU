import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%

class LoadData:

    def __init__(self):
        pass

    @staticmethod
    def DanWood():


        """load DanWood dataset

        Returns
        =======

        DanWood (pd.DataFrame)

        """
        return pd.read_table("DanWood.dt",sep=" ",names=["x","y"])

    @staticmethod
    def MNIST():

        """ Load MNIST dataset

        Returns
        =======
        df_train, df_test (pd.DataFrame)

        """

        #For training set
        training_features = np.genfromtxt("MNIST-cropped-txt//MNIST-Train-cropped.txt")
        training_features = training_features.reshape(10000,784)

        training_labels = np.genfromtxt("MNIST-cropped-txt//MNIST-Train-Labels-cropped.txt")
        training_labels = training_labels.reshape(10000,1)

        feature_names = ["f{}".format(i) for i in range(len(training_features.T))] #names for X_variable
        df_train = pd.DataFrame(data=training_features, columns=feature_names)
        df_train['labs'] = training_labels

        df_train.to_csv('MNIST_df_train.csv', index=False) # saving database to csv

        #For test set
        test_features = np.genfromtxt("MNIST-cropped-txt//MNIST-Test-cropped.txt")
        test_features = test_features.reshape(2000,784)

        test_labels = np.genfromtxt("MNIST-cropped-txt//MNIST-Test-Labels-cropped.txt")
        test_labels = test_labels.reshape(2000,1)

        assert len(training_features.T)==len(test_features.T) # asserting dimensions

        feature_names = ["f{}".format(i) for i in range(len(test_features.T))] #names for X_variable
        df_test = pd.DataFrame(data=test_features, columns=feature_names)
        df_test['labs'] = test_labels

        df_test.to_csv('MNIST_df_test.csv', index=False) # saving database to csv
        #return df_train, df_test

def ShowImage(df_digits,row):

    """shows image of given row

    parameters
    ==========
    df_digits : (pd.DataFrame) dataframe with digits (MNIST)
    row : (int) which row to show iamge from
    """
    df_temp=df_digits.drop(labels="labs",axis=1)
    plt.imshow(np.array(df_temp.iloc[row:row+1]).reshape(28,28).T, cmap="Greys")
    plt.show()
# %%

def label_split(df):
    """ Splits MNIST into X and Y

    Parameters
    ==========
    pandas dataframe

    Returns
    =======
    x, y : numpy array where x = features, y = labels"""

    x = np.array(df.drop('labs',axis=1))
    y = np.array(df['labs'])
    return x, y

def kNearestClassifier(x,y,value_to_predict,k=1):

    ones = np.ones(x.shape[0])
    data = np.array(x)

    subtractor = np.outer(ones,value_to_predict)

    matrix = data - subtractor

    dist_before_sq = np.diag(np.inner(matrix,matrix)) # distance before squareing
    dist = np.sqrt(dist_before_sq) # all distances

    sorter = np.argsort(dist)

    sorted_y = y[sorter] # labels sorted according to distance
    counts = np.bincount(sorted_y[:k-1])
    return np.argmax(counts)
