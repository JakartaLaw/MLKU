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

def BinaryRespons(df_input, val_a, val_b):
    """ Removes all rows where repons variable is not val_a or val_b

    Parameters
    ==========
    df_input : (pd.DataFrame)
    val_a, val_b : (integers) allowed values of respons variables

    Returns
    =======
    df : (pd.DataFrame) dataframe where repons variable is only val_a or val_b

    """
    df = df_input.loc[(df_input['labs']==val_a)|(df_input['labs']==val_b)]
    return df

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
    """ Classifies
    """

    ones = np.ones(x.shape[0])
    data = np.array(x)

    subtractor = np.outer(ones,value_to_predict)

    matrix = data - subtractor

    dist_before_sq = np.diag(np.inner(matrix,matrix)) # distance before squareing
    dist = np.sqrt(dist_before_sq) # all distances

    sorter = np.argsort(dist)

    sorted_y = y[sorter] # labels sorted according to distance
    counts = np.bincount(sorted_y[:k])
    return np.argmax(counts)

def evaluation_kNearest(df):
    labs = np.array(df['labs'])
    preds = np.array(df['prediction'])

    diff = abs(labs-preds)

    def dummy_generator(x):
        if x==0:
            return 0
        else:
            return 1

    dummy = [dummy_generator(x) for x in diff]
    return sum(dummy)/len(diff)

def ValidationSplit(df_input, frac_of_training_obs):

    """ Splits dataset into training and  validation dataset

    Parameters
    ==========
    df_input : (pd.DataFrame) raw training dataset
    frac_of_training_obs : (float) a float in the interval 0 to 1

    Returns
    =======
    df_training, df_validation
    """

    s = df_input.shape[0] # number of cases
    train_up_bound = int(s*frac_of_training_obs)

    df_train = df_input.iloc[:train_up_bound]
    df_validation = df_input.iloc[train_up_bound:]

    return df_train, df_validation

def DevelopData(df_input, num_rows):

    """ For developing purposes
    """

    return df_input.iloc[0:num_rows]

def OptimalK(k_list, k_pred_list):

    """ Finding Optimal K

    Parameters
    ==========
    k_list = list of k
    k_pred_list = list of prediction errer from kNearestClassifier

    Returns
    =======
    K (int) with best performance

    """

    temp_list = np.array(k_pred_list) - np.array([0.0000001*i for i in range(len(k_list))]) # punish complixity
    opt_k = k_list[np.argmin(temp_list)]
    error_of_opt_k = k_pred_list[np.argmin(temp_list)]
    return opt_k, error_of_opt_k
