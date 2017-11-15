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

        #For test set
        test_features = np.genfromtxt("MNIST-cropped-txt//MNIST-Test-cropped.txt")
        test_features = test_features.reshape(2000,784)

        test_labels = np.genfromtxt("MNIST-cropped-txt//MNIST-Test-Labels-cropped.txt")
        test_labels = test_labels.reshape(2000,1)

        assert len(training_features.T)==len(test_features.T) # asserting dimensions

        feature_names = ["f{}".format(i) for i in range(len(test_features.T))] #names for X_variable
        df_test = pd.DataFrame(data=test_features, columns=feature_names)
        df_test['labs'] = test_labels

        return df_train, df_test

def ShowImage(df_digits,row):

    """shows image of given row

    parameters
    ==========
    df_digits : (pd.DataFrame) dataframe with digits (MNIST)
    row : (int) which row to show iamge from
    """
    df_temp=df_digits.drop(labels="labels",axis=1)
    plt.imshow(np.array(df_temp.iloc[row:row+1]).reshape(28,28).T, cmap="Greys")
    plt.show()
# %%


df = pd.DataFrame({'a':[1,2,3,4],'b':[3,4,5,6],'c':[1,1,7,8]})

df.head()

test = np.array([1,2,3])

ones = np.ones(df.shape[0])

data = np.array(df)
subtractor = np.outer(ones,test)

y = data - subtractor


y
d_before_sq = np.diag(np.inner(y,y)) # distance before squareing
d = np.sqrt(d_before_sq) # all distances
d


A = np.array([5, 2, 3, 4])
B = np.array([9, 7, 8, 6])
print("lars")
C = np.array([A,B])

print(C)
C.T
D = np.argsort(C[:1])
print(D)
C[:1]

a = np.argsort(A)
print(a)
print(B)

sorted = B[a]

print(sorted)
