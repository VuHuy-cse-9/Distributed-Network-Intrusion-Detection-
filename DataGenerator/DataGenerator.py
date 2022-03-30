import pandas as pd
import numpy as np
from DataGenerator.convertdata import convert
import hyper

def loadFile(csvPathFile):
    df = pd.read_csv (csvPathFile) # load file using pandas
    return df

def convert_to_one_hot(df, feature_names):
    for feature in feature_names:
        labels, data_one_hots = convert(df[feature].to_numpy())
        data_one_hots = np.transpose(data_one_hots)
        for label, data_one_hot in zip(labels, data_one_hots):
            df[feature + "_" + str(label)] = data_one_hot
        del df[feature]
        #print(f">> feature: {feature} to {labels}")
    return df
        
def get_data(
            path, 
            category_features = None,     #Name of category to be remove
            option = 0):                  #Option to process category data
    df = loadFile(path)
    print(f"shape: {df.to_numpy().shape}")
    print(f"Option: {option}")
    if category_features != None:
        if option == 0: #One hot
            df = convert_to_one_hot(df, category_features)
        else: #Remove all category data
            for feature in category_features:
                del df[feature]
    Y = df["label"].to_numpy()
    del df["label"]
    X = df.to_numpy()
    Y, labels_to_index, labels = _convert_symbol_to_index(Y)
    return X, Y, labels_to_index, labels

def _convert_symbol_to_index(Y):
    """ Function to convert labels to index, which value 1 for normal, <= -1 for attack
    args Y (N_samples, )
    """
    labels = np.unique(Y)
    labels_to_index = np.zeros(labels.shape)
    for index, label in zip(np.arange(len(labels)), labels):
        if label == "normal.":
            Y[Y == label] = 1
            labels_to_index[index] = 1
        elif label in hyper.dos:
            Y[Y == label] = -1
            labels_to_index[index] = -1
        elif label in hyper.r2l:
            Y[Y == label] = -2
            labels_to_index[index] = -2
        elif label in hyper.u2r:
            Y[Y == label] = -3
            labels_to_index[index] = -3
        elif label in hyper.prob:
            Y[Y == label] = -4
            labels_to_index[index] = -4
        else:
            Y[Y == label] = -5
            labels_to_index[index] = -5
    return Y, labels_to_index, labels


def load_fake_data(N_train, N_test, N_attribute):
    #train dataset
    X0 = np.random.normal(loc=2.0, scale=1.0, size=(N_train, N_attribute))
    X1 = np.random.normal(loc=5.0, scale=1.0, size=(N_train, N_attribute))
    X_train = np.concatenate((X0, X1))
    Y_train = np.concatenate((np.ones((N_train, ), dtype=np.int32),
                        -1*np.ones((N_train, ), dtype=np.int32)))
    

    shuffle_train_index = np.arange(2*N_train)
    np.random.shuffle(shuffle_train_index)
    X_train  = X_train[shuffle_train_index]
    Y_train = Y_train[shuffle_train_index]

    

    #Test dataset
    X0 = np.random.normal(loc=2.0, scale=0.3, size=(N_test, N_attribute))
    X1 = np.random.normal(loc=4.0, scale=0.3, size=(N_test, N_attribute))
    X_test = np.concatenate((X0, X1), axis=0)
    Y_test = np.concatenate((np.ones((N_test, ), dtype=np.int32), 
                            -1*np.ones((N_test, ), dtype=np.int32)))
    
    shuffle_test_index = np.arange(2*N_test)
    np.random.shuffle(shuffle_test_index)
    X_test  = X_test[shuffle_test_index]
    Y_test = Y_test[shuffle_test_index]
    
    
    if X_train.shape != (2*N_train, N_attribute) or \
        Y_train.shape != (2*N_train, ):
            raise Exception(f"X_train, or Y_train has changed their shape: {X_train.shape}, {Y_train.shape}")
        
    if X_test.shape != (2*N_test, N_attribute) or \
        Y_test.shape != (2*N_test, ):
            raise Exception(f"X_test, or Y_test has changed their shape: {X_test.shape}, {Y_test.shape}")
        
    return X_train, X_test, Y_train, Y_test

def split_train_test(X, y, ratio):
    """Split dataset into train, testset 

    Args:
        X (_N_samples, N_features_): Samples_
        y (_N_samples_): Labels
        ratio: N_train / N_samples
    """
    if X.shape[0] != y.shape[0]:
        raise Exception(f"Number of sample in X not the same in y: {X.shape[0]}, {y.shape[0]}")
    
    if ratio >= 1.0:
        raise Exception(f"Ratio cannot larger than 1.0")
    
    N_samples = X.shape[0]
    N_train = int(ratio * N_samples)
    shuffle_index = np.arange(N_samples)
    np.random.shuffle(shuffle_index)
    train_index = shuffle_index[:N_train]
    test_index = shuffle_index[N_train:]
    
    X_train, y_train = np.squeeze(X[train_index]), np.squeeze(y[train_index])
    X_test, y_test = np.squeeze(X[test_index]), np.squeeze(y[test_index])
    
    if X_train.shape != (N_train, X.shape[1]) or y_train.shape != (N_train, ):
        raise Exception(f"Shape of X_train or y_train is not right, {X_train.shape}, {y_train.shape}")
    
    if X_test.shape != (N_samples - N_train, X.shape[1]) or y_test.shape != (N_samples - N_train, ):
        raise Exception(f"Shape of X_test or y_test is not right, {X_test.shape}, {y_test.shape}")
    
    return X_train, X_test, y_train, y_test
