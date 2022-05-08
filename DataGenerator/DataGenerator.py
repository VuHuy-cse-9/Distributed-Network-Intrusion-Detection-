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

def convert_to_int_value(df, feature_names):
    for feature_name in feature_names:
        if feature_name == "protocol_type":
            datas = df[feature_name].to_numpy()
            datas[datas == "icmp"] = - 1
            datas[datas == "tcp"] = 0
            datas[datas == "udp"] = 1
            df[feature_name] = datas
        if feature_name == "service":
            datas = df[feature_name].to_numpy()
            datas[:] = 3
            datas[datas == "ecr_i"] = -2
            datas[datas == "private"] = -1
            datas[datas == "http"] = 0
            datas[datas == "smtp"] = 1
            datas[datas == "pop_3"] = 2
            df[feature_name] = datas
        if feature_name == "flag":
            datas = df[feature_name].to_numpy()
            datas[:] = 3
            datas[datas == "SF"] = -2
            datas[datas == "REJ"] = -1
            datas[datas == "S0"] = 0
            datas[datas == "RSTO"] = 1
            datas[datas == "RSTR"] = 2
            df[feature_name] = datas
    return df

def log_transform(df, skew_features):
    for feature_name in skew_features:
        data = df[feature_name].to_numpy()
        data[data > 0] = np.log(data[data > 0])
        df[feature_name] = data
    return df

def get_local_test_dataset(
                nodeid,
                path, 
                is_log_transform, 
                option):
    df = loadFile(path)
    if option == "integer":
        df = convert_to_int_value(df, hyper.category_features)
    elif option == "remove": 
        for feature in hyper.category_features:
            del df[feature]
    #ln transform skew features:
    if is_log_transform == "True":
        df = log_transform(df, hyper.skew_features)
    feature_names = list(df.columns)
    
    label = df["label"].to_numpy()
    del df["label"]
    data = np.array(df.to_numpy(), dtype=np.float64)
   
    data_normal, label_normal = data[label == "normal."], label[label == "normal."]
    data_neptune, label_neptune = data[label == "neptune."], label[label =="neptune."]
    data_snmpgetattack, label_snmpgetattack = data[label == "snmpgetattack."], label[label == "snmpgetattack."]
    data_mailbomb, label_mailbomb= data[label == "mailbomb."], label[label == "mailbomb."]
    data_smurf, label_smurf = data[label == "smurf."], label[label == "smurf."]
    
    X = np.concatenate((data_normal, data_neptune, data_snmpgetattack, data_mailbomb, data_smurf), axis=0)
    Y = np.concatenate((label_normal, label_neptune, label_snmpgetattack, label_mailbomb, label_smurf))
    
    Y, labels_to_index, labels = _convert_attack_to_index(Y)
    
    #Shuffle
    array_index = np.arange(X.shape[0])
    np.random.shuffle(array_index)
    X = X[array_index]
    Y = Y[array_index]
    
    return X, Y, labels_to_index, labels, feature_names 
        
def get_local_train_dataset(
                nodeid,
                path, 
                is_log_transform, 
                option): #For Global training
    df = loadFile(path)
    if option == "integer":
        df = convert_to_int_value(df, hyper.category_features)
    elif option == "remove": 
        for feature in hyper.category_features:
            del df[feature]
    #ln transform skew features:
    if is_log_transform == True:
        df = log_transform(df, hyper.skew_features)
    feature_names = list(df.columns)
    
    labels = df["label"].to_numpy()
    del df["label"]
    datas = np.array(df.to_numpy(), dtype=np.float64)
    
    X, Y = None, None
    for attack_name in hyper.attack_global_name:
        data, label = datas[labels == attack_name], labels[labels == attack_name]
        data, label = get_random(data, label, hyper.limit[nodeid][attack_name])  
        if X is None:
            X = data
            Y = label
        else:
            X = np.concatenate((X, data), axis=0)
            Y = np.concatenate((Y, label))
            
    Y, labels_to_index, labels = _convert_attack_to_index(Y)
    
    #Shuffle
    array_index = np.arange(X.shape[0])
    np.random.shuffle(array_index)
    X = X[array_index]
    Y = Y[array_index]
    
    return X, Y, labels_to_index, labels, feature_names

def get_part_local_train_dataset(nodeid, path, is_log_transform, option):
    X_train, Y_train, labels_idx, labels, feature_names = \
        get_local_train_dataset(nodeid, path, is_log_transform, option)
    X_train_normal = X_train[Y_train == 1][:1000, :]
    X_train_1 = X_train[Y_train == -1][:500, :]
    X_train_2 = X_train[Y_train == -2][:500, :]
    X_train_3 = X_train[Y_train == -3][:500, :]
    X_train_4 = X_train[Y_train == -4][:500, :]
    Y_train_normal = Y_train[Y_train == 1][:1000]
    Y_train_1 = Y_train[Y_train == -1][:500]
    Y_train_2 = Y_train[Y_train == -2][:500]
    Y_train_3 = Y_train[Y_train == -3][:500]
    Y_train_4 = Y_train[Y_train == -4][:500]
    X_train = np.concatenate((X_train_normal, X_train_1, X_train_2, X_train_3, X_train_4), axis=0)
    Y_train = np.concatenate((Y_train_normal, Y_train_1, Y_train_2, Y_train_3, Y_train_4), axis=0)
    array_index = np.arange(len(Y_train))
    np.random.shuffle(array_index)
    X_train = X_train[array_index,:]
    Y_train = Y_train[array_index]
    return X_train, Y_train, labels_idx, labels, feature_names
        
def get_full_data(path, is_log_transform, option):                 
    df = loadFile(path)
    #Process Category feature
    if option == "integer":
        df = convert_to_int_value(df, hyper.category_features)
    elif option == "remove": #Remove all category data
        for feature in hyper.category_features:
            del df[feature]
    #ln transform skew features:
    if is_log_transform == True:
        df = log_transform(df, hyper.skew_features)
    feature_names = list(df.columns)
    Y = df["label"].to_numpy()
    del df["label"]
    X = np.array(df.to_numpy(), dtype=np.float64)
    Y, labels_to_index, labels = _convert_symbol_to_index(Y)
    return X, Y, labels_to_index, labels, feature_names

def get_part_data(path, is_log_transform, option):
    X_train, Y_train, labels_idx, labels, feature_names = \
        get_full_data(path, is_log_transform, option)
    X_train_normal = X_train[Y_train == 1][:1000, :]
    X_train_1 = X_train[Y_train == -1][:500, :]
    X_train_2 = X_train[Y_train == -2][:500, :]
    X_train_3 = X_train[Y_train == -3][:500, :]
    X_train_4 = X_train[Y_train == -4][:500, :]
    Y_train_normal = Y_train[Y_train == 1][:1000]
    Y_train_1 = Y_train[Y_train == -1][:500]
    Y_train_2 = Y_train[Y_train == -2][:500]
    Y_train_3 = Y_train[Y_train == -3][:500]
    Y_train_4 = Y_train[Y_train == -4][:500]
    X_train = np.concatenate((X_train_normal, X_train_1, X_train_2, X_train_3, X_train_4), axis=0)
    Y_train = np.concatenate((Y_train_normal, Y_train_1, Y_train_2, Y_train_3, Y_train_4), axis=0)
    array_index = np.arange(len(Y_train))
    np.random.shuffle(array_index)
    X_train = X_train[array_index,:]
    Y_train = Y_train[array_index]
    return X_train, Y_train, labels_idx, labels, feature_names

def get_random(data, labels, limit):
        array_index = np.arange(data.shape[0])
        np.random.shuffle(array_index)
        data = data[array_index[:limit],:]
        labels = labels[array_index[:limit]]
        return data, labels
    
def _convert_attack_to_index(Y):
    """ Function to convert labels to index, which value 1 for normal, <= -1 for attack
    args Y (N_samples, )
    """
    labels = np.unique(Y)
    labels_to_index = np.zeros(labels.shape)
    for index, label in enumerate(hyper.attack_global_train):
        Y[Y == label] = hyper.attack_global_train[label]
        labels_to_index[index] = hyper.attack_global_train[label]
    print(f"unique: {np.unique(Y)}")
    return Y, labels_to_index, labels

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
    X0 = np.random.normal(loc=0.0, scale=5.0, size=(N_train, N_attribute))
    X1 = np.random.normal(loc=6.0, scale=4.0, size=(N_train, N_attribute))
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